//! PyO3 bindings for Soma — exposes Graph, Study, and Filter to Python.
//!
//! Bridges Python Filter classes to the Rust Filter trait, converts
//! between Python lists/dicts and Soma Values, and wraps the StudyRunner
//! for hyperparameter optimization from Python.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyBytes, PyDict, PyList};
use std::collections::HashMap;
use std::sync::Arc;

use somatize_compiler::{CompileMode, compile};
use somatize_core::cache::CacheKey;
use somatize_core::error::{Result as SomaResult, SomaError};
use somatize_core::event::MetricRecord;
use somatize_core::filter::{Filter, FilterKind, FilterMeta, StreamMode};
use somatize_core::graph::{Edge, Graph, Node};
use somatize_core::search::{Scale, SearchDimension, SearchSpace};
use somatize_core::study::{Direction, Objective, SearchStrategy, Study};
use somatize_core::value::Value;
use somatize_runtime::EventBus;
use somatize_runtime::cache::{LocalCache, MemoryCache, TieredCache};
use somatize_runtime::executor::{self, Context, GraphInfo};
use somatize_runtime::executors::study::{FnTrialExecutor, StudyRunner, TrialOutcome};
use somatize_runtime::filter_library::FilterLibrary;
use somatize_runtime::runner::{LocalRunner, Runner};
use somatize_runtime::sampler::{BayesianSampler, GridSampler, RandomSampler, Sampler};

fn soma_err_to_py(e: SomaError) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

fn py_err_to_soma(e: PyErr) -> SomaError {
    SomaError::Other(e.to_string())
}

// ── Value conversion ──

fn py_to_value(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if let Ok(lists) = obj.extract::<Vec<Vec<f64>>>() {
        let rows = lists.len();
        let cols = if rows > 0 { lists[0].len() } else { 0 };
        let flat: Vec<f64> = lists.into_iter().flatten().collect();
        return Ok(Value::tensor(flat, vec![rows, cols]));
    }

    if let Ok(arr) = obj.extract::<Vec<f64>>() {
        let len = arr.len();
        return Ok(Value::tensor(arr, vec![len]));
    }

    if obj.is_instance_of::<PyDict>() {
        let pickle = py.import("pickle")?;
        let data: Vec<u8> = pickle.call_method1("dumps", (obj, 5i32))?.extract()?;
        return Ok(Value::object(data));
    }

    if let Ok(s) = obj.extract::<String>()
        && let Ok(val) = serde_json::from_str(&s)
    {
        return Ok(Value::json(val));
    }

    Err(PyRuntimeError::new_err(
        "Cannot convert Python object to Value. Expected list, 2D list, dict, or JSON string.",
    ))
}

fn value_to_py(py: Python<'_>, val: &Value) -> PyResult<PyObject> {
    match val {
        Value::Tensor { values, shape } => {
            if shape.len() == 2 {
                let rows = shape[0];
                let cols = shape[1];
                let result = PyList::empty(py);
                for r in 0..rows {
                    let row: Vec<f64> = values[r * cols..(r + 1) * cols].to_vec();
                    result.append(row)?;
                }
                Ok(result.into_any().unbind())
            } else {
                Ok(values.as_slice().into_pyobject(py)?.into_any().unbind())
            }
        }
        Value::Json(v) => {
            let json_str = v.to_string();
            let json_mod = py.import("json")?;
            let obj = json_mod.call_method1("loads", (json_str,))?;
            Ok(obj.unbind())
        }
        Value::Object(data) => {
            let pickle = py.import("pickle")?;
            let py_bytes = PyBytes::new(py, data.as_slice());
            let obj = pickle.call_method1("loads", (py_bytes,))?;
            Ok(obj.unbind())
        }
        Value::Bytes(b) => Ok(b.as_slice().into_pyobject(py)?.into_any().unbind()),
        Value::Empty => Ok(py.None()),
        _ => Ok(py.None()),
    }
}

// ── Python Filter wrapper ──

struct PyFilterBridge {
    py_obj: PyObject,
    name: String,
    config_hash_val: CacheKey,
    /// cloudpickle.dumps() bytes — serializes the full object (bytecode + closures + deps).
    pickled_bytes: Vec<u8>,
    /// Full module source code (imports + classes + helpers) for introspection by Nous agents.
    source: String,
    /// Pip requirements detected from the filter's imports.
    requirements: Vec<String>,
    /// Whether this filter is trainable (has meaningful fit()).
    trainable: bool,
}

impl PyFilterBridge {
    fn new(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let name: String = obj.get_type().name()?.to_string();

        // Build config hash from public attributes only.
        let dict = obj.getattr("__dict__")?;
        let dict_ref = dict.downcast::<pyo3::types::PyDict>()?;
        let params = pyo3::types::PyDict::new(py);
        for (key, value) in dict_ref.iter() {
            let k: String = key.extract()?;
            if !k.starts_with('_') {
                params.set_item(key, value)?;
            }
        }

        let json_mod = py.import("json")?;
        let dict_sorted = json_mod.call_method(
            "dumps",
            (params,),
            Some(&[("sort_keys", true)].into_py_dict(py)?),
        )?;
        let params_json: String = dict_sorted.extract()?;
        let config_hash = CacheKey::from_parts(&[name.as_bytes(), params_json.as_bytes()]);

        // Serialize with cloudpickle (like Spark/Dask/Ray).
        // Register the filter's module AND all its transitive local dependencies
        // for by-value pickling. Without this, the worker would need the exact
        // same source tree installed (e.g. src.filters.classifiers, src.utils).
        let cloudpickle = py.import("cloudpickle")?;
        let inspect = py.import("inspect")?;
        let module = inspect.call_method1("getmodule", (obj.get_type(),))?;
        // Python helper: walk module globals, find all non-stdlib imported modules,
        // register them for by-value serialization (transitive).
        py.run(
            c"
import sys, types, sysconfig, site, os, cloudpickle as _cp

# Python 3.10+ exposes the canonical stdlib module name set. Fall back to
# empty frozenset on older versions (combined with other heuristics below).
_STDLIB = getattr(sys, 'stdlib_module_names', frozenset())
_BUILTINS = frozenset(sys.builtin_module_names)
# Never pickle-by-value modules the worker already has installed.
_NEVER = {'soma', 'cloudpickle', 'numpy', 'pandas', 'torch', 'sklearn', 'scipy'}
# Site-packages directories (platform/installer independent: works on
# Debian, RHEL/Fedora with lib64, Windows, conda, virtualenvs, --user installs).
_site_dirs = {sysconfig.get_paths().get('purelib'), sysconfig.get_paths().get('platlib')}
_site_dirs.add(getattr(site, 'getusersitepackages', lambda: None)())
_SITE_PREFIXES = tuple(
    os.path.realpath(p) + os.sep
    for p in _site_dirs
    if p
)

def _is_stdlib_or_installed(mod):
    name = getattr(mod, '__name__', '') or ''
    top = name.split('.')[0]
    if top in _STDLIB or top in _BUILTINS or top in _NEVER:
        return True
    f = getattr(mod, '__file__', None)
    if f is None:
        # C extension or frozen without __file__: treat as stdlib/builtin.
        return True
    # Python 3.11+ frozen stdlib modules: __file__ == '<frozen io>' etc.
    if f.startswith('<'):
        return True
    # Authoritative site-packages check via sysconfig (handles lib64,
    # virtualenvs, conda). Fall back to substring match for pathological
    # cases where realpath resolution differs.
    rf = os.path.realpath(f)
    if _SITE_PREFIXES and rf.startswith(_SITE_PREFIXES):
        return True
    if 'site-packages' in f or 'dist-packages' in f:
        return True
    return False

def _register_transitive(mod, visited=None):
    if visited is None:
        visited = set()
    name = getattr(mod, '__name__', '')
    if not name or name in visited or name == '__main__':
        return
    visited.add(name)
    if _is_stdlib_or_installed(mod):
        return
    _cp.register_pickle_by_value(mod)
    # Walk globals for imported modules
    for v in vars(mod).values():
        if isinstance(v, types.ModuleType):
            _register_transitive(v, visited)
        elif isinstance(v, type):
            m = sys.modules.get(v.__module__)
            if m and m.__name__ not in visited:
                _register_transitive(m, visited)

if _soma_module is not None:
    _register_transitive(_soma_module)
",
            Some(&[("_soma_module", &module)].into_py_dict(py)?),
            None,
        )?;
        let pickled = cloudpickle.call_method1("dumps", (obj,))?;
        let pickled_bytes: Vec<u8> = pickled.extract()?;
        let source = if !module.is_none() {
            inspect
                .call_method1("getsource", (&module,))
                .and_then(|s| s.extract::<String>())
                .unwrap_or_default()
        } else {
            inspect
                .call_method1("getsource", (obj.get_type(),))
                .and_then(|s| s.extract::<String>())
                .unwrap_or_default()
        };

        // Detect pip requirements from the filter module's imports.
        // Collects top-level package names of all site-packages imports.
        let reqs_result = py.run(
            c"
import types, sys
_reqs = set()
if _mod is not None:
    for v in vars(_mod).values():
        if isinstance(v, types.ModuleType):
            f = getattr(v, '__file__', '') or ''
            if 'site-packages' in f:
                _reqs.add(v.__name__.split('.')[0])
        elif isinstance(v, type):
            m = sys.modules.get(v.__module__)
            if m:
                f = getattr(m, '__file__', '') or ''
                if 'site-packages' in f:
                    _reqs.add(m.__name__.split('.')[0])
_reqs = sorted(_reqs)
",
            Some(&[("_mod", &module)].into_py_dict(py)?),
            None,
        );
        let requirements: Vec<String> = if reqs_result.is_ok() {
            py.eval(c"_reqs", None, None)
                .and_then(|r| r.extract())
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        // Detect if the filter is trainable (has _kind = "trainable")
        let trainable = obj
            .get_type()
            .getattr("_kind")
            .and_then(|v| v.extract::<String>())
            .map(|s| s != "stateless")
            .unwrap_or(true); // default to trainable if not specified

        Ok(Self {
            py_obj: obj.clone().unbind(),
            name,
            config_hash_val: config_hash,
            pickled_bytes,
            source,
            requirements,
            trainable,
        })
    }
}

impl Filter for PyFilterBridge {
    fn config_hash(&self) -> CacheKey {
        self.config_hash_val.clone()
    }

    fn fit(&self, x: &Value, y: Option<&Value>) -> SomaResult<Value> {
        Python::with_gil(|py| {
            let py_x = value_to_py(py, x).map_err(py_err_to_soma)?;
            let py_y = match y {
                Some(v) => value_to_py(py, v).map_err(py_err_to_soma)?,
                None => py.None(),
            };
            let result = self
                .py_obj
                .call_method1(py, "fit", (py_x, py_y))
                .map_err(|e| SomaError::Other(format!("Python fit() error: {e}")))?;
            let bound = result.bind(py);
            py_to_value(py, bound).map_err(py_err_to_soma)
        })
    }

    fn forward(&self, x: &Value, state: &Value) -> SomaResult<Value> {
        Python::with_gil(|py| {
            let py_x = value_to_py(py, x).map_err(py_err_to_soma)?;
            let py_state = value_to_py(py, state).map_err(py_err_to_soma)?;
            let result = self
                .py_obj
                .call_method1(py, "forward", (py_x, py_state))
                .map_err(|e| SomaError::Other(format!("Python forward() error: {e}")))?;
            let bound = result.bind(py);
            py_to_value(py, bound).map_err(py_err_to_soma)
        })
    }

    fn meta(&self) -> FilterMeta {
        // Read optional meta attributes from the Python class.
        // Users can set: _cacheable = False, _differentiable = False,
        //                _kind = "stateless", _stream_mode = "evolving"
        let (kind, cacheable, differentiable, stream_mode) = Python::with_gil(|py| {
            let obj = self.py_obj.bind(py);

            let cacheable = obj
                .getattr("_cacheable")
                .and_then(|v| v.extract::<bool>())
                .unwrap_or(true);

            let differentiable = obj
                .getattr("_differentiable")
                .and_then(|v| v.extract::<bool>())
                .unwrap_or(false);

            let kind = obj
                .getattr("_kind")
                .and_then(|v| v.extract::<String>())
                .map(|s| match s.as_str() {
                    "stateless" => FilterKind::Stateless,
                    _ => FilterKind::Trainable,
                })
                .unwrap_or(FilterKind::Trainable);

            let stream_mode = obj
                .getattr("_stream_mode")
                .and_then(|v| v.extract::<String>())
                .map(|s| match s.as_str() {
                    "evolving" => StreamMode::Evolving {
                        checkpoint_every: 100,
                    },
                    "barrier" => StreamMode::Barrier,
                    _ => StreamMode::FixedState,
                })
                .unwrap_or(StreamMode::FixedState);

            (kind, cacheable, differentiable, stream_mode)
        });

        FilterMeta {
            name: self.name.clone(),
            kind,
            cacheable,
            differentiable,
            stream_mode,
            distribution: somatize_core::filter::Distribution::Local,
            input_schema: None,
            output_schema: None,
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn composite_fit(
        &self,
        peers: &[(String, std::sync::Arc<dyn Filter>)],
        x: &Value,
        y: Option<&Value>,
    ) -> Option<SomaResult<(Value, std::collections::HashMap<String, Value>)>> {
        Python::with_gil(|py| {
            let obj = self.py_obj.bind(py);

            // Only trigger when the user has overridden composite_fit.
            // The base Filter class does not declare the method, so
            // ``hasattr`` is the simplest "is it overridden?" probe.
            match obj.hasattr("composite_fit") {
                Ok(true) => {}
                _ => return None,
            }

            // Build a Python dict {node_id: filter_py_obj} for the whole block.
            // If any peer isn't a PyFilterBridge (non-Python filter) we bail
            // out so the runner falls back to sequential execution.
            let peers_dict = PyDict::new(py);
            for (node_id, filter) in peers {
                let bridge = filter.as_any().downcast_ref::<PyFilterBridge>()?;
                if let Err(e) = peers_dict.set_item(node_id, bridge.py_obj.clone_ref(py)) {
                    return Some(Err(SomaError::Other(format!(
                        "composite_fit: building peers dict: {e}"
                    ))));
                }
            }

            // Convert x and y for the Python call.
            let py_x = match value_to_py(py, x) {
                Ok(v) => v,
                Err(e) => return Some(Err(py_err_to_soma(e))),
            };
            let py_y = match y {
                Some(v) => match value_to_py(py, v) {
                    Ok(v) => v,
                    Err(e) => return Some(Err(py_err_to_soma(e))),
                },
                None => py.None(),
            };

            let result = match obj.call_method1("composite_fit", (peers_dict, py_x, py_y)) {
                Ok(r) => r,
                Err(e) => {
                    return Some(Err(SomaError::Other(format!(
                        "Python composite_fit() error: {e}"
                    ))));
                }
            };

            // Expected return: (output, {node_id: state}).
            let tuple = match result.downcast::<pyo3::types::PyTuple>() {
                Ok(t) => t.clone(),
                Err(_) => {
                    return Some(Err(SomaError::Other(
                        "composite_fit must return (output, states_dict)".into(),
                    )));
                }
            };
            if tuple.len() != 2 {
                return Some(Err(SomaError::Other(format!(
                    "composite_fit must return a 2-tuple, got length {}",
                    tuple.len()
                ))));
            }
            let output_py = tuple.get_item(0).ok()?;
            let states_py = tuple.get_item(1).ok()?;

            let output = match py_to_value(py, &output_py) {
                Ok(v) => v,
                Err(e) => return Some(Err(py_err_to_soma(e))),
            };

            let states_dict = match states_py.downcast::<PyDict>() {
                Ok(d) => d.clone(),
                Err(_) => {
                    return Some(Err(SomaError::Other(
                        "composite_fit states must be a dict[node_id, state]".into(),
                    )));
                }
            };
            let mut states_map: std::collections::HashMap<String, Value> =
                std::collections::HashMap::new();
            for (k, v) in states_dict.iter() {
                let key: String = match k.extract() {
                    Ok(s) => s,
                    Err(e) => {
                        return Some(Err(SomaError::Other(format!(
                            "composite_fit state key: {e}"
                        ))));
                    }
                };
                let val = match py_to_value(py, &v) {
                    Ok(v) => v,
                    Err(e) => return Some(Err(py_err_to_soma(e))),
                };
                states_map.insert(key, val);
            }

            Some(Ok((output, states_map)))
        })
    }
}

// ── Search dimension parsing ──

fn parse_py_search_dim(_py: Python<'_>, item: &Bound<'_, PyAny>) -> PyResult<SearchDimension> {
    let dict = item.downcast::<PyDict>()?;
    let dtype: String = dict
        .get_item("type")?
        .ok_or_else(|| PyRuntimeError::new_err("missing 'type'"))?
        .extract()?;
    let name: String = dict
        .get_item("name")?
        .ok_or_else(|| PyRuntimeError::new_err("missing 'name'"))?
        .extract()?;

    match dtype.as_str() {
        "float" => {
            let low: f64 = dict.get_item("low")?.unwrap().extract()?;
            let high: f64 = dict.get_item("high")?.unwrap().extract()?;
            let scale_str: String = dict
                .get_item("scale")?
                .map(|s| s.extract().unwrap_or_default())
                .unwrap_or("linear".into());
            let scale = match scale_str.as_str() {
                "log" => Scale::Log,
                "reverse_log" => Scale::ReverseLog,
                _ => Scale::Linear,
            };
            Ok(SearchDimension::Float {
                name,
                low,
                high,
                scale,
                default: None,
            })
        }
        "int" => {
            let low: i64 = dict.get_item("low")?.unwrap().extract()?;
            let high: i64 = dict.get_item("high")?.unwrap().extract()?;
            Ok(SearchDimension::Int {
                name,
                low,
                high,
                scale: Scale::Linear,
            })
        }
        "categorical" => {
            let choices_py = dict.get_item("choices")?.unwrap();
            let choices_list = choices_py.downcast::<PyList>()?;
            let choices: Vec<serde_json::Value> = choices_list
                .iter()
                .map(|c| {
                    if let Ok(s) = c.extract::<String>() {
                        serde_json::json!(s)
                    } else if let Ok(b) = c.extract::<bool>() {
                        serde_json::json!(b)
                    } else if let Ok(i) = c.extract::<i64>() {
                        serde_json::json!(i)
                    } else if let Ok(f) = c.extract::<f64>() {
                        serde_json::json!(f)
                    } else {
                        serde_json::json!(c.to_string())
                    }
                })
                .collect();
            Ok(SearchDimension::Categorical { name, choices })
        }
        _ => Err(PyRuntimeError::new_err(format!(
            "unknown search dim type: {dtype}"
        ))),
    }
}

// ── PyStudy ──

#[pyclass(name = "Study")]
struct PyStudy {
    study: Study,
}

#[pymethods]
impl PyStudy {
    #[new]
    #[pyo3(signature = (name, search_space, strategy, n_trials, objectives, seed=None))]
    fn new(
        _py: Python<'_>,
        name: String,
        search_space: &Bound<'_, PyList>,
        strategy: String,
        n_trials: usize,
        objectives: Vec<(String, String)>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let mut space = SearchSpace::new();
        for item in search_space.iter() {
            let dim = parse_py_search_dim(item.py(), &item)?;
            space.add(dim);
        }

        let strat = match strategy.as_str() {
            "grid" => SearchStrategy::Grid {
                points_per_dim: n_trials,
            },
            "random" => SearchStrategy::Random { n_trials, seed },
            "bayesian" => SearchStrategy::Bayesian {
                n_trials,
                n_startup: (n_trials / 5).max(5),
                seed,
            },
            _ => {
                return Err(PyRuntimeError::new_err(format!(
                    "Unknown strategy: {strategy}. Use 'grid', 'random', or 'bayesian'."
                )));
            }
        };

        let objs: Vec<Objective> = objectives
            .into_iter()
            .map(|(metric, dir)| Objective {
                metric,
                direction: if dir == "minimize" {
                    Direction::Minimize
                } else {
                    Direction::Maximize
                },
            })
            .collect();

        Ok(Self {
            study: Study::new(name, space, strat, objs),
        })
    }

    fn run(&mut self, _py: Python<'_>, executor: &Bound<'_, PyAny>) -> PyResult<()> {
        let bus = Arc::new(EventBus::new(256));
        let runner = StudyRunner::new(bus);
        let executor_obj = executor.clone().unbind();

        let trial_executor = FnTrialExecutor(
            move |params: &HashMap<String, serde_json::Value>| -> SomaResult<TrialOutcome> {
                Python::with_gil(|py| {
                    let py_params = PyDict::new(py);
                    for (k, v) in params {
                        let py_val: PyObject = match v {
                            serde_json::Value::Number(n) => {
                                if let Some(f) = n.as_f64() {
                                    f.into_pyobject(py).unwrap().into_any().unbind()
                                } else {
                                    py.None()
                                }
                            }
                            serde_json::Value::String(s) => {
                                s.into_pyobject(py).unwrap().into_any().unbind()
                            }
                            serde_json::Value::Bool(b) => (*b)
                                .into_pyobject(py)
                                .unwrap()
                                .to_owned()
                                .into_any()
                                .unbind(),
                            _ => v.to_string().into_pyobject(py).unwrap().into_any().unbind(),
                        };
                        py_params.set_item(k, py_val).map_err(py_err_to_soma)?;
                    }

                    let result = executor_obj
                        .call1(py, (py_params,))
                        .map_err(|e| SomaError::Other(format!("Python executor error: {e}")))?;
                    let bound = result.bind(py);
                    let dict = bound
                        .downcast::<PyDict>()
                        .map_err(|_| SomaError::Other("executor must return a dict".into()))?;

                    let mut metrics = Vec::new();
                    for (k, v) in dict.iter() {
                        let name: String = k.extract().map_err(py_err_to_soma)?;
                        let value: f64 = v.extract().map_err(py_err_to_soma)?;
                        metrics.push(MetricRecord {
                            name,
                            value,
                            step: 0,
                            timestamp: chrono::Utc::now(),
                        });
                    }
                    Ok(TrialOutcome::Completed(metrics))
                })
            },
        );

        let mut sampler: Box<dyn Sampler> = match &self.study.strategy {
            SearchStrategy::Grid { points_per_dim } => Box::new(GridSampler::new(*points_per_dim)),
            SearchStrategy::Random { n_trials, seed } => {
                Box::new(RandomSampler::new(*n_trials, *seed))
            }
            SearchStrategy::Bayesian {
                n_trials,
                n_startup,
                seed,
                ..
            } => Box::new(BayesianSampler::new(*n_trials, *n_startup, *seed)),
            _ => return Err(PyRuntimeError::new_err("Unsupported strategy")),
        };

        runner
            .run(&mut self.study, sampler.as_mut(), &trial_executor)
            .map_err(soma_err_to_py)
    }

    #[getter]
    fn best_trial(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        match self.study.best_trial() {
            Some(trial) => {
                let dict = PyDict::new(py);
                dict.set_item("id", &trial.id)?;
                let params_dict = PyDict::new(py);
                for (k, v) in &trial.params {
                    let py_val: PyObject = match v {
                        serde_json::Value::Number(n) => {
                            if let Some(f) = n.as_f64() {
                                f.into_pyobject(py).unwrap().into_any().unbind()
                            } else {
                                py.None()
                            }
                        }
                        serde_json::Value::String(s) => {
                            s.into_pyobject(py).unwrap().into_any().unbind()
                        }
                        serde_json::Value::Bool(b) => (*b)
                            .into_pyobject(py)
                            .unwrap()
                            .to_owned()
                            .into_any()
                            .unbind(),
                        _ => v.to_string().into_pyobject(py).unwrap().into_any().unbind(),
                    };
                    params_dict.set_item(k, py_val)?;
                }
                dict.set_item("params", params_dict)?;
                let metrics_dict = PyDict::new(py);
                for m in &trial.metrics {
                    metrics_dict.set_item(&m.name, m.value)?;
                }
                dict.set_item("metrics", metrics_dict)?;
                Ok(Some(dict.into_any().unbind()))
            }
            None => Ok(None),
        }
    }

    #[getter]
    fn n_trials(&self) -> usize {
        self.study.trials.len()
    }

    #[getter]
    fn progress(&self) -> f64 {
        self.study.progress()
    }
}

// ── PyGraph ──

#[pyclass(name = "Graph")]
struct PyGraph {
    graph: Graph,
    library: FilterLibrary,
    cache: Arc<dyn somatize_core::cache::CacheStore>,
    event_bus: Arc<EventBus>,
    fitted: bool,
    /// Registered remote workers: (address, token, tags).
    workers: Vec<(String, Option<String>, Vec<String>)>,
    /// Coordinator URL + token.
    coordinator: Option<(String, Option<String>)>,
    /// Pickled filter bytes + requirements for remote serialization.
    /// node_id → (cloudpickle bytes, pip requirements)
    pickled_filters: std::collections::HashMap<String, (Vec<u8>, Vec<String>)>,
    /// Module source code per filter for Nous agent introspection/editing.
    /// node_id → full module source (imports + classes + helpers)
    filter_sources: std::collections::HashMap<String, String>,
    /// Optional DataStore for persistent data transport (opt-in, costs storage).
    data_store: Option<Arc<dyn somatize_core::store::DataStore>>,
    /// Whether each filter is trainable (node_id → bool).
    filter_trainable: std::collections::HashMap<String, bool>,
}

impl PyGraph {
    /// Split a Value into batches along the first axis.
    /// For Tensor: splits rows. For Json dict with lists: splits list values.
    #[allow(dead_code)]
    fn split_value_into_batches(value: &Value, batch_size: usize) -> Vec<Value> {
        match value {
            Value::Tensor { values, shape } if !shape.is_empty() && shape[0] > batch_size => {
                let rows = shape[0];
                let row_size: usize = shape[1..].iter().product::<usize>().max(1);
                let mut batches = Vec::new();
                for start in (0..rows).step_by(batch_size) {
                    let end = (start + batch_size).min(rows);
                    let flat_start = start * row_size;
                    let flat_end = end * row_size;
                    let batch_vals = values[flat_start..flat_end].to_vec();
                    let mut batch_shape = shape.clone();
                    batch_shape[0] = end - start;
                    batches.push(Value::tensor(batch_vals, batch_shape));
                }
                batches
            }
            Value::Json(json_val) if json_val.is_object() => {
                let map = json_val.as_object().unwrap();
                let total = map
                    .values()
                    .find_map(|v| v.as_array().map(|a| a.len()))
                    .unwrap_or(0);
                if total <= batch_size {
                    return vec![value.clone()];
                }
                let mut batches = Vec::new();
                for start in (0..total).step_by(batch_size) {
                    let end = (start + batch_size).min(total);
                    let mut batch_map = serde_json::Map::new();
                    for (k, v) in map {
                        if let Some(arr) = v.as_array() {
                            let slice = arr[start..end.min(arr.len())].to_vec();
                            batch_map.insert(k.clone(), serde_json::Value::Array(slice));
                        } else {
                            batch_map.insert(k.clone(), v.clone());
                        }
                    }
                    batches.push(Value::json(serde_json::Value::Object(batch_map)));
                }
                batches
            }
            _ => vec![value.clone()],
        }
    }

    /// Build a transport from the first registered worker (if any).
    fn make_transport(&self) -> Option<Arc<dyn somatize_runtime::runner::Transport>> {
        if self.workers.is_empty() {
            return None;
        }
        let (addr, token, _tags) = &self.workers[0];
        let transport = somatize_worker::WsTransport::new(addr, token.clone());
        Some(Arc::new(transport))
    }

    /// Send a Shutdown message to a worker via WebSocket.
    fn send_shutdown(address: &str, token: Option<&str>, reason: &str) -> PyResult<()> {
        use somatize_worker::protocol::CoordinatorToWorker;

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("tokio: {e}")))?;

        rt.block_on(async {
            let url = if let Some(t) = token {
                format!("{address}/ws?token={t}")
            } else {
                format!("{address}/ws")
            };

            let (mut ws, _) = tokio_tungstenite::connect_async(&url)
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("WS connect: {e}")))?;

            use futures_util::SinkExt;
            use tokio_tungstenite::tungstenite::Message;

            let msg = CoordinatorToWorker::Shutdown {
                reason: reason.to_string(),
            };
            let json = serde_json::to_string(&msg)
                .map_err(|e| PyRuntimeError::new_err(format!("serialize: {e}")))?;

            ws.send(Message::Text(json.into()))
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("WS send: {e}")))?;

            Ok(())
        })
    }

    /// Decide how to transport input data to the worker.
    ///
    /// - DataStore configured → upload to store, return Reference
    /// - Large payload (≥ 10MB) → HTTP bulk upload to worker, return Reference
    /// - Small payload → Inline (current WS behavior)
    fn resolve_transport(
        &self,
        x: &Value,
        addr: &str,
        token: &Option<String>,
    ) -> Result<somatize_worker::protocol::InputSource, PyErr> {
        use somatize_core::cache::CacheKey;
        use somatize_worker::protocol::InputSource;

        // DataStore configured → always use it (user opted in)
        if let Some(store) = &self.data_store {
            let data_bytes = serde_json::to_vec(x).unwrap_or_default();
            let key = CacheKey::hash_data(&data_bytes);
            let data_ref = store.put(&key, x).map_err(soma_err_to_py)?;
            return Ok(InputSource::Reference { data_ref });
        }

        // Estimate payload size
        let size_bytes = serde_json::to_vec(x).map(|v| v.len()).unwrap_or(0);
        if size_bytes >= somatize_core::store::INLINE_THRESHOLD_BYTES {
            let data_ref = self.http_upload(x, addr, token)?;
            return Ok(InputSource::Reference { data_ref });
        }

        Ok(InputSource::Inline { value: x.clone() })
    }

    /// Upload a Value to the worker via HTTP POST /upload.
    /// Runs in a dedicated thread to avoid tokio runtime nesting.
    fn http_upload(
        &self,
        value: &Value,
        addr: &str,
        token: &Option<String>,
    ) -> Result<somatize_core::store::DataRef, PyErr> {
        let http_addr = addr
            .replace("ws://", "http://")
            .replace("wss://", "https://");
        let url = format!("{http_addr}/upload");

        let body = serde_json::to_vec(value)
            .map_err(|e| PyRuntimeError::new_err(format!("json serialize: {e}")))?;

        let token = token.clone();

        // Run blocking HTTP in a dedicated thread to avoid tokio runtime conflicts
        std::thread::spawn(move || {
            let client = reqwest::blocking::Client::new();
            let mut req = client
                .post(&url)
                .header("Content-Type", "application/json")
                .body(body);

            if let Some(t) = &token {
                req = req.query(&[("token", t.as_str())]);
            }

            let resp = req
                .send()
                .map_err(|e| PyRuntimeError::new_err(format!("HTTP upload: {e}")))?;

            if !resp.status().is_success() {
                return Err(PyRuntimeError::new_err(format!(
                    "HTTP upload failed: {}",
                    resp.status()
                )));
            }

            resp.json::<somatize_core::store::DataRef>()
                .map_err(|e| PyRuntimeError::new_err(format!("parse upload response: {e}")))
        })
        .join()
        .map_err(|_| PyRuntimeError::new_err("HTTP upload thread panicked"))?
    }

    /// Resolve an OutputDelivery into a Value.
    ///
    /// Inline → return value directly.
    /// Reference → download from worker via HTTP GET /download.
    fn resolve_output(
        delivery: somatize_worker::protocol::OutputDelivery,
        addr: &str,
        token: &Option<String>,
    ) -> Result<Value, PyErr> {
        use somatize_worker::protocol::OutputDelivery;
        match delivery {
            OutputDelivery::Inline { value } => Ok(value),
            OutputDelivery::Reference { data_ref } => Self::http_download(&data_ref, addr, token),
            _ => Err(PyRuntimeError::new_err("unknown OutputDelivery variant")),
        }
    }

    /// Download a Value from a worker via HTTP GET /download.
    fn http_download(
        data_ref: &somatize_core::store::DataRef,
        addr: &str,
        token: &Option<String>,
    ) -> Result<Value, PyErr> {
        let http_addr = addr
            .replace("ws://", "http://")
            .replace("wss://", "https://");
        let url = format!("{http_addr}/download");

        let ref_json = serde_json::to_string(data_ref)
            .map_err(|e| PyRuntimeError::new_err(format!("serialize data_ref: {e}")))?;

        let token = token.clone();

        // Run blocking HTTP in a dedicated thread to avoid tokio runtime conflicts
        std::thread::spawn(move || {
            let client = reqwest::blocking::Client::new();
            let mut req = client.get(&url).query(&[("ref", &ref_json)]);

            if let Some(t) = &token {
                req = req.query(&[("token", t.as_str())]);
            }

            let resp = req
                .send()
                .map_err(|e| PyRuntimeError::new_err(format!("HTTP download: {e}")))?;

            if !resp.status().is_success() {
                return Err(PyRuntimeError::new_err(format!(
                    "HTTP download failed: {}",
                    resp.status()
                )));
            }

            let bytes = resp
                .bytes()
                .map_err(|e| PyRuntimeError::new_err(format!("read download response: {e}")))?;

            rmp_serde::from_slice(&bytes).or_else(|_| {
                serde_json::from_slice(&bytes)
                    .map_err(|e| PyRuntimeError::new_err(format!("deserialize download: {e}")))
            })
        })
        .join()
        .map_err(|_| PyRuntimeError::new_err("HTTP download thread panicked"))?
    }

    /// Send a plan to a remote worker via WebSocket.
    /// Returns (output, trained_states) — states are non-empty after Fit mode.
    fn dispatch_to_worker(
        &self,
        x: &Value,
        mode: somatize_worker::protocol::ExecutionMode,
    ) -> Result<(Value, std::collections::HashMap<String, Value>), PyErr> {
        use somatize_worker::protocol::*;

        let compile_mode = match &mode {
            ExecutionMode::Fit { .. } => CompileMode::NoCache,
            ExecutionMode::Forward => CompileMode::Inference,
            _ => CompileMode::Inference,
        };

        let compile_result = somatize_compiler::compile(
            &self.graph,
            &self.library,
            compile_mode,
            Some(self.cache.as_ref()),
        )
        .map_err(soma_err_to_py)?;

        // Find the best worker by target tag or first available
        let first_target = self
            .graph
            .nodes
            .iter()
            .find_map(|n| n.target.as_deref())
            .unwrap_or("default");

        let (addr, token) = self
            .workers
            .iter()
            .find(|(_, _, tags)| {
                first_target == "default" || tags.contains(&first_target.to_string())
            })
            .or_else(|| self.workers.first())
            .map(|(a, t, _)| (a.clone(), t.clone()))
            .ok_or_else(|| PyRuntimeError::new_err("no workers available"))?;

        // Serialize filters with cloudpickle bytes so the worker can reconstruct them
        let filters: Vec<SerializedFilter> = self
            .graph
            .nodes
            .iter()
            .filter_map(|node| {
                let (pickled, reqs) = self.pickled_filters.get(&node.id)?;
                let state = self.library.get_state(&node.id).map(|arc| (*arc).clone());
                let trainable = self.filter_trainable.get(&node.id).copied().unwrap_or(true);
                Some(SerializedFilter {
                    node_id: node.id.clone(),
                    pickled_filter: pickled.clone(),
                    state,
                    requirements: reqs.clone(),
                    trainable,
                })
            })
            .collect();

        let input_source = self.resolve_transport(x, &addr, &token)?;
        let plan = SerializedPlan {
            plan_id: somatize_core::util::timestamp_id("remote_plan"),
            plan: compile_result.plan,
            input: Some(input_source),
            filters,
            mode,
            metadata: serde_json::json!({}),
        };

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("tokio: {e}")))?;

        rt.block_on(async {
            let url = if let Some(t) = &token {
                format!("{addr}/ws?token={t}")
            } else {
                format!("{addr}/ws")
            };

            let (mut ws, _) = tokio_tungstenite::connect_async(&url)
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("WS connect: {e}")))?;

            use futures_util::{SinkExt, StreamExt};
            use tokio_tungstenite::tungstenite::Message;

            let msg = CoordinatorToWorker::AssignPlan { plan };
            let json = serde_json::to_string(&msg)
                .map_err(|e| PyRuntimeError::new_err(format!("serialize: {e}")))?;

            ws.send(Message::Text(json.into()))
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("WS send: {e}")))?;

            while let Some(Ok(Message::Text(response))) = ws.next().await {
                if let Ok(result) = serde_json::from_str::<WorkerToCoordinator>(&response) {
                    match result {
                        WorkerToCoordinator::PlanResult { result, .. } => match result {
                            PlanResult::Success { output, states, .. } => {
                                let _ = ws.close(None).await;
                                let value = Self::resolve_output(output, &addr, &token)?;
                                return Ok((value, states));
                            }
                            PlanResult::Failed { error, .. } => {
                                let _ = ws.close(None).await;
                                return Err(PyRuntimeError::new_err(format!("remote: {error}")));
                            }
                        },
                        _ => continue,
                    }
                }
            }

            Err(PyRuntimeError::new_err("worker closed without result"))
        })
    }

    /// Stream data to a worker in chunks via WebSocket Binary frames.
    /// Chunks are processed by StreamExecutor on the worker — no full materialization.
    fn dispatch_streamed(&self, x: &Value, chunk_size: usize) -> Result<Value, PyErr> {
        use somatize_worker::protocol::*;

        let compile_result = somatize_compiler::compile(
            &self.graph,
            &self.library,
            CompileMode::Inference,
            Some(self.cache.as_ref()),
        )
        .map_err(soma_err_to_py)?;

        let first_target = self
            .graph
            .nodes
            .iter()
            .find_map(|n| n.target.as_deref())
            .unwrap_or("default");

        let (addr, token) = self
            .workers
            .iter()
            .find(|(_, _, tags)| {
                first_target == "default" || tags.contains(&first_target.to_string())
            })
            .or_else(|| self.workers.first())
            .map(|(a, t, _)| (a.clone(), t.clone()))
            .ok_or_else(|| PyRuntimeError::new_err("no workers available"))?;

        let filters: Vec<SerializedFilter> = self
            .graph
            .nodes
            .iter()
            .filter_map(|node| {
                let (pickled, reqs) = self.pickled_filters.get(&node.id)?;
                let state = self.library.get_state(&node.id).map(|arc| (*arc).clone());
                let trainable = self.filter_trainable.get(&node.id).copied().unwrap_or(true);
                Some(SerializedFilter {
                    node_id: node.id.clone(),
                    pickled_filter: pickled.clone(),
                    state,
                    requirements: reqs.clone(),
                    trainable,
                })
            })
            .collect();

        // Split Value into chunks
        let chunks = Self::chunk_value(x, chunk_size);
        let total_chunks = chunks.len();
        let stream_id = somatize_core::util::timestamp_id("stream");

        let plan = SerializedPlan {
            plan_id: stream_id.clone(),
            plan: compile_result.plan,
            input: None, // input comes via chunks
            filters,
            mode: ExecutionMode::Forward,
            metadata: serde_json::json!({}),
        };

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("tokio: {e}")))?;

        rt.block_on(async {
            let url = if let Some(t) = &token {
                format!("{addr}/ws?token={t}")
            } else {
                format!("{addr}/ws")
            };

            let (mut ws, _) = tokio_tungstenite::connect_async(&url)
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("WS connect: {e}")))?;

            use futures_util::{SinkExt, StreamExt};
            use tokio_tungstenite::tungstenite::Message;

            // 1. Send StreamBegin
            let begin = StreamMessage::StreamBegin {
                stream_id: stream_id.clone(),
                plan_id: stream_id.clone(),
                total_chunks: Some(total_chunks),
                plan: Box::new(plan),
            };
            let bytes = rmp_serde::to_vec(&begin)
                .map_err(|e| PyRuntimeError::new_err(format!("msgpack: {e}")))?;
            ws.send(Message::Binary(bytes.into()))
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("WS send: {e}")))?;

            // 2. Send chunks, collect ChunkResults as they arrive
            let mut chunk_results: Vec<Value> = Vec::new();

            for (i, chunk) in chunks.into_iter().enumerate() {
                let chunk_msg = StreamMessage::ChunkData {
                    stream_id: stream_id.clone(),
                    chunk_index: i,
                    value: chunk,
                };
                let bytes = rmp_serde::to_vec(&chunk_msg)
                    .map_err(|e| PyRuntimeError::new_err(format!("msgpack: {e}")))?;
                ws.send(Message::Binary(bytes.into()))
                    .await
                    .map_err(|e| PyRuntimeError::new_err(format!("WS send chunk: {e}")))?;

                // Drain ChunkResults that arrived so far
                while let Ok(Some(Ok(Message::Binary(resp)))) =
                    tokio::time::timeout(std::time::Duration::from_millis(1), ws.next()).await
                {
                    if let Ok(StreamMessage::ChunkResult { value, .. }) =
                        rmp_serde::from_slice(&resp)
                    {
                        chunk_results.push(value);
                    }
                }
            }

            // 3. Send StreamEnd
            let end = StreamMessage::StreamEnd {
                stream_id: stream_id.clone(),
            };
            let bytes = rmp_serde::to_vec(&end)
                .map_err(|e| PyRuntimeError::new_err(format!("msgpack: {e}")))?;
            ws.send(Message::Binary(bytes.into()))
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("WS send end: {e}")))?;

            // 4. Drain remaining ChunkResults + wait for StreamComplete
            let mut flush_value: Option<Value> = None;
            while let Some(Ok(msg)) = ws.next().await {
                if let Message::Binary(resp) = msg {
                    match rmp_serde::from_slice::<StreamMessage>(&resp) {
                        Ok(StreamMessage::ChunkResult { value, .. }) => {
                            chunk_results.push(value);
                        }
                        Ok(StreamMessage::StreamComplete { result, .. }) => match result {
                            PlanResult::Success { output, .. } => {
                                let v = Self::resolve_output(output, &addr, &token)?;
                                if !v.is_empty() {
                                    flush_value = Some(v);
                                }
                                break;
                            }
                            PlanResult::Failed { error, .. } => {
                                return Err(PyRuntimeError::new_err(format!(
                                    "stream error: {error}"
                                )));
                            }
                        },
                        _ => {}
                    }
                }
            }

            // 5. Combine: chunk results (progressive) + flush result (barrier)
            if let Some(flushed) = flush_value {
                chunk_results.push(flushed);
            }
            if chunk_results.is_empty() {
                return Ok(Value::Empty);
            }
            if chunk_results.len() == 1 {
                return Ok(chunk_results.into_iter().next().unwrap());
            }
            // Concatenate tensors along first dimension
            somatize_runtime::executors::materialize_buffer(&chunk_results).map_err(soma_err_to_py)
        })
    }

    /// Split a Value into chunks for streaming.
    fn chunk_value(x: &Value, chunk_size: usize) -> Vec<Value> {
        match x {
            Value::Tensor { values, shape } if !values.is_empty() => {
                // Split along first dimension
                let row_size = if shape.len() > 1 {
                    shape[1..].iter().product()
                } else {
                    1
                };
                let n_rows = shape[0];
                let mut chunks = Vec::new();
                for start in (0..n_rows).step_by(chunk_size) {
                    let end = (start + chunk_size).min(n_rows);
                    let flat_start = start * row_size;
                    let flat_end = end * row_size;
                    let chunk_vals = values[flat_start..flat_end].to_vec();
                    let mut chunk_shape = shape.clone();
                    chunk_shape[0] = end - start;
                    chunks.push(Value::tensor(chunk_vals, chunk_shape));
                }
                chunks
            }
            // For non-tensor or small data, single chunk
            _ => vec![x.clone()],
        }
    }
}

#[pymethods]
impl PyGraph {
    /// Create a new Graph.
    ///
    /// Optional keyword arguments:
    ///
    /// * `cache` — cache backend: `"memory"` (default), `"local"`, or `"tiered"`.
    /// * `cache_path` — directory for `"local"` / `"tiered"` cache (required for those).
    /// * `cache_max_bytes` — max bytes for the in-memory LRU (default 1 GB).
    #[new]
    #[pyo3(signature = (*, cache=None, cache_path=None, cache_max_bytes=None))]
    fn new(
        cache: Option<&str>,
        cache_path: Option<String>,
        cache_max_bytes: Option<usize>,
    ) -> PyResult<Self> {
        let max_bytes = cache_max_bytes.unwrap_or(1024 * 1024 * 1024);
        let cache_store: Arc<dyn somatize_core::cache::CacheStore> = match cache.unwrap_or("memory")
        {
            "memory" => Arc::new(MemoryCache::new(max_bytes)),
            "local" => {
                let path = cache_path.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "cache_path is required for cache=\"local\"",
                    )
                })?;
                Arc::new(
                    LocalCache::new(path)
                        .map_err(|e| PyRuntimeError::new_err(format!("cache init: {e}")))?,
                )
            }
            "tiered" => {
                let path = cache_path.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "cache_path is required for cache=\"tiered\"",
                    )
                })?;
                let local = LocalCache::new(path)
                    .map_err(|e| PyRuntimeError::new_err(format!("cache init: {e}")))?;
                Arc::new(TieredCache::memory_and_local(
                    Box::new(MemoryCache::new(max_bytes)),
                    Box::new(local),
                ))
            }
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown cache type: {other:?} (expected \"memory\", \"local\", or \"tiered\")"
                )));
            }
        };

        Ok(Self {
            graph: Graph::new(),
            library: FilterLibrary::new(),
            cache: cache_store,
            event_bus: Arc::new(EventBus::new(256)),
            fitted: false,
            workers: Vec::new(),
            coordinator: None,
            pickled_filters: std::collections::HashMap::new(),
            filter_sources: std::collections::HashMap::new(),
            data_store: None,
            filter_trainable: std::collections::HashMap::new(),
        })
    }

    /// Add a filter node. Returns the node id.
    ///
    /// Usage:
    ///   g.node(MyFilter())                        # auto-named
    ///   g.node("scaler", MyFilter())              # explicit id
    ///   g.node(MyFilter(), target="gpu")           # route to gpu worker
    ///   g.node("model", MyFilter(), target="local") # force local execution
    #[pyo3(signature = (*args, target=None))]
    fn node(
        &mut self,
        py: Python<'_>,
        args: &Bound<'_, pyo3::types::PyTuple>,
        target: Option<String>,
    ) -> PyResult<String> {
        let (node_id, filter_obj) = match args.len() {
            1 => {
                let filter_obj = args.get_item(0)?;
                let class_name = filter_obj
                    .getattr("__class__")?
                    .getattr("__name__")?
                    .extract::<String>()?;
                let snake = to_snake_case(&class_name);
                (snake, filter_obj.to_owned())
            }
            2 => {
                let id = args.get_item(0)?.extract::<String>()?;
                let filter_obj = args.get_item(1)?;
                (id, filter_obj.to_owned())
            }
            n => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "node() takes 1 or 2 positional arguments, got {n}"
                )));
            }
        };

        let bridge = PyFilterBridge::new(py, &filter_obj)?;

        // Handle duplicate node ids
        let actual_id = if self.graph.node(&node_id).is_some() {
            let mut i = 2;
            loop {
                let candidate = format!("{node_id}_{i}");
                if self.graph.node(&candidate).is_none() {
                    break candidate;
                }
                i += 1;
            }
        } else {
            node_id.clone()
        };

        let mut node = Node::filter_with_id(&actual_id, &bridge.name);
        if let Some(t) = target {
            node = node.with_target(t);
        }
        self.graph.add_node(node);

        // Store pickled bytes + requirements for remote execution, source for Nous
        self.pickled_filters.insert(
            actual_id.clone(),
            (bridge.pickled_bytes.clone(), bridge.requirements.clone()),
        );
        self.filter_sources
            .insert(actual_id.clone(), bridge.source.clone());
        self.filter_trainable
            .insert(actual_id.clone(), bridge.trainable);
        self.library.register(actual_id.clone(), Box::new(bridge));

        Ok(actual_id)
    }

    /// Connect two nodes with a data edge.
    fn edge(&mut self, source: String, target: String) {
        let id = format!("e_{}", self.graph.edges.len());
        self.graph.add_edge(Edge::data(id, source, target));
    }

    /// Alias for edge().
    fn connect(&mut self, source: String, target: String) {
        self.edge(source, target);
    }

    /// Fit all trainable filters in topological order.
    ///
    /// If `batch_size` is set, the input is split into batches and each batch
    /// is processed through the entire pipeline (encoder → classifier) before
    /// moving to the next. This keeps memory bounded.
    ///
    /// If workers are registered and no node forces local, training is
    /// dispatched to a remote worker.
    #[pyo3(signature = (x, y=None, batch_size=None, mode="inference"))]
    fn fit(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, pyo3::types::PyAny>,
        y: Option<&Bound<'_, pyo3::types::PyAny>>,
        batch_size: Option<usize>,
        mode: &str,
    ) -> PyResult<()> {
        let x_val = py_to_value(py, x)?;
        let y_val = match y {
            Some(v) => Some(py_to_value(py, v)?),
            None => None,
        };

        // Differentiable mode: compile with CompileMode::Differentiable (which
        // collapses consecutive differentiable filters into a Composite block)
        // and execute via LocalRunner, which delegates the block to the first
        // filter's ``composite_fit``. Gradients flow end-to-end inside the
        // user-provided composite_fit implementation.
        if mode == "differentiable" {
            if !self.workers.is_empty() {
                return Err(PyRuntimeError::new_err(
                    "mode='differentiable' is only supported for local execution \
                     (no workers). Remote differentiable training is not yet implemented.",
                ));
            }
            self.graph.validate().map_err(soma_err_to_py)?;
            let compile_result = compile(
                &self.graph,
                &self.library,
                CompileMode::Differentiable,
                Some(self.cache.as_ref()),
            )
            .map_err(soma_err_to_py)?;
            let runner = LocalRunner;
            let (_output, states) = runner
                .fit(
                    &compile_result.plan,
                    &self.library,
                    self.cache.as_ref(),
                    &self.event_bus,
                    &x_val,
                    y_val.as_ref(),
                )
                .map_err(soma_err_to_py)?;
            // LocalRunner tags composite-produced states with "__state_{id}".
            // Regular sequential states appear under the bare node_id.
            for (key, state) in states {
                let node_id = key.strip_prefix("__state_").unwrap_or(&key).to_string();
                self.library.set_state(node_id, state);
            }
            self.fitted = true;
            return Ok(());
        }
        if mode != "inference" {
            return Err(PyRuntimeError::new_err(format!(
                "Unknown mode={mode:?}. Use 'inference' or 'differentiable'."
            )));
        }

        // Dispatch fit to worker if possible.
        // Release GIL during WS dispatch so worker thread can acquire it for Python execution.
        if !self.workers.is_empty() && self.graph.nodes.iter().all(|n| !n.is_local()) {
            if let Some(bs) = batch_size {
                // Batched fit: dispatch once with batch_size so the worker handles batching
                let mode = somatize_worker::protocol::ExecutionMode::Fit {
                    y: y_val.clone(),
                    batch_size: Some(bs),
                };
                let result = py.allow_threads(|| self.dispatch_to_worker(&x_val, mode));
                let (_output, states) = result?;
                for (node_id, state) in states {
                    self.library.set_state(&node_id, state);
                }
                self.fitted = true;
                return Ok(());
            }

            let mode = somatize_worker::protocol::ExecutionMode::Fit {
                y: y_val.clone(),
                batch_size: None,
            };
            let result = py.allow_threads(|| self.dispatch_to_worker(&x_val, mode));
            let (_output, states) = result?;
            for (node_id, state) in states {
                self.library.set_state(&node_id, state);
            }
            self.fitted = true;
            return Ok(());
        }

        // Local fit
        self.graph.validate().map_err(soma_err_to_py)?;
        let sorted = self.graph.topological_sort().map_err(soma_err_to_py)?;
        let graph_info = GraphInfo::from_graph(&self.graph);
        let run_id = somatize_core::util::timestamp_id("graph_fit");

        let roots = self.graph.roots();
        let mut outputs: std::collections::HashMap<String, Value> =
            std::collections::HashMap::new();
        for root_id in &roots {
            outputs.insert(format!("__input_{root_id}"), x_val.clone());
        }

        for node_id in &sorted {
            let filter = self
                .library
                .get(node_id)
                .ok_or_else(|| PyRuntimeError::new_err(format!("filter not found: {node_id}")))?;

            self.event_bus
                .emit(somatize_core::event::Event::NodeStarted {
                    run_id: run_id.clone(),
                    node_id: node_id.to_string(),
                    kind: filter.meta().kind,
                });

            let preds = graph_info.predecessors(node_id);
            let input = match preds.len() {
                0 => x_val.clone(),
                1 => outputs
                    .get(&preds[0])
                    .cloned()
                    .unwrap_or_else(|| x_val.clone()),
                _ => {
                    let mut merged = serde_json::Map::new();
                    for pred_id in preds {
                        if let Some(val) = outputs.get(pred_id.as_str()) {
                            let json_val =
                                serde_json::to_value(val).unwrap_or(serde_json::Value::Null);
                            merged.insert(pred_id.clone(), json_val);
                        }
                    }
                    Value::json(serde_json::Value::Object(merged))
                }
            };

            let meta = filter.meta();
            let start = std::time::Instant::now();

            let output = if meta.kind == somatize_core::filter::FilterKind::Trainable {
                let data_hash = somatize_core::cache::CacheKey::hash_data(
                    &serde_json::to_vec(&input).unwrap_or_default(),
                );
                let state_key =
                    somatize_core::cache::CacheKey::for_state(&filter.config_hash(), &data_hash);

                let state = if let Ok(Some(cached)) = self.cache.get(&state_key) {
                    cached
                } else {
                    let s = filter.fit(&input, y_val.as_ref()).map_err(soma_err_to_py)?;
                    let _ = self.cache.put(&state_key, &s);
                    s
                };

                let out = filter.forward(&input, &state).map_err(soma_err_to_py)?;
                self.library.set_state(node_id.to_string(), state);
                out
            } else {
                filter
                    .forward(&input, &Value::Empty)
                    .map_err(soma_err_to_py)?
            };

            self.event_bus
                .emit(somatize_core::event::Event::NodeCompleted {
                    run_id: run_id.clone(),
                    node_id: node_id.to_string(),
                    duration: start.elapsed(),
                    output_summary: format!("{output}"),
                });

            outputs.insert(node_id.to_string(), output);
        }

        self.fitted = true;
        Ok(())
    }

    /// Forward data through the compiled graph (inference mode).
    ///
    /// Routing:
    /// - stream=True → chunks sent via WS Binary to StreamExecutor on worker
    /// - No workers → local execution
    /// - Workers + all nodes non-local → entire plan dispatched to worker
    /// - Workers + mixed (some local) → local execution with remote fallback
    #[pyo3(signature = (x, stream=false, chunk_size=1024))]
    fn forward(
        &self,
        py: Python<'_>,
        x: &Bound<'_, pyo3::types::PyAny>,
        stream: bool,
        chunk_size: usize,
    ) -> PyResult<PyObject> {
        if !self.fitted {
            return Err(PyRuntimeError::new_err(
                "graph must be fitted before forward",
            ));
        }
        let x_val = py_to_value(py, x)?;

        // Streaming mode: remote via WS Binary, local via StreamExecutor
        if stream {
            if !self.workers.is_empty() {
                // Release GIL during WS dispatch so worker thread can acquire
                // it for Python execution.
                let output = py.allow_threads(|| self.dispatch_streamed(&x_val, chunk_size))?;
                return value_to_py(py, &output);
            }
            // Local streaming: compile as Stream plan, execute via StreamExecutor
            let compile_result =
                somatize_compiler::compile_stream(&self.graph, &self.library, chunk_size)
                    .map_err(soma_err_to_py)?;

            let graph_info = GraphInfo::from_graph(&self.graph);
            let mut ctx = Context::new(
                self.event_bus.clone(),
                somatize_core::util::timestamp_id("stream_forward"),
            )
            .with_graph_info(graph_info);

            let roots = self.graph.roots();
            if roots.len() == 1 {
                ctx.set(format!("__input_{}", roots[0]), x_val.clone());
            }
            ctx.set("__input__", x_val);

            executor::execute(
                &compile_result.plan,
                &mut ctx,
                &self.library,
                self.cache.as_ref(),
            )
            .map_err(soma_err_to_py)?;

            let leaves = self.graph.leaves();
            let output = if let Some(leaf_id) = leaves.first() {
                ctx.store
                    .remove(*leaf_id)
                    .and_then(|vv| vv.as_value().cloned())
                    .unwrap_or(Value::Empty)
            } else {
                ctx.execution_order
                    .last()
                    .and_then(|id| ctx.store.remove(id))
                    .and_then(|vv| vv.as_value().cloned())
                    .unwrap_or(Value::Empty)
            };

            return value_to_py(py, &output);
        }

        // Dispatch entire plan remotely if workers registered and no node forces local
        if !self.workers.is_empty() && self.graph.nodes.iter().all(|n| !n.is_local()) {
            let (output, _states) = py.allow_threads(|| {
                self.dispatch_to_worker(&x_val, somatize_worker::protocol::ExecutionMode::Forward)
            })?;
            return value_to_py(py, &output);
        }

        // Local execution (with optional remote executor for mixed graphs)
        let compile_result = somatize_compiler::compile(
            &self.graph,
            &self.library,
            CompileMode::Inference,
            Some(self.cache.as_ref()),
        )
        .map_err(soma_err_to_py)?;

        let graph_info = GraphInfo::from_graph(&self.graph);
        let mut ctx = Context::new(
            self.event_bus.clone(),
            somatize_core::util::timestamp_id("graph_forward"),
        )
        .with_graph_info(graph_info);

        if let Some(transport) = self.make_transport() {
            ctx = ctx.with_transport(transport);
        }

        let roots = self.graph.roots();
        if roots.len() == 1 {
            ctx.set(format!("__input_{}", roots[0]), x_val.clone());
        }
        ctx.set("__input__", x_val);

        executor::execute(
            &compile_result.plan,
            &mut ctx,
            &self.library,
            self.cache.as_ref(),
        )
        .map_err(soma_err_to_py)?;

        let leaves = self.graph.leaves();
        let output = if let Some(leaf_id) = leaves.first() {
            ctx.store
                .remove(*leaf_id)
                .and_then(|vv| vv.as_value().cloned())
                .unwrap_or(Value::Empty)
        } else {
            ctx.execution_order
                .last()
                .and_then(|id| ctx.store.remove(id))
                .and_then(|vv| vv.as_value().cloned())
                .unwrap_or(Value::Empty)
        };

        value_to_py(py, &output)
    }

    /// Compile and execute, returning all node outputs as a dict.
    fn run(&self, py: Python<'_>) -> PyResult<PyObject> {
        let compile_result = somatize_compiler::compile(
            &self.graph,
            &self.library,
            CompileMode::NoCache,
            Some(self.cache.as_ref()),
        )
        .map_err(soma_err_to_py)?;

        let graph_info = GraphInfo::from_graph(&self.graph);
        let mut ctx = Context::new(
            self.event_bus.clone(),
            somatize_core::util::timestamp_id("graph_run"),
        )
        .with_graph_info(graph_info);

        if let Some(transport) = self.make_transport() {
            ctx = ctx.with_transport(transport);
        }

        executor::execute(
            &compile_result.plan,
            &mut ctx,
            &self.library,
            self.cache.as_ref(),
        )
        .map_err(soma_err_to_py)?;

        let dict = PyDict::new(py);
        for (k, vv) in &ctx.store {
            if let Some(v) = vv.as_value() {
                dict.set_item(k, value_to_py(py, v)?)?;
            }
        }
        Ok(dict.into_any().unbind())
    }

    /// Compile the graph and return diagnostic information.
    #[pyo3(signature = (mode="inference"))]
    fn compile(&self, py: Python<'_>, mode: &str) -> PyResult<PyObject> {
        let compile_mode = match mode {
            "inference" => CompileMode::Inference,
            "differentiable" => CompileMode::Differentiable,
            "no_cache" => CompileMode::NoCache,
            _ => {
                return Err(PyRuntimeError::new_err(format!(
                    "Unknown mode: {mode}. Use 'inference', 'differentiable', or 'no_cache'."
                )));
            }
        };

        let result = somatize_compiler::compile(
            &self.graph,
            &self.library,
            compile_mode,
            Some(self.cache.as_ref()),
        )
        .map_err(soma_err_to_py)?;

        let dict = PyDict::new(py);
        let summary = result.plan.summary();
        dict.set_item("total_nodes", summary.total_nodes)?;
        dict.set_item("cached_nodes", summary.cached_nodes)?;
        dict.set_item("parallel_branches", summary.parallel_branches)?;

        let diags = PyList::empty(py);
        for d in &result.diagnostics {
            diags.append(format!("{d:?}"))?;
        }
        dict.set_item("diagnostics", diags)?;
        dict.set_item("plan_text", format!("{}", result.plan))?;
        dict.set_item("plan_mermaid", result.plan.to_mermaid())?;

        Ok(dict.into_any().unbind())
    }

    // ── Visualization ──

    /// Render the graph as a Mermaid diagram string.
    fn to_mermaid(&self) -> String {
        self.graph.to_mermaid()
    }

    /// Render the graph as a Graphviz DOT string.
    fn to_graphviz(&self) -> String {
        self.graph.to_graphviz()
    }

    /// Render the graph as an ASCII text tree.
    fn to_text(&self) -> String {
        self.graph.to_text()
    }

    // ── Events ──

    /// Register a Python callback to receive events during execution.
    ///
    /// The callback is called with a dict for each event. Events are
    /// delivered in a background thread; the callback must be thread-safe.
    ///
    /// Usage:
    /// ```python
    /// def on_event(event):
    ///     print(event["event_type"], event.get("node_id", ""))
    /// g.on_event(on_event)
    /// g.fit(data)
    /// ```
    fn on_event(&self, callback: PyObject) -> PyResult<()> {
        let mut rx = self.event_bus.subscribe();
        std::thread::spawn(move || {
            while let Ok(event) = rx.blocking_recv() {
                if let Ok(json_str) = serde_json::to_string(&event) {
                    Python::with_gil(|py| {
                        // Parse JSON string into Python dict via json.loads
                        let json_mod = py.import("json").unwrap();
                        if let Ok(dict) = json_mod.call_method1("loads", (json_str,)) {
                            let _ = callback.call1(py, (dict,));
                        }
                    });
                }
            }
        });
        Ok(())
    }

    // ── Workers ──

    /// Register a remote worker for direct connection (mode B).
    ///
    /// Usage:
    ///   g.add_worker("ws://gpu-0:8080", token="sk-xxx", tags=["gpu"])
    #[pyo3(signature = (address, token=None, tags=None))]
    fn add_worker(&mut self, address: String, token: Option<String>, tags: Option<Vec<String>>) {
        self.workers
            .push((address, token, tags.unwrap_or_default()));
    }

    /// Configure a DataStore for persistent data transport (opt-in).
    ///
    /// When set, large payloads are uploaded to the store and workers read
    /// via DataRef instead of receiving data inline or via HTTP upload.
    ///
    /// Usage:
    ///   g.set_data_store("local", path="/data/soma")
    ///   g.set_data_store("s3", bucket="my-lab", prefix="exp/",
    ///                    endpoint="s3.amazonaws.com",
    ///                    access_key="AK...", secret_key="SK...")
    #[pyo3(signature = (store_type, path=None, bucket=None, prefix=None, endpoint=None, access_key=None, secret_key=None, cache_dir=None))]
    #[allow(clippy::too_many_arguments)]
    fn set_data_store(
        &mut self,
        store_type: String,
        path: Option<String>,
        bucket: Option<String>,
        prefix: Option<String>,
        endpoint: Option<String>,
        access_key: Option<String>,
        secret_key: Option<String>,
        cache_dir: Option<String>,
    ) -> PyResult<()> {
        match store_type.as_str() {
            "local" => {
                let p = path.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("local store requires 'path'")
                })?;
                let store = somatize_core::store::LocalDataStore::new(p);
                self.data_store = Some(Arc::new(store));
                Ok(())
            }
            "s3" => {
                let bucket = bucket.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("s3 store requires 'bucket'")
                })?;
                let prefix = prefix.unwrap_or_default();
                let endpoint = endpoint.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("s3 store requires 'endpoint'")
                })?;
                let ak = access_key
                    .or_else(|| std::env::var("AWS_ACCESS_KEY_ID").ok())
                    .ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(
                            "s3 store requires 'access_key' or AWS_ACCESS_KEY_ID env var",
                        )
                    })?;
                let sk = secret_key
                    .or_else(|| std::env::var("AWS_SECRET_ACCESS_KEY").ok())
                    .ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(
                            "s3 store requires 'secret_key' or AWS_SECRET_ACCESS_KEY env var",
                        )
                    })?;
                let cache = cache_dir.unwrap_or_else(|| {
                    std::env::temp_dir()
                        .join(format!("soma-s3-cache-{bucket}"))
                        .to_string_lossy()
                        .to_string()
                });
                let store =
                    somatize_core::store::S3DataStore::new(bucket, prefix, endpoint, ak, sk, cache)
                        .map_err(soma_err_to_py)?;
                self.data_store = Some(Arc::new(store));
                Ok(())
            }
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown store type: '{other}'. Available: local, s3"
            ))),
        }
    }

    /// Shutdown a specific worker by address.
    ///
    /// Usage:
    ///   g.shutdown_worker("ws://worker:8080")
    ///   g.shutdown_worker("ws://worker:8080", reason="maintenance")
    #[pyo3(signature = (address, reason=None))]
    fn shutdown_worker(&self, address: String, reason: Option<String>) -> PyResult<()> {
        let token = self
            .workers
            .iter()
            .find(|(a, _, _)| *a == address)
            .and_then(|(_, t, _)| t.clone());
        Self::send_shutdown(&address, token.as_deref(), &reason.unwrap_or_default())
    }

    /// Shutdown all registered workers.
    ///
    /// Usage:
    ///   g.shutdown_workers()
    ///   g.shutdown_workers(reason="end of experiment")
    #[pyo3(signature = (reason=None))]
    fn shutdown_workers(&self, reason: Option<String>) -> PyResult<()> {
        let reason = reason.unwrap_or_default();
        for (addr, token, _) in &self.workers {
            if let Err(e) = Self::send_shutdown(addr, token.as_deref(), &reason) {
                eprintln!("Warning: failed to shutdown {addr}: {e}");
            }
        }
        Ok(())
    }

    /// Set a coordinator for auto-discovery (mode C).
    ///
    /// Usage:
    ///   g.set_coordinator("http://coord:9090", token="sk-xxx")
    #[pyo3(signature = (url, token=None))]
    fn set_coordinator(&mut self, url: String, token: Option<String>) {
        self.coordinator = Some((url, token));
    }

    /// List known workers (from add_worker or coordinator).
    ///
    /// Returns a list of dicts with worker info.
    fn workers(&self, py: Python<'_>) -> PyResult<PyObject> {
        let list = PyList::empty(py);

        // Direct workers
        for (addr, _token, tags) in &self.workers {
            let dict = PyDict::new(py);
            dict.set_item("address", addr)?;
            dict.set_item("tags", tags)?;
            dict.set_item("source", "direct")?;
            list.append(dict)?;
        }

        // If coordinator is set, query it for registered workers
        if let Some((url, token)) = &self.coordinator {
            let workers_url = format!("{url}/workers");
            // Synchronous HTTP request (blocking in Python context is fine)
            let client = reqwest::blocking::Client::new();
            let mut request = client.get(&workers_url);
            if let Some(t) = token {
                request = request.query(&[("token", t.as_str())]);
            }
            if let Ok(resp) = request.send()
                && let Ok(text) = resp.text()
            {
                let json_mod = py.import("json")?;
                if let Ok(parsed) = json_mod.call_method1("loads", (text,))
                    && let Ok(items) = parsed.downcast::<PyList>()
                {
                    for item in items.iter() {
                        list.append(item)?;
                    }
                }
            }
        }

        Ok(list.into_any().unbind())
    }

    /// Get the full module source code for a filter node (for Nous agent introspection).
    /// Returns None if the node has no captured source.
    fn filter_source(&self, node_id: String) -> Option<String> {
        self.filter_sources.get(&node_id).cloned()
    }

    /// Get all filter sources as a dict: {node_id: source_code}.
    fn filter_sources_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        for (node_id, source) in &self.filter_sources {
            dict.set_item(node_id, source)?;
        }
        Ok(dict.into_any().unbind())
    }

    /// Number of nodes in the graph.
    fn __len__(&self) -> usize {
        self.graph.nodes.len()
    }

    fn __repr__(&self) -> String {
        let n = self.graph.nodes.len();
        let e = self.graph.edges.len();
        format!(
            "Graph({n} nodes, {e} edges, fitted={fitted})",
            fitted = self.fitted
        )
    }

    fn __str__(&self) -> String {
        self.graph.to_text()
    }
}

fn to_snake_case(name: &str) -> String {
    name.chars()
        .enumerate()
        .fold(String::new(), |mut s, (i, c)| {
            if c.is_uppercase() && i > 0 {
                s.push('_');
            }
            s.push(c.to_ascii_lowercase());
            s
        })
}

// ── PyWorker ──

/// A Soma worker that can be started from Python.
///
/// Usage:
///   from soma import Worker
///   w = Worker(port=8080, tags=["gpu", "training"], token="sk-xxx")
///   w.serve()  # blocks, serving requests
#[pyclass(name = "Worker")]
struct PyWorker {
    port: u16,
    tags: Vec<String>,
    token: Option<String>,
    cpus: Option<usize>,
    memory: Option<u64>,
    gpus: Option<usize>,
    max_concurrent: usize,
    worker_id: Option<String>,
    coordinator: Option<String>,
}

#[allow(clippy::too_many_arguments)]
#[pymethods]
impl PyWorker {
    #[new]
    #[pyo3(signature = (port=8080, tags=None, token=None, cpus=None, memory=None, gpus=None, max_concurrent=4, worker_id=None, coordinator=None))]
    fn new(
        port: u16,
        tags: Option<Vec<String>>,
        token: Option<String>,
        cpus: Option<usize>,
        memory: Option<u64>,
        gpus: Option<usize>,
        max_concurrent: usize,
        worker_id: Option<String>,
        coordinator: Option<String>,
    ) -> Self {
        Self {
            port,
            tags: tags.unwrap_or_default(),
            token,
            cpus,
            memory,
            gpus,
            max_concurrent,
            worker_id,
            coordinator,
        }
    }

    /// Start the worker server (blocking). Releases the GIL so other threads can run.
    fn serve(&self, py: Python<'_>) -> PyResult<()> {
        let port = self.port;
        let tags = self.tags.clone();
        let token = self.token.clone();
        let cpus = self.cpus;
        let memory = self.memory;
        let gpus = self.gpus;
        let max_concurrent = self.max_concurrent;
        let worker_id = self.worker_id.clone();
        let coordinator = self.coordinator.clone();

        // Build the runtime in a new thread; release GIL so other Python threads can run.
        let handle = std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async move {
                // Auto-detect capabilities
                let mut caps = somatize_worker::protocol::Capabilities::detect();
                let limits = somatize_worker::detect::ResourceLimits {
                    max_cpus: cpus,
                    max_memory_bytes: memory,
                    max_gpus: gpus,
                    max_concurrent,
                };
                caps = caps.with_limits(&limits);
                for tag in &tags {
                    if !caps.tags.contains(tag) {
                        caps.tags.push(tag.clone());
                    }
                }

                let id = worker_id.unwrap_or_else(|| {
                    hostname::get()
                        .map(|h| h.to_string_lossy().to_string())
                        .unwrap_or_else(|_| format!("worker_{}", std::process::id()))
                });

                eprintln!("Soma worker '{id}' starting on port {port}");
                eprintln!("Capabilities: {}", caps.summary());

                let worker = somatize_worker::Worker::new(&id, caps.clone());
                let addr = format!("0.0.0.0:{port}");

                // Register with coordinator if configured
                if let Some(coord_url) = &coordinator {
                    let url = format!("{coord_url}/register");
                    let body = serde_json::json!({
                        "worker_id": id,
                        "address": format!("ws://0.0.0.0:{port}"),
                        "capabilities": caps,
                    });
                    let mut req = reqwest::Client::new().post(&url).json(&body);
                    if let Some(t) = &token {
                        req = req.query(&[("token", t.as_str())]);
                    }
                    match req.send().await {
                        Ok(resp) if resp.status().is_success() => {
                            eprintln!("Registered with coordinator at {coord_url}");
                        }
                        Ok(resp) => {
                            eprintln!("Coordinator registration failed: {}", resp.status());
                        }
                        Err(e) => {
                            eprintln!("Could not reach coordinator: {e}");
                        }
                    }
                }

                if let Some(t) = token {
                    eprintln!("Authentication enabled");
                    somatize_worker::serve_worker_authenticated(worker, &addr, &t)
                        .await
                        .unwrap();
                } else {
                    somatize_worker::serve_worker(worker, &addr).await.unwrap();
                }
            });
        });

        // Release GIL while waiting for server thread (allows other Python threads to proceed)
        py.allow_threads(|| {
            handle
                .join()
                .map_err(|_| PyRuntimeError::new_err("worker thread panicked"))
        })?;

        Ok(())
    }

    /// Get the worker info as a dict.
    fn info(&self) -> PyResult<String> {
        let caps = somatize_worker::protocol::Capabilities::detect();
        Ok(caps.summary())
    }

    fn __repr__(&self) -> String {
        format!(
            "Worker(port={}, tags={:?}, auth={})",
            self.port,
            self.tags,
            self.token.is_some()
        )
    }
}

// ── Module ──

#[pymodule]
fn _soma(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGraph>()?;
    m.add_class::<PyStudy>()?;
    m.add_class::<PyWorker>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
