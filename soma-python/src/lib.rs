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
use somatize_core::study::{Direction, Objective, PruningStrategy, SearchStrategy, Study};
use somatize_core::tracking::{GraphSummaryInfo, RunKind, RunState, Tracker};
use somatize_core::value::Value;
use somatize_runtime::EventBus;
use somatize_runtime::cache::{FsActionStore, MemoryCache, TieredCache};
use somatize_runtime::executor::{self, Context, GraphInfo};
use somatize_runtime::executors::study::{
    FnTrialExecutor, StudyRunner, TrialContext, TrialOutcome,
};
use somatize_runtime::filter_library::FilterLibrary;
use somatize_runtime::runner::{LocalRunner, Runner};
use somatize_runtime::sampler::{BayesianSampler, GridSampler, RandomSampler, Sampler};
use somatize_runtime::tracking::{LocalTracker, load_manifest};

fn soma_err_to_py(e: SomaError) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

fn py_err_to_soma(e: PyErr) -> SomaError {
    SomaError::Other(e.to_string())
}

/// The shared persistent cache root: `$SOMA_CACHE_DIR`, else `~/.soma/cache`.
/// Shared across processes and projects so a re-run (or another
/// investigation over the same data) reuses previous compute.
fn default_cache_dir() -> Option<std::path::PathBuf> {
    if let Ok(dir) = std::env::var("SOMA_CACHE_DIR")
        && !dir.is_empty()
    {
        return Some(std::path::PathBuf::from(dir));
    }
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(|home| std::path::PathBuf::from(home).join(".soma").join("cache"))
}

/// Forward with output memoization: key = hash(config + state + input).
/// An unhashable state or input means "uncacheable", not an error.
#[allow(clippy::too_many_arguments)]
fn forward_with_cache(
    cache: &dyn somatize_core::cache::CacheStore,
    filter: &dyn somatize_core::filter::Filter,
    meta: &somatize_core::filter::FilterMeta,
    input: &Value,
    input_hash: Option<&somatize_core::cache::CacheKey>,
    state: &Value,
    origin: &somatize_core::cache::Origin,
    seed: Option<i64>,
) -> somatize_core::error::Result<Value> {
    // Nondeterministic forwards are excluded: serving a recorded result
    // would freeze what the user expects to vary. A seeded run may cache
    // them: the seed is in the key, so results vary across seeds but are
    // stable within one.
    let out_key = if meta.cacheable && (meta.deterministic || seed.is_some()) {
        match (somatize_core::cache::CacheKey::for_value(state), input_hash) {
            (Ok(state_hash), Some(input_hash)) => Some(somatize_runtime::executor::salt_with_seed(
                somatize_core::cache::CacheKey::for_output(
                    &filter.config_hash(),
                    &state_hash,
                    input_hash,
                ),
                seed,
            )),
            _ => None,
        }
    } else {
        None
    };
    if let Some(key) = &out_key
        && let Ok(Some(cached)) = cache.get(key)
    {
        return Ok(cached);
    }
    let start = std::time::Instant::now();
    let output = filter.forward(input, state)?;
    if let Some(key) = &out_key {
        let _ = cache.put_computed(key, &output, origin, start.elapsed(), meta.deterministic);
    }
    Ok(output)
}

/// Convert a JSON value into the closest Python object.
/// Convert a Python object to a JSON value via the json module.
fn py_any_to_json(obj: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    let py = obj.py();
    let json_mod = py.import("json")?;
    let text: String = json_mod.call_method1("dumps", (obj,))?.extract()?;
    serde_json::from_str(&text)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("not JSON-serializable: {e}")))
}

fn json_to_py(py: Python<'_>, v: &serde_json::Value) -> PyObject {
    match v {
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_pyobject(py).unwrap().into_any().unbind()
            } else if let Some(f) = n.as_f64() {
                f.into_pyobject(py).unwrap().into_any().unbind()
            } else {
                py.None()
            }
        }
        serde_json::Value::String(s) => s.into_pyobject(py).unwrap().into_any().unbind(),
        serde_json::Value::Bool(b) => (*b)
            .into_pyobject(py)
            .unwrap()
            .to_owned()
            .into_any()
            .unbind(),
        serde_json::Value::Null => py.None(),
        other => other
            .to_string()
            .into_pyobject(py)
            .unwrap()
            .into_any()
            .unbind(),
    }
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

        // Deterministic identity: qualified class name + canonical
        // config (search defaults merged, sorted keys, typed
        // CacheConfigError on unhashable attrs) + code fingerprint
        // (_cache_version → source hash → cloudpickle ladder).
        // See python/soma/_identity.py.
        let identity_mod = py.import("soma._identity")?;
        let identity = identity_mod.call_method1("filter_identity", (obj,))?;
        let qualname: String = identity.get_item("qualname")?.extract()?;
        let config_json: String = identity.get_item("config_json")?.extract()?;
        let code_fp: String = identity.get_item("code_fp")?.extract()?;

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

        // The environment is part of the identity: the same code under
        // different dependency sets can produce different results.
        let env = requirements.join(",");
        let config_hash = CacheKey::from_parts(&[
            b"soma-id-v2",
            qualname.as_bytes(),
            config_json.as_bytes(),
            code_fp.as_bytes(),
            env.as_bytes(),
        ]);

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
            // Filter forward may return either ``out`` directly (legacy)
            // or ``(out, aux_dict)`` (new DifferentiableFilter contract).
            // Auxiliary signals are runtime-only — drop them at the
            // serialization boundary so cached/remote callers see the
            // same shape as before.
            if let Ok(tuple) = bound.downcast::<pyo3::types::PyTuple>()
                && tuple.len() == 2
                && tuple
                    .get_item(1)
                    .is_ok_and(|v| v.is_instance_of::<PyDict>())
            {
                let out = tuple
                    .get_item(0)
                    .map_err(|e| SomaError::Other(format!("forward tuple unpack: {e}")))?;
                return py_to_value(py, &out).map_err(py_err_to_soma);
            }
            py_to_value(py, bound).map_err(py_err_to_soma)
        })
    }

    fn meta(&self) -> FilterMeta {
        // Read optional meta attributes from the Python class.
        // Users can set: _cacheable = False, _differentiable = False,
        //                _deterministic = False, _kind = "stateless",
        //                _stream_mode = "evolving"
        let (kind, cacheable, differentiable, deterministic, stream_mode) =
            Python::with_gil(|py| {
                let obj = self.py_obj.bind(py);

                let cacheable = obj
                    .getattr("_cacheable")
                    .and_then(|v| v.extract::<bool>())
                    .unwrap_or(true);

                let differentiable = obj
                    .getattr("_differentiable")
                    .and_then(|v| v.extract::<bool>())
                    .unwrap_or(false);

                let deterministic = obj
                    .getattr("_deterministic")
                    .and_then(|v| v.extract::<bool>())
                    .unwrap_or(true);

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

                (kind, cacheable, differentiable, deterministic, stream_mode)
            });

        FilterMeta {
            name: self.name.clone(),
            kind,
            cacheable,
            differentiable,
            deterministic,
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

// ── PyTrial ──

/// Handle passed to a study's objective function.
///
/// Behaves like a read-only params mapping (`trial["Encoder.lr"]`,
/// `trial.get("x", 0.5)`) so legacy `fn(params) -> dict` executors keep
/// working, and adds `report(name, value, step)` / `should_prune()`
/// for pruning-aware training loops.
#[pyclass(name = "Trial")]
struct PyTrial {
    ctx: TrialContext,
    params: HashMap<String, serde_json::Value>,
}

#[pymethods]
impl PyTrial {
    /// Sampled parameters as a dict.
    #[getter]
    fn params(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        for (k, v) in &self.params {
            dict.set_item(k, json_to_py(py, v))?;
        }
        Ok(dict.unbind())
    }

    /// Trial id (`trial_0003`).
    #[getter]
    fn id(&self) -> String {
        self.ctx.trial_id().to_string()
    }

    /// Report an intermediate metric. Returns True when the trial
    /// should stop (the pruner decided against it).
    fn report(&self, name: &str, value: f64, step: usize) -> bool {
        self.ctx.report(name, value, step)
    }

    /// Whether the pruner has decided to stop this trial.
    fn should_prune(&self) -> bool {
        self.ctx.should_prune()
    }

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<PyObject> {
        self.params
            .get(key)
            .map(|v| json_to_py(py, v))
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(key.to_string()))
    }

    fn __contains__(&self, key: &str) -> bool {
        self.params.contains_key(key)
    }

    #[pyo3(signature = (key, default=None))]
    fn get(&self, py: Python<'_>, key: &str, default: Option<PyObject>) -> PyObject {
        match self.params.get(key) {
            Some(v) => json_to_py(py, v),
            None => default.unwrap_or_else(|| py.None()),
        }
    }

    fn keys(&self) -> Vec<String> {
        self.params.keys().cloned().collect()
    }
}

// ── PyStudy ──

#[pyclass(name = "Study")]
struct PyStudy {
    study: Study,
    /// Python callable metrics-dict -> float; recorded as metric "score".
    objective_cb: Option<PyObject>,
    tracking: bool,
    root: std::path::PathBuf,
    run_dir: Option<std::path::PathBuf>,
}

fn parse_pruning(pruning: &Bound<'_, PyAny>) -> PyResult<PruningStrategy> {
    if let Ok(s) = pruning.extract::<String>() {
        return match s.as_str() {
            "median" => Ok(PruningStrategy::Median { n_warmup_steps: 0 }),
            other => Err(PyRuntimeError::new_err(format!(
                "unknown pruning '{other}'; use 'median', ('median', warmup) or ('percentile', pct, warmup)"
            ))),
        };
    }
    let tuple = pruning.downcast::<pyo3::types::PyTuple>().map_err(|_| {
        PyRuntimeError::new_err(
            "pruning must be 'median', ('median', warmup) or ('percentile', pct, warmup)",
        )
    })?;
    let kind: String = tuple.get_item(0)?.extract()?;
    match kind.as_str() {
        "median" => Ok(PruningStrategy::Median {
            n_warmup_steps: tuple.get_item(1)?.extract()?,
        }),
        "percentile" => Ok(PruningStrategy::Percentile {
            percentile: tuple.get_item(1)?.extract()?,
            n_warmup_steps: tuple.get_item(2)?.extract()?,
        }),
        other => Err(PyRuntimeError::new_err(format!(
            "unknown pruning '{other}'"
        ))),
    }
}

/// Append a completed study's summary as an ExperimentRecord to
/// `<root>/experiments.jsonl`. Best-effort: tracking must never fail
/// the training path.
fn record_study_experiment(root: &std::path::Path, study: &Study, run_id: &str) {
    use somatize_memory::{ExperimentRecord, FileKnowledgeBase, KnowledgeBase};

    let Some(best) = study.best_trial() else {
        return;
    };
    let mut metrics: HashMap<String, f64> = HashMap::new();
    for m in &best.metrics {
        metrics.insert(m.name.clone(), m.value);
    }
    let total_ms: u64 = study.trials.iter().filter_map(|t| t.duration_ms).sum();
    let mut tags = study.tags.clone();
    tags.push(format!("run:{run_id}"));

    let record = ExperimentRecord::new(study.id.clone(), study.name.clone())
        .with_pipeline(format!("study over {} trials", study.trials.len()))
        .with_params(best.params.clone())
        .with_metrics(metrics)
        .with_duration(std::time::Duration::from_millis(total_ms))
        .with_tags(tags);

    match FileKnowledgeBase::open(root.join("experiments.jsonl")) {
        Ok(mut kb) => {
            if let Err(e) = kb.record(record) {
                eprintln!("soma: failed to record experiment: {e}");
            }
        }
        Err(e) => eprintln!("soma: failed to open experiments.jsonl: {e}"),
    }
}

fn trial_to_py(py: Python<'_>, trial: &somatize_core::study::Trial) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("id", &trial.id)?;
    let params_dict = PyDict::new(py);
    for (k, v) in &trial.params {
        params_dict.set_item(k, json_to_py(py, v))?;
    }
    dict.set_item("params", params_dict)?;
    dict.set_item(
        "state",
        match &trial.state {
            somatize_core::study::TrialState::Pending => "pending".to_string(),
            somatize_core::study::TrialState::Running => "running".to_string(),
            somatize_core::study::TrialState::Completed => "completed".to_string(),
            somatize_core::study::TrialState::Pruned { .. } => "pruned".to_string(),
            somatize_core::study::TrialState::Failed { .. } => "failed".to_string(),
        },
    )?;
    let metrics_dict = PyDict::new(py);
    for m in &trial.metrics {
        metrics_dict.set_item(&m.name, m.value)?;
    }
    dict.set_item("metrics", metrics_dict)?;
    // Full metric series (one record per report), for learning curves —
    // "metrics" above keeps only the last value per name.
    let series = pyo3::types::PyList::empty(py);
    for m in &trial.metrics {
        let rec = PyDict::new(py);
        rec.set_item("name", &m.name)?;
        rec.set_item("value", m.value)?;
        rec.set_item("step", m.step)?;
        rec.set_item("timestamp", m.timestamp.to_rfc3339())?;
        series.append(rec)?;
    }
    dict.set_item("series", series)?;
    dict.set_item("started_at", trial.started_at.map(|t| t.to_rfc3339()))?;
    dict.set_item("finished_at", trial.finished_at.map(|t| t.to_rfc3339()))?;
    dict.set_item("duration_ms", trial.duration_ms)?;
    Ok(dict.unbind())
}

#[pymethods]
impl PyStudy {
    #[new]
    #[pyo3(signature = (name, search_space=None, strategy="grid".to_string(), n_trials=10,
                        objectives=None, seed=None, objective=None, direction="maximize".to_string(),
                        pruning=None, tracking=true, root=".soma".to_string(), tags=None,
                        seeds=None, frozen=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        _py: Python<'_>,
        name: String,
        search_space: Option<&Bound<'_, PyList>>,
        strategy: String,
        n_trials: usize,
        objectives: Option<Vec<(String, String)>>,
        seed: Option<u64>,
        objective: Option<PyObject>,
        direction: String,
        pruning: Option<&Bound<'_, PyAny>>,
        tracking: bool,
        root: String,
        tags: Option<Vec<String>>,
        seeds: Option<Vec<i64>>,
        frozen: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let mut space = SearchSpace::new();
        if let Some(dims) = search_space {
            for item in dims.iter() {
                let dim = parse_py_search_dim(item.py(), &item)?;
                space.add(dim);
            }
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

        // Strict: a typo like "minimise" must not silently maximize.
        let parse_dir = |d: &str| -> PyResult<Direction> {
            match d {
                "minimize" => Ok(Direction::Minimize),
                "maximize" => Ok(Direction::Maximize),
                other => Err(PyRuntimeError::new_err(format!(
                    "unknown direction '{other}'; use 'maximize' or 'minimize'"
                ))),
            }
        };

        // With a Python objective callable, the composite score is
        // recorded as metric "score" — that becomes the objective.
        let objs: Vec<Objective> = match (&objective, objectives) {
            (Some(_), _) => vec![Objective {
                metric: "score".into(),
                direction: parse_dir(&direction)?,
            }],
            (None, Some(list)) => list
                .into_iter()
                .map(|(metric, dir)| {
                    Ok(Objective {
                        metric,
                        direction: parse_dir(&dir)?,
                    })
                })
                .collect::<PyResult<Vec<_>>>()?,
            (None, None) => vec![],
        };

        let mut study = Study::new(name, space, strat, objs);
        if let Some(p) = pruning {
            study.pruning = parse_pruning(p)?;
        }
        study.tags = tags.unwrap_or_default();
        // Experiment seeds: every sampled config runs once per seed;
        // trial params carry "seed" (wire torch.manual_seed(trial["seed"])
        // in the trial callable — the cache keys per seed automatically).
        study.seeds = seeds.unwrap_or_default();
        if let Some(f) = frozen {
            for (k, v) in f.iter() {
                let key: String = k.extract()?;
                let value: serde_json::Value = py_any_to_json(&v)?;
                study.frozen.insert(key, value);
            }
        }

        Ok(Self {
            study,
            objective_cb: objective,
            tracking,
            root: std::path::PathBuf::from(root),
            run_dir: None,
        })
    }

    /// Load a study from a run directory (its `study.json`).
    ///
    /// Continue it from anywhere::
    ///
    ///     study = soma.Study.load(".soma/runs/study_.../")
    ///     study.run(train, resume=True)
    ///
    /// A composite `objective=` callable cannot be persisted — re-pass
    /// it here when resuming a study that was created with one, or the
    /// new trials won't produce the "score" metric.
    #[staticmethod]
    #[pyo3(signature = (run_dir, objective=None))]
    fn load(run_dir: String, objective: Option<PyObject>) -> PyResult<Self> {
        let dir = std::path::PathBuf::from(run_dir);
        let study = Study::load(dir.join("study.json")).map_err(soma_err_to_py)?;
        // <root>/runs/<run_id> → root
        let root = dir
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| std::path::PathBuf::from(".soma"));
        Ok(Self {
            study,
            objective_cb: objective,
            tracking: true,
            root,
            run_dir: Some(dir),
        })
    }

    /// Save the study as JSON (defaults to `<run_dir>/study.json`).
    #[pyo3(signature = (path=None))]
    fn save(&self, path: Option<String>) -> PyResult<()> {
        let path = match (path, &self.run_dir) {
            (Some(p), _) => std::path::PathBuf::from(p),
            (None, Some(dir)) => dir.join("study.json"),
            (None, None) => {
                return Err(PyRuntimeError::new_err(
                    "no path given and the study has no run directory yet",
                ));
            }
        };
        self.study.save(path).map_err(soma_err_to_py)
    }

    /// Run the study.
    ///
    /// `fn` receives a `Trial` handle: read params via `trial.params`,
    /// `trial["name"]` or `trial.get(...)`; report intermediate metrics
    /// with `trial.report(name, value, step)` (returns True → stop);
    /// return a dict of final metrics (or None / a bare float).
    #[pyo3(signature = (executor, on_event=None, resume=false))]
    fn run(
        &mut self,
        py: Python<'_>,
        executor: &Bound<'_, PyAny>,
        on_event: Option<PyObject>,
        resume: bool,
    ) -> PyResult<()> {
        // A study created with objective= scores through the "score"
        // metric; resuming it without re-passing the callable would
        // silently stop producing that metric. Only prior trials that
        // actually carry "score" are evidence a callable existed.
        if self.objective_cb.is_none()
            && self
                .study
                .objectives
                .first()
                .is_some_and(|o| o.metric == "score")
            && self
                .study
                .trials
                .iter()
                .any(|t| t.metrics.iter().any(|m| m.name == "score"))
        {
            let warnings = py.import("warnings")?;
            warnings.call_method1(
                "warn",
                (
                    "this study scores on the 'score' metric but no objective= callable \
                     is set — pass it to Study.load(run_dir, objective=...) when resuming, \
                     or return a {'score': ...} dict from the executor",
                ),
            )?;
        }

        let tracker: Option<Arc<LocalTracker>> = if self.tracking {
            let t = match (&self.run_dir, resume) {
                (Some(dir), true) => LocalTracker::open(dir).map_err(soma_err_to_py)?,
                _ => {
                    let t = LocalTracker::create(&self.root, RunKind::Study, &self.study.name)
                        .map_err(soma_err_to_py)?;
                    // Enrich the manifest exactly like begin_run does.
                    if let Ok(mut manifest) = load_manifest(t.run_dir()) {
                        manifest.tags = self.study.tags.clone();
                        manifest.python_version =
                            Some(py.version().split_whitespace().next().unwrap_or("").into());
                        // Record experiment seeds (and the sampler seed)
                        // so a reader can see what this run covers.
                        for (i, s) in self.study.seeds.iter().enumerate() {
                            manifest.seeds.insert(format!("seed_{i}"), *s);
                        }
                        match &self.study.strategy {
                            somatize_core::study::SearchStrategy::Random {
                                seed: Some(s), ..
                            }
                            | somatize_core::study::SearchStrategy::Bayesian {
                                seed: Some(s),
                                ..
                            } => {
                                manifest.seeds.insert("sampler".into(), *s as i64);
                            }
                            _ => {}
                        }
                        let _ = t.save_manifest(&manifest);
                    }
                    t
                }
            };
            self.run_dir = Some(t.run_dir().to_path_buf());
            Some(Arc::new(t))
        } else {
            None
        };

        let bus = Arc::new(EventBus::new(1024));
        if let Some(t) = &tracker {
            bus.add_sink(t.sink());
        }
        if let Some(callback) = on_event {
            let mut rx = bus.subscribe();
            std::thread::spawn(move || {
                while let Ok(event) = rx.blocking_recv() {
                    if let Ok(json_str) = serde_json::to_string(&event) {
                        Python::with_gil(|py| {
                            let json_mod = py.import("json").unwrap();
                            if let Ok(dict) = json_mod.call_method1("loads", (json_str,)) {
                                let _ = callback.call1(py, (dict,));
                            }
                        });
                    }
                }
            });
        }

        let mut runner = StudyRunner::new(bus.clone());
        if let Some(t) = &tracker {
            runner = runner.with_tracker(t.clone() as Arc<dyn Tracker>);
        }

        let executor_obj = executor.clone().unbind();
        let objective_cb = self
            .objective_cb
            .as_ref()
            .map(|cb| Python::with_gil(|py| cb.clone_ref(py)));

        let trial_executor = FnTrialExecutor(
            move |params: &HashMap<String, serde_json::Value>,
                  ctx: &TrialContext|
                  -> SomaResult<TrialOutcome> {
                Python::with_gil(|py| {
                    let trial = PyTrial {
                        ctx: ctx.clone(),
                        params: params.clone(),
                    };
                    let result = executor_obj
                        .call1(py, (trial,))
                        .map_err(|e| SomaError::Other(format!("Python executor error: {e}")))?;
                    let bound = result.bind(py);

                    // Accepted returns: None, a bare number, or a dict
                    // of final metrics.
                    let mut metrics: Vec<MetricRecord> = Vec::new();
                    if bound.is_none() {
                        // metrics were reported via trial.report()
                    } else if let Ok(v) = bound.extract::<f64>() {
                        metrics.push(MetricRecord {
                            name: "score".into(),
                            value: v,
                            step: 0,
                            timestamp: chrono::Utc::now(),
                        });
                    } else if let Ok(dict) = bound.downcast::<PyDict>() {
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
                    } else {
                        return Err(SomaError::Other(
                            "executor must return None, a number, or a dict of metrics".into(),
                        ));
                    }

                    // Composite objective callable → recorded as "score".
                    if let Some(cb) = &objective_cb
                        && !metrics.iter().any(|m| m.name == "score")
                    {
                        let merged = PyDict::new(py);
                        for m in ctx.metrics() {
                            merged.set_item(&m.name, m.value).map_err(py_err_to_soma)?;
                        }
                        for m in &metrics {
                            merged.set_item(&m.name, m.value).map_err(py_err_to_soma)?;
                        }
                        let score: f64 = cb
                            .call1(py, (merged,))
                            .map_err(|e| SomaError::Other(format!("objective error: {e}")))?
                            .extract(py)
                            .map_err(|_| {
                                SomaError::Other("objective must return a number".into())
                            })?;
                        metrics.push(MetricRecord {
                            name: "score".into(),
                            value: score,
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

        // Release the GIL for the whole study: the executor re-acquires
        // it per trial, and on_event callbacks can run between trials.
        let study = &mut self.study;
        let result = py.allow_threads(move || runner.run(study, sampler.as_mut(), &trial_executor));

        if let Some(t) = &tracker {
            let state = if result.is_ok() {
                RunState::Completed
            } else {
                RunState::Failed
            };
            let _ = t.finalize(state);
            if result.is_ok() {
                record_study_experiment(&self.root, &self.study, t.run_id());
            }
        }
        result.map_err(soma_err_to_py)
    }

    #[getter]
    fn best_trial(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        match self.study.best_trial() {
            Some(trial) => Ok(Some(trial_to_py(py, trial)?)),
            None => Ok(None),
        }
    }

    /// All recorded trials as dicts.
    #[getter]
    fn trials(&self, py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
        self.study
            .trials
            .iter()
            .map(|t| trial_to_py(py, t))
            .collect()
    }

    #[getter]
    fn n_trials(&self) -> usize {
        self.study.trials.len()
    }

    #[getter]
    fn progress(&self) -> f64 {
        self.study.progress()
    }

    /// Declared objectives as `(metric, direction)` pairs, e.g.
    /// `[("f1", "maximize")]`. A composite objective reports as
    /// `("score", direction)`.
    #[getter]
    fn objectives(&self) -> Vec<(String, String)> {
        let dir_str = |d: &somatize_core::study::Direction| match d {
            somatize_core::study::Direction::Maximize => "maximize".to_string(),
            somatize_core::study::Direction::Minimize => "minimize".to_string(),
        };
        if self.study.objectives.is_empty()
            && let Some(composite) = &self.study.composite
        {
            return vec![("score".to_string(), dir_str(&composite.direction))];
        }
        self.study
            .objectives
            .iter()
            .map(|o| (o.metric.clone(), dir_str(&o.direction)))
            .collect()
    }

    /// The study's display name.
    #[getter]
    fn name(&self) -> String {
        self.study.name.clone()
    }

    /// Run directory holding study.json/events.jsonl (None if tracking
    /// is disabled and the study never ran).
    #[getter]
    fn run_dir(&self) -> Option<String> {
        self.run_dir.as_ref().map(|p| p.display().to_string())
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
    /// Live Python filter instances retained by node id. Used by the
    /// in-process training path (graph.train/forward/freeze) so that a
    /// filter's persistent state (e.g. an nn.Module attached to self)
    /// survives across forward calls instead of being deserialised each
    /// time. Distinct from `pickled_filters`, which exists only for
    /// remote-worker dispatch.
    live_filters: std::collections::HashMap<String, Py<PyAny>>,
    /// Generic Python-side scratch dict for orchestration state that
    /// doesn't belong on the Rust struct (e.g. the registered optimiser).
    /// Lazily initialised on first access. PyGraph deliberately doesn't
    /// expose `__dict__`, so this dict is the supported way to attach
    /// per-graph Python state.
    py_state: Option<Py<PyDict>>,
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
                let config_hash = self.library.get(&node.id).map(|f| f.config_hash());
                Some(SerializedFilter {
                    node_id: node.id.clone(),
                    pickled_filter: pickled.clone(),
                    state,
                    requirements: reqs.clone(),
                    trainable,
                    config_hash,
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
                let config_hash = self.library.get(&node.id).map(|f| f.config_hash());
                Some(SerializedFilter {
                    node_id: node.id.clone(),
                    pickled_filter: pickled.clone(),
                    state,
                    requirements: reqs.clone(),
                    trainable,
                    config_hash,
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
    /// * `cache` — cache backend: `"memory"`, `"local"`, or `"tiered"`.
    ///   Default: a persistent tiered cache (memory LRU over a shared
    ///   on-disk store at `$SOMA_CACHE_DIR` or `~/.soma/cache`), so fit
    ///   states and forward outputs survive crashes and are shared
    ///   across processes and projects. Pass `cache="memory"` to opt out.
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
        let cache_store: Arc<dyn somatize_core::cache::CacheStore> = match cache {
            None => match default_cache_dir().and_then(|dir| FsActionStore::new(dir).ok()) {
                Some(local) => Arc::new(TieredCache::memory_and_local(
                    Box::new(MemoryCache::new(max_bytes)),
                    Box::new(local),
                )),
                // No writable cache dir (sandbox, read-only home):
                // degrade to memory-only rather than failing.
                None => Arc::new(MemoryCache::new(max_bytes)),
            },
            Some("memory") => Arc::new(MemoryCache::new(max_bytes)),
            Some("local") => {
                let path = cache_path.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "cache_path is required for cache=\"local\"",
                    )
                })?;
                Arc::new(
                    FsActionStore::new(path)
                        .map_err(|e| PyRuntimeError::new_err(format!("cache init: {e}")))?,
                )
            }
            Some("tiered") => {
                let path = cache_path.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "cache_path is required for cache=\"tiered\"",
                    )
                })?;
                let local = FsActionStore::new(path)
                    .map_err(|e| PyRuntimeError::new_err(format!("cache init: {e}")))?;
                Arc::new(TieredCache::memory_and_local(
                    Box::new(MemoryCache::new(max_bytes)),
                    Box::new(local),
                ))
            }
            Some(other) => {
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
            live_filters: std::collections::HashMap::new(),
            py_state: None,
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
        // Retain the live Python instance so it can be retrieved by
        // graph.filter(node_id) for the in-process training path. The
        // FilterLibrary owns a Box<dyn Filter> wrapping the bridge; we
        // keep an independent strong reference to the original PyObject
        // here so callers can mutate it (e.g. set self.training=True or
        // attach an nn.Module).
        self.live_filters
            .insert(actual_id.clone(), filter_obj.clone().unbind());
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
    #[pyo3(signature = (x, y=None, batch_size=None, mode="inference", seed=None))]
    fn fit(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, pyo3::types::PyAny>,
        y: Option<&Bound<'_, pyo3::types::PyAny>>,
        batch_size: Option<usize>,
        mode: &str,
        seed: Option<i64>,
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
            let run_id = somatize_core::util::timestamp_id("fit");
            self.event_bus
                .emit(somatize_core::event::Event::RunStarted {
                    run_id: run_id.clone(),
                    plan_summary: compile_result.plan.summary(),
                });
            let run_start = std::time::Instant::now();
            let result = runner.fit(
                &compile_result.plan,
                &self.library,
                self.cache.as_ref(),
                &self.event_bus,
                &run_id,
                &x_val,
                y_val.as_ref(),
            );
            let (_output, states) = match result {
                Ok(out) => {
                    self.event_bus
                        .emit(somatize_core::event::Event::RunCompleted {
                            run_id,
                            duration: run_start.elapsed(),
                        });
                    out
                }
                Err(e) => {
                    self.event_bus.emit(somatize_core::event::Event::RunFailed {
                        run_id,
                        error: e.to_string(),
                    });
                    return Err(soma_err_to_py(e));
                }
            };
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

        self.event_bus
            .emit(somatize_core::event::Event::RunStarted {
                run_id: run_id.clone(),
                plan_summary: somatize_core::event::PlanSummary {
                    total_nodes: sorted.len(),
                    cached_nodes: 0,
                    parallel_branches: 0,
                },
            });
        let run_start = std::time::Instant::now();
        let bus = self.event_bus.clone();

        let fit_result: PyResult<()> = (|| {
            for node_id in &sorted {
                let filter = self.library.get(node_id).ok_or_else(|| {
                    PyRuntimeError::new_err(format!("filter not found: {node_id}"))
                })?;

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
                                let json_val = val.to_plain_json();
                                merged.insert(pred_id.clone(), json_val);
                            }
                        }
                        Value::json(serde_json::Value::Object(merged))
                    }
                };

                let meta = filter.meta();
                let start = std::time::Instant::now();

                let origin = somatize_core::cache::Origin::Computed {
                    node_id: node_id.to_string(),
                    run_id: run_id.clone(),
                };
                let input_hash = somatize_core::cache::CacheKey::for_value(&input).ok();

                let output = if meta.kind == somatize_core::filter::FilterKind::Trainable {
                    // Unhashable x/y ⇒ skip caching, never a degenerate key.
                    // Labels are part of the state key.
                    let y_hash = match y_val.as_ref() {
                        Some(y) => somatize_core::cache::CacheKey::for_value(y).ok().map(Some),
                        None => Some(None),
                    };
                    let state_key = match (&input_hash, y_hash) {
                        (Some(x_hash), Some(y_hash)) => {
                            Some(somatize_runtime::executor::salt_with_seed(
                                somatize_core::cache::CacheKey::for_state(
                                    &filter.config_hash(),
                                    x_hash,
                                    y_hash.as_ref(),
                                ),
                                seed,
                            ))
                        }
                        _ => None,
                    };

                    let cached_state = state_key
                        .as_ref()
                        .and_then(|key| self.cache.get(key).ok().flatten());
                    let state = if let Some(cached) = cached_state {
                        cached
                    } else {
                        let fit_start = std::time::Instant::now();
                        let s = filter.fit(&input, y_val.as_ref()).map_err(soma_err_to_py)?;
                        if let Some(key) = &state_key {
                            let _ = self.cache.put_computed(
                                key,
                                &s,
                                &origin,
                                fit_start.elapsed(),
                                true,
                            );
                        }
                        s
                    };

                    let out = forward_with_cache(
                        self.cache.as_ref(),
                        filter.as_ref(),
                        &meta,
                        &input,
                        input_hash.as_ref(),
                        &state,
                        &origin,
                        seed,
                    )
                    .map_err(soma_err_to_py)?;
                    self.library.set_state(node_id.to_string(), state);
                    out
                } else {
                    forward_with_cache(
                        self.cache.as_ref(),
                        filter.as_ref(),
                        &meta,
                        &input,
                        input_hash.as_ref(),
                        &Value::Empty,
                        &origin,
                        seed,
                    )
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
            Ok(())
        })();

        match &fit_result {
            Ok(()) => bus.emit(somatize_core::event::Event::RunCompleted {
                run_id,
                duration: run_start.elapsed(),
            }),
            Err(e) => bus.emit(somatize_core::event::Event::RunFailed {
                run_id,
                error: e.to_string(),
            }),
        };
        fit_result?;

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
    #[pyo3(signature = (x, stream=false, chunk_size=1024, seed=None))]
    fn forward(
        &self,
        py: Python<'_>,
        x: &Bound<'_, pyo3::types::PyAny>,
        stream: bool,
        chunk_size: usize,
        seed: Option<i64>,
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
            .with_graph_info(graph_info)
            .with_seed(seed)
            .with_cache_arc(self.cache.clone());

            let roots = self.graph.roots();
            if roots.len() == 1 {
                ctx.set(format!("__input_{}", roots[0]), x_val.clone());
            }
            ctx.set("__input__", x_val);

            py.allow_threads(|| {
                executor::execute(
                    &compile_result.plan,
                    &mut ctx,
                    &self.library,
                    self.cache.as_ref(),
                )
            })
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
        .with_graph_info(graph_info)
        .with_seed(seed);

        if let Some(transport) = self.make_transport() {
            ctx = ctx.with_transport(transport);
        }

        let roots = self.graph.roots();
        if roots.len() == 1 {
            ctx.set(format!("__input_{}", roots[0]), x_val.clone());
        }
        ctx.set("__input__", x_val);

        // Release the GIL: Parallel plans run branches on scoped threads
        // whose Python filters must acquire it — holding it here would
        // deadlock the join.
        py.allow_threads(|| {
            executor::execute(
                &compile_result.plan,
                &mut ctx,
                &self.library,
                self.cache.as_ref(),
            )
        })
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
        let run_id = somatize_core::util::timestamp_id("graph_run");
        let mut ctx =
            Context::new(self.event_bus.clone(), run_id.clone()).with_graph_info(graph_info);

        if let Some(transport) = self.make_transport() {
            ctx = ctx.with_transport(transport);
        }

        self.event_bus
            .emit(somatize_core::event::Event::RunStarted {
                run_id: run_id.clone(),
                plan_summary: compile_result.plan.summary(),
            });
        let run_start = std::time::Instant::now();

        // Release the GIL: Parallel plans run branches on scoped threads
        // whose Python filters must acquire it — holding it here would
        // deadlock the join.
        let result = py.allow_threads(|| {
            executor::execute(
                &compile_result.plan,
                &mut ctx,
                &self.library,
                self.cache.as_ref(),
            )
        });
        match &result {
            Ok(()) => self
                .event_bus
                .emit(somatize_core::event::Event::RunCompleted {
                    run_id,
                    duration: run_start.elapsed(),
                }),
            Err(e) => self.event_bus.emit(somatize_core::event::Event::RunFailed {
                run_id,
                error: e.to_string(),
            }),
        };
        result.map_err(soma_err_to_py)?;

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
    ///
    /// `overlay` is an optional dict of per-node execution annotations
    /// (the shape `RunView.overlay()` returns): status coloring plus a
    /// duration/cache/flags label line per node.
    #[pyo3(signature = (overlay=None))]
    fn to_mermaid(&self, py: Python<'_>, overlay: Option<&Bound<'_, PyDict>>) -> PyResult<String> {
        match overlay {
            None => Ok(self.graph.to_mermaid()),
            Some(ov) => Ok(self.graph.to_mermaid_with(&py_overlay(py, ov)?)),
        }
    }

    /// Render the graph as a self-contained SVG diagram (same optional
    /// `overlay` as `to_mermaid`). No JavaScript — displays inline in
    /// any notebook viewer.
    #[pyo3(signature = (overlay=None))]
    fn to_svg(&self, py: Python<'_>, overlay: Option<&Bound<'_, PyDict>>) -> PyResult<String> {
        match overlay {
            None => Ok(self.graph.to_svg()),
            Some(ov) => Ok(self.graph.to_svg_with(&py_overlay(py, ov)?)),
        }
    }

    /// Notebook display: the architecture as an inline SVG diagram
    /// (falls back to the text tree for very large graphs).
    fn _repr_html_(&self) -> String {
        if self.graph.nodes.is_empty() {
            return "<i>empty graph — add nodes with g.node(...)</i>".to_string();
        }
        if self.graph.nodes.len() > 80 {
            return format!(
                "<pre style='font-family:ui-monospace,monospace'>{}</pre>",
                self.graph
                    .to_text()
                    .replace('&', "&amp;")
                    .replace('<', "&lt;")
            );
        }
        self.graph.to_svg()
    }

    /// Render the graph as a Graphviz DOT string (same optional
    /// `overlay` as `to_mermaid`).
    #[pyo3(signature = (overlay=None))]
    fn to_graphviz(&self, py: Python<'_>, overlay: Option<&Bound<'_, PyDict>>) -> PyResult<String> {
        match overlay {
            None => Ok(self.graph.to_graphviz()),
            Some(ov) => Ok(self.graph.to_graphviz_with(&py_overlay(py, ov)?)),
        }
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

    /// Emit an event onto the graph's bus from Python.
    ///
    /// The dict must carry an `event_type` matching a Soma event
    /// variant (e.g. `StepCompleted`, `MetricReported`, `HealthFlag`)
    /// plus that variant's fields. Used by the native training loop and
    /// the gradient audit to make Python-side progress visible to
    /// trackers and subscribers.
    fn emit_event(&self, py: Python<'_>, event: &Bound<'_, PyDict>) -> PyResult<()> {
        let json_mod = py.import("json")?;
        let json_str: String = json_mod.call_method1("dumps", (event,))?.extract()?;
        let value: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| PyRuntimeError::new_err(format!("invalid event JSON: {e}")))?;
        let event: somatize_core::event::Event = serde_json::from_value(value)
            .map_err(|e| PyRuntimeError::new_err(format!("unknown or malformed event: {e}")))?;
        self.event_bus.emit(event);
        Ok(())
    }

    /// Serialized graph topology (nodes/edges) as JSON — written into
    /// run directories so a front-end can draw the architecture.
    fn graph_json(&self) -> PyResult<String> {
        serde_json::to_string_pretty(&self.graph)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Start a tracked run: creates `.soma/runs/<run_id>/` and attaches
    /// its lossless sink to this graph's event bus. Prefer the
    /// `graph.track_run(...)` context manager from Python.
    #[pyo3(signature = (name, root=".soma".to_string(), kind="train".to_string(), tags=None))]
    fn begin_run(
        &self,
        py: Python<'_>,
        name: String,
        root: String,
        kind: String,
        tags: Option<Vec<String>>,
    ) -> PyResult<PyRun> {
        let kind = match kind.as_str() {
            "fit" => RunKind::Fit,
            "train" => RunKind::Train,
            "study" => RunKind::Study,
            "trial" => RunKind::Trial,
            _ => RunKind::Other,
        };
        let tracker = LocalTracker::create(&root, kind, &name).map_err(soma_err_to_py)?;

        // Enrich the manifest with Python-side context.
        let mut manifest = load_manifest(tracker.run_dir()).map_err(soma_err_to_py)?;
        manifest.tags = tags.unwrap_or_default();
        manifest.python_version = Some(py.version().split_whitespace().next().unwrap_or("").into());
        manifest.graph = Some(GraphSummaryInfo {
            n_nodes: self.graph.nodes.len(),
            node_ids: self.graph.nodes.iter().map(|n| n.id.clone()).collect(),
            graph_path: Some("graph.json".into()),
            mermaid_path: Some("graph.mmd".into()),
        });
        tracker.save_manifest(&manifest).map_err(soma_err_to_py)?;

        let sink = tracker.sink();
        self.event_bus.add_sink(sink.clone());
        Ok(PyRun {
            tracker: Arc::new(tracker),
            bus: self.event_bus.clone(),
            sink,
            finished: std::sync::atomic::AtomicBool::new(false),
            summary: std::sync::Mutex::new(HashMap::new()),
        })
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

    /// Retrieve the live Python filter instance registered under `node_id`.
    ///
    /// Returns ``None`` if the node doesn't exist or wasn't added through
    /// the Python `node()` API (e.g. nodes materialised from a serialised
    /// graph have only pickled bytes, not a live instance).
    ///
    /// Used by the in-process training path so callers can manipulate the
    /// filter directly — e.g. toggle `self.training`, read `_module`, or
    /// extract `state_dict()` — without round-tripping through a pickle.
    fn filter(&self, py: Python<'_>, node_id: String) -> Option<PyObject> {
        self.live_filters.get(&node_id).map(|o| o.clone_ref(py))
    }

    /// List node ids with live Python filter instances, in topological order.
    ///
    /// Falls back to insertion order if topological sort fails (e.g. the
    /// graph hasn't been validated yet — possible during construction).
    /// Callers that drive training need the topo order so output of one
    /// filter feeds the next.
    fn filter_ids(&self) -> Vec<String> {
        match self.graph.topological_sort() {
            Ok(sorted) => sorted
                .into_iter()
                .filter(|id| self.live_filters.contains_key(*id))
                .map(|id| id.to_string())
                .collect(),
            Err(_) => self.live_filters.keys().cloned().collect(),
        }
    }

    /// Return live Python filter instances as an ordered list of
    /// ``(node_id, filter)`` tuples in topological order.
    ///
    /// Returning a list (vs. a dict) preserves the order — callers
    /// iterating to chain forwards get inputs threaded correctly.
    fn filters(&self, py: Python<'_>) -> PyResult<PyObject> {
        let list = PyList::empty(py);
        for node_id in self.filter_ids() {
            if let Some(obj) = self.live_filters.get(&node_id) {
                let tuple = (node_id, obj.clone_ref(py));
                list.append(tuple)?;
            }
        }
        Ok(list.into_any().unbind())
    }

    /// Store a Python state value for a filter node.
    ///
    /// Used by ``Graph.freeze()`` (Python side) to push each live
    /// ``DifferentiableFilter`` module's serialised ``state_dict`` into
    /// the runtime's filter-state library, so subsequent eval calls go
    /// through the Rust forward path with state pre-populated.
    fn set_node_state(
        &mut self,
        py: Python<'_>,
        node_id: String,
        state: Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let value = py_to_value(py, &state)?;
        self.library.set_state(node_id, value);
        Ok(())
    }

    /// List data edges as ``[(source, target), ...]`` in insertion order.
    ///
    /// Used by :meth:`Graph.save` to record topology in the manifest so
    /// :meth:`Graph.load` can reconstruct non-linear graphs (forks,
    /// joins) instead of falling back to a linear chain.
    fn edges(&self) -> Vec<(String, String)> {
        self.graph
            .edges
            .iter()
            .map(|e| (e.source.clone(), e.target.clone()))
            .collect()
    }

    /// Retrieve the stored state value for a filter node, or ``None``.
    ///
    /// Mirror of :meth:`set_node_state`. Used by ``Graph.state()`` to
    /// snapshot every node's state for checkpointing.
    fn get_node_state(&self, py: Python<'_>, node_id: String) -> PyResult<Option<PyObject>> {
        match self.library.get_state(&node_id) {
            Some(arc) => Ok(Some(value_to_py(py, arc.as_ref())?)),
            None => Ok(None),
        }
    }

    /// Mark the graph as fitted without running ``fit()``.
    ///
    /// The Rust ``forward`` path refuses to run on an un-fitted graph.
    /// When the user trains via the Python autograd loop (``train()`` /
    /// ``forward(x)`` / ``backward`` / ``step``) and then calls
    /// ``freeze()``, no Rust ``fit()`` ran — but state has been pushed
    /// via ``set_node_state``. ``freeze()`` calls this so the
    /// subsequent eval ``forward`` is allowed.
    fn mark_fitted(&mut self) {
        self.fitted = true;
    }

    /// Per-graph scratch dict for Python-side orchestration state.
    ///
    /// PyGraph doesn't expose ``__dict__``, so callers (e.g. the
    /// _orchestrator module) use this dict to attach things like the
    /// registered optimiser without monkey-patching the class.
    /// Lazily created on first access.
    #[getter]
    fn py_state(&mut self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        if self.py_state.is_none() {
            self.py_state = Some(PyDict::new(py).unbind());
        }
        Ok(self.py_state.as_ref().unwrap().clone_ref(py))
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

// ── PyRun ──

/// A tracked run bound to a graph's event bus.
///
/// Created by `Graph.begin_run` (or the `graph.track_run(...)` context
/// manager). Metrics logged here become `MetricReported` events, which
/// the run's sink persists to `metrics.jsonl`/`events.jsonl`.
#[pyclass(name = "Run")]
struct PyRun {
    tracker: Arc<LocalTracker>,
    bus: Arc<EventBus>,
    sink: Arc<dyn somatize_core::tracking::EventSink>,
    finished: std::sync::atomic::AtomicBool,
    /// W&B-style summary: last logged value per metric name, written
    /// into the experiments journal on finish.
    summary: std::sync::Mutex<HashMap<String, f64>>,
}

#[pymethods]
impl PyRun {
    #[getter]
    fn id(&self) -> String {
        self.tracker.run_id().to_string()
    }

    /// Absolute path of the run directory.
    #[getter]
    fn dir(&self) -> String {
        self.tracker.run_dir().display().to_string()
    }

    /// Log a scalar metric (optionally scoped to a node).
    #[pyo3(signature = (name, value, step=None, node=None))]
    fn log(&self, name: String, value: f64, step: Option<usize>, node: Option<String>) {
        if let Ok(mut summary) = self.summary.lock() {
            summary.insert(name.clone(), value);
        }
        self.bus.emit(somatize_core::event::Event::MetricReported {
            run_id: self.tracker.run_id().to_string(),
            metric: MetricRecord {
                name,
                value,
                step: step.unwrap_or(0),
                timestamp: chrono::Utc::now(),
            },
            node_id: node,
            trial_id: None,
        });
    }

    /// Mark the start of an epoch.
    #[pyo3(signature = (epoch, total=None))]
    fn log_epoch(&self, epoch: usize, total: Option<usize>) {
        self.bus.emit(somatize_core::event::Event::EpochStarted {
            run_id: self.tracker.run_id().to_string(),
            epoch,
            total_epochs: total,
        });
        let _ = self.tracker.heartbeat();
    }

    /// Mark the end of an epoch with its summary metrics.
    #[pyo3(signature = (epoch, metrics=None))]
    fn log_epoch_completed(
        &self,
        epoch: usize,
        metrics: Option<std::collections::HashMap<String, f64>>,
    ) {
        let now = chrono::Utc::now();
        let records: Vec<MetricRecord> = metrics
            .unwrap_or_default()
            .into_iter()
            .map(|(name, value)| {
                if let Ok(mut summary) = self.summary.lock() {
                    summary.insert(name.clone(), value);
                }
                MetricRecord {
                    name,
                    value,
                    step: epoch,
                    timestamp: now,
                }
            })
            .collect();
        self.bus.emit(somatize_core::event::Event::EpochCompleted {
            run_id: self.tracker.run_id().to_string(),
            epoch,
            metrics: records,
        });
        let _ = self.tracker.heartbeat();
    }

    /// Mark one optimizer step (used by the native training loop).
    #[pyo3(signature = (step, epoch=None))]
    fn step_completed(&self, step: usize, epoch: Option<usize>) {
        self.bus.emit(somatize_core::event::Event::StepCompleted {
            run_id: self.tracker.run_id().to_string(),
            step,
            epoch,
        });
    }

    /// Refresh the run's heartbeat (liveness for external readers).
    fn heartbeat(&self) -> PyResult<()> {
        self.tracker.heartbeat().map_err(soma_err_to_py)
    }

    /// Finalize the run: flush logs, set terminal status, detach the
    /// sink from the graph's bus, and append the run's summary to the
    /// experiments journal. Idempotent.
    #[pyo3(signature = (status="completed".to_string()))]
    fn finish(&self, status: String) -> PyResult<()> {
        if self
            .finished
            .swap(true, std::sync::atomic::Ordering::SeqCst)
        {
            return Ok(());
        }
        self.bus.remove_sink(&self.sink);
        let state = match status.as_str() {
            "failed" => RunState::Failed,
            _ => RunState::Completed,
        };
        self.tracker.finalize(state).map_err(soma_err_to_py)?;
        if matches!(state, RunState::Completed) {
            self.record_experiment();
        }
        Ok(())
    }
}

impl PyRun {
    /// Append this run's summary to `<root>/experiments.jsonl`.
    /// Best-effort — never fails the training path.
    fn record_experiment(&self) {
        use somatize_memory::{ExperimentRecord, FileKnowledgeBase, KnowledgeBase};

        let run_dir = self.tracker.run_dir();
        let Some(root) = run_dir.parent().and_then(|p| p.parent()) else {
            return;
        };
        let (name, mut tags) = match load_manifest(run_dir) {
            Ok(m) => (m.name, m.tags),
            Err(_) => (self.tracker.run_id().to_string(), Vec::new()),
        };
        tags.push(format!("run:{}", self.tracker.run_id()));
        let metrics = self.summary.lock().map(|s| s.clone()).unwrap_or_default();

        let record = ExperimentRecord::new(self.tracker.run_id().to_string(), name)
            .with_pipeline("tracked run")
            .with_metrics(metrics)
            .with_tags(tags);
        match FileKnowledgeBase::open(root.join("experiments.jsonl")) {
            Ok(mut kb) => {
                if let Err(e) = kb.record(record) {
                    eprintln!("soma: failed to record experiment: {e}");
                }
            }
            Err(e) => eprintln!("soma: failed to open experiments.jsonl: {e}"),
        }
    }
}

// ── Cache management (backs the `soma cache` CLI) ──

fn resolve_cache_root(path: Option<String>) -> PyResult<std::path::PathBuf> {
    match path {
        Some(p) => Ok(p.into()),
        None => default_cache_dir().ok_or_else(|| {
            PyRuntimeError::new_err("no cache directory: set SOMA_CACHE_DIR or HOME")
        }),
    }
}

fn open_store(path: Option<String>) -> PyResult<FsActionStore> {
    let root = resolve_cache_root(path)?;
    FsActionStore::new(root).map_err(|e| PyRuntimeError::new_err(format!("cache open: {e}")))
}

/// Stats about the shared persistent cache.
#[pyfunction]
#[pyo3(signature = (path=None))]
fn cache_stats(py: Python<'_>, path: Option<String>) -> PyResult<PyObject> {
    let store = open_store(path)?;
    let actions = store
        .actions()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let cas_bytes = store
        .cas_bytes()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pinned = store
        .pinned()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let total_compute_ms: u64 = actions.iter().map(|a| a.compute_ms).sum();
    let mut blob_hashes = std::collections::HashSet::new();
    let mut available = 0usize;
    for record in &actions {
        for hash in record.outputs.values() {
            if blob_hashes.insert(*hash)
                && somatize_core::action::BlobStore::contains(&store, hash).unwrap_or(false)
            {
                available += 1;
            }
        }
    }

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("root", store.root().display().to_string())?;
    dict.set_item("actions", actions.len())?;
    dict.set_item("unique_outputs", blob_hashes.len())?;
    dict.set_item("blobs_available", available)?;
    dict.set_item("blobs_evicted", blob_hashes.len() - available)?;
    dict.set_item("cas_bytes", cas_bytes)?;
    dict.set_item("pinned", pinned.len())?;
    dict.set_item("saved_compute_ms", total_compute_ms)?;
    Ok(dict.into_any().unbind())
}

/// Cost-aware GC: evict blobs (records retained) down to `max_bytes`.
#[pyfunction]
#[pyo3(signature = (max_bytes, min_age_secs=3600, path=None))]
fn cache_gc(
    py: Python<'_>,
    max_bytes: u64,
    min_age_secs: u64,
    path: Option<String>,
) -> PyResult<PyObject> {
    let store = open_store(path)?;
    let policy = somatize_runtime::cache::gc::GcPolicy {
        max_bytes,
        min_age: std::time::Duration::from_secs(min_age_secs),
    };
    let report = somatize_runtime::cache::gc::collect(&store, &policy)
        .map_err(|e| PyRuntimeError::new_err(format!("gc: {e}")))?;
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("bytes_before", report.bytes_before)?;
    dict.set_item("bytes_after", report.bytes_after)?;
    dict.set_item("blobs_evicted", report.blobs_evicted)?;
    dict.set_item("blobs_kept", report.blobs_kept)?;
    dict.set_item("pinned_blobs", report.pinned_blobs)?;
    Ok(dict.into_any().unbind())
}

/// Pin an action as a GC root under a human-readable name.
#[pyfunction]
#[pyo3(signature = (name, key_hex, path=None))]
fn cache_pin(name: &str, key_hex: &str, path: Option<String>) -> PyResult<()> {
    if key_hex.len() != 64 || !key_hex.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "key must be a 64-char hex action key",
        ));
    }
    let mut digest = [0u8; 32];
    for (i, byte) in digest.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&key_hex[i * 2..i * 2 + 2], 16)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    }
    let store = open_store(path)?;
    store
        .pin(name, &somatize_core::cache::CacheKey(digest))
        .map_err(|e| PyRuntimeError::new_err(format!("pin: {e}")))
}

/// Verify every referenced blob against its content hash.
#[pyfunction]
#[pyo3(signature = (path=None))]
fn cache_verify(py: Python<'_>, path: Option<String>) -> PyResult<PyObject> {
    use somatize_core::action::BlobStore;
    let store = open_store(path)?;
    let actions = store
        .actions()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let (mut ok, mut missing, mut corrupt) = (0usize, 0usize, 0usize);
    let mut seen = std::collections::HashSet::new();
    for record in &actions {
        for hash in record.outputs.values() {
            if !seen.insert(*hash) {
                continue;
            }
            // get_bytes verifies content and maps corrupt blobs to None,
            // so distinguish via raw existence.
            match (
                store.contains(hash).unwrap_or(false),
                store.get_bytes(hash).ok().flatten(),
            ) {
                (true, Some(_)) => ok += 1,
                (true, None) => corrupt += 1,
                (false, _) => missing += 1,
            }
        }
    }
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("ok", ok)?;
    dict.set_item("missing", missing)?;
    dict.set_item("corrupt", corrupt)?;
    Ok(dict.into_any().unbind())
}

/// Remove legacy Phase-1 (v1) cache entries: `<hh>/<hh>/<hex>.json`
/// shard trees at the store root. The v2 key namespace makes them
/// unreachable — they are pure dead weight.
#[pyfunction]
#[pyo3(signature = (path=None))]
fn cache_purge_v1(py: Python<'_>, path: Option<String>) -> PyResult<PyObject> {
    let root = resolve_cache_root(path)?;
    let mut removed_files = 0u64;
    let mut removed_bytes = 0u64;
    if root.exists() {
        for entry in std::fs::read_dir(&root).map_err(|e| PyRuntimeError::new_err(e.to_string()))? {
            let entry = entry.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let name = entry.file_name().to_string_lossy().to_string();
            // v1 shard dirs are exactly two lowercase hex chars.
            let is_v1_shard = name.len() == 2
                && name.chars().all(|c| c.is_ascii_hexdigit())
                && entry.path().is_dir();
            if is_v1_shard {
                fn walk(dir: &std::path::Path, files: &mut u64, bytes: &mut u64) {
                    if let Ok(entries) = std::fs::read_dir(dir) {
                        for e in entries.filter_map(|e| e.ok()) {
                            let p = e.path();
                            if p.is_dir() {
                                walk(&p, files, bytes);
                            } else {
                                *files += 1;
                                *bytes += e.metadata().map(|m| m.len()).unwrap_or(0);
                            }
                        }
                    }
                }
                walk(&entry.path(), &mut removed_files, &mut removed_bytes);
                std::fs::remove_dir_all(entry.path())
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            }
        }
    }
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("removed_files", removed_files)?;
    dict.set_item("removed_bytes", removed_bytes)?;
    Ok(dict.into_any().unbind())
}

// ── Module ──

// ── Run-directory readers (back `soma.runs()` / `soma.RunView`) ──
//
// Each function returns a JSON string (the serde shape of the
// soma-runtime reader structs); the Python wrapper in `soma/_runs.py`
// parses it. Same pattern as `Graph.graph_json`.

/// Convert an overlay dict (the `RunView.overlay()` shape) into the
/// core overlay struct, via JSON like `Graph.emit_event`.
fn py_overlay(
    py: Python<'_>,
    overlay: &Bound<'_, PyDict>,
) -> PyResult<somatize_core::viz::GraphOverlay> {
    let json_mod = py.import("json")?;
    let json_str: String = json_mod.call_method1("dumps", (overlay,))?.extract()?;
    serde_json::from_str(&json_str)
        .map_err(|e| PyRuntimeError::new_err(format!("invalid overlay: {e}")))
}

fn open_run_reader(dir: &str) -> PyResult<somatize_runtime::tracking::RunReader> {
    somatize_runtime::tracking::RunReader::open(dir)
        .map_err(|e| PyRuntimeError::new_err(format!("open run dir {dir}: {e}")))
}

fn to_json_py<T: serde::Serialize>(value: &T) -> PyResult<String> {
    serde_json::to_string(value).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// All runs under `<root>/runs/`, newest first (JSON array of RunInfo).
#[pyfunction]
#[pyo3(signature = (root=".soma".to_string()))]
fn list_runs_json(root: String) -> PyResult<String> {
    let infos = somatize_runtime::tracking::list_runs(&root)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    to_json_py(&infos)
}

/// Listing entry (manifest identity + derived state) for one run dir.
#[pyfunction]
fn run_info_json(dir: String) -> PyResult<String> {
    let info = open_run_reader(&dir)?
        .info()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    to_json_py(&info)
}

/// The run's manifest.json contents.
#[pyfunction]
fn run_manifest_json(dir: String) -> PyResult<String> {
    let manifest = open_run_reader(&dir)?
        .manifest()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    to_json_py(&manifest)
}

/// All parseable event envelopes from events.jsonl, in log order.
#[pyfunction]
fn run_events_json(dir: String) -> PyResult<String> {
    let events = open_run_reader(&dir)?
        .events()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    to_json_py(&events)
}

/// Metric time series (optionally filtered by name).
#[pyfunction]
#[pyo3(signature = (dir, name=None))]
fn run_metric_series_json(dir: String, name: Option<String>) -> PyResult<String> {
    let points = open_run_reader(&dir)?
        .metric_series(name.as_deref())
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    to_json_py(&points)
}

/// Per-node execution spans (gantt/overlay substrate).
#[pyfunction]
fn run_node_timings_json(dir: String) -> PyResult<String> {
    let spans = open_run_reader(&dir)?
        .node_timings()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    to_json_py(&spans)
}

/// Cache hit/miss counts, total and per node.
#[pyfunction]
fn run_cache_activity_json(dir: String) -> PyResult<String> {
    let activity = open_run_reader(&dir)?
        .cache_activity()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    to_json_py(&activity)
}

/// HealthFlag events with wall time.
#[pyfunction]
fn run_health_flags_json(dir: String) -> PyResult<String> {
    let flags = open_run_reader(&dir)?
        .health_flags()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    to_json_py(&flags)
}

/// Trial lifetimes from study.json (empty for non-study runs).
#[pyfunction]
fn run_trial_timeline_json(dir: String) -> PyResult<String> {
    let spans = open_run_reader(&dir)?
        .trial_timeline()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    to_json_py(&spans)
}

/// Rendering overlay aggregated from this run's events (JSON
/// GraphOverlay: per-node status, duration, cache tier, flags).
#[pyfunction]
fn run_overlay_json(dir: String) -> PyResult<String> {
    let overlay = open_run_reader(&dir)?
        .overlay()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    to_json_py(&overlay)
}

/// Mermaid diagram of the run's graph, annotated with its overlay.
#[pyfunction]
#[pyo3(signature = (dir, overlay=true))]
fn run_to_mermaid(dir: String, overlay: bool) -> PyResult<String> {
    let reader = open_run_reader(&dir)?;
    if overlay {
        reader
            .to_mermaid()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    } else {
        let graph = reader
            .graph()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .ok_or_else(|| PyRuntimeError::new_err(format!("run dir {dir} has no graph.json")))?;
        Ok(graph.to_mermaid())
    }
}

/// Render a serialized soma-core Graph (the `graph.json` schema) to
/// mermaid, with an optional overlay dict (the `RunView.overlay()`
/// shape). Backs inner-architecture rendering (`diagnostics/modules/`).
#[pyfunction]
#[pyo3(signature = (graph_json, overlay=None))]
fn graph_json_to_mermaid(
    py: Python<'_>,
    graph_json: &str,
    overlay: Option<&Bound<'_, PyDict>>,
) -> PyResult<String> {
    let graph: somatize_core::graph::Graph = serde_json::from_str(graph_json)
        .map_err(|e| PyRuntimeError::new_err(format!("invalid graph JSON: {e}")))?;
    match overlay {
        None => Ok(graph.to_mermaid()),
        Some(ov) => Ok(graph.to_mermaid_with(&py_overlay(py, ov)?)),
    }
}

/// Render a serialized soma-core Graph to a self-contained SVG diagram
/// (no JavaScript), with an optional overlay dict.
#[pyfunction]
#[pyo3(signature = (graph_json, overlay=None))]
fn graph_json_to_svg(
    py: Python<'_>,
    graph_json: &str,
    overlay: Option<&Bound<'_, PyDict>>,
) -> PyResult<String> {
    let graph: somatize_core::graph::Graph = serde_json::from_str(graph_json)
        .map_err(|e| PyRuntimeError::new_err(format!("invalid graph JSON: {e}")))?;
    match overlay {
        None => Ok(graph.to_svg()),
        Some(ov) => Ok(graph.to_svg_with(&py_overlay(py, ov)?)),
    }
}

/// Self-contained SVG of the run's graph, annotated with its overlay.
#[pyfunction]
#[pyo3(signature = (dir, overlay=true))]
fn run_to_svg(dir: String, overlay: bool) -> PyResult<String> {
    let reader = open_run_reader(&dir)?;
    if overlay {
        reader
            .to_svg()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    } else {
        let graph = reader
            .graph()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .ok_or_else(|| PyRuntimeError::new_err(format!("run dir {dir} has no graph.json")))?;
        Ok(graph.to_svg())
    }
}

/// Graphviz DOT of the run's graph, annotated with its overlay.
#[pyfunction]
#[pyo3(signature = (dir, overlay=true))]
fn run_to_graphviz(dir: String, overlay: bool) -> PyResult<String> {
    let reader = open_run_reader(&dir)?;
    if overlay {
        reader
            .to_graphviz()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    } else {
        let graph = reader
            .graph()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .ok_or_else(|| PyRuntimeError::new_err(format!("run dir {dir} has no graph.json")))?;
        Ok(graph.to_graphviz())
    }
}

#[pymodule]
fn _soma(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGraph>()?;
    m.add_class::<PyStudy>()?;
    m.add_class::<PyTrial>()?;
    m.add_class::<PyRun>()?;
    m.add_class::<PyWorker>()?;
    m.add_function(wrap_pyfunction!(cache_stats, m)?)?;
    m.add_function(wrap_pyfunction!(cache_gc, m)?)?;
    m.add_function(wrap_pyfunction!(cache_pin, m)?)?;
    m.add_function(wrap_pyfunction!(cache_verify, m)?)?;
    m.add_function(wrap_pyfunction!(cache_purge_v1, m)?)?;
    m.add_function(wrap_pyfunction!(list_runs_json, m)?)?;
    m.add_function(wrap_pyfunction!(run_info_json, m)?)?;
    m.add_function(wrap_pyfunction!(run_manifest_json, m)?)?;
    m.add_function(wrap_pyfunction!(run_events_json, m)?)?;
    m.add_function(wrap_pyfunction!(run_metric_series_json, m)?)?;
    m.add_function(wrap_pyfunction!(run_node_timings_json, m)?)?;
    m.add_function(wrap_pyfunction!(run_cache_activity_json, m)?)?;
    m.add_function(wrap_pyfunction!(run_health_flags_json, m)?)?;
    m.add_function(wrap_pyfunction!(run_trial_timeline_json, m)?)?;
    m.add_function(wrap_pyfunction!(run_overlay_json, m)?)?;
    m.add_function(wrap_pyfunction!(run_to_mermaid, m)?)?;
    m.add_function(wrap_pyfunction!(run_to_graphviz, m)?)?;
    m.add_function(wrap_pyfunction!(graph_json_to_mermaid, m)?)?;
    m.add_function(wrap_pyfunction!(graph_json_to_svg, m)?)?;
    m.add_function(wrap_pyfunction!(run_to_svg, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
