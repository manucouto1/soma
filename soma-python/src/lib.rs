//! PyO3 bindings for Soma — exposes Graph, Study, and Filter to Python.
//!
//! Bridges Python Filter classes to the Rust Filter trait, converts
//! between Python lists/dicts and Soma Values, and wraps the StudyRunner
//! for hyperparameter optimization from Python.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict, PyList};
use std::collections::HashMap;
use std::sync::Arc;

use soma_compiler::CompileMode;
use soma_core::cache::CacheKey;
use soma_core::error::{Result as SomaResult, SomaError};
use soma_core::event::MetricRecord;
use soma_core::filter::{Filter, FilterKind, FilterMeta, StreamMode};
use soma_core::graph::{Edge, Graph, Node};
use soma_core::search::{Scale, SearchDimension, SearchSpace};
use soma_core::study::{Direction, Objective, SearchStrategy, Study};
use soma_core::value::Value;
use soma_runtime::EventBus;
use soma_runtime::cache::MemoryCache;
use soma_runtime::filter_library::FilterLibrary;
use soma_runtime::graph_session;
use soma_runtime::sampler::{BayesianSampler, GridSampler, RandomSampler, Sampler};
use soma_runtime::study_runner::{FnTrialExecutor, StudyRunner, TrialOutcome};

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
        let json_mod = py.import("json")?;
        let json_str: String = json_mod.call_method1("dumps", (obj,))?.extract()?;
        let val: serde_json::Value =
            serde_json::from_str(&json_str).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        return Ok(Value::Json(val));
    }

    if let Ok(s) = obj.extract::<String>()
        && let Ok(val) = serde_json::from_str(&s)
    {
        return Ok(Value::Json(val));
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
                Ok(values.into_pyobject(py)?.into_any().unbind())
            }
        }
        Value::Json(v) => {
            let json_str = v.to_string();
            let json_mod = py.import("json")?;
            let obj = json_mod.call_method1("loads", (json_str,))?;
            Ok(obj.unbind())
        }
        Value::Bytes(b) => Ok(b.into_pyobject(py)?.into_any().unbind()),
        Value::Empty => Ok(py.None()),
        _ => Ok(py.None()),
    }
}

// ── Python Filter wrapper ──

struct PyFilterBridge {
    py_obj: PyObject,
    name: String,
    config_hash_val: CacheKey,
}

impl PyFilterBridge {
    fn new(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let name: String = obj.get_type().name()?.to_string();

        // Build config hash from public attributes only.
        // Private attrs (_name) are internal state, not parameters.
        // This ensures: same params → same cache key, regardless of internal state.
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
        let dict_str: String = dict_sorted.extract()?;
        let config_hash = CacheKey::from_parts(&[name.as_bytes(), dict_str.as_bytes()]);

        Ok(Self {
            py_obj: obj.clone().unbind(),
            name,
            config_hash_val: config_hash,
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
            distribution: soma_core::filter::Distribution::Local,
            input_schema: None,
            output_schema: None,
        }
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
    cache: Arc<dyn soma_core::cache::CacheStore>,
    fitted: bool,
}

#[pymethods]
impl PyGraph {
    #[new]
    fn new() -> Self {
        Self {
            graph: Graph::new(),
            library: FilterLibrary::new(),
            cache: Arc::new(MemoryCache::default()),
            fitted: false,
        }
    }

    /// Add a filter node. If only a filter is given, the node id defaults
    /// to the snake_case class name. Returns the node id.
    ///
    /// Usage:
    ///   g.node(MyFilter())           # id = "my_filter"
    ///   g.node("scaler", MyFilter()) # id = "scaler"
    #[pyo3(signature = (*args))]
    fn node(&mut self, py: Python<'_>, args: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<String> {
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
                    "node() takes 1 or 2 arguments, got {n}"
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

        self.graph
            .add_node(Node::filter_with_id(&actual_id, &bridge.name));
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
    #[pyo3(signature = (x, y=None))]
    fn fit(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, pyo3::types::PyAny>,
        y: Option<&Bound<'_, pyo3::types::PyAny>>,
    ) -> PyResult<()> {
        let x_val = py_to_value(py, x)?;
        let y_val = match y {
            Some(v) => Some(py_to_value(py, v)?),
            None => None,
        };
        graph_session::graph_fit(
            &self.graph,
            &self.library,
            &x_val,
            y_val.as_ref(),
            self.cache.as_ref(),
        )
        .map_err(soma_err_to_py)?;
        self.fitted = true;
        Ok(())
    }

    /// Forward data through the compiled graph (inference mode).
    fn forward(&self, py: Python<'_>, x: &Bound<'_, pyo3::types::PyAny>) -> PyResult<PyObject> {
        if !self.fitted {
            return Err(PyRuntimeError::new_err(
                "graph must be fitted before forward",
            ));
        }
        let x_val = py_to_value(py, x)?;
        let result =
            graph_session::graph_predict(&self.graph, &self.library, &x_val, self.cache.as_ref())
                .map_err(soma_err_to_py)?;
        value_to_py(py, &result)
    }

    /// Compile and execute, returning all node outputs as a dict.
    fn run(&self, py: Python<'_>) -> PyResult<PyObject> {
        let outputs = graph_session::graph_run(
            &self.graph,
            &self.library,
            CompileMode::NoCache,
            self.cache.as_ref(),
        )
        .map_err(soma_err_to_py)?;

        let dict = PyDict::new(py);
        for (k, v) in &outputs {
            dict.set_item(k, value_to_py(py, v)?)?;
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

        let result = soma_compiler::compile(
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

// ── Module ──

#[pymodule]
fn _soma(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGraph>()?;
    m.add_class::<PyStudy>()?;
    m.add("__version__", "0.1.0")?;
    Ok(())
}
