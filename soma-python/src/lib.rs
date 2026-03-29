use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict, PyList};
use std::collections::HashMap;
use std::sync::Arc;

use soma_core::cache::CacheKey;
use soma_core::error::{Result as SomaResult, SomaError};
use soma_core::event::MetricRecord;
use soma_core::filter::{Filter, FilterKind, FilterMeta, StreamMode};
use soma_core::search::{Scale, SearchDimension, SearchSpace};
use soma_core::study::{Direction, Objective, SearchStrategy, Study};
use soma_core::value::Value;
use soma_runtime::sampler::{GridSampler, RandomSampler, Sampler};
use soma_runtime::sampler_bayesian::BayesianSampler;
use soma_runtime::study_runner::{FnTrialExecutor, StudyRunner, TrialOutcome};
use soma_runtime::{EventBus, Pipeline};

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
        let dict = obj.getattr("__dict__")?;
        let json_mod = py.import("json")?;
        let dict_sorted = json_mod.call_method(
            "dumps",
            (dict.clone(),),
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
                    "evolving" => StreamMode::Evolving { checkpoint_every: 100 },
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

// ── PyPipeline ──

#[pyclass(name = "Pipeline")]
struct PyPipeline {
    pipeline: Pipeline,
}

#[pymethods]
impl PyPipeline {
    #[new]
    fn new(py: Python<'_>, filters: &Bound<'_, PyList>) -> PyResult<Self> {
        let mut named_filters: Vec<(String, Box<dyn Filter>)> = Vec::new();
        for (i, item) in filters.iter().enumerate() {
            let name = item
                .getattr("__class__")
                .and_then(|c| c.getattr("__name__"))
                .and_then(|n| n.extract::<String>())
                .unwrap_or_else(|_| format!("filter_{i}"));
            let bridge = PyFilterBridge::new(py, &item)?;
            named_filters.push((name, Box::new(bridge)));
        }
        Ok(Self {
            pipeline: Pipeline::new(named_filters),
        })
    }

    #[pyo3(signature = (x, y=None))]
    fn fit(
        &mut self,
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        y: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let x_val = py_to_value(py, x)?;
        let y_val = match y {
            Some(v) => Some(py_to_value(py, v)?),
            None => None,
        };
        self.pipeline
            .fit(&x_val, y_val.as_ref())
            .map_err(soma_err_to_py)
    }

    fn predict(&self, py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let x_val = py_to_value(py, x)?;
        let result = self.pipeline.predict(&x_val).map_err(soma_err_to_py)?;
        value_to_py(py, &result)
    }

    fn is_fitted(&self) -> bool {
        self.pipeline.is_fitted()
    }

    fn filter_names(&self) -> Vec<String> {
        self.pipeline
            .filter_names()
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// Get the aggregated search space from all filters as a list of dicts.
    fn search_space(&self, py: Python<'_>) -> PyResult<PyObject> {
        // Collect _soma_search_space from each filter's Python class
        // (this is stored at the Pipeline level via the bridge)
        // For now return the filter names that have search spaces
        let result = PyList::empty(py);
        // Pipeline doesn't currently track search spaces, return empty
        // (search spaces are accessed via Filter class directly)
        Ok(result.into_any().unbind())
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

// ── Module ──

#[pymodule]
fn _soma(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPipeline>()?;
    m.add_class::<PyStudy>()?;
    m.add("__version__", "0.1.0")?;
    Ok(())
}
