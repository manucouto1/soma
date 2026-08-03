//! `Study` and `Trial`: hyperparameter search from Python.

use crate::prelude::*;
use crate::run::append_run_record;

// ── Search dimension parsing ──

/// Read one search dimension out of a Python dict.
///
/// Every required key is reported, none is unwrapped. `low`, `high` and
/// `choices` used to be `.unwrap()` while `type` and `name` — two lines
/// above them — used `ok_or_else`. So a plausible typo produced a
/// `PanicException`, which does not inherit `Exception` and is therefore
/// not caught by `except Exception`: a study builder would take the
/// interpreter down instead of reporting a bad search space.
fn parse_py_search_dim(_py: Python<'_>, item: &Bound<'_, PyAny>) -> PyResult<SearchDimension> {
    let dict = item.downcast::<PyDict>()?;

    /// The key, or a message naming the dimension and what it needs.
    fn required<'a>(
        dict: &Bound<'a, PyDict>,
        key: &str,
        name: &str,
        dtype: &str,
    ) -> PyResult<Bound<'a, PyAny>> {
        dict.get_item(key)?.ok_or_else(|| {
            PyValueError::new_err(format!(
                "search dimension `{name}` of type `{dtype}` needs `{key}`"
            ))
        })
    }

    let dtype: String = dict
        .get_item("type")?
        .ok_or_else(|| PyValueError::new_err("a search dimension needs `type`"))?
        .extract()?;
    let name: String = dict
        .get_item("name")?
        .ok_or_else(|| PyValueError::new_err("a search dimension needs `name`"))?
        .extract()?;

    match dtype.as_str() {
        "float" => {
            let low: f64 = required(dict, "low", &name, "float")?.extract()?;
            let high: f64 = required(dict, "high", &name, "float")?.extract()?;
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
            let low: i64 = required(dict, "low", &name, "int")?.extract()?;
            let high: i64 = required(dict, "high", &name, "int")?.extract()?;
            Ok(SearchDimension::Int {
                name,
                low,
                high,
                scale: Scale::Linear,
            })
        }
        "categorical" => {
            let choices_py = required(dict, "choices", &name, "categorical")?;
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
pub(crate) struct PyTrial {
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
            dict.set_item(k, json_to_py(py, v)?)?;
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
        match self.params.get(key) {
            Some(v) => json_to_py(py, v),
            None => Err(pyo3::exceptions::PyKeyError::new_err(key.to_string())),
        }
    }

    fn __contains__(&self, key: &str) -> bool {
        self.params.contains_key(key)
    }

    #[pyo3(signature = (key, default=None))]
    fn get(&self, py: Python<'_>, key: &str, default: Option<PyObject>) -> PyResult<PyObject> {
        match self.params.get(key) {
            Some(v) => json_to_py(py, v),
            None => Ok(default.unwrap_or_else(|| py.None())),
        }
    }

    fn keys(&self) -> Vec<String> {
        self.params.keys().cloned().collect()
    }
}

// ── PyStudy ──

#[pyclass(name = "Study")]
pub(crate) struct PyStudy {
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

/// Append a completed study's summary as an ExperimentRecord.
///
/// Goes through the same run-directory path as every other run, so a
/// study lands in the pool with a conclusion, a lineage and a trial
/// breakdown instead of a hand-rolled record. The best trial's
/// configuration and metrics are the extras only the study knows.
pub(crate) fn record_study_experiment(run_dir: &std::path::Path, study: &Study) {
    let (params, metrics) = match study.best_trial() {
        Some(best) => (
            best.params.clone().into_iter().collect(),
            best.metrics
                .iter()
                .map(|m| (m.name.clone(), m.value))
                .collect(),
        ),
        None => (HashMap::new(), HashMap::new()),
    };
    append_run_record(run_dir, params, metrics);
}

pub(crate) fn trial_to_py(
    py: Python<'_>,
    trial: &somatize_core::study::Trial,
) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("id", &trial.id)?;
    let params_dict = PyDict::new(py);
    for (k, v) in &trial.params {
        params_dict.set_item(k, json_to_py(py, v)?)?;
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
                        manifest.parent_run_id = resolve_parent(&self.root, None);
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
                record_study_experiment(t.run_dir(), &self.study);
                advance_head(&self.root, t.run_id());
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
