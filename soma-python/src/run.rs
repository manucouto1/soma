//! `Run` — a tracked run and what it records.

use crate::prelude::*;

// ── PyRun ──

/// A tracked run bound to a graph's event bus.
///
/// Created by `Graph.begin_run` (or the `graph.track_run(...)` context
/// manager). Metrics logged here become `MetricReported` events, which
/// the run's sink persists to `metrics.jsonl`/`events.jsonl`.
#[pyclass(name = "Run")]
pub(crate) struct PyRun {
    pub(crate) tracker: Arc<LocalTracker>,
    pub(crate) bus: Arc<EventBus>,
    pub(crate) sink: Arc<dyn somatize_core::tracking::EventSink>,
    pub(crate) finished: std::sync::atomic::AtomicBool,
    /// W&B-style summary: last logged value per metric name, written
    /// into the experiments journal on finish.
    pub(crate) summary: std::sync::Mutex<HashMap<String, f64>>,
}

#[pymethods]
impl PyRun {
    #[getter]
    fn id(&self) -> String {
        self.tracker.run_id().to_string()
    }

    /// Path of the run directory — relative whenever the pool root was.
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
            let metrics = self.summary.lock().map(|s| s.clone()).unwrap_or_default();
            append_run_record(self.tracker.run_dir(), HashMap::new(), metrics);
            // HEAD advances only on success: a run that died must never
            // become the parent of everything that follows it.
            if let Some(root) = tracking_root(self.tracker.run_dir()) {
                advance_head(root, self.tracker.run_id());
            }
        }
        Ok(())
    }
}

/// The tracking root (`.soma`) a run directory lives under.
pub(crate) fn tracking_root(run_dir: &std::path::Path) -> Option<&std::path::Path> {
    run_dir.parent().and_then(|p| p.parent())
}

/// Append a finished run's summary to `<root>/experiments.jsonl`,
/// linked to its parent by the derivation move between them.
///
/// This is the single path from "a run happened" to "the pool knows
/// about it": `RunReader` → `summarize` → `ExperimentRecord::from_run`.
/// `extra_params` and `extra_metrics` carry what the run directory does
/// not know (a study's best-trial configuration, a training loop's
/// W&B-style summary metrics).
///
/// Best-effort throughout: recording an experiment must never fail a
/// training run that already produced its results.
pub(crate) fn append_run_record(
    run_dir: &std::path::Path,
    extra_params: HashMap<String, serde_json::Value>,
    extra_metrics: HashMap<String, f64>,
) {
    use somatize_memory::{ExperimentRecord, FileKnowledgeBase, KnowledgeBase};

    let Some(root) = tracking_root(run_dir) else {
        return;
    };
    let reader = match RunReader::open(run_dir) {
        Ok(reader) => reader,
        Err(e) => {
            eprintln!(
                "soma: cannot read run {} to record it: {e}",
                run_dir.display()
            );
            return;
        }
    };
    let summary = match summarize(&reader) {
        Ok(summary) => summary,
        Err(e) => {
            eprintln!("soma: cannot summarize run {}: {e}", run_dir.display());
            return;
        }
    };

    let mut record = ExperimentRecord::from_run(&summary)
        .with_extra_params(extra_params)
        .with_extra_metrics(extra_metrics);

    let mut kb = match FileKnowledgeBase::open(root.join("experiments.jsonl")) {
        Ok(kb) => kb,
        Err(e) => {
            eprintln!("soma: failed to open experiments.jsonl: {e}");
            return;
        }
    };
    if let Some(parent_id) = summary.parent_run_id.clone()
        && let Ok(Some(parent)) = kb.get(&parent_id)
    {
        record = record.descended_from(&parent);
    }
    if let Err(e) = kb.record(record) {
        eprintln!("soma: failed to record experiment: {e}");
    }
}
