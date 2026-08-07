//! Worker — receives and executes plans from a coordinator.

use crate::error::{Result, WorkerError};
use crate::protocol::*;
use somatize_core::cache::CacheStore;
use somatize_core::data::store::{DataStore, LocalDataStore};
use somatize_core::data::value::Value;
use somatize_core::graph::filter::Filter;
use somatize_core::tracking::event::Event;
use somatize_runtime::{EventBus, MemoryCache, NodeCatalog, Runner};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// Worker state: manages execution of plans received from a coordinator.
///
/// Five fields, grouped by what each is *for*. They used to be nine loose
/// ones, and the three that a run needs sat beside the two that decide
/// where data lives beside the two that start Python — so every method
/// reached past the concerns it did not care about to find the one it did.
pub struct Worker {
    /// The identity this worker registers and reports under.
    pub id: WorkerId,
    /// What this worker can run, announced to the coordinator at
    /// registration.
    pub capabilities: Capabilities,
    execution: Execution,
    stores: Stores,
    python: PythonRuntime,
}

/// What running a plan needs.
///
/// Exactly the three things [`RunContext::linear`] is built from, which is
/// why they are one field: no method has ever wanted two of them.
///
/// [`RunContext::linear`]: somatize_runtime::execution::runner::RunContext::linear
struct Execution {
    /// Implementations and trained states for every node the worker knows.
    catalog: NodeCatalog,
    /// Output cache consulted and filled by the runner.
    cache: Arc<dyn CacheStore>,
    /// Bus the worker's node events go out on.
    events: Arc<EventBus>,
}

impl Execution {
    /// A run over this worker's catalog, cache and bus.
    ///
    /// `linear`, explicitly: a worker receives a serialized plan and no
    /// graph, so it has no topology to consult. Correct for the pipelines
    /// that get dispatched, and stated here rather than assumed inside the
    /// runner.
    fn run_context<'a>(
        &'a self,
        plan: &'a SerializedPlan,
        run_id: &'a str,
    ) -> somatize_runtime::execution::runner::RunContext<'a> {
        let mut ctx = somatize_runtime::execution::runner::RunContext::linear(
            &self.catalog,
            self.cache.as_ref(),
            &self.events,
            run_id,
            &plan.plan,
        );
        ctx.seed = plan.seed;
        ctx
    }
}

/// Where a plan's data comes from and where its output goes.
///
/// Two stores, and the difference matters: one is the user's and may not
/// exist, the other is the worker's own and always does.
struct Stores {
    /// S3, Zarr, ... — configured by the user. `None` means inline only.
    persistent: Option<Arc<dyn DataStore>>,
    /// Backs the HTTP bulk-upload endpoint. Auto-created, auto-cleaned.
    temp: Arc<LocalDataStore>,
}

/// How this worker runs Python.
struct PythonRuntime {
    /// Isolated environments per requirement set.
    envs: crate::env_manager::EnvManager,
    /// Which interpreter to unpickle filters in, when no venv is needed.
    ///
    /// A cloudpickled filter can only be reconstructed by an interpreter
    /// close enough to the one that pickled it. Defaulting to `python3`
    /// off `PATH` means the worker will happily pick a different minor
    /// version from the process that sent the work, and cloudpickle then
    /// returns the class's `__dict__` instead of an instance — which
    /// surfaces as `'dict' object is not callable`, from inside a
    /// subprocess, with nothing pointing at the version gap.
    interpreter: String,
}

/// Rows per chunk when auto-streaming from a DataStore — and the threshold
/// that triggers that path at all.
///
/// One constant, deliberately: the decision to stream and the size of the
/// chunks it streams are the same number, and a plan just over the line
/// producing a single chunk is the intended behaviour, not a coincidence.
const STREAM_CHUNK_ROWS: usize = 1024;

/// `$SOMA_PYTHON`, else `python3` off `PATH`.
///
/// The env var exists because a worker started from a shell has no other
/// way to be told, and `python3` is frequently not the interpreter whose
/// pickles it will be asked to read.
fn default_python() -> String {
    std::env::var("SOMA_PYTHON")
        .ok()
        .filter(|p| !p.is_empty())
        .unwrap_or_else(|| "python3".to_string())
}

impl Worker {
    /// A worker with an in-memory cache, an empty catalog, and per-worker
    /// temp/env directories derived from `id`. Filters arrive later, with
    /// the plans; the interpreter defaults to `$SOMA_PYTHON`, then
    /// `python3` — see [`Worker::with_python`] for why that matters.
    pub fn new(id: impl Into<String>, capabilities: Capabilities) -> Self {
        let worker_id: String = id.into();
        let temp_path = std::env::temp_dir().join(format!("soma-uploads-{worker_id}"));
        let temp_store = LocalDataStore::new(temp_path);
        let env_path = std::env::temp_dir().join(format!("soma-envs-{worker_id}"));
        Self {
            id: worker_id,
            capabilities,
            execution: Execution {
                catalog: NodeCatalog::new(),
                cache: Arc::new(MemoryCache::default()),
                events: Arc::new(EventBus::new(256)),
            },
            stores: Stores {
                persistent: None,
                temp: Arc::new(temp_store),
            },
            python: PythonRuntime {
                envs: crate::env_manager::EnvManager::new(
                    env_path,
                    crate::env_manager::EnvType::Venv,
                ),
                interpreter: default_python(),
            },
        }
    }

    /// Run filters in this interpreter rather than whatever `python3`
    /// resolves to.
    ///
    /// An embedding process should pass its own `sys.executable`: it is
    /// the interpreter that pickled the filters, so it is the only one
    /// certain to unpickle them.
    pub fn with_python(mut self, python: impl Into<String>) -> Self {
        self.python.interpreter = python.into();
        self
    }

    /// Set a custom cache store (e.g. tiered or shared).
    pub fn with_cache(mut self, cache: Arc<dyn CacheStore>) -> Self {
        self.execution.cache = cache;
        self
    }

    /// Set a persistent DataStore (S3, Zarr, etc.) for large data references.
    pub fn with_data_store(mut self, store: Arc<dyn DataStore>) -> Self {
        self.stores.persistent = Some(store);
        self
    }

    /// Set a custom temp directory for HTTP bulk uploads.
    pub fn with_temp_dir(mut self, path: std::path::PathBuf) -> Self {
        self.stores.temp = Arc::new(LocalDataStore::new(path));
        self
    }

    /// Get the temp store (for HTTP upload endpoint).
    pub fn temp_store(&self) -> &Arc<LocalDataStore> {
        &self.stores.temp
    }

    /// Register a filter that this worker can execute.
    pub fn register_filter(&mut self, node_id: impl Into<String>, filter: Box<dyn Filter>) {
        self.execution.catalog.register(node_id, filter);
    }

    /// Get a filter by node_id.
    pub fn get_filter(&self, node_id: &str) -> Option<Arc<dyn Filter>> {
        self.execution.catalog.get(node_id)
    }

    /// The node catalog — what a stream driver is built over.
    pub fn catalog(&self) -> &NodeCatalog {
        &self.execution.catalog
    }

    /// The worker's event bus.
    pub fn event_bus(&self) -> &Arc<EventBus> {
        &self.execution.events
    }

    /// The worker's cache store.
    pub fn cache(&self) -> &Arc<dyn CacheStore> {
        &self.execution.cache
    }

    /// Get trained state for a filter.
    pub fn get_filter_state(&self, node_id: &str) -> Arc<Value> {
        self.execution
            .catalog
            .get_state(node_id)
            .unwrap_or_else(|| Arc::new(Value::Empty))
    }

    /// Reach the live Python process behind a node, if it has one.
    ///
    /// Every per-node operation that has to talk to the subprocess goes
    /// through here rather than repeating the downcast.
    fn subprocess_for(
        &self,
        node_id: &str,
    ) -> Option<Arc<std::sync::Mutex<crate::python_process::PythonProcess>>> {
        let filter = self.execution.catalog.get(node_id)?;
        let sf = filter
            .as_any()
            .downcast_ref::<crate::python_process::SubprocessFilter>()?;
        Some(sf.process.clone())
    }

    /// Trained state of one or more nodes, read from the Python process.
    ///
    /// The four methods below back the wire messages of the same names.
    /// They existed on `PythonProcess` and in the daemon script from the
    /// start; what was missing was anything calling them, so
    /// `soma-worker/src/server.rs` answered all four with "not implemented
    /// for SubprocessFilter" and `DataParallel` could not run.
    pub fn read_states(&self, node_ids: &[String]) -> Result<HashMap<String, Value>> {
        let mut out = HashMap::new();
        for node_id in node_ids {
            let Some(proc) = self.subprocess_for(node_id) else {
                // A Rust filter keeps its state in the catalog.
                out.insert(node_id.clone(), (*self.get_filter_state(node_id)).clone());
                continue;
            };
            let mut guard = proc
                .lock()
                .map_err(|e| WorkerError::Concurrency(format!("process mutex poisoned: {e}")))?;
            out.insert(node_id.clone(), guard.get_state(node_id)?);
        }
        Ok(out)
    }

    /// Load states into the Python process (and the catalog beside it).
    pub fn write_states(&mut self, states: &HashMap<String, Value>) -> Result<()> {
        for (node_id, state) in states {
            if let Some(proc) = self.subprocess_for(node_id) {
                let mut guard = proc.lock().map_err(|e| {
                    WorkerError::Concurrency(format!("process mutex poisoned: {e}"))
                })?;
                guard.set_state(node_id, state)?;
            }
            self.set_filter_state(node_id, state.clone())?;
        }
        Ok(())
    }

    /// Gradients currently held by each node's parameters.
    pub fn read_gradients(&self, node_ids: &[String]) -> Result<HashMap<String, Value>> {
        let mut out = HashMap::new();
        for node_id in node_ids {
            let proc = self.subprocess_for(node_id).ok_or_else(|| {
                WorkerError::Env(format!(
                    "`{node_id}` has no Python process, so it has no gradients to read"
                ))
            })?;
            let mut guard = proc
                .lock()
                .map_err(|e| WorkerError::Concurrency(format!("process mutex poisoned: {e}")))?;
            out.insert(node_id.clone(), guard.get_gradients(node_id)?);
        }
        Ok(out)
    }

    /// Apply aggregated gradients to each node's parameters.
    pub fn write_gradients(&self, gradients: &HashMap<String, Value>) -> Result<()> {
        for (node_id, grads) in gradients {
            let proc = self.subprocess_for(node_id).ok_or_else(|| {
                WorkerError::Env(format!(
                    "`{node_id}` has no Python process, so gradients cannot be applied"
                ))
            })?;
            let mut guard = proc
                .lock()
                .map_err(|e| WorkerError::Concurrency(format!("process mutex poisoned: {e}")))?;
            guard.apply_gradients(node_id, grads)?;
        }
        Ok(())
    }

    /// Set trained state for a filter.
    pub fn set_filter_state(&mut self, node_id: &str, state: Value) -> Result<()> {
        self.execution
            .catalog
            .try_set_state(node_id, state)
            .map_err(|e| {
                WorkerError::Env(format!(
                    "could not store the trained state for `{node_id}`: {e}. \
                 Refusing to continue: the node would run from random weights."
                ))
            })
    }

    /// Wrap output in the right delivery: inline for small, DataRef for large.
    pub fn wrap_output(&self, output: Value) -> OutputDelivery {
        let size = serde_json::to_vec(&output).map(|v| v.len()).unwrap_or(0);
        if size >= somatize_core::data::store::INLINE_THRESHOLD_BYTES {
            let key = somatize_core::cache::CacheKey::hash_data(
                &serde_json::to_vec(&output).unwrap_or_default(),
            );
            if let Ok(data_ref) = self.stores.temp.put(&key, &output) {
                return OutputDelivery::Reference { data_ref };
            }
        }
        OutputDelivery::Inline { value: output }
    }

    /// Subscribe to execution events.
    pub fn subscribe(&self) -> tokio::sync::broadcast::Receiver<Event> {
        self.execution.events.subscribe()
    }

    /// Build a registration message.
    pub fn registration_message(&self) -> WorkerToCoordinator {
        WorkerToCoordinator::Register {
            worker_id: self.id.clone(),
            capabilities: self.capabilities.clone(),
        }
    }

    /// Execute a serialized plan.
    ///
    /// If the plan contains serialized filter definitions, they are registered
    /// temporarily for this execution (alongside any pre-registered filters).
    ///
    /// In **Fit** mode: fits each filter (topological order), stores trained states,
    /// then forwards to propagate outputs. Returns states so the client can cache them.
    ///
    /// In **Forward** mode: executes the compiled plan directly.
    pub fn execute_plan(&mut self, plan: &SerializedPlan) -> PlanResult {
        let start = Instant::now();

        // Before anything else. A plan this build only partly understands
        // must be refused, not executed with the parts it recognised.
        if let Err(message) = plan.check_version() {
            tracing::error!("{message}");
            return PlanResult::Failed {
                error: message,
                duration_ms: start.elapsed().as_millis() as u64,
            };
        }

        let _span = tracing::info_span!(
            "execute_plan",
            plan_id = %plan.plan_id,
            n_filters = plan.filters.len(),
            mode = ?plan.mode,
        )
        .entered();

        tracing::info!(
            "Plan received: {} filters, mode={:?}",
            plan.filters.len(),
            plan.mode
        );

        let python_path = self.interpreter_for(plan);
        if let Err(failed) = self.install_python_filters(plan, &python_path, start) {
            return *failed;
        }

        let input = match self.resolve_plan_input(plan, start) {
            Ok(value) => value,
            Err(failed) => return *failed,
        };

        if let Some(streamed) = self.try_streamed_from_store(plan, start) {
            return streamed;
        }

        let outcome = self.run_in_mode(plan, &input.unwrap_or(Value::Empty));
        self.finish(outcome, start)
    }

    /// Which interpreter this plan's filters will be unpickled in.
    ///
    /// A plan with pip requirements gets a venv; one without gets whatever
    /// [`Worker::with_python`] was told. A venv that cannot be built falls
    /// back to the system interpreter with a `warn!` — that fallback is
    /// [D-24](/soma/internals/debt#d-24--venv-provisioning-fails-into-the-system-interpreter)
    /// and is deliberately left as it was here; changing it is its own
    /// finding, not a side effect of splitting this function.
    fn interpreter_for(&mut self, plan: &SerializedPlan) -> String {
        // Collect all requirements from serialized filters
        let all_reqs: Vec<String> = plan
            .filters
            .iter()
            .flat_map(|sf| sf.requirements.iter().cloned())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        // Create/reuse venv if there are pip requirements, otherwise use system python
        let python_path = if all_reqs.is_empty() {
            self.python.interpreter.clone()
        } else {
            let reqs_str = all_reqs.join("\n");
            // Keyed by the requirements, not by the plan. A plan id is a
            // fresh timestamp, so keying on it meant every plan built its
            // own venv and pip-installed into it — never reusing anything,
            // and never cleaning up. Plans that need the same packages now
            // share one environment, which is what the lockfile inside it
            // was already written to support.
            let env_id = crate::env_manager::EnvManager::env_id_for(&reqs_str);
            match self.python.envs.ensure_env(&env_id, &reqs_str) {
                Ok(path) => {
                    tracing::info!("Using venv {env_id} for plan {}: {:?}", plan.plan_id, path);
                    path.to_string_lossy().to_string()
                }
                Err(e) => {
                    tracing::warn!("Failed to create venv, falling back to system python: {e}");
                    self.python.interpreter.clone()
                }
            }
        };

        // No site-packages resolution needed — subprocess uses the venv python directly
        python_path
    }

    /// Spawn the plan's Python process, restore any trained states into it,
    /// and register every filter in the catalog.
    ///
    /// One process for all of a plan's filters, deliberately: `Composite`
    /// autograd needs them to share an interpreter.
    ///
    /// `Err` carries the reply to send, already timed — the three ways this
    /// stage fails (no interpreter, a state that will not load into the
    /// process, a state that will not store in the catalog) are all fatal,
    /// and none of them has anything to add after the fact.
    fn install_python_filters(
        &mut self,
        plan: &SerializedPlan,
        python_path: &str,
        start: Instant,
    ) -> std::result::Result<(), Box<PlanResult>> {
        let filter_specs: Vec<(String, Vec<u8>, bool)> = plan
            .filters
            .iter()
            .map(|sf| (sf.node_id.clone(), sf.pickled_filter.clone(), sf.trainable))
            .collect();

        if !filter_specs.is_empty() {
            let filter_names: Vec<&str> =
                plan.filters.iter().map(|sf| sf.node_id.as_str()).collect();
            tracing::info!(
                python = %python_path,
                filters = ?filter_names,
                "Spawning Python process for {} filters",
                filter_specs.len()
            );

            let mut proc = crate::python_process::PythonProcess::spawn(python_path, &filter_specs)
                .map_err(|e| {
                    // Not `.expect`: this runs on a tokio worker thread, so
                    // a panic here took the whole worker down — every other
                    // pipeline it was holding with it — because one plan
                    // named an interpreter that would not start.
                    tracing::error!(python = %python_path, "failed to spawn Python: {e}");
                    Box::new(PlanResult::Failed {
                        error: format!("could not start `{python_path}`: {e}"),
                        duration_ms: start.elapsed().as_millis() as u64,
                    })
                })?;

            // Load trained states from previous epochs (SET_STATE)
            for sf in &plan.filters {
                if let Some(state) = &sf.state {
                    let size = match state {
                        Value::Bytes(b) => b.len(),
                        _ => 0,
                    };
                    tracing::info!(
                        node_id = %sf.node_id,
                        size_bytes = size,
                        "Loading trained state from previous epoch"
                    );
                    if let Err(e) = proc.set_state(&sf.node_id, state) {
                        return Err(Box::new(state_restore_failed(
                            &sf.node_id,
                            "the Python process that owns the weights",
                            &e,
                            start,
                        )));
                    }
                }
            }

            let process = Arc::new(std::sync::Mutex::new(proc));

            for sf in &plan.filters {
                let config_hash = sf.config_hash.clone().unwrap_or_else(|| {
                    crate::python_process::SubprocessFilter::fallback_config_hash(
                        &sf.node_id,
                        &sf.pickled_filter,
                    )
                });
                let filter = Box::new(crate::python_process::SubprocessFilter::new(
                    process.clone(),
                    sf.node_id.clone(),
                    sf.trainable,
                    config_hash,
                ));
                self.execution.catalog.register(&sf.node_id, filter);
                if let Some(state) = &sf.state
                    && let Err(e) = self.set_filter_state(&sf.node_id, state.clone())
                {
                    return Err(Box::new(state_restore_failed(
                        &sf.node_id,
                        "the catalog the executor reads",
                        &e,
                        start,
                    )));
                }
            }

            tracing::info!("Filters registered, Python process ready");
        }
        Ok(())
    }

    /// The plan's input, materialized.
    ///
    /// A reference that resolves nowhere fails HERE, naming what it looked
    /// in — it used to become an empty value and travel on into the filter,
    /// where it surfaced as a `TypeError` in the user's own code.
    fn resolve_plan_input(
        &self,
        plan: &SerializedPlan,
        start: Instant,
    ) -> std::result::Result<Option<Value>, Box<PlanResult>> {
        plan.input
            .as_ref()
            .map(|src| src.resolve(self.stores.persistent.as_deref(), &self.stores.temp))
            .transpose()
            .map_err(|e| {
                Box::new(PlanResult::Failed {
                    error: e.to_string(),
                    duration_ms: start.elapsed().as_millis() as u64,
                })
            })
    }

    /// Take the DataStore-backed streaming path, if this plan qualifies:
    /// read chunks via `get_rows()` and stream them, never materializing
    /// the dataset.
    ///
    /// `None` means "not this path" — the caller runs the plan normally.
    ///
    /// **Forward only, and the mode check is the whole point.** This branch
    /// used to be taken before `plan.mode` was ever looked at, so a Fit over
    /// a large reference was silently executed as a stream of forwards:
    /// nothing was fitted, no state was stored, and the fit reported
    /// success. The next forward then found no state and died inside the
    /// user's filter — while the same graph under the 1024-row threshold
    /// fitted normally and gave the right answer. A stream has no fit
    /// semantics (`compile_stream` refuses one locally, for the same
    /// reason), so a Fit falls through to the path that can honour it.
    fn try_streamed_from_store(
        &mut self,
        plan: &SerializedPlan,
        start: Instant,
    ) -> Option<PlanResult> {
        if matches!(plan.mode, ExecutionMode::Forward)
            && let Some(InputSource::Reference { data_ref }) = &plan.input
            && let Some(store) = self.stores.persistent.clone()
            && let Ok(meta) = store.meta(data_ref)
            && meta.total_rows > STREAM_CHUNK_ROWS
        {
            return Some(self.execute_streamed_from_store(plan, &store, data_ref, &meta, start));
        }
        None
    }

    /// Run the plan in whichever mode it asked for.
    ///
    /// The three arms are the three things a worker can be told to do, and
    /// they differ in more than a flag: a batched fit talks to the
    /// subprocess directly, a plain fit goes through the runner, and a
    /// forward learns nothing. All three answer with the same pair — the
    /// output, and whatever was learned.
    fn run_in_mode(
        &mut self,
        plan: &SerializedPlan,
        x: &Value,
    ) -> somatize_core::error::Result<(Value, HashMap<String, Value>)> {
        match &plan.mode {
            ExecutionMode::Fit {
                y,
                batch_size: Some(bs),
            } => self.run_batched_fit(plan, x, y.as_ref(), *bs),
            ExecutionMode::Fit { y, .. } => self.run_fit(plan, x, y.as_ref()),
            ExecutionMode::Forward => self.run_forward(plan, x),
        }
    }

    /// Fit in batches, driving the subprocess directly rather than the
    /// runner — the batching lives on the Python side.
    fn run_batched_fit(
        &mut self,
        plan: &SerializedPlan,
        x: &Value,
        y: Option<&Value>,
        batch_size: usize,
    ) -> somatize_core::error::Result<(Value, HashMap<String, Value>)> {
        tracing::info!(batch_size, "Using batched fit");
        let node_ids = plan
            .plan
            .node_ids()
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        let Some(filter) = node_ids
            .first()
            .and_then(|id| self.execution.catalog.get(id))
        else {
            return Err(somatize_core::error::SomaError::Other(
                "no filters found".into(),
            ));
        };
        let Some(sf) = filter
            .as_any()
            .downcast_ref::<crate::python_process::SubprocessFilter>()
        else {
            return Err(somatize_core::error::SomaError::Other(
                "batched_fit requires SubprocessFilter".into(),
            ));
        };

        let (output, states) = sf
            .process
            .lock()
            .map_err(|e| WorkerError::Concurrency(format!("process mutex poisoned: {e}")))
            .and_then(|mut proc| proc.batched_fit(&node_ids, x, y, batch_size))?;

        self.keep_what_was_learned(&states)?;
        Ok((output, states))
    }

    /// Fit through the runner — the same execution path a local fit takes.
    fn run_fit(
        &mut self,
        plan: &SerializedPlan,
        x: &Value,
        y: Option<&Value>,
    ) -> somatize_core::error::Result<(Value, HashMap<String, Value>)> {
        let run_id = format!("worker_fit_{}", plan.plan_id);
        let ctx = self.execution.run_context(plan, &run_id);
        let fitted = somatize_runtime::LocalRunner.fit(&plan.plan, &ctx, x, y)?;
        // The runner already told states and outputs apart — this used to
        // re-derive the split from a key prefix, as did three other callers.
        self.keep_what_was_learned(&fitted.states)?;
        Ok((fitted.last, fitted.states))
    }

    /// Forward through the runner. Learns nothing, so it answers with an
    /// empty state map rather than pretending it might have one.
    fn run_forward(
        &mut self,
        plan: &SerializedPlan,
        x: &Value,
    ) -> somatize_core::error::Result<(Value, HashMap<String, Value>)> {
        let run_id = format!("worker_forward_{}", plan.plan_id);
        let ctx = self.execution.run_context(plan, &run_id);
        let output = somatize_runtime::LocalRunner.forward(&plan.plan, &ctx, x)?;
        Ok((output, HashMap::new()))
    }

    /// Store what a fit learned, so a later plan on this worker sees it.
    ///
    /// Both fit paths used to log a failure here and carry on, which is the
    /// same shape as [D-25](/soma/internals/debt#d-25--state-load-failure-silently-restarts-from-random-init)
    /// pointing the other way: the states still travel back to the client,
    /// but this worker's catalog keeps the *old* ones, so the next Forward
    /// sent here runs the previous epoch's weights and looks like a model
    /// that stopped learning.
    fn keep_what_was_learned(
        &mut self,
        states: &HashMap<String, Value>,
    ) -> somatize_core::error::Result<()> {
        for (node_id, state) in states {
            self.set_filter_state(node_id, state.clone())?;
        }
        Ok(())
    }

    /// Turn what the run returned into the reply, and time it.
    fn finish(
        &self,
        outcome: somatize_core::error::Result<(Value, HashMap<String, Value>)>,
        start: Instant,
    ) -> PlanResult {
        let duration_ms = start.elapsed().as_millis() as u64;
        match outcome {
            Ok((output, states)) => {
                tracing::info!(
                    duration_ms,
                    n_states = states.len(),
                    "Plan completed successfully"
                );
                PlanResult::Success {
                    output: self.wrap_output(output),
                    duration_ms,
                    states,
                }
            }
            Err(e) => {
                tracing::error!(duration_ms, error = %e, "Plan failed");
                PlanResult::Failed {
                    error: e.to_string(),
                    duration_ms,
                }
            }
        }
    }

    /// DataStore-backed streaming: read chunks via get_rows() and drive
    /// them through the runtime's `StreamRun` — the same primitives,
    /// cache, and per-node events as a local stream, without loading the
    /// dataset into memory. The concatenated output is the plan result.
    fn execute_streamed_from_store(
        &mut self,
        plan: &SerializedPlan,
        store: &Arc<dyn DataStore>,
        data_ref: &somatize_core::data::store::DataRef,
        meta: &somatize_core::data::store::StoreMeta,
        start: Instant,
    ) -> PlanResult {
        use somatize_runtime::{Context, StreamOutput, StreamRun};

        let node_ids: Vec<String> = plan.plan.node_ids().into_iter().map(String::from).collect();

        // StreamRun refuses a node the catalog does not know — a failed
        // plan, never a silently shorter chain (a `filter_map` here once
        // streamed a 3-node plan through 2 filters and reported success).
        let mut run = match StreamRun::new(&node_ids, &self.execution.catalog) {
            Ok(run) => run,
            Err(e) => {
                return PlanResult::Failed {
                    error: e.to_string(),
                    duration_ms: start.elapsed().as_millis() as u64,
                };
            }
        };

        let chunk_size = STREAM_CHUNK_ROWS;
        let run_id = format!("worker_stream_{}", plan.plan_id);
        let mut ctx =
            Context::new(self.execution.events.clone(), run_id.clone()).with_seed(plan.seed);

        self.execution.events.emit(Event::RunStarted {
            run_id: run_id.clone(),
            plan_summary: somatize_core::tracking::event::PlanSummary {
                total_nodes: node_ids.len(),
                cached_nodes: 0,
                parallel_branches: 0,
            },
        });
        // Every early return below is a failed run; say so on the bus
        // instead of leaving the RunStarted bracket open.
        let fail = |bus: &EventBus, error: String, start: Instant| {
            bus.emit(Event::RunFailed {
                run_id: run_id.clone(),
                error: error.clone(),
            });
            PlanResult::Failed {
                error,
                duration_ms: start.elapsed().as_millis() as u64,
            }
        };

        let mut output = StreamOutput::new();
        let total = meta.total_rows;
        let mut chunk_idx = 0;

        for row_start in (0..total).step_by(chunk_size) {
            let len = chunk_size.min(total - row_start);
            let chunk = match store.get_rows(data_ref, row_start, len) {
                Ok(c) => c,
                Err(e) => {
                    let error = format!("get_rows({row_start}..{}): {e}", row_start + len);
                    return fail(&self.execution.events, error, start);
                }
            };

            match run.process_chunk(chunk, &mut ctx, self.execution.cache.as_ref()) {
                Ok(Some(out)) => output.push(out),
                Ok(None) => {} // Barrier — accumulating
                Err(e) => {
                    return fail(
                        &self.execution.events,
                        format!("stream chunk {chunk_idx}: {e}"),
                        start,
                    );
                }
            }
            chunk_idx += 1;
        }

        // Flush barrier filters.
        match run.flush(&mut ctx, self.execution.cache.as_ref()) {
            Ok(Some(out)) => output.push(out),
            Ok(None) => {}
            Err(e) => {
                return fail(&self.execution.events, format!("stream flush: {e}"), start);
            }
        }
        run.finish(&ctx);

        tracing::info!(
            "Streamed {chunk_idx} chunks ({total} rows) in {}ms",
            start.elapsed().as_millis()
        );

        self.execution.events.emit(Event::RunCompleted {
            run_id,
            duration: start.elapsed(),
        });
        PlanResult::Success {
            output: self.wrap_output(output.finish()),
            duration_ms: start.elapsed().as_millis() as u64,
            states: std::collections::HashMap::new(),
        }
    }

    /// Check if this worker matches a remote target.
    pub fn matches_target(&self, target: &somatize_core::graph::filter::RemoteTarget) -> bool {
        match target {
            somatize_core::graph::filter::RemoteTarget::WorkerId(id) => &self.id == id,
            somatize_core::graph::filter::RemoteTarget::Tag(tag) => {
                self.capabilities.tags.contains(tag)
            }
        }
    }
}

/// A resume that cannot resume is not a resume.
///
/// Restoring a trained state has two halves — the Python process that holds
/// the weights, and the catalog the executor reads — and both used to log
/// the failure and carry on. Carrying on restarts the epoch from random
/// initialization, and *nothing in the returned metrics distinguishes that
/// from a genuinely bad run*: the loss curve is simply worse, which is
/// indistinguishable from a bad hyperparameter. The state only exists
/// because a previous epoch produced it, so failing to apply it means the
/// plan cannot do the thing it was sent to do.
///
/// The `target` names which half refused, because the two fail for
/// different reasons: the process rejects a state it cannot decode, the
/// catalog rejects one it cannot store.
fn state_restore_failed(
    node_id: &str,
    target: &str,
    error: &dyn std::fmt::Display,
    start: Instant,
) -> PlanResult {
    let error = format!(
        "could not restore the trained state for `{node_id}` into {target}: {error}. \
         Refusing to run: continuing would silently retrain from random weights."
    );
    tracing::error!(node_id = %node_id, "{error}");
    PlanResult::Failed {
        error,
        duration_ms: start.elapsed().as_millis() as u64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use somatize_compiler::ExecutionPlan;
    use somatize_core::cache::CacheKey;
    use somatize_core::data::value::Value;
    use somatize_core::error::Result as SomaResult;
    use somatize_core::graph::filter::{FilterKind, FilterMeta, StreamMode};

    struct TestDoubler;

    impl Filter for TestDoubler {
        fn config_hash(&self) -> CacheKey {
            CacheKey::from_parts(&[b"TestDoubler"])
        }
        fn fit(&self, _x: &Value, _y: Option<&Value>) -> SomaResult<Value> {
            Ok(Value::Empty)
        }
        fn forward(&self, x: &Value, _state: &Value) -> SomaResult<Value> {
            match x {
                Value::Tensor { values, shape } => {
                    let doubled: Vec<f64> = values.iter().map(|v| v * 2.0).collect();
                    Ok(Value::tensor(doubled, shape.clone()))
                }
                _ => Ok(x.clone()),
            }
        }
        fn meta(&self) -> FilterMeta {
            FilterMeta {
                name: "TestDoubler".into(),
                kind: FilterKind::Stateless,
                cacheable: true,
                differentiable: true,
                deterministic: true,
                stream_mode: StreamMode::FixedState,
                distribution: somatize_core::graph::filter::Distribution::Local,
                input_schema: None,
                output_schema: None,
            }
        }
    }

    fn make_worker() -> Worker {
        Worker::new(
            "test_worker",
            Capabilities {
                cpu_cores: 4,
                ram_bytes: 8_000_000_000,
                gpus: vec![],
                python_envs: vec![],
                tags: vec!["cpu".into(), "test".into()],
            },
        )
    }

    #[test]
    fn worker_registration() {
        let worker = make_worker();
        let msg = worker.registration_message();
        if let WorkerToCoordinator::Register {
            worker_id,
            capabilities,
        } = msg
        {
            assert_eq!(worker_id, "test_worker");
            assert_eq!(capabilities.cpu_cores, 4);
        } else {
            panic!("wrong message type");
        }
    }

    #[test]
    fn worker_executes_plan_successfully() {
        let mut worker = make_worker();
        worker.register_filter("doubler", Box::new(TestDoubler));

        let plan = SerializedPlan {
            protocol_version: PROTOCOL_VERSION,
            plan_id: "p_001".into(),
            plan: ExecutionPlan::Execute {
                node_id: "doubler".into(),
            },
            input: Some(crate::protocol::InputSource::Inline {
                value: Value::tensor(vec![1.0, 2.0, 3.0], vec![3]),
            }),
            filters: vec![],
            mode: ExecutionMode::default(),
            seed: None,
            metadata: serde_json::json!({}),
        };

        let result = worker.execute_plan(&plan);

        if let PlanResult::Success {
            output,
            duration_ms,
            ..
        } = result
        {
            let value = match output {
                OutputDelivery::Inline { value } => value,
                _ => panic!("expected inline output"),
            };
            let (data, _) = value.as_tensor().unwrap();
            assert_eq!(data, &[2.0, 4.0, 6.0]);
            assert!(duration_ms < 1000);
        } else {
            panic!("expected success, got: {result:?}");
        }
    }

    #[test]
    fn worker_handles_missing_filter() {
        let mut worker = make_worker();
        // Don't register any filters

        let plan = SerializedPlan {
            protocol_version: PROTOCOL_VERSION,
            plan_id: "p_002".into(),
            plan: ExecutionPlan::Execute {
                node_id: "nonexistent".into(),
            },
            input: None,
            filters: vec![],
            mode: ExecutionMode::default(),
            seed: None,
            metadata: serde_json::json!({}),
        };

        let result = worker.execute_plan(&plan);
        assert!(matches!(result, PlanResult::Failed { .. }));
    }

    #[test]
    fn worker_matches_target_by_id() {
        let worker = make_worker();
        assert!(
            worker.matches_target(&somatize_core::graph::filter::RemoteTarget::WorkerId(
                "test_worker".into()
            ))
        );
        assert!(
            !worker.matches_target(&somatize_core::graph::filter::RemoteTarget::WorkerId(
                "other".into()
            ))
        );
    }

    #[test]
    fn worker_matches_target_by_tag() {
        let worker = make_worker();
        assert!(
            worker.matches_target(&somatize_core::graph::filter::RemoteTarget::Tag(
                "cpu".into()
            ))
        );
        assert!(
            worker.matches_target(&somatize_core::graph::filter::RemoteTarget::Tag(
                "test".into()
            ))
        );
        assert!(
            !worker.matches_target(&somatize_core::graph::filter::RemoteTarget::Tag(
                "gpu".into()
            ))
        );
    }

    #[test]
    fn worker_executes_sequence() {
        let mut worker = make_worker();
        worker.register_filter("d1", Box::new(TestDoubler));
        worker.register_filter("d2", Box::new(TestDoubler));

        let plan = SerializedPlan {
            protocol_version: PROTOCOL_VERSION,
            plan_id: "p_003".into(),
            plan: ExecutionPlan::Sequence(vec![
                ExecutionPlan::Execute {
                    node_id: "d1".into(),
                },
                ExecutionPlan::Execute {
                    node_id: "d2".into(),
                },
            ]),
            input: Some(crate::protocol::InputSource::Inline {
                value: Value::tensor(vec![5.0], vec![1]),
            }),
            filters: vec![],
            mode: ExecutionMode::default(),
            seed: None,
            metadata: serde_json::json!({}),
        };

        let result = worker.execute_plan(&plan);
        if let PlanResult::Success { output, .. } = result {
            let value = match output {
                OutputDelivery::Inline { value } => value,
                _ => panic!("expected inline output"),
            };
            let (data, _) = value.as_tensor().unwrap();
            assert_eq!(data, &[20.0]); // 5 * 2 * 2
        } else {
            panic!("expected success");
        }
    }

    #[test]
    fn worker_emits_events() {
        let mut worker = make_worker();
        worker.register_filter("doubler", Box::new(TestDoubler));
        let mut rx = worker.subscribe();

        let plan = SerializedPlan {
            protocol_version: PROTOCOL_VERSION,
            plan_id: "p_004".into(),
            plan: ExecutionPlan::Execute {
                node_id: "doubler".into(),
            },
            input: Some(crate::protocol::InputSource::Inline {
                value: Value::tensor(vec![1.0], vec![1]),
            }),
            filters: vec![],
            mode: ExecutionMode::default(),
            seed: None,
            metadata: serde_json::json!({}),
        };

        worker.execute_plan(&plan);

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }
        assert!(
            events
                .iter()
                .any(|e| matches!(e, Event::NodeStarted { .. }))
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, Event::NodeCompleted { .. }))
        );
    }

    /// A Fit over a large reference must FIT, not stream.
    ///
    /// The auto-stream branch used to be taken before `plan.mode` was
    /// looked at, so a fit whose input happened to exceed 1024 rows was
    /// silently executed as a stream of forwards: nothing was fitted, no
    /// state was stored, and the plan reported success. The next forward
    /// then found no state and died inside the user's filter — while the
    /// same graph under the threshold fitted normally and was correct.
    #[test]
    fn a_fit_over_a_large_reference_is_not_silently_streamed() {
        struct Counter;
        impl Filter for Counter {
            fn config_hash(&self) -> CacheKey {
                CacheKey::from_parts(&[b"Counter"])
            }
            fn fit(&self, x: &Value, _y: Option<&Value>) -> SomaResult<Value> {
                // The state a real fit produces: something derived from
                // the WHOLE input, which is exactly what a stream cannot
                // give you.
                let n = match x {
                    Value::Tensor { values, .. } => values.len() as f64,
                    _ => 0.0,
                };
                Ok(Value::tensor(vec![n], vec![1]))
            }
            fn forward(&self, x: &Value, _state: &Value) -> SomaResult<Value> {
                Ok(x.clone())
            }
            fn meta(&self) -> FilterMeta {
                FilterMeta {
                    name: "Counter".into(),
                    kind: FilterKind::Trainable,
                    cacheable: true,
                    differentiable: false,
                    deterministic: true,
                    stream_mode: StreamMode::FixedState,
                    distribution: somatize_core::graph::filter::Distribution::Local,
                    input_schema: None,
                    output_schema: None,
                }
            }
        }

        let dir = tempfile::tempdir().unwrap();
        let store: Arc<dyn DataStore> = Arc::new(LocalDataStore::new(dir.path().join("data")));
        let n = 2048usize; // over the auto-stream threshold
        let key = somatize_core::cache::CacheKey::hash_data(b"fit-input");
        let data_ref = store
            .put(&key, &Value::tensor(vec![1.0; n], vec![n]))
            .unwrap();

        let mut worker = make_worker().with_data_store(store);
        worker.register_filter("counter", Box::new(Counter));

        let mut plan = SerializedPlan::new(
            "p_fit_large",
            ExecutionPlan::Execute {
                node_id: "counter".into(),
            },
        );
        plan.input = Some(InputSource::Reference { data_ref });
        plan.mode = ExecutionMode::Fit {
            y: None,
            batch_size: None,
        };

        let result = worker.execute_plan(&plan);
        assert!(
            matches!(result, PlanResult::Success { .. }),
            "the fit failed: {result:?}"
        );
        // It fitted over everything, so the state says 2048 — not a chunk.
        let state = worker.get_filter_state("counter");
        let (values, _) = state
            .as_tensor()
            .expect("no state was stored: the fit was streamed");
        assert_eq!(
            values[0], n as f64,
            "the fit saw a chunk, not the whole input"
        );
    }

    /// A stateful filter, streamed from a DataStore.
    ///
    /// The path above it only ever ran `TestDoubler`, which ignores its
    /// state, so "the stream reaches the filter" was proved and "the
    /// stream reaches the filter WITH its state" was not. Against a real
    /// worker, any filter whose `forward` reads `state["..."]` died with a
    /// KeyError on chunk 0 — while the same graph under the 1024-row
    /// threshold worked and gave the right answer.
    #[test]
    fn a_stateful_filter_keeps_its_state_across_a_streamed_data_ref() {
        struct Centre;
        impl Filter for Centre {
            fn config_hash(&self) -> CacheKey {
                CacheKey::from_parts(&[b"Centre"])
            }
            fn fit(&self, _x: &Value, _y: Option<&Value>) -> SomaResult<Value> {
                Ok(Value::Empty)
            }
            fn forward(&self, x: &Value, state: &Value) -> SomaResult<Value> {
                // The state a fit would have produced. Reading it is the
                // whole point: an empty state has to be an error here, not
                // a silently different answer.
                let mean = match state {
                    Value::Tensor { values, .. } if !values.is_empty() => values[0],
                    other => {
                        return Err(somatize_core::error::SomaError::Execution {
                            node_id: "centre".into(),
                            message: format!("no state reached the filter: {other:?}"),
                        });
                    }
                };
                match x {
                    Value::Tensor { values, shape } => Ok(Value::tensor(
                        values.iter().map(|v| v - mean).collect(),
                        shape.clone(),
                    )),
                    other => Ok(other.clone()),
                }
            }
            fn meta(&self) -> FilterMeta {
                FilterMeta {
                    name: "Centre".into(),
                    kind: FilterKind::Trainable,
                    cacheable: true,
                    differentiable: false,
                    deterministic: true,
                    stream_mode: StreamMode::FixedState,
                    distribution: somatize_core::graph::filter::Distribution::Local,
                    input_schema: None,
                    output_schema: None,
                }
            }
        }

        let dir = tempfile::tempdir().unwrap();
        let store: Arc<dyn DataStore> = Arc::new(LocalDataStore::new(dir.path().join("data")));

        let n = 2048usize; // over the auto-stream threshold: two chunks
        let values: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let mean = values.iter().sum::<f64>() / n as f64;
        let key = somatize_core::cache::CacheKey::hash_data(b"stateful-stream-input");
        let data_ref = store.put(&key, &Value::tensor(values, vec![n])).unwrap();

        let mut worker = make_worker().with_data_store(store);
        worker.register_filter("centre", Box::new(Centre));
        worker
            .set_filter_state("centre", Value::tensor(vec![mean], vec![1]))
            .unwrap();

        let mut plan = SerializedPlan::new(
            "p_stateful_stream",
            ExecutionPlan::Execute {
                node_id: "centre".into(),
            },
        );
        plan.input = Some(InputSource::Reference { data_ref });

        let result = worker.execute_plan(&plan);
        let PlanResult::Success { output, .. } = result else {
            panic!("a stateful filter must keep its state when streamed: {result:?}");
        };
        let value = match output {
            OutputDelivery::Inline { value } => value,
            OutputDelivery::Reference { data_ref } => worker.temp_store().get(&data_ref).unwrap(),
        };
        let (data, shape) = value.as_tensor().unwrap();
        assert_eq!(shape, &[n]);
        // Centred on the mean of the WHOLE input, not of a chunk.
        assert!((data[0] - (0.0 - mean)).abs() < 1e-9, "{}", data[0]);
        assert!(
            (data[n - 1] - ((n - 1) as f64 - mean)).abs() < 1e-9,
            "{}",
            data[n - 1]
        );
    }

    /// The DataStore auto-stream path runs through the runtime's
    /// `StreamRun`: the plan output is the CONCATENATED stream (the old
    /// executor returned only the last chunk's output), events carry a
    /// closed Run bracket, and the plan's seed salts the chunk cache.
    #[test]
    fn a_large_data_ref_streams_concatenated_through_stream_run() {
        let dir = tempfile::tempdir().unwrap();
        let store: Arc<dyn DataStore> = Arc::new(LocalDataStore::new(dir.path().join("data")));

        // 2048 rows > the 1024-row auto-stream threshold: two chunks.
        let n = 2048usize;
        let values: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let key = somatize_core::cache::CacheKey::hash_data(b"stream-input");
        let data_ref = store.put(&key, &Value::tensor(values, vec![n])).unwrap();

        let mut worker = make_worker().with_data_store(store);
        let mut rx = worker.subscribe();
        worker.register_filter("doubler", Box::new(TestDoubler));

        let mut plan = SerializedPlan::new(
            "p_stream",
            ExecutionPlan::Execute {
                node_id: "doubler".into(),
            },
        );
        plan.input = Some(InputSource::Reference { data_ref });
        plan.seed = Some(7);

        let result = worker.execute_plan(&plan);
        let PlanResult::Success { output, .. } = result else {
            panic!("stream plan failed: {result:?}");
        };
        let value = match output {
            OutputDelivery::Inline { value } => value,
            OutputDelivery::Reference { data_ref } => worker.temp_store().get(&data_ref).unwrap(),
        };
        let (data, shape) = value.as_tensor().unwrap();
        assert_eq!(
            shape,
            &[n],
            "the output is the whole stream, not the last chunk"
        );
        assert_eq!(data[0], 0.0);
        assert_eq!(data[n - 1], (n - 1) as f64 * 2.0);

        let mut started = 0;
        let mut node_completed = 0;
        let mut run_completed = 0;
        while let Ok(event) = rx.try_recv() {
            match event {
                Event::NodeStarted { node_id, .. } => {
                    assert_eq!(node_id, "doubler");
                    started += 1;
                }
                Event::NodeCompleted {
                    node_id,
                    output_summary,
                    ..
                } => {
                    assert_eq!(node_id, "doubler");
                    assert!(output_summary.contains("2 chunks"), "{output_summary}");
                    node_completed += 1;
                }
                Event::RunCompleted { .. } => run_completed += 1,
                _ => {}
            }
        }
        assert_eq!((started, node_completed), (1, 1), "one bracket per node");
        assert_eq!(run_completed, 1, "the run bracket must close");
    }
}
