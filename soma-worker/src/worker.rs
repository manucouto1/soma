//! Worker — receives and executes plans from a coordinator.

use crate::protocol::*;
use somatize_core::cache::CacheStore;
use somatize_core::event::Event;
use somatize_core::filter::Filter;
use somatize_core::store::{DataStore, LocalDataStore};
use somatize_core::value::Value;
use somatize_runtime::{EventBus, MemoryCache, NodeCatalog, Runner};
use std::sync::Arc;
use std::time::Instant;

/// Worker state: manages execution of plans received from a coordinator.
pub struct Worker {
    pub id: WorkerId,
    pub capabilities: Capabilities,
    event_bus: Arc<EventBus>,
    cache: Arc<dyn CacheStore>,
    filters: NodeCatalog,
    /// Optional persistent DataStore (S3, Zarr, etc.) — configured by user.
    data_store: Option<Arc<dyn DataStore>>,
    /// Temporary local store for HTTP bulk uploads — auto-created, auto-cleaned.
    temp_store: Arc<LocalDataStore>,
    /// Environment manager for creating venvs with filter dependencies.
    env_manager: crate::env_manager::EnvManager,
    /// Which interpreter to unpickle filters in, when no venv is needed.
    ///
    /// A cloudpickled filter can only be reconstructed by an interpreter
    /// close enough to the one that pickled it. Defaulting to `python3`
    /// off `PATH` means the worker will happily pick a different minor
    /// version from the process that sent the work, and cloudpickle then
    /// returns the class's `__dict__` instead of an instance — which
    /// surfaces as `'dict' object is not callable`, from inside a
    /// subprocess, with nothing pointing at the version gap.
    python: String,
}

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
    pub fn new(id: impl Into<String>, capabilities: Capabilities) -> Self {
        let worker_id: String = id.into();
        let temp_path = std::env::temp_dir().join(format!("soma-uploads-{worker_id}"));
        let temp_store = LocalDataStore::new(temp_path);
        let env_path = std::env::temp_dir().join(format!("soma-envs-{worker_id}"));
        Self {
            id: worker_id,
            capabilities,
            event_bus: Arc::new(EventBus::new(256)),
            cache: Arc::new(MemoryCache::default()),
            filters: NodeCatalog::new(),
            data_store: None,
            temp_store: Arc::new(temp_store),
            env_manager: crate::env_manager::EnvManager::new(
                env_path,
                crate::env_manager::EnvType::Venv,
            ),
            python: default_python(),
        }
    }

    /// Run filters in this interpreter rather than whatever `python3`
    /// resolves to.
    ///
    /// An embedding process should pass its own `sys.executable`: it is
    /// the interpreter that pickled the filters, so it is the only one
    /// certain to unpickle them.
    pub fn with_python(mut self, python: impl Into<String>) -> Self {
        self.python = python.into();
        self
    }

    /// Set a custom cache store (e.g. tiered or shared).
    pub fn with_cache(mut self, cache: Arc<dyn CacheStore>) -> Self {
        self.cache = cache;
        self
    }

    /// Set a persistent DataStore (S3, Zarr, etc.) for large data references.
    pub fn with_data_store(mut self, store: Arc<dyn DataStore>) -> Self {
        self.data_store = Some(store);
        self
    }

    /// Set a custom temp directory for HTTP bulk uploads.
    pub fn with_temp_dir(mut self, path: std::path::PathBuf) -> Self {
        self.temp_store = Arc::new(LocalDataStore::new(path));
        self
    }

    /// Get the temp store (for HTTP upload endpoint).
    pub fn temp_store(&self) -> &Arc<LocalDataStore> {
        &self.temp_store
    }

    /// Register a filter that this worker can execute.
    pub fn register_filter(&mut self, node_id: impl Into<String>, filter: Box<dyn Filter>) {
        self.filters.register(node_id, filter);
    }

    /// Get a filter by node_id (for stream executor construction).
    pub fn get_filter(&self, node_id: &str) -> Option<Arc<dyn Filter>> {
        self.filters.get(node_id)
    }

    /// Get trained state for a filter.
    pub fn get_filter_state(&self, node_id: &str) -> Arc<Value> {
        self.filters
            .get_state(node_id)
            .unwrap_or_else(|| Arc::new(Value::Empty))
    }

    /// Set trained state for a filter.
    pub fn set_filter_state(&mut self, node_id: &str, state: Value) {
        if let Err(e) = self.filters.try_set_state(node_id, state) {
            tracing::error!(node_id, "storing filter state failed: {e}");
        }
    }

    /// Wrap output in the right delivery: inline for small, DataRef for large.
    pub fn wrap_output(&self, output: Value) -> OutputDelivery {
        let size = serde_json::to_vec(&output).map(|v| v.len()).unwrap_or(0);
        if size >= somatize_core::store::INLINE_THRESHOLD_BYTES {
            let key = somatize_core::cache::CacheKey::hash_data(
                &serde_json::to_vec(&output).unwrap_or_default(),
            );
            if let Ok(data_ref) = self.temp_store.put(&key, &output) {
                return OutputDelivery::Reference { data_ref };
            }
        }
        OutputDelivery::Inline { value: output }
    }

    /// Subscribe to execution events.
    pub fn subscribe(&self) -> tokio::sync::broadcast::Receiver<Event> {
        self.event_bus.subscribe()
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
            self.python.clone()
        } else {
            let reqs_str = all_reqs.join("\n");
            match self.env_manager.ensure_env(&plan.plan_id, &reqs_str) {
                Ok(path) => {
                    tracing::info!("Using venv for plan {}: {:?}", plan.plan_id, path);
                    path.to_string_lossy().to_string()
                }
                Err(e) => {
                    tracing::warn!("Failed to create venv, falling back to system python: {e}");
                    self.python.clone()
                }
            }
        };

        // No site-packages resolution needed — subprocess uses the venv python directly

        // Spawn ONE Python subprocess for all filters in this plan.
        // All filters share the same process (needed for Composite autograd).
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

            let proc = crate::python_process::PythonProcess::spawn(&python_path, &filter_specs)
                .map_err(|e| {
                    // Not `.expect`: this runs on a tokio worker thread, so
                    // a panic here took the whole worker down — every other
                    // pipeline it was holding with it — because one plan
                    // named an interpreter that would not start.
                    tracing::error!(python = %python_path, "failed to spawn Python: {e}");
                    e
                });
            let mut proc = match proc {
                Ok(p) => p,
                Err(e) => {
                    return PlanResult::Failed {
                        error: format!("could not start `{python_path}`: {e}"),
                        duration_ms: start.elapsed().as_millis() as u64,
                    };
                }
            };

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
                        tracing::warn!(
                            node_id = %sf.node_id,
                            error = %e,
                            "Failed to load state (will use fresh weights)"
                        );
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
                self.filters.register(&sf.node_id, filter);
                if let Some(state) = &sf.state
                    && let Err(e) = self.filters.try_set_state(&sf.node_id, state.clone())
                {
                    tracing::error!(node_id = %sf.node_id, "storing filter state failed: {e}");
                }
            }

            tracing::info!("Filters registered, Python process ready");
        }

        // Resolve input via InputSource::resolve()
        let input_value = plan
            .input
            .as_ref()
            .map(|src| src.resolve(self.data_store.as_deref(), &self.temp_store));

        // DataStore-backed streaming: if input is a large DataRef and we have a store,
        // read chunks via get_rows() and process with StreamExecutor (no full materialization).
        if let Some(InputSource::Reference { data_ref }) = &plan.input
            && let Some(store) = self.data_store.clone()
            && let Ok(meta) = store.meta(data_ref)
            && meta.total_rows > 1024
        {
            return self.execute_streamed_from_store(plan, &store, data_ref, &meta, start);
        }

        // Delegate to LocalRunner (same execution path as local)
        let runner = somatize_runtime::LocalRunner;
        let x = input_value.unwrap_or(Value::Empty);

        let result = match &plan.mode {
            ExecutionMode::Fit { y, batch_size } => {
                // If batch_size is set, use BATCHED_FIT on the subprocess directly
                if let Some(bs) = batch_size {
                    tracing::info!(batch_size = bs, "Using batched fit");
                    let node_ids = plan
                        .plan
                        .node_ids()
                        .iter()
                        .map(|s| s.to_string())
                        .collect::<Vec<_>>();
                    if let Some(filter) = self.filters.get(&node_ids[0]) {
                        if let Some(sf) = filter
                            .as_any()
                            .downcast_ref::<crate::python_process::SubprocessFilter>()
                        {
                            let result = sf
                                .process
                                .lock()
                                .map_err(|e| {
                                    somatize_core::error::SomaError::Other(format!("mutex: {e}"))
                                })
                                .and_then(|mut proc| {
                                    proc.batched_fit(&node_ids, &x, y.as_ref(), *bs)
                                });
                            match result {
                                Ok((output, states)) => {
                                    for (id, state) in &states {
                                        if let Err(e) =
                                            self.filters.try_set_state(id, state.clone())
                                        {
                                            tracing::error!(
                                                node_id = %id,
                                                "storing filter state failed: {e}"
                                            );
                                        }
                                    }
                                    Ok((output, states))
                                }
                                Err(e) => Err(e),
                            }
                        } else {
                            Err(somatize_core::error::SomaError::Other(
                                "batched_fit requires SubprocessFilter".into(),
                            ))
                        }
                    } else {
                        Err(somatize_core::error::SomaError::Other(
                            "no filters found".into(),
                        ))
                    }
                } else {
                    let run_id = format!("worker_fit_{}", plan.plan_id);
                    // `linear`, explicitly: a worker receives a serialized
                    // plan and no graph, so it has no topology to consult.
                    // Correct for the pipelines that get dispatched, and
                    // stated here rather than assumed inside the runner.
                    let ctx = somatize_runtime::runner::RunContext::linear(
                        &self.filters,
                        self.cache.as_ref(),
                        &self.event_bus,
                        &run_id,
                        &plan.plan,
                    );
                    runner
                        .fit(&plan.plan, &ctx, &x, y.as_ref())
                        .map(|(output, all_outputs)| {
                            // Extract trained states (prefixed __state_) and store in library
                            let mut trained_states = std::collections::HashMap::new();
                            for (key, value) in &all_outputs {
                                if let Some(node_id) = key.strip_prefix("__state_") {
                                    if let Err(e) =
                                        self.filters.try_set_state(node_id, value.clone())
                                    {
                                        tracing::error!(
                                            node_id,
                                            "storing filter state failed: {e}"
                                        );
                                    }
                                    trained_states.insert(node_id.to_string(), value.clone());
                                }
                            }
                            (output, trained_states)
                        })
                }
            }
            ExecutionMode::Forward => {
                let run_id = format!("worker_forward_{}", plan.plan_id);
                let ctx = somatize_runtime::runner::RunContext::linear(
                    &self.filters,
                    self.cache.as_ref(),
                    &self.event_bus,
                    &run_id,
                    &plan.plan,
                );
                runner
                    .forward(&plan.plan, &ctx, &x)
                    .map(|output| (output, std::collections::HashMap::new()))
            }
        };

        let elapsed = start.elapsed().as_millis() as u64;
        match result {
            Ok((output, states)) => {
                tracing::info!(
                    duration_ms = elapsed,
                    n_states = states.len(),
                    "Plan completed successfully"
                );
                PlanResult::Success {
                    output: self.wrap_output(output),
                    duration_ms: elapsed,
                    states,
                }
            }
            Err(e) => {
                tracing::error!(duration_ms = elapsed, error = %e, "Plan failed");
                PlanResult::Failed {
                    error: e.to_string(),
                    duration_ms: elapsed,
                }
            }
        }
    }

    /// DataStore-backed streaming: read chunks via get_rows(), process with StreamExecutor.
    /// Avoids loading the entire dataset into memory.
    fn execute_streamed_from_store(
        &mut self,
        plan: &SerializedPlan,
        store: &Arc<dyn DataStore>,
        data_ref: &somatize_core::store::DataRef,
        meta: &somatize_core::store::StoreMeta,
        start: Instant,
    ) -> PlanResult {
        use somatize_runtime::executors::stream::{FittedFilter, StreamExecutor};

        let node_ids: Vec<String> = plan.plan.node_ids().into_iter().map(String::from).collect();
        let fitted: Vec<FittedFilter> = node_ids
            .iter()
            .filter_map(|id| {
                let filter = self.filters.get(id)?;
                let state = self
                    .filters
                    .get_state(id)
                    .unwrap_or_else(|| Arc::new(Value::Empty));
                Some(FittedFilter {
                    name: id.clone(),
                    filter,
                    state,
                })
            })
            .collect();

        let mut executor = StreamExecutor::new(fitted);
        let chunk_size = 1024;
        let run_id = format!("worker_stream_{}", plan.plan_id);

        self.event_bus.emit(Event::RunStarted {
            run_id: run_id.clone(),
            plan_summary: somatize_core::event::PlanSummary {
                total_nodes: node_ids.len(),
                cached_nodes: 0,
                parallel_branches: 0,
            },
        });

        let mut last_output = Value::Empty;
        let total = meta.total_rows;
        let mut chunk_idx = 0;

        for row_start in (0..total).step_by(chunk_size) {
            let len = chunk_size.min(total - row_start);
            let chunk = match store.get_rows(data_ref, row_start, len) {
                Ok(c) => c,
                Err(e) => {
                    return PlanResult::Failed {
                        error: format!("get_rows({row_start}..{}): {e}", row_start + len),
                        duration_ms: start.elapsed().as_millis() as u64,
                    };
                }
            };

            match executor.process_chunk(chunk) {
                Ok(Some(output)) => last_output = output,
                Ok(None) => {} // Barrier — accumulating
                Err(e) => {
                    return PlanResult::Failed {
                        error: format!("stream chunk {chunk_idx}: {e}"),
                        duration_ms: start.elapsed().as_millis() as u64,
                    };
                }
            }
            chunk_idx += 1;
        }

        // Flush barrier filters
        match executor.flush() {
            Ok(Some(output)) => last_output = output,
            Ok(None) => {}
            Err(e) => {
                return PlanResult::Failed {
                    error: format!("stream flush: {e}"),
                    duration_ms: start.elapsed().as_millis() as u64,
                };
            }
        }

        tracing::info!(
            "Streamed {chunk_idx} chunks ({total} rows) in {}ms",
            start.elapsed().as_millis()
        );

        PlanResult::Success {
            output: self.wrap_output(last_output),
            duration_ms: start.elapsed().as_millis() as u64,
            states: std::collections::HashMap::new(),
        }
    }

    /// Check if this worker matches a remote target.
    pub fn matches_target(&self, target: &somatize_core::filter::RemoteTarget) -> bool {
        match target {
            somatize_core::filter::RemoteTarget::WorkerId(id) => &self.id == id,
            somatize_core::filter::RemoteTarget::Tag(tag) => self.capabilities.tags.contains(tag),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use somatize_compiler::ExecutionPlan;
    use somatize_core::cache::CacheKey;
    use somatize_core::error::Result as SomaResult;
    use somatize_core::filter::{FilterKind, FilterMeta, StreamMode};
    use somatize_core::value::Value;

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
                distribution: somatize_core::filter::Distribution::Local,
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
            plan_id: "p_001".into(),
            plan: ExecutionPlan::Execute {
                node_id: "doubler".into(),
            },
            input: Some(crate::protocol::InputSource::Inline {
                value: Value::tensor(vec![1.0, 2.0, 3.0], vec![3]),
            }),
            filters: vec![],
            mode: ExecutionMode::default(),
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
            plan_id: "p_002".into(),
            plan: ExecutionPlan::Execute {
                node_id: "nonexistent".into(),
            },
            input: None,
            filters: vec![],
            mode: ExecutionMode::default(),
            metadata: serde_json::json!({}),
        };

        let result = worker.execute_plan(&plan);
        assert!(matches!(result, PlanResult::Failed { .. }));
    }

    #[test]
    fn worker_matches_target_by_id() {
        let worker = make_worker();
        assert!(
            worker.matches_target(&somatize_core::filter::RemoteTarget::WorkerId(
                "test_worker".into()
            ))
        );
        assert!(
            !worker.matches_target(&somatize_core::filter::RemoteTarget::WorkerId(
                "other".into()
            ))
        );
    }

    #[test]
    fn worker_matches_target_by_tag() {
        let worker = make_worker();
        assert!(worker.matches_target(&somatize_core::filter::RemoteTarget::Tag("cpu".into())));
        assert!(worker.matches_target(&somatize_core::filter::RemoteTarget::Tag("test".into())));
        assert!(!worker.matches_target(&somatize_core::filter::RemoteTarget::Tag("gpu".into())));
    }

    #[test]
    fn worker_executes_sequence() {
        let mut worker = make_worker();
        worker.register_filter("d1", Box::new(TestDoubler));
        worker.register_filter("d2", Box::new(TestDoubler));

        let plan = SerializedPlan {
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
            plan_id: "p_004".into(),
            plan: ExecutionPlan::Execute {
                node_id: "doubler".into(),
            },
            input: Some(crate::protocol::InputSource::Inline {
                value: Value::tensor(vec![1.0], vec![1]),
            }),
            filters: vec![],
            mode: ExecutionMode::default(),
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
}
