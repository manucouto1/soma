//! Worker — receives and executes plans from a coordinator.

use crate::protocol::*;
use somatize_core::cache::{CacheKey, CacheStore};
use somatize_core::error::Result as SomaResult;
use somatize_core::event::Event;
use somatize_core::filter::{Filter, FilterKind, FilterMeta, StreamMode};
use somatize_core::store::{DataStore, LocalDataStore};
use somatize_core::value::Value;
use somatize_runtime::{Context, EventBus, FilterLibrary, MemoryCache, execute};
use std::sync::Arc;
use std::time::Instant;

/// A filter reconstructed from cloudpickle bytes.
/// Deserializes the Python object on the worker and executes methods via subprocess.
pub(crate) struct PickledFilterRunner {
    /// cloudpickle.dumps() bytes of the original Python filter object.
    pub(crate) pickled_bytes: Vec<u8>,
    /// Node ID (for error messages).
    pub(crate) node_id: String,
    /// Path to the Python interpreter (venv or system).
    pub(crate) python_path: String,
    /// Pip requirements for retry-on-import-error.
    pub(crate) requirements: Vec<String>,
    /// Whether this filter is trainable (has meaningful fit()).
    pub(crate) trainable: bool,
}

impl Filter for PickledFilterRunner {
    fn config_hash(&self) -> CacheKey {
        CacheKey::from_parts(&[&self.pickled_bytes])
    }

    fn fit(&self, x: &Value, _y: Option<&Value>) -> SomaResult<Value> {
        self.run_python("fit", x)
    }

    fn forward(&self, x: &Value, state: &Value) -> SomaResult<Value> {
        let input = if matches!(state, Value::Empty) {
            x.clone()
        } else {
            Value::json(serde_json::json!({
                "x": serde_json::to_value(x).unwrap_or_default(),
                "state": serde_json::to_value(state).unwrap_or_default(),
            }))
        };
        self.run_python("forward", &input)
    }

    fn meta(&self) -> FilterMeta {
        FilterMeta {
            name: self.node_id.clone(),
            kind: if self.trainable {
                FilterKind::Trainable
            } else {
                FilterKind::Stateless
            },
            cacheable: true,
            differentiable: false,
            stream_mode: StreamMode::FixedState,
            distribution: somatize_core::filter::Distribution::Local,
            input_schema: None,
            output_schema: None,
        }
    }
}

impl PickledFilterRunner {
    fn run_python(&self, method: &str, input: &Value) -> SomaResult<Value> {
        self.run_python_with_retry(method, input, true)
    }

    fn run_python_with_retry(
        &self,
        method: &str,
        input: &Value,
        allow_retry: bool,
    ) -> SomaResult<Value> {
        use base64::engine::{Engine, general_purpose::STANDARD};
        use std::io::Write;

        let input_json = serde_json::to_string(input)
            .map_err(|e| somatize_core::error::SomaError::Other(format!("serialize input: {e}")))?;
        let pickled_b64 = STANDARD.encode(&self.pickled_bytes);

        let script = format!(
            r#"
import json, sys, base64, cloudpickle

def unwrap_value(v):
    """Convert Soma Value JSON to native Python types."""
    if isinstance(v, dict) and "type" in v and "data" in v:
        t = v["type"]
        d = v["data"]
        if t == "Tensor":
            return d.get("values", [])
        if t == "Json":
            return d
        if t == "Empty":
            return {{}}
        if t == "Bytes":
            return bytes(d)
    return v

pickled_b64 = sys.stdin.readline().strip()
input_line = sys.stdin.read()

pickled = base64.b64decode(pickled_b64)
obj = cloudpickle.loads(pickled)
raw = json.loads(input_line)
input_data = unwrap_value(raw)

if isinstance(input_data, dict) and "x" in input_data and "state" in input_data:
    x = unwrap_value(input_data["x"])
    state = unwrap_value(input_data["state"])
    result = obj.{method}(x, state)
else:
    result = obj.{method}(input_data, {{}})

print(json.dumps(result))
"#,
        );

        let mut child = std::process::Command::new(&self.python_path)
            .args(["-c", &script])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| {
                somatize_core::error::SomaError::Other(format!("python spawn failed: {e}"))
            })?;

        if let Some(mut stdin) = child.stdin.take() {
            let _ = writeln!(stdin, "{pickled_b64}");
            let _ = write!(stdin, "{input_json}");
        }

        let output = child.wait_with_output().map_err(|e| {
            somatize_core::error::SomaError::Other(format!("python exec failed: {e}"))
        })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);

            // Retry on ModuleNotFoundError: install known requirements + missing package
            if allow_retry && stderr.contains("ModuleNotFoundError") {
                let missing = Self::parse_missing_module(&stderr);
                // Collect packages to install: known requirements + the missing one
                let mut to_install: Vec<String> = self.requirements.clone();
                if let Some(ref m) = missing
                    && !to_install.iter().any(|r| r == m)
                {
                    to_install.push(m.clone());
                }
                if !to_install.is_empty() {
                    let names = to_install.join(", ");
                    tracing::warn!(
                        "Missing module for filter '{}', installing: {names}",
                        self.node_id
                    );
                    let mut args = vec!["-m", "pip", "install", "--quiet"];
                    let refs: Vec<&str> = to_install.iter().map(|s| s.as_str()).collect();
                    args.extend(refs);
                    let install = std::process::Command::new(&self.python_path)
                        .args(&args)
                        .output();
                    if let Ok(res) = install
                        && res.status.success()
                    {
                        tracing::info!("Installed [{names}], retrying...");
                        return self.run_python_with_retry(method, input, false);
                    }
                }
            }

            return Err(somatize_core::error::SomaError::Execution {
                node_id: self.node_id.clone(),
                message: format!("Python error: {stderr}"),
            });
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let result: serde_json::Value = serde_json::from_str(stdout.trim()).map_err(|e| {
            somatize_core::error::SomaError::Other(format!(
                "parse python output: {e}\nstdout: {stdout}"
            ))
        })?;

        if let Some(arr) = result.as_array() {
            let values: Vec<f64> = arr.iter().filter_map(|v| v.as_f64()).collect();
            if !values.is_empty() {
                return Ok(Value::tensor(values.clone(), vec![values.len()]));
            }
        }

        Ok(Value::json(result))
    }

    /// Parse "ModuleNotFoundError: No module named 'xxx'" from stderr.
    fn parse_missing_module(stderr: &str) -> Option<String> {
        for line in stderr.lines().rev() {
            if line.contains("ModuleNotFoundError") {
                // "ModuleNotFoundError: No module named 'xxx'"
                if let Some(start) = line.find('\'') {
                    let rest = &line[start + 1..];
                    if let Some(end) = rest.find('\'') {
                        return Some(rest[..end].split('.').next()?.to_string());
                    }
                }
            }
        }
        None
    }
}

/// Worker state: manages execution of plans received from a coordinator.
pub struct Worker {
    pub id: WorkerId,
    pub capabilities: Capabilities,
    event_bus: Arc<EventBus>,
    cache: Arc<dyn CacheStore>,
    filters: FilterLibrary,
    /// Optional persistent DataStore (S3, Zarr, etc.) — configured by user.
    data_store: Option<Arc<dyn DataStore>>,
    /// Temporary local store for HTTP bulk uploads — auto-created, auto-cleaned.
    temp_store: Arc<LocalDataStore>,
    /// Environment manager for creating venvs with filter dependencies.
    env_manager: crate::env_manager::EnvManager,
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
            filters: FilterLibrary::new(),
            data_store: None,
            temp_store: Arc::new(temp_store),
            env_manager: crate::env_manager::EnvManager::new(
                env_path,
                crate::env_manager::EnvType::Venv,
            ),
        }
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
    pub fn get_filter_state(&self, node_id: &str) -> Value {
        self.filters
            .get_state(node_id)
            .cloned()
            .unwrap_or(Value::Empty)
    }

    /// Set trained state for a filter.
    pub fn set_filter_state(&mut self, node_id: &str, state: Value) {
        self.filters.set_state(node_id, state);
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
            "python3".to_string()
        } else {
            let reqs_str = all_reqs.join("\n");
            match self.env_manager.ensure_env(&plan.plan_id, &reqs_str) {
                Ok(path) => {
                    tracing::info!("Using venv for plan {}: {:?}", plan.plan_id, path);
                    path.to_string_lossy().to_string()
                }
                Err(e) => {
                    tracing::warn!("Failed to create venv, falling back to system python: {e}");
                    "python3".to_string()
                }
            }
        };

        // Register pickled filters (from remote client via cloudpickle)
        for sf in &plan.filters {
            let filter = Box::new(PickledFilterRunner {
                pickled_bytes: sf.pickled_filter.clone(),
                node_id: sf.node_id.clone(),
                python_path: python_path.clone(),
                requirements: sf.requirements.clone(),
                trainable: sf.trainable,
            });
            self.filters.register(&sf.node_id, filter);
            if let Some(state) = &sf.state {
                self.filters.set_state(&sf.node_id, state.clone());
            }
        }

        // Resolve input: inline, DataStore, or temp store (HTTP upload)
        let input_value = plan.input.as_ref().map(|src| match src {
            InputSource::Inline { value } => value.clone(),
            InputSource::Reference { data_ref } => {
                // Try persistent DataStore first, then temp store (HTTP uploads)
                if let Some(store) = &self.data_store
                    && let Ok(val) = store.get(data_ref)
                {
                    return val;
                }
                self.temp_store.get(data_ref).unwrap_or_else(|e| {
                    tracing::warn!("Failed to resolve DataRef: {e}");
                    Value::Empty
                })
            }
        });

        // DataStore-backed streaming: if input is a large DataRef and we have a store,
        // read chunks via get_rows() and process with StreamExecutor (no full materialization).
        if let Some(InputSource::Reference { data_ref }) = &plan.input
            && let Some(store) = self.data_store.clone()
            && let Ok(meta) = store.meta(data_ref)
            && meta.total_rows > 1024
        {
            return self.execute_streamed_from_store(plan, &store, data_ref, &meta, start);
        }

        match &plan.mode {
            ExecutionMode::Fit { y } => self.execute_fit(plan, input_value, y.as_ref(), start),
            ExecutionMode::Forward => self.execute_forward(plan, input_value, start),
        }
    }

    /// Forward mode: run the compiled execution plan.
    fn execute_forward(
        &mut self,
        plan: &SerializedPlan,
        input: Option<Value>,
        start: Instant,
    ) -> PlanResult {
        let mut ctx = Context::new(
            self.event_bus.clone(),
            format!("worker_run_{}", plan.plan_id),
        );

        if let Some(val) = input {
            ctx.set("input", val.clone());
            // Also set per-root input
            if let somatize_compiler::ExecutionPlan::Execute { node_id } = &plan.plan {
                ctx.set(format!("__input_{node_id}"), val);
            }
        }

        match execute(&plan.plan, &mut ctx, &self.filters, self.cache.as_ref()) {
            Ok(()) => {
                let output = ctx
                    .execution_order
                    .last()
                    .and_then(|id| ctx.get(id))
                    .cloned()
                    .unwrap_or(Value::Empty);

                PlanResult::Success {
                    output,
                    duration_ms: start.elapsed().as_millis() as u64,
                    states: std::collections::HashMap::new(),
                }
            }
            Err(e) => PlanResult::Failed {
                error: e.to_string(),
                duration_ms: start.elapsed().as_millis() as u64,
            },
        }
    }

    /// Fit mode: train each filter in topological order, return trained states.
    fn execute_fit(
        &mut self,
        plan: &SerializedPlan,
        input: Option<Value>,
        y: Option<&Value>,
        start: Instant,
    ) -> PlanResult {
        let run_id = format!("worker_fit_{}", plan.plan_id);
        let x = input.unwrap_or(Value::Empty);

        // Extract node execution order from plan
        let node_ids: Vec<String> = plan.plan.node_ids().into_iter().map(String::from).collect();
        let mut outputs: std::collections::HashMap<String, Value> =
            std::collections::HashMap::new();
        let mut trained_states: std::collections::HashMap<String, Value> =
            std::collections::HashMap::new();

        for node_id in &node_ids {
            let filter = match self.filters.get(node_id) {
                Some(f) => f,
                None => {
                    return PlanResult::Failed {
                        error: format!("filter not found: {node_id}"),
                        duration_ms: start.elapsed().as_millis() as u64,
                    };
                }
            };

            let meta = filter.meta();

            self.event_bus.emit(Event::NodeStarted {
                run_id: run_id.clone(),
                node_id: node_id.to_string(),
                kind: meta.kind,
            });

            let node_start = Instant::now();

            // Resolve input: predecessor output or original input
            let node_input = outputs
                .values()
                .last()
                .cloned()
                .unwrap_or_else(|| x.clone());

            // Fit trainable filters, get/use state for forward
            let state = if meta.kind == FilterKind::Trainable {
                match filter.fit(&node_input, y) {
                    Ok(s) => {
                        self.filters.set_state(node_id, s.clone());
                        trained_states.insert(node_id.clone(), s.clone());
                        s
                    }
                    Err(e) => {
                        return PlanResult::Failed {
                            error: format!("fit({node_id}): {e}"),
                            duration_ms: start.elapsed().as_millis() as u64,
                        };
                    }
                }
            } else {
                self.filters
                    .get_state(node_id)
                    .cloned()
                    .unwrap_or(Value::Empty)
            };

            // Forward with trained state
            match filter.forward(&node_input, &state) {
                Ok(output) => {
                    self.event_bus.emit(Event::NodeCompleted {
                        run_id: run_id.clone(),
                        node_id: node_id.to_string(),
                        duration: node_start.elapsed(),
                        output_summary: format!("{output}"),
                    });
                    outputs.insert(node_id.clone(), output);
                }
                Err(e) => {
                    return PlanResult::Failed {
                        error: format!("forward({node_id}): {e}"),
                        duration_ms: start.elapsed().as_millis() as u64,
                    };
                }
            }
        }

        let output = outputs.values().last().cloned().unwrap_or(Value::Empty);

        PlanResult::Success {
            output,
            duration_ms: start.elapsed().as_millis() as u64,
            states: trained_states,
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
        use somatize_runtime::stream::{FittedFilter, StreamExecutor};

        let node_ids: Vec<String> = plan.plan.node_ids().into_iter().map(String::from).collect();
        let fitted: Vec<FittedFilter> = node_ids
            .iter()
            .filter_map(|id| {
                let filter = self.filters.get(id)?;
                let state = self.filters.get_state(id).cloned().unwrap_or(Value::Empty);
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
            output: last_output,
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
            let (data, _) = output.as_tensor().unwrap();
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
            let (data, _) = output.as_tensor().unwrap();
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
