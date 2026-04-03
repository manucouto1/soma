//! Worker — receives and executes plans from a coordinator.

use crate::protocol::*;
use somatize_core::cache::{CacheKey, CacheStore};
use somatize_core::error::Result as SomaResult;
use somatize_core::event::Event;
use somatize_core::filter::{Filter, FilterKind, FilterMeta, StreamMode};
use somatize_core::value::Value;
use somatize_runtime::{Context, EventBus, FilterLibrary, MemoryCache, execute};
use std::sync::Arc;
use std::time::Instant;

/// A filter reconstructed from serialized source code.
/// Executes Python code via subprocess on the worker.
struct SerializedFilterRunner {
    source: String,
    class_name: String,
    params: serde_json::Value,
}

impl Filter for SerializedFilterRunner {
    fn config_hash(&self) -> CacheKey {
        CacheKey::from_parts(&[
            self.class_name.as_bytes(),
            self.params.to_string().as_bytes(),
        ])
    }

    fn fit(&self, x: &Value, _y: Option<&Value>) -> SomaResult<Value> {
        self.run_python("fit", x)
    }

    fn forward(&self, x: &Value, state: &Value) -> SomaResult<Value> {
        // Pass state as second input via JSON
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
            name: self.class_name.clone(),
            kind: FilterKind::Stateless,
            cacheable: true,
            differentiable: false,
            stream_mode: StreamMode::FixedState,
            distribution: somatize_core::filter::Distribution::Local,
            input_schema: None,
            output_schema: None,
        }
    }
}

impl SerializedFilterRunner {
    fn run_python(&self, method: &str, input: &Value) -> SomaResult<Value> {
        let input_json = serde_json::to_string(input)
            .map_err(|e| somatize_core::error::SomaError::Other(format!("serialize input: {e}")))?;

        let script = format!(
            r#"
import json, sys
{}

obj = {}(**json.loads('{}'))
input_data = json.loads(sys.argv[1])

if isinstance(input_data, dict) and "x" in input_data and "state" in input_data:
    result = obj.{}(input_data["x"], input_data["state"])
else:
    result = obj.{}(input_data, {{}})

print(json.dumps(result))
"#,
            self.source,
            self.class_name,
            self.params.to_string().replace('\'', "\\'"),
            method,
            method,
        );

        let output = std::process::Command::new("python3")
            .args(["-c", &script, &input_json])
            .output()
            .map_err(|e| {
                somatize_core::error::SomaError::Other(format!("python exec failed: {e}"))
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(somatize_core::error::SomaError::Execution {
                node_id: self.class_name.clone(),
                message: format!("Python error: {stderr}"),
            });
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let result: serde_json::Value = serde_json::from_str(stdout.trim()).map_err(|e| {
            somatize_core::error::SomaError::Other(format!(
                "parse python output: {e}\nstdout: {stdout}"
            ))
        })?;

        // Convert JSON result back to Value
        if let Some(arr) = result.as_array() {
            let values: Vec<f64> = arr.iter().filter_map(|v| v.as_f64()).collect();
            if !values.is_empty() {
                return Ok(Value::tensor(values.clone(), vec![values.len()]));
            }
        }

        Ok(Value::json(result))
    }
}

/// Worker state: manages execution of plans received from a coordinator.
pub struct Worker {
    pub id: WorkerId,
    pub capabilities: Capabilities,
    event_bus: Arc<EventBus>,
    cache: Arc<dyn CacheStore>,
    filters: FilterLibrary,
}

impl Worker {
    pub fn new(id: impl Into<String>, capabilities: Capabilities) -> Self {
        Self {
            id: id.into(),
            capabilities,
            event_bus: Arc::new(EventBus::new(256)),
            cache: Arc::new(MemoryCache::default()),
            filters: FilterLibrary::new(),
        }
    }

    /// Set a custom cache store (e.g. tiered or shared).
    pub fn with_cache(mut self, cache: Arc<dyn CacheStore>) -> Self {
        self.cache = cache;
        self
    }

    /// Register a filter that this worker can execute.
    pub fn register_filter(&mut self, node_id: impl Into<String>, filter: Box<dyn Filter>) {
        self.filters.register(node_id, filter);
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
    pub fn execute_plan(&mut self, plan: &SerializedPlan) -> PlanResult {
        let start = Instant::now();

        // Register serialized filters (from remote client)
        for sf in &plan.filters {
            let filter = Box::new(SerializedFilterRunner {
                source: sf.source.clone(),
                class_name: sf.class_name.clone(),
                params: sf.params.clone(),
            });
            self.filters.register(&sf.node_id, filter);
            if let Some(state) = &sf.state {
                self.filters.set_state(&sf.node_id, state.clone());
            }
        }

        let mut ctx = Context::new(
            self.event_bus.clone(),
            format!("worker_run_{}", plan.plan_id),
        );

        // Resolve input data (inline or from DataStore)
        if let Some(input_source) = &plan.input {
            use crate::protocol::InputSource;
            let input_value = match input_source {
                InputSource::Inline { value } => value.clone(),
                InputSource::Reference { data_ref } => {
                    if let Some(store) = &ctx.data_store {
                        store
                            .get(data_ref)
                            .unwrap_or(somatize_core::value::Value::Empty)
                    } else {
                        tracing::warn!("DataRef input but no DataStore configured on worker");
                        somatize_core::value::Value::Empty
                    }
                }
            };
            ctx.set("input", input_value);
        }

        match execute(&plan.plan, &mut ctx, &self.filters, self.cache.as_ref()) {
            Ok(()) => {
                // Find the last output
                let output = ctx
                    .execution_order
                    .last()
                    .and_then(|id| ctx.get(id))
                    .cloned()
                    .unwrap_or(somatize_core::value::Value::Empty);

                PlanResult::Success {
                    output,
                    duration_ms: start.elapsed().as_millis() as u64,
                }
            }
            Err(e) => PlanResult::Failed {
                error: e.to_string(),
                duration_ms: start.elapsed().as_millis() as u64,
            },
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
            metadata: serde_json::json!({}),
        };

        let result = worker.execute_plan(&plan);

        if let PlanResult::Success {
            output,
            duration_ms,
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
