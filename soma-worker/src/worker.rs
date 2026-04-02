//! Worker — receives and executes plans from a coordinator.

use crate::protocol::*;
use soma_core::cache::CacheStore;
use soma_core::event::Event;
use soma_core::filter::Filter;
use soma_runtime::{Context, EventBus, FilterStore, MemoryCache, execute};
use std::sync::Arc;
use std::time::Instant;

/// Worker state: manages execution of plans received from a coordinator.
pub struct Worker {
    pub id: WorkerId,
    pub capabilities: Capabilities,
    event_bus: Arc<EventBus>,
    cache: Arc<dyn CacheStore>,
    filters: FilterStore,
}

impl Worker {
    pub fn new(id: impl Into<String>, capabilities: Capabilities) -> Self {
        Self {
            id: id.into(),
            capabilities,
            event_bus: Arc::new(EventBus::new(256)),
            cache: Arc::new(MemoryCache::default()),
            filters: FilterStore::new(),
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
    pub fn execute_plan(&mut self, plan: &SerializedPlan) -> PlanResult {
        let start = Instant::now();

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
                    // Try to load from context's data store
                    if let Some(store) = &ctx.data_store {
                        store.get(data_ref).unwrap_or(soma_core::value::Value::Empty)
                    } else {
                        tracing::warn!("DataRef input but no DataStore configured on worker");
                        soma_core::value::Value::Empty
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
                    .unwrap_or(soma_core::value::Value::Empty);

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
    pub fn matches_target(&self, target: &soma_core::filter::RemoteTarget) -> bool {
        match target {
            soma_core::filter::RemoteTarget::WorkerId(id) => &self.id == id,
            soma_core::filter::RemoteTarget::Tag(tag) => self.capabilities.tags.contains(tag),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use soma_compiler::ExecutionPlan;
    use soma_core::cache::CacheKey;
    use soma_core::error::Result as SomaResult;
    use soma_core::filter::{FilterKind, FilterMeta, StreamMode};
    use soma_core::value::Value;

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
                distribution: soma_core::filter::Distribution::Local,
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
            input: Some(crate::protocol::InputSource::Inline { value: Value::tensor(vec![1.0, 2.0, 3.0], vec![3]) }),
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
            metadata: serde_json::json!({}),
        };

        let result = worker.execute_plan(&plan);
        assert!(matches!(result, PlanResult::Failed { .. }));
    }

    #[test]
    fn worker_matches_target_by_id() {
        let worker = make_worker();
        assert!(
            worker.matches_target(&soma_core::filter::RemoteTarget::WorkerId(
                "test_worker".into()
            ))
        );
        assert!(!worker.matches_target(&soma_core::filter::RemoteTarget::WorkerId("other".into())));
    }

    #[test]
    fn worker_matches_target_by_tag() {
        let worker = make_worker();
        assert!(worker.matches_target(&soma_core::filter::RemoteTarget::Tag("cpu".into())));
        assert!(worker.matches_target(&soma_core::filter::RemoteTarget::Tag("test".into())));
        assert!(!worker.matches_target(&soma_core::filter::RemoteTarget::Tag("gpu".into())));
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
            input: Some(crate::protocol::InputSource::Inline { value: Value::tensor(vec![5.0], vec![1]) }),
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
            input: Some(crate::protocol::InputSource::Inline { value: Value::tensor(vec![1.0], vec![1]) }),
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
