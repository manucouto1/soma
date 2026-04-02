//! Plan executor — walks [`ExecutionPlan`] trees and runs filter nodes.
//!
//! Handles sequential, parallel (scoped threads), cached, remote, loop,
//! and branch execution. Uses [`GraphInfo`] for topology-aware input resolution.

use crate::event_bus::EventBus;
use crate::filter_library::FilterLibrary;
use somatize_compiler::ExecutionPlan;
use somatize_core::cache::CacheStore;
use somatize_core::error::{Result, SomaError};
use somatize_core::event::Event;
use somatize_core::store::DataStore;
use somatize_core::value::Value;
use somatize_core::virtual_value::VirtualValue;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// Graph topology information for input resolution.
///
/// Maps each node to its predecessor node IDs so the executor knows
/// where to read inputs from in the context store.
#[derive(Debug, Clone, Default)]
pub struct GraphInfo {
    /// node_id → list of predecessor node IDs
    predecessors: HashMap<String, Vec<String>>,
}

impl GraphInfo {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register predecessors for a node.
    pub fn set_predecessors(&mut self, node_id: impl Into<String>, preds: Vec<String>) {
        self.predecessors.insert(node_id.into(), preds);
    }

    /// Build GraphInfo from a somatize_core::graph::Graph.
    pub fn from_graph(graph: &somatize_core::graph::Graph) -> Self {
        let mut info = Self::new();
        for node in &graph.nodes {
            let preds: Vec<String> = graph
                .predecessors(&node.id)
                .into_iter()
                .map(|s| s.to_string())
                .collect();
            info.set_predecessors(node.id.clone(), preds);
        }
        info
    }

    /// Build GraphInfo for a linear pipeline (each node depends on the previous).
    pub fn for_linear(node_ids: &[&str]) -> Self {
        let mut info = Self::new();
        for (i, &id) in node_ids.iter().enumerate() {
            let preds = if i > 0 {
                vec![node_ids[i - 1].to_string()]
            } else {
                vec![]
            };
            info.set_predecessors(id, preds);
        }
        info
    }

    /// Get predecessors for a node.
    pub fn predecessors(&self, node_id: &str) -> &[String] {
        self.predecessors
            .get(node_id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }
}

/// Trait for executing plan nodes on remote workers.
///
/// When set on Context, `ExecutionPlan::Remote` nodes delegate to this
/// instead of executing locally. The implementation sends the sub-plan
/// to a worker and returns the result.
pub trait RemoteExecutor: Send + Sync {
    /// Execute a sub-plan remotely and return the output value.
    fn execute_remote(
        &self,
        node_id: &str,
        target: &somatize_core::filter::RemoteTarget,
        input: Option<&Value>,
    ) -> Result<Value>;
}

/// Execution context passed to filters during runtime.
///
/// Node outputs are stored as [`VirtualValue`]s — they may be materialized
/// in memory, cached on disk, or deferred (not yet computed). The executor
/// resolves them on demand when a downstream node needs the data.
pub struct Context {
    /// Node outputs as virtual values (may be lazy).
    pub store: HashMap<String, VirtualValue>,
    /// Event bus for emitting runtime events.
    pub event_bus: Arc<EventBus>,
    /// Current run ID.
    pub run_id: String,
    /// Track execution order.
    pub execution_order: Vec<String>,
    /// Graph topology for input resolution.
    pub graph_info: GraphInfo,
    /// Optional remote executor for distributed plans.
    pub remote_executor: Option<Arc<dyn RemoteExecutor>>,
    /// Optional data store for persisting intermediate results.
    pub data_store: Option<Arc<dyn DataStore>>,
    /// Minimum value size (bytes) to spill to DataStore instead of keeping in memory.
    /// Default: 0 (disabled — all values stay in memory).
    pub spill_threshold: usize,
}

impl Context {
    pub fn new(event_bus: Arc<EventBus>, run_id: impl Into<String>) -> Self {
        Self {
            store: HashMap::new(),
            event_bus,
            run_id: run_id.into(),
            execution_order: Vec::new(),
            graph_info: GraphInfo::new(),
            remote_executor: None,
            data_store: None,
            spill_threshold: 0,
        }
    }

    pub fn with_graph_info(mut self, info: GraphInfo) -> Self {
        self.graph_info = info;
        self
    }

    pub fn with_remote_executor(mut self, executor: Arc<dyn RemoteExecutor>) -> Self {
        self.remote_executor = Some(executor);
        self
    }

    pub fn with_data_store(mut self, store: Arc<dyn DataStore>) -> Self {
        self.data_store = Some(store);
        self
    }

    /// Set spill threshold: values larger than this (in bytes) are offloaded
    /// to the DataStore and replaced with a VirtualValue::Cached reference.
    /// Requires a DataStore to be set via `with_data_store()`.
    pub fn with_spill_threshold(mut self, bytes: usize) -> Self {
        self.spill_threshold = bytes;
        self
    }

    /// If a DataStore and spill threshold are configured, check if the value
    /// should be offloaded. Returns VirtualValue (materialized or cached ref).
    fn maybe_spill(&self, node_id: &str, value: Value) -> VirtualValue {
        if self.spill_threshold > 0
            && let Some(store) = &self.data_store
        {
            let size = value.size() * 8; // approximate bytes (f64 = 8 bytes)
            if size >= self.spill_threshold {
                let key = somatize_core::cache::CacheKey::from_parts(&[
                    self.run_id.as_bytes(),
                    node_id.as_bytes(),
                ]);
                let vv_for_schema = VirtualValue::materialized(value.clone());
                let schema = vv_for_schema.schema().clone();
                if let Ok(_data_ref) = store.put(&key, &value) {
                    tracing::debug!("spilled node `{node_id}` ({size} bytes) to DataStore");
                    return VirtualValue::cached(key, schema);
                }
            }
        }
        VirtualValue::materialized(value)
    }

    /// Get the materialized Value for a node, if present and materialized.
    pub fn get(&self, node_id: &str) -> Option<&Value> {
        self.store.get(node_id).and_then(|vv| vv.as_value())
    }

    /// Get the raw VirtualValue for a node.
    pub fn get_virtual(&self, node_id: &str) -> Option<&VirtualValue> {
        self.store.get(node_id)
    }

    /// Store a materialized value for a node.
    pub fn set(&mut self, node_id: impl Into<String>, value: Value) {
        let id = node_id.into();
        self.execution_order.push(id.clone());
        self.store.insert(id, VirtualValue::materialized(value));
    }

    /// Store a virtual value (which may be deferred or cached).
    pub fn set_virtual(&mut self, node_id: impl Into<String>, vv: VirtualValue) {
        let id = node_id.into();
        self.execution_order.push(id.clone());
        self.store.insert(id, vv);
    }

    fn snapshot(&self) -> Self {
        Self {
            store: self.store.clone(),
            event_bus: self.event_bus.clone(),
            run_id: self.run_id.clone(),
            execution_order: self.execution_order.clone(),
            graph_info: self.graph_info.clone(),
            remote_executor: self.remote_executor.clone(),
            data_store: self.data_store.clone(),
            spill_threshold: self.spill_threshold,
        }
    }
}

/// Execute a compiled plan synchronously.
/// For parallel branches, uses the async executor under the hood.
pub fn execute(
    plan: &ExecutionPlan,
    ctx: &mut Context,
    filters: &FilterLibrary,
    cache: &dyn CacheStore,
) -> Result<()> {
    match plan {
        ExecutionPlan::Empty => Ok(()),

        ExecutionPlan::Execute { node_id } => execute_node(node_id, ctx, filters, cache),

        ExecutionPlan::Cached { node_id, key } => {
            let start = Instant::now();
            let value = cache.get(key)?.ok_or_else(|| {
                SomaError::Cache(format!(
                    "expected cached value for node `{node_id}` not found"
                ))
            })?;
            ctx.set(node_id.clone(), value);
            ctx.event_bus.emit(Event::NodeCacheHit {
                run_id: ctx.run_id.clone(),
                node_id: node_id.clone(),
                key: key.clone(),
                tier: somatize_core::cache::CacheTier::Memory,
                load_time: start.elapsed(),
            });
            Ok(())
        }

        ExecutionPlan::Sequence(steps) => {
            for step in steps {
                execute(step, ctx, filters, cache)?;
            }
            Ok(())
        }

        ExecutionPlan::Parallel(branches) => execute_parallel(branches, ctx, filters, cache),

        ExecutionPlan::Loop {
            node_id,
            body,
            max_iterations,
        } => {
            let max = max_iterations.unwrap_or(100);
            for i in 0..max {
                execute(body, ctx, filters, cache)?;

                // Check termination: if the last executed node produced a Value
                // that indicates "done" (true, "done", "stop", or empty), break.
                let should_stop = ctx
                    .execution_order
                    .last()
                    .and_then(|last_id| ctx.get(last_id))
                    .map(|v| match v {
                        Value::Json(j) => {
                            j.as_bool() == Some(true)
                                || j.as_str().map(|s| s == "done" || s == "stop") == Some(true)
                                || j.get("done").and_then(|d| d.as_bool()) == Some(true)
                        }
                        Value::Empty => true,
                        _ => false,
                    })
                    .unwrap_or(false);

                if should_stop {
                    ctx.event_bus.emit(Event::NodeCompleted {
                        run_id: ctx.run_id.clone(),
                        node_id: node_id.clone(),
                        duration: std::time::Duration::ZERO,
                        output_summary: format!("Loop terminated at iteration {}", i + 1),
                    });
                    break;
                }
            }
            Ok(())
        }

        ExecutionPlan::Branch { node_id, arms } => {
            // Execute the branch node first (it produces the condition value)
            execute_node(node_id, ctx, filters, cache)?;

            // Get the condition result
            let condition = ctx.get(node_id).cloned().unwrap_or(Value::Empty);

            // Match against arm labels
            let selected_arm = match &condition {
                Value::Json(j) => {
                    // Try matching by string value, bool, or "branch" field
                    let selector = j
                        .as_str()
                        .map(String::from)
                        .or_else(|| j.as_bool().map(|b| b.to_string()))
                        .or_else(|| j.get("branch").and_then(|b| b.as_str()).map(String::from))
                        .unwrap_or_else(|| "true".to_string());

                    arms.iter()
                        .find(|(label, _)| label == &selector)
                        .or_else(|| {
                            arms.iter()
                                .find(|(label, _)| label == "default" || label == "else")
                        })
                        .or_else(|| arms.first())
                }
                _ => arms.first(),
            };

            if let Some((label, plan)) = selected_arm {
                ctx.event_bus.emit(Event::NodeCompleted {
                    run_id: ctx.run_id.clone(),
                    node_id: node_id.clone(),
                    duration: std::time::Duration::ZERO,
                    output_summary: format!("Branch selected: {label}"),
                });
                execute(plan, ctx, filters, cache)?;
            }
            Ok(())
        }

        ExecutionPlan::Remote {
            node_id,
            target,
            plan,
        } => {
            if let Some(remote) = &ctx.remote_executor {
                // Gather input from predecessors
                let input = ctx
                    .graph_info
                    .predecessors(node_id)
                    .first()
                    .and_then(|pred| ctx.get(pred));

                let result = remote.execute_remote(node_id, target, input)?;
                ctx.set(node_id.clone(), result);
                ctx.execution_order.push(node_id.clone());
                Ok(())
            } else {
                // No remote executor — fall back to local execution
                execute(plan, ctx, filters, cache)
            }
        }

        _ => {
            tracing::warn!("Unhandled ExecutionPlan variant");
            Ok(())
        }
    }
}

/// Execute a single filter node.
fn execute_node(
    node_id: &str,
    ctx: &mut Context,
    filters: &FilterLibrary,
    _cache: &dyn CacheStore,
) -> Result<()> {
    let start = Instant::now();

    let filter = filters
        .get(node_id)
        .ok_or_else(|| SomaError::NodeNotFound(node_id.to_string()))?;

    ctx.event_bus.emit(Event::NodeStarted {
        run_id: ctx.run_id.clone(),
        node_id: node_id.to_string(),
        kind: filter.meta().kind,
    });

    let input = resolve_input(node_id, ctx);
    let state = filters.get_state(node_id).cloned().unwrap_or(Value::Empty);
    let result = filter.forward(&input, &state);

    match result {
        Ok(output) => {
            let duration = start.elapsed();
            let summary = format!("{output}");
            let vv = ctx.maybe_spill(node_id, output);
            ctx.set_virtual(node_id, vv);
            ctx.event_bus.emit(Event::NodeCompleted {
                run_id: ctx.run_id.clone(),
                node_id: node_id.to_string(),
                duration,
                output_summary: summary,
            });
            Ok(())
        }
        Err(e) => {
            ctx.event_bus.emit(Event::NodeFailed {
                run_id: ctx.run_id.clone(),
                node_id: node_id.to_string(),
                error: e.to_string(),
            });
            Err(e)
        }
    }
}

/// Execute parallel branches concurrently using std::thread::scope.
///
/// Each branch gets a snapshot of the context. After all branches complete,
/// their new outputs are merged back into the main context.
fn execute_parallel(
    branches: &[ExecutionPlan],
    ctx: &mut Context,
    filters: &FilterLibrary,
    cache: &dyn CacheStore,
) -> Result<()> {
    let snapshot_keys: Arc<std::collections::HashSet<String>> =
        Arc::new(ctx.store.keys().cloned().collect());

    // Use scoped threads for true parallelism without Send requirements
    let results: Vec<Result<Vec<(String, VirtualValue)>>> = std::thread::scope(|s| {
        let handles: Vec<_> = branches
            .iter()
            .map(|branch| {
                let mut branch_ctx = ctx.snapshot();
                let keys = snapshot_keys.clone();
                s.spawn(move || {
                    execute(branch, &mut branch_ctx, filters, cache)?;
                    let new_entries: Vec<(String, VirtualValue)> = branch_ctx
                        .store
                        .into_iter()
                        .filter(|(k, _)| !keys.contains(k))
                        .collect();
                    Ok(new_entries)
                })
            })
            .collect();

        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // Merge results and propagate first error
    for result in results {
        let entries = result?;
        for (key, vv) in entries {
            ctx.set_virtual(key, vv);
        }
    }

    Ok(())
}

/// Resolve a VirtualValue to a concrete Value, loading from DataStore if needed.
fn resolve_value(vv: &VirtualValue, data_store: &Option<Arc<dyn DataStore>>) -> Option<Value> {
    match vv {
        VirtualValue::Materialized { value, .. } => Some(value.clone()),
        VirtualValue::Cached { key, .. } => {
            // Try to load from DataStore
            if let Some(store) = data_store {
                let data_ref = somatize_core::store::DataRef::Cached {
                    cache_key: key.clone(),
                };
                store.get(&data_ref).ok()
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Resolve the input for a node from the context store using graph topology.
/// If a predecessor was spilled to DataStore, loads it back.
pub fn resolve_input(node_id: &str, ctx: &Context) -> Value {
    let preds = ctx.graph_info.predecessors(node_id);

    let resolve_node = |id: &str| -> Option<Value> {
        ctx.store
            .get(id)
            .and_then(|vv| resolve_value(vv, &ctx.data_store))
    };

    match preds.len() {
        0 => ctx
            .execution_order
            .last()
            .and_then(|id| resolve_node(id))
            .unwrap_or(Value::Empty),
        1 => resolve_node(&preds[0]).unwrap_or(Value::Empty),
        _ => {
            let mut merged = serde_json::Map::new();
            for pred_id in preds {
                if let Some(val) = resolve_node(pred_id) {
                    let json_val = serde_json::to_value(&val).unwrap_or(serde_json::Value::Null);
                    merged.insert(pred_id.clone(), json_val);
                }
            }
            Value::Json(serde_json::Value::Object(merged))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::MemoryCache;
    use somatize_core::cache::CacheKey;
    use somatize_core::filter::{Filter, FilterKind, FilterMeta, StreamMode};

    struct DoublerFilter;

    impl Filter for DoublerFilter {
        fn config_hash(&self) -> CacheKey {
            CacheKey::from_parts(&[b"Doubler"])
        }
        fn fit(&self, _x: &Value, _y: Option<&Value>) -> Result<Value> {
            Ok(Value::Empty)
        }
        fn forward(&self, x: &Value, _state: &Value) -> Result<Value> {
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
                name: "Doubler".into(),
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

    struct AdderFilter {
        amount: f64,
    }

    impl Filter for AdderFilter {
        fn config_hash(&self) -> CacheKey {
            CacheKey::from_parts(&[b"Adder", &self.amount.to_le_bytes()])
        }
        fn fit(&self, _x: &Value, _y: Option<&Value>) -> Result<Value> {
            Ok(Value::Empty)
        }
        fn forward(&self, x: &Value, _state: &Value) -> Result<Value> {
            match x {
                Value::Tensor { values, shape } => {
                    let added: Vec<f64> = values.iter().map(|v| v + self.amount).collect();
                    Ok(Value::tensor(added, shape.clone()))
                }
                _ => Ok(x.clone()),
            }
        }
        fn meta(&self) -> FilterMeta {
            FilterMeta {
                name: "Adder".into(),
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

    /// Slow filter that sleeps to verify parallelism.
    struct SlowFilter {
        id: String,
        delay_ms: u64,
    }

    impl Filter for SlowFilter {
        fn config_hash(&self) -> CacheKey {
            CacheKey::from_parts(&[b"Slow", self.id.as_bytes()])
        }
        fn fit(&self, _x: &Value, _y: Option<&Value>) -> Result<Value> {
            Ok(Value::Empty)
        }
        fn forward(&self, x: &Value, _state: &Value) -> Result<Value> {
            std::thread::sleep(std::time::Duration::from_millis(self.delay_ms));
            Ok(x.clone())
        }
        fn meta(&self) -> FilterMeta {
            FilterMeta {
                name: format!("Slow_{}", self.id),
                kind: FilterKind::Stateless,
                cacheable: false,
                differentiable: true,
                stream_mode: StreamMode::FixedState,
                distribution: somatize_core::filter::Distribution::Local,
                input_schema: None,
                output_schema: None,
            }
        }
    }

    fn setup() -> (Arc<EventBus>, MemoryCache) {
        (Arc::new(EventBus::new(64)), MemoryCache::default())
    }

    #[test]
    fn execute_single_node() {
        let (bus, cache) = setup();
        let mut ctx = Context::new(bus, "run_1");
        ctx.set("input", Value::tensor(vec![1.0, 2.0, 3.0], vec![3]));
        ctx.graph_info
            .set_predecessors("doubler", vec!["input".into()]);

        let mut filters = FilterLibrary::new();
        filters.register("doubler", Box::new(DoublerFilter));

        let plan = ExecutionPlan::Execute {
            node_id: "doubler".into(),
        };

        execute(&plan, &mut ctx, &filters, &cache).unwrap();

        let result = ctx.get("doubler").unwrap();
        let (data, _) = result.as_tensor().unwrap();
        assert_eq!(data, &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn execute_sequence_with_graph_info() {
        let (bus, cache) = setup();
        let mut ctx = Context::new(bus, "run_1");
        ctx.set("input", Value::tensor(vec![1.0, 2.0], vec![2]));

        let graph_info = GraphInfo::for_linear(&["input", "add", "double"]);
        ctx.graph_info = graph_info;

        let mut filters = FilterLibrary::new();
        filters.register("add", Box::new(AdderFilter { amount: 10.0 }));
        filters.register("double", Box::new(DoublerFilter));

        let plan = ExecutionPlan::Sequence(vec![
            ExecutionPlan::Execute {
                node_id: "add".into(),
            },
            ExecutionPlan::Execute {
                node_id: "double".into(),
            },
        ]);

        execute(&plan, &mut ctx, &filters, &cache).unwrap();

        let result = ctx.get("double").unwrap();
        let (data, _) = result.as_tensor().unwrap();
        assert_eq!(data, &[22.0, 24.0]);
    }

    #[test]
    fn execute_cached_node() {
        let (bus, cache) = setup();
        let key = CacheKey::hash_data(b"cached_result");
        let cached_value = Value::tensor(vec![99.0], vec![1]);
        cache.put(&key, &cached_value).unwrap();

        let mut ctx = Context::new(bus, "run_1");
        let filters = FilterLibrary::new();

        let plan = ExecutionPlan::Cached {
            node_id: "cached_node".into(),
            key,
        };

        execute(&plan, &mut ctx, &filters, &cache).unwrap();
        assert_eq!(*ctx.get("cached_node").unwrap(), cached_value);
    }

    #[test]
    fn execute_emits_events() {
        let bus = Arc::new(EventBus::new(64));
        let cache = MemoryCache::default();
        let mut rx = bus.subscribe();

        let mut ctx = Context::new(bus, "run_1");
        ctx.set("input", Value::tensor(vec![1.0], vec![1]));
        ctx.graph_info
            .set_predecessors("double", vec!["input".into()]);

        let mut filters = FilterLibrary::new();
        filters.register("double", Box::new(DoublerFilter));

        execute(
            &ExecutionPlan::Execute {
                node_id: "double".into(),
            },
            &mut ctx,
            &filters,
            &cache,
        )
        .unwrap();

        let e1 = rx.try_recv().unwrap();
        assert!(matches!(e1, Event::NodeStarted { .. }));
        let e2 = rx.try_recv().unwrap();
        assert!(matches!(e2, Event::NodeCompleted { .. }));
    }

    #[test]
    fn execute_missing_filter_errors() {
        let (bus, cache) = setup();
        let mut ctx = Context::new(bus, "run_1");
        let filters = FilterLibrary::new();

        let result = execute(
            &ExecutionPlan::Execute {
                node_id: "nonexistent".into(),
            },
            &mut ctx,
            &filters,
            &cache,
        );
        assert!(matches!(result, Err(SomaError::NodeNotFound(_))));
    }

    #[test]
    fn execute_empty_plan() {
        let (bus, cache) = setup();
        let mut ctx = Context::new(bus, "run_1");
        let filters = FilterLibrary::new();
        execute(&ExecutionPlan::Empty, &mut ctx, &filters, &cache).unwrap();
    }

    #[test]
    fn execute_parallel_branches_merge_outputs() {
        let (bus, cache) = setup();
        let mut ctx = Context::new(bus, "run_1");
        ctx.set("input", Value::tensor(vec![5.0], vec![1]));
        ctx.graph_info
            .set_predecessors("double", vec!["input".into()]);
        ctx.graph_info.set_predecessors("add", vec!["input".into()]);

        let mut filters = FilterLibrary::new();
        filters.register("double", Box::new(DoublerFilter));
        filters.register("add", Box::new(AdderFilter { amount: 100.0 }));

        let plan = ExecutionPlan::Parallel(vec![
            ExecutionPlan::Execute {
                node_id: "double".into(),
            },
            ExecutionPlan::Execute {
                node_id: "add".into(),
            },
        ]);

        execute(&plan, &mut ctx, &filters, &cache).unwrap();

        let double_out = ctx.get("double").unwrap().as_tensor().unwrap().0;
        assert_eq!(double_out, &[10.0]);
        let add_out = ctx.get("add").unwrap().as_tensor().unwrap().0;
        assert_eq!(add_out, &[105.0]);
    }

    #[test]
    fn parallel_branches_run_concurrently() {
        let (bus, cache) = setup();
        let mut ctx = Context::new(bus, "run_1");
        ctx.set("input", Value::tensor(vec![1.0], vec![1]));
        ctx.graph_info
            .set_predecessors("slow_a", vec!["input".into()]);
        ctx.graph_info
            .set_predecessors("slow_b", vec!["input".into()]);

        let mut filters = FilterLibrary::new();
        filters.register(
            "slow_a",
            Box::new(SlowFilter {
                id: "a".into(),
                delay_ms: 50,
            }),
        );
        filters.register(
            "slow_b",
            Box::new(SlowFilter {
                id: "b".into(),
                delay_ms: 50,
            }),
        );

        let plan = ExecutionPlan::Parallel(vec![
            ExecutionPlan::Execute {
                node_id: "slow_a".into(),
            },
            ExecutionPlan::Execute {
                node_id: "slow_b".into(),
            },
        ]);

        let start = Instant::now();
        execute(&plan, &mut ctx, &filters, &cache).unwrap();
        let elapsed = start.elapsed();

        // If truly parallel: ~50ms. If sequential: ~100ms.
        // Use 90ms as threshold to account for overhead.
        assert!(
            elapsed.as_millis() < 90,
            "parallel branches took {}ms, expected <90ms (sequential would be ~100ms)",
            elapsed.as_millis()
        );

        assert!(ctx.get("slow_a").is_some());
        assert!(ctx.get("slow_b").is_some());
    }

    #[test]
    fn resolve_input_single_predecessor() {
        let bus = Arc::new(EventBus::new(8));
        let mut ctx = Context::new(bus, "r");
        ctx.set("A", Value::tensor(vec![42.0], vec![1]));
        ctx.graph_info.set_predecessors("B", vec!["A".into()]);

        let input = resolve_input("B", &ctx);
        let (data, _) = input.as_tensor().unwrap();
        assert_eq!(data, &[42.0]);
    }

    #[test]
    fn resolve_input_multiple_predecessors() {
        let bus = Arc::new(EventBus::new(8));
        let mut ctx = Context::new(bus, "r");
        ctx.set("A", Value::tensor(vec![1.0], vec![1]));
        ctx.set("B", Value::tensor(vec![2.0], vec![1]));
        ctx.graph_info
            .set_predecessors("C", vec!["A".into(), "B".into()]);

        let input = resolve_input("C", &ctx);
        let json = input.as_json().unwrap();
        assert!(json.get("A").is_some());
        assert!(json.get("B").is_some());
    }

    #[test]
    fn resolve_input_no_predecessors_fallback() {
        let bus = Arc::new(EventBus::new(8));
        let mut ctx = Context::new(bus, "r");
        ctx.set("prev", Value::tensor(vec![7.0], vec![1]));

        let input = resolve_input("root", &ctx);
        let (data, _) = input.as_tensor().unwrap();
        assert_eq!(data, &[7.0]);
    }

    #[test]
    fn graph_info_from_linear() {
        let info = GraphInfo::for_linear(&["a", "b", "c"]);
        assert!(info.predecessors("a").is_empty());
        assert_eq!(info.predecessors("b"), &["a"]);
        assert_eq!(info.predecessors("c"), &["b"]);
    }
}
