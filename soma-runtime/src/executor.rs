use crate::event_bus::EventBus;
use soma_compiler::ExecutionPlan;
use soma_core::cache::CacheStore;
use soma_core::error::{Result, SomaError};
use soma_core::event::Event;
use soma_core::filter::Filter;
use soma_core::value::Value;
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

    /// Build GraphInfo from a soma_core::graph::Graph.
    pub fn from_graph(graph: &soma_core::graph::Graph) -> Self {
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

/// Execution context passed to filters during runtime.
pub struct Context {
    /// Outputs of completed nodes, keyed by node ID.
    pub store: HashMap<String, Value>,
    /// Event bus for emitting runtime events.
    pub event_bus: Arc<EventBus>,
    /// Current run ID.
    pub run_id: String,
    /// Track execution order.
    pub execution_order: Vec<String>,
    /// Graph topology for input resolution.
    pub graph_info: GraphInfo,
}

impl Context {
    pub fn new(event_bus: Arc<EventBus>, run_id: impl Into<String>) -> Self {
        Self {
            store: HashMap::new(),
            event_bus,
            run_id: run_id.into(),
            execution_order: Vec::new(),
            graph_info: GraphInfo::new(),
        }
    }

    pub fn with_graph_info(mut self, info: GraphInfo) -> Self {
        self.graph_info = info;
        self
    }

    pub fn get(&self, node_id: &str) -> Option<&Value> {
        self.store.get(node_id)
    }

    pub fn set(&mut self, node_id: impl Into<String>, value: Value) {
        let id = node_id.into();
        self.execution_order.push(id.clone());
        self.store.insert(id, value);
    }
}

/// Registry of filter implementations, keyed by node ID.
pub struct FilterStore {
    filters: HashMap<String, Box<dyn Filter>>,
    states: HashMap<String, Value>,
}

impl FilterStore {
    pub fn new() -> Self {
        Self {
            filters: HashMap::new(),
            states: HashMap::new(),
        }
    }

    pub fn register(&mut self, node_id: impl Into<String>, filter: Box<dyn Filter>) {
        self.filters.insert(node_id.into(), filter);
    }

    pub fn get(&self, node_id: &str) -> Option<&dyn Filter> {
        self.filters.get(node_id).map(|f| f.as_ref())
    }

    pub fn set_state(&mut self, node_id: impl Into<String>, state: Value) {
        self.states.insert(node_id.into(), state);
    }

    pub fn get_state(&self, node_id: &str) -> Option<&Value> {
        self.states.get(node_id)
    }
}

impl Default for FilterStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Execute a compiled plan.
pub fn execute(
    plan: &ExecutionPlan,
    ctx: &mut Context,
    filters: &FilterStore,
    cache: &dyn CacheStore,
) -> Result<()> {
    match plan {
        ExecutionPlan::Empty => Ok(()),

        ExecutionPlan::Execute { node_id } => {
            let start = Instant::now();

            ctx.event_bus.emit(Event::NodeStarted {
                run_id: ctx.run_id.clone(),
                node_id: node_id.clone(),
                kind: filters
                    .get(node_id)
                    .map(|f| f.meta().kind)
                    .unwrap_or(soma_core::filter::FilterKind::Opaque),
            });

            let filter = filters.get(node_id).ok_or_else(|| SomaError::NodeNotFound(node_id.clone()))?;

            // Resolve input from predecessors
            let input = resolve_input(node_id, ctx);

            // Get or compute state
            let state = match filters.get_state(node_id) {
                Some(s) => s.clone(),
                None => Value::Empty,
            };

            // Execute forward
            let result = filter.forward(&input, &state);

            match result {
                Ok(output) => {
                    let duration = start.elapsed();
                    let summary = format!("{output}");

                    ctx.set(node_id.clone(), output);

                    ctx.event_bus.emit(Event::NodeCompleted {
                        run_id: ctx.run_id.clone(),
                        node_id: node_id.clone(),
                        duration,
                        output_summary: summary,
                    });
                    Ok(())
                }
                Err(e) => {
                    ctx.event_bus.emit(Event::NodeFailed {
                        run_id: ctx.run_id.clone(),
                        node_id: node_id.clone(),
                        error: e.to_string(),
                    });
                    Err(e)
                }
            }
        }

        ExecutionPlan::Cached { node_id, key } => {
            let start = Instant::now();

            let value = cache.get(key)?.ok_or_else(|| {
                SomaError::Cache(format!("expected cached value for node `{node_id}` not found"))
            })?;

            ctx.set(node_id.clone(), value);

            ctx.event_bus.emit(Event::NodeCacheHit {
                run_id: ctx.run_id.clone(),
                node_id: node_id.clone(),
                key: key.clone(),
                tier: soma_core::cache::CacheTier::Memory,
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

        ExecutionPlan::Parallel(branches) => {
            // Synchronous parallel for now (true async parallelism in future)
            let mut branch_outputs: Vec<(String, Value)> = Vec::new();

            for branch in branches {
                let mut branch_ctx = Context {
                    store: ctx.store.clone(),
                    event_bus: ctx.event_bus.clone(),
                    run_id: ctx.run_id.clone(),
                    execution_order: ctx.execution_order.clone(),
                    graph_info: ctx.graph_info.clone(),
                };
                execute(branch, &mut branch_ctx, filters, cache)?;

                for (key, value) in &branch_ctx.store {
                    if !ctx.store.contains_key(key) {
                        branch_outputs.push((key.clone(), value.clone()));
                    }
                }
            }

            for (key, value) in branch_outputs {
                ctx.set(key, value);
            }
            Ok(())
        }

        ExecutionPlan::Loop { node_id: _, body, max_iterations } => {
            let max = max_iterations.unwrap_or(usize::MAX);
            for _i in 0..max {
                execute(body, ctx, filters, cache)?;
            }
            Ok(())
        }

        ExecutionPlan::Branch { node_id: _, arms } => {
            if let Some((_, plan)) = arms.first() {
                execute(plan, ctx, filters, cache)?;
            }
            Ok(())
        }

        ExecutionPlan::Remote { plan, .. } => {
            execute(plan, ctx, filters, cache)
        }

        _ => Ok(()),
    }
}

/// Resolve the input for a node from the context store using graph topology.
///
/// - If the node has one predecessor, use that predecessor's output.
/// - If the node has multiple predecessors, merge them into a JSON object.
/// - If the node has no predecessors, use Value::Empty (root node).
/// - Fallback: if no graph info, use the most recently produced output.
fn resolve_input(node_id: &str, ctx: &Context) -> Value {
    let preds = ctx.graph_info.predecessors(node_id);

    match preds.len() {
        0 => {
            // Root node or no graph info: fall back to last output
            ctx.execution_order
                .last()
                .and_then(|last_id| ctx.store.get(last_id))
                .cloned()
                .unwrap_or(Value::Empty)
        }
        1 => {
            // Single predecessor: use its output directly
            ctx.store
                .get(&preds[0])
                .cloned()
                .unwrap_or(Value::Empty)
        }
        _ => {
            // Multiple predecessors: merge into JSON object keyed by node ID
            let mut merged = serde_json::Map::new();
            for pred_id in preds {
                if let Some(val) = ctx.store.get(pred_id) {
                    let json_val = serde_json::to_value(val).unwrap_or(serde_json::Value::Null);
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
    use soma_core::cache::CacheKey;
    use soma_core::filter::{FilterKind, FilterMeta, StreamMode};

    // ── Test filters ──

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
                distribution: soma_core::filter::Distribution::Local,
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
                distribution: soma_core::filter::Distribution::Local,
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

        // Tell executor that "doubler" reads from "input"
        ctx.graph_info
            .set_predecessors("doubler", vec!["input".into()]);

        let mut filters = FilterStore::new();
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

        // Linear: input → add → double
        let graph_info = GraphInfo::for_linear(&["input", "add", "double"]);
        ctx.graph_info = graph_info;

        let mut filters = FilterStore::new();
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
        assert_eq!(data, &[22.0, 24.0]); // (1+10)*2=22, (2+10)*2=24
    }

    #[test]
    fn execute_cached_node() {
        let (bus, cache) = setup();
        let key = CacheKey::hash_data(b"cached_result");
        let cached_value = Value::tensor(vec![99.0], vec![1]);
        cache.put(&key, &cached_value).unwrap();

        let mut ctx = Context::new(bus, "run_1");
        let filters = FilterStore::new();

        let plan = ExecutionPlan::Cached {
            node_id: "cached_node".into(),
            key,
        };

        execute(&plan, &mut ctx, &filters, &cache).unwrap();

        let result = ctx.get("cached_node").unwrap();
        assert_eq!(*result, cached_value);
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

        let mut filters = FilterStore::new();
        filters.register("double", Box::new(DoublerFilter));

        let plan = ExecutionPlan::Execute {
            node_id: "double".into(),
        };

        execute(&plan, &mut ctx, &filters, &cache).unwrap();

        let e1 = rx.try_recv().unwrap();
        assert!(matches!(e1, Event::NodeStarted { .. }));
        let e2 = rx.try_recv().unwrap();
        assert!(matches!(e2, Event::NodeCompleted { .. }));
    }

    #[test]
    fn execute_missing_filter_errors() {
        let (bus, cache) = setup();
        let mut ctx = Context::new(bus, "run_1");
        let filters = FilterStore::new();

        let plan = ExecutionPlan::Execute {
            node_id: "nonexistent".into(),
        };

        let result = execute(&plan, &mut ctx, &filters, &cache);
        assert!(matches!(result, Err(SomaError::NodeNotFound(_))));
    }

    #[test]
    fn execute_empty_plan() {
        let (bus, cache) = setup();
        let mut ctx = Context::new(bus, "run_1");
        let filters = FilterStore::new();

        execute(&ExecutionPlan::Empty, &mut ctx, &filters, &cache).unwrap();
    }

    #[test]
    fn execute_parallel_branches_merge_outputs() {
        let (bus, cache) = setup();
        let mut ctx = Context::new(bus, "run_1");
        ctx.set("input", Value::tensor(vec![5.0], vec![1]));

        // Both branches read from "input"
        ctx.graph_info
            .set_predecessors("double", vec!["input".into()]);
        ctx.graph_info
            .set_predecessors("add", vec!["input".into()]);

        let mut filters = FilterStore::new();
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

        // Both branches produced their own outputs
        let double_out = ctx.get("double").unwrap().as_tensor().unwrap().0;
        assert_eq!(double_out, &[10.0]); // 5*2

        let add_out = ctx.get("add").unwrap().as_tensor().unwrap().0;
        assert_eq!(add_out, &[105.0]); // 5+100
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
        // Should be a JSON object with both predecessor outputs
        let json = input.as_json().unwrap();
        assert!(json.get("A").is_some());
        assert!(json.get("B").is_some());
    }

    #[test]
    fn resolve_input_no_predecessors_fallback() {
        let bus = Arc::new(EventBus::new(8));
        let mut ctx = Context::new(bus, "r");
        ctx.set("prev", Value::tensor(vec![7.0], vec![1]));
        // No graph info for "root" → falls back to last output
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
