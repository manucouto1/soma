//! Graph session — the primary orchestrator for Graph → Compile → Execute.
//!
//! [`GraphSession`] binds a [`Graph`] with its [`NodeCatalog`], cache,
//! event bus, and optional distributed components into a single object
//! that can compile, fit, and execute.

use crate::cache::MemoryCache;
use crate::event_bus::EventBus;
use crate::executor::{self, Context, GraphInfo};
use crate::node_catalog::NodeCatalog;
use crate::runner::Runner;
use crate::runner::Transport;
use somatize_compiler::{CompileMode, CompileResult, compile};
use somatize_core::cache::{CacheKey, CacheStore};
use somatize_core::error::{Result, SomaError};
use somatize_core::event::Event;
use somatize_core::fingerprint::ArchitectureFingerprint;
use somatize_core::graph::Graph;
use somatize_core::store::{DataRef, DataStore};
use somatize_core::util::timestamp_id;
use somatize_core::value::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// The primary orchestrator: Graph + filters + cache + events.
///
/// ```ignore
/// let mut lib = NodeCatalog::new();
/// lib.register("scaler", Box::new(MyScaler::new()));
/// lib.register("model", Box::new(MyModel::new()));
///
/// let mut session = GraphSession::new(graph, lib);
/// session.fit(&train_x, Some(&train_y))?;
/// let output = session.forward(&test_x)?;
/// ```
pub struct GraphSession {
    graph: Graph,
    library: NodeCatalog,
    cache: Arc<dyn CacheStore>,
    event_bus: Arc<EventBus>,
    data_store: Option<Arc<dyn DataStore>>,
    transport: Option<Arc<dyn Transport>>,
    fitted: bool,
}

impl GraphSession {
    pub fn new(graph: Graph, library: NodeCatalog) -> Self {
        Self {
            graph,
            library,
            cache: Arc::new(MemoryCache::default()),
            event_bus: Arc::new(EventBus::new(256)),
            data_store: None,
            transport: None,
            fitted: false,
        }
    }

    pub fn with_cache(mut self, cache: Arc<dyn CacheStore>) -> Self {
        self.cache = cache;
        self
    }

    pub fn with_event_bus(mut self, bus: Arc<EventBus>) -> Self {
        self.event_bus = bus;
        self
    }

    pub fn with_data_store(mut self, store: Arc<dyn DataStore>) -> Self {
        self.data_store = Some(store);
        self
    }

    pub fn with_transport(mut self, transport: Arc<dyn Transport>) -> Self {
        self.transport = Some(transport);
        self
    }

    // ── Core operations ──

    /// Compile the graph and return diagnostics without executing.
    pub fn compile(&self, mode: CompileMode) -> Result<CompileResult> {
        compile(&self.graph, &self.library, mode, Some(self.cache.as_ref()))
    }

    /// Compile and execute the graph, returning all node outputs.
    ///
    /// Emits a `RunStarted`/`RunCompleted` (or `RunFailed`) bracket
    /// around the node events so readers can compute total duration
    /// and group the run.
    pub fn run(&mut self, mode: CompileMode) -> Result<HashMap<String, Value>> {
        let CompileResult { plan, diagnostics } =
            compile(&self.graph, &self.library, mode, Some(self.cache.as_ref()))?;

        for diag in &diagnostics {
            tracing::warn!("compile diagnostic: {:?}", diag);
        }

        let graph_info = GraphInfo::from_graph(&self.graph);
        let run_id = timestamp_id("graph_run");
        let mut ctx =
            Context::new(self.event_bus.clone(), run_id.clone()).with_graph_info(graph_info);

        if let Some(store) = &self.data_store {
            ctx = ctx.with_data_store(store.clone());
        }
        if let Some(transport) = &self.transport {
            ctx = ctx.with_transport(transport.clone());
        }

        self.event_bus.emit(Event::RunStarted {
            run_id: run_id.clone(),
            plan_summary: plan.summary(),
        });
        let start = std::time::Instant::now();
        if let Err(e) = executor::execute(&plan, &mut ctx, &self.library, self.cache.as_ref()) {
            self.event_bus.emit(Event::RunFailed {
                run_id,
                error: e.to_string(),
            });
            return Err(e);
        }
        self.event_bus.emit(Event::RunCompleted {
            run_id,
            duration: start.elapsed(),
        });

        Ok(ctx.into_outputs())
    }

    /// Fit all trainable filters in topological order.
    /// Delegates to LocalRunner — same execution path as remote workers.
    ///
    /// Emits a `RunStarted`/`RunCompleted` (or `RunFailed`) bracket
    /// tagged with the same run id as the node events inside it.
    pub fn fit(&mut self, x: &Value, y: Option<&Value>) -> Result<HashMap<String, Value>> {
        self.graph.validate()?;

        let CompileResult { plan, .. } = compile(
            &self.graph,
            &self.library,
            CompileMode::NoCache,
            Some(self.cache.as_ref()),
        )?;

        let run_id = timestamp_id("fit");
        self.event_bus.emit(Event::RunStarted {
            run_id: run_id.clone(),
            plan_summary: plan.summary(),
        });
        let start = std::time::Instant::now();

        let runner = crate::runner::LocalRunner;
        let ctx = crate::runner::RunContext::new(
            &self.library,
            self.cache.as_ref(),
            &self.event_bus,
            &run_id,
            GraphInfo::from_graph(&self.graph),
        );
        let result = runner.fit(&plan, &ctx, x, y);
        let (_last_output, mut all_outputs) = match result {
            Ok(out) => {
                self.event_bus.emit(Event::RunCompleted {
                    run_id,
                    duration: start.elapsed(),
                });
                out
            }
            Err(e) => {
                self.event_bus.emit(Event::RunFailed {
                    run_id,
                    error: e.to_string(),
                });
                return Err(e);
            }
        };

        // Store trained states from __state_ keys into NodeCatalog
        for (key, value) in &all_outputs {
            if let Some(node_id) = somatize_core::keys::node_of_state_key(key) {
                self.library.try_set_state(node_id, value.clone())?;
            }
        }

        // Remove __state_ keys from returned outputs (callers expect node IDs only)
        all_outputs.retain(|k, _| somatize_core::keys::node_of_state_key(k).is_none());

        self.fitted = true;
        Ok(all_outputs)
    }

    /// Forward pass using the given strategy.
    ///
    /// Strategies define HOW data flows through the compiled graph:
    /// - [`crate::forward::Standard`] — full input at once with inference caching (default)
    /// - [`crate::forward::Stream`] — chunked input through StreamExecutor
    /// - [`crate::forward::Batched`] — rows from DataStore, batch by batch
    pub fn forward_with(
        &self,
        x: &Value,
        strategy: &dyn crate::forward::ForwardStrategy,
    ) -> Result<Value> {
        strategy.forward(
            &self.graph,
            &self.library,
            self.cache.as_ref(),
            &self.event_bus,
            self.data_store.as_ref(),
            x,
        )
    }

    /// Standard forward pass (shortcut for `forward_with(x, &Standard)`).
    pub fn forward(&self, x: &Value) -> Result<Value> {
        self.forward_with(x, &crate::forward::Standard)
    }

    // ── State persistence ──

    /// Persist all trained states to the data store.
    pub fn persist_states(&self) -> Result<DataRef> {
        let store = self
            .data_store
            .as_ref()
            .ok_or_else(|| SomaError::Execution {
                node_id: "session".into(),
                message: "persist_states requires a data store".into(),
            })?;

        let sorted = self.graph.topological_sort()?;
        let mut states_map = serde_json::Map::new();
        for node_id in &sorted {
            if let Some(state) = self.library.get_state(node_id) {
                let json = serde_json::to_value(&*state)
                    .map_err(|e| SomaError::Other(format!("state serialize: {e}")))?;
                states_map.insert(node_id.to_string(), json);
            }
        }

        let states_value = Value::json(serde_json::Value::Object(states_map));
        let fingerprint = self.graph_config_hash()?;
        let key = CacheKey::from_parts(&[b"graph_states", fingerprint.as_bytes()]);
        store.put(&key, &states_value)
    }

    /// Load previously persisted states from a data store reference.
    pub fn load_states(&mut self, data_ref: &DataRef) -> Result<()> {
        let store = self
            .data_store
            .as_ref()
            .ok_or_else(|| SomaError::Execution {
                node_id: "session".into(),
                message: "load_states requires a data store".into(),
            })?;

        let states_value = store.get(data_ref)?;
        let states_json = states_value
            .as_json()
            .ok_or_else(|| SomaError::Other("persisted states must be JSON".into()))?;
        let obj = states_json
            .as_object()
            .ok_or_else(|| SomaError::Other("persisted states must be a JSON object".into()))?;

        for (node_id, json_val) in obj {
            let value: Value = serde_json::from_value(json_val.clone())
                .map_err(|e| SomaError::Other(format!("state deserialize: {e}")))?;
            self.library.try_set_state(node_id.clone(), value)?;
        }

        self.fitted = true;
        Ok(())
    }

    // ── Observability ──

    /// Subscribe to execution events.
    pub fn subscribe(&self) -> tokio::sync::broadcast::Receiver<Event> {
        self.event_bus.subscribe()
    }

    /// Access the event bus directly.
    pub fn event_bus(&self) -> &Arc<EventBus> {
        &self.event_bus
    }

    /// Whether the session has been fitted.
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// Access the graph.
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Access the filter library.
    pub fn library(&self) -> &NodeCatalog {
        &self.library
    }

    /// Mutable access to the filter library (for registering filters after creation).
    pub fn library_mut(&mut self) -> &mut NodeCatalog {
        &mut self.library
    }

    // ── Private helpers ──

    /// The address under which this graph's trained states are persisted.
    ///
    /// It has to follow the graph's *shape*, not just its node names. The
    /// previous form was `node_ids.join(",")`, so two graphs that shared
    /// node ids but wired them differently — or configured them
    /// differently — persisted to one address and read back each other's
    /// states. [`ArchitectureFingerprint`] already computes exactly this,
    /// canonically, for the experiment pool.
    fn graph_config_hash(&self) -> Result<String> {
        Ok(ArchitectureFingerprint::of(&self.graph)?.digest)
    }
}

// ── Convenience free functions ──
//
// One-liners over [`GraphSession`]. They used to be separate
// implementations, and `graph_fit` was the worst of them: a topological
// loop written from scratch that never compiled a plan, so it ignored
// parallelism, loops, branches and steps outright — and then discarded
// every state it fitted instead of storing it. A graph that ran fine
// through `GraphSession::fit` did something else here.

/// Compile and execute a graph, returning all node outputs.
pub fn graph_run(
    graph: &Graph,
    library: &NodeCatalog,
    mode: CompileMode,
    cache: Arc<dyn CacheStore>,
) -> Result<HashMap<String, Value>> {
    GraphSession::new(graph.clone(), library.clone())
        .with_cache(cache)
        .run(mode)
}

/// Fit all trainable filters, returning every node's output.
pub fn graph_fit(
    graph: &Graph,
    library: &NodeCatalog,
    x: &Value,
    y: Option<&Value>,
    cache: Arc<dyn CacheStore>,
) -> Result<HashMap<String, Value>> {
    GraphSession::new(graph.clone(), library.clone())
        .with_cache(cache)
        .fit(x, y)
}

/// Compile in Inference mode and execute, returning the output.
pub fn graph_predict(
    graph: &Graph,
    library: &NodeCatalog,
    x: &Value,
    cache: Arc<dyn CacheStore>,
) -> Result<Value> {
    GraphSession::new(graph.clone(), library.clone())
        .with_cache(cache)
        .forward(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::MemoryCache;
    use somatize_compiler::NodeRegistry;
    use somatize_core::cache::CacheKey;
    use somatize_core::error::Result;
    use somatize_core::filter::{FilterKind, FilterMeta, StreamMode};
    use somatize_core::graph::{Edge, Node};

    // ── Test filters ──

    struct DoublerFilter;
    impl somatize_core::filter::Filter for DoublerFilter {
        fn config_hash(&self) -> CacheKey {
            CacheKey::from_parts(&[b"Doubler"])
        }
        fn fit(&self, _x: &Value, _y: Option<&Value>) -> Result<Value> {
            Ok(Value::Empty)
        }
        fn forward(&self, x: &Value, _state: &Value) -> Result<Value> {
            let (data, shape) = x
                .as_tensor()
                .ok_or(SomaError::Other("need tensor".into()))?;
            Ok(Value::tensor(
                data.iter().map(|v| v * 2.0).collect(),
                shape.to_vec(),
            ))
        }
        fn meta(&self) -> FilterMeta {
            FilterMeta {
                name: "Doubler".into(),
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

    struct AdderFilter(f64);
    impl somatize_core::filter::Filter for AdderFilter {
        fn config_hash(&self) -> CacheKey {
            CacheKey::from_parts(&[b"Adder", &self.0.to_le_bytes()])
        }
        fn fit(&self, _x: &Value, _y: Option<&Value>) -> Result<Value> {
            Ok(Value::Empty)
        }
        fn forward(&self, x: &Value, _state: &Value) -> Result<Value> {
            let (data, shape) = x
                .as_tensor()
                .ok_or(SomaError::Other("need tensor".into()))?;
            Ok(Value::tensor(
                data.iter().map(|v| v + self.0).collect(),
                shape.to_vec(),
            ))
        }
        fn meta(&self) -> FilterMeta {
            FilterMeta {
                name: "Adder".into(),
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

    struct MeanFilter;
    impl somatize_core::filter::Filter for MeanFilter {
        fn config_hash(&self) -> CacheKey {
            CacheKey::from_parts(&[b"Mean"])
        }
        fn fit(&self, x: &Value, _y: Option<&Value>) -> Result<Value> {
            let (data, _) = x
                .as_tensor()
                .ok_or(SomaError::Other("need tensor".into()))?;
            let mean = data.iter().sum::<f64>() / data.len() as f64;
            Ok(Value::json(serde_json::json!({ "mean": mean })))
        }
        fn forward(&self, x: &Value, state: &Value) -> Result<Value> {
            let (data, shape) = x
                .as_tensor()
                .ok_or(SomaError::Other("need tensor".into()))?;
            let mean = state
                .as_json()
                .and_then(|j| j["mean"].as_f64())
                .unwrap_or(0.0);
            Ok(Value::tensor(
                data.iter().map(|v| v - mean).collect(),
                shape.to_vec(),
            ))
        }
        fn meta(&self) -> FilterMeta {
            FilterMeta {
                name: "Mean".into(),
                kind: FilterKind::Trainable,
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

    fn linear_graph(ids: &[&str]) -> Graph {
        let mut g = Graph::new();
        for &id in ids {
            g.nodes.push(Node::new(id, id, id));
        }
        for (i, pair) in ids.windows(2).enumerate() {
            g.edges.push(Edge::data(format!("e{i}"), pair[0], pair[1]));
        }
        g
    }

    // ── GraphSession tests ──

    #[test]
    fn session_run_linear() {
        let graph = linear_graph(&["double", "add"]);
        let mut lib = NodeCatalog::new();
        lib.register("double", Box::new(DoublerFilter));
        lib.register("add", Box::new(AdderFilter(10.0)));

        let mut session = GraphSession::new(graph, lib);

        let cache = MemoryCache::default();
        session = session.with_cache(Arc::new(cache));

        // Manual compile + execute via run
        let CompileResult { plan, .. } = session.compile(CompileMode::NoCache).unwrap();
        let bus = Arc::new(EventBus::new(64));
        let mut ctx =
            Context::new(bus, "test").with_graph_info(GraphInfo::from_graph(session.graph()));
        ctx.set(
            somatize_core::keys::GRAPH_INPUT,
            Value::tensor(vec![1.0, 2.0, 3.0], vec![3]),
        );
        executor::execute(&plan, &mut ctx, session.library(), &MemoryCache::default()).unwrap();

        let outputs: HashMap<String, Value> = ctx.into_outputs();

        let result = outputs.get("add").unwrap();
        let (data, _) = result.as_tensor().unwrap();
        assert_eq!(data, &[12.0, 14.0, 16.0]);
    }

    #[test]
    fn session_fit_and_forward() {
        let graph = linear_graph(&["mean", "double"]);
        let mut lib = NodeCatalog::new();
        lib.register("mean", Box::new(MeanFilter));
        lib.register("double", Box::new(DoublerFilter));

        let mut session = GraphSession::new(graph, lib);

        let x = Value::tensor(vec![10.0, 20.0, 30.0], vec![3]);
        let outputs = session.fit(&x, None).unwrap();

        // mean: fit learns mean=20, forward: [10-20, 20-20, 30-20] = [-10, 0, 10]
        // double: [-10, 0, 10] → [-20, 0, 20]
        let result = outputs.get("double").unwrap();
        let (data, _) = result.as_tensor().unwrap();
        assert_eq!(data, &[-20.0, 0.0, 20.0]);

        assert!(session.is_fitted());
    }

    #[test]
    fn session_compile_diagnostics() {
        let graph = linear_graph(&["double"]);
        let mut lib = NodeCatalog::new();
        lib.register("double", Box::new(DoublerFilter));

        let session = GraphSession::new(graph, lib);
        let result = session.compile(CompileMode::NoCache).unwrap();
        assert!(result.plan.node_count() > 0);
    }

    // ── Free function tests (backward compat) ──

    #[test]
    fn graph_run_linear() {
        let graph = linear_graph(&["double", "add"]);
        let mut lib = NodeCatalog::new();
        lib.register("double", Box::new(DoublerFilter));
        lib.register("add", Box::new(AdderFilter(10.0)));

        let cache = MemoryCache::default();

        let outputs = {
            let CompileResult { plan, .. } =
                compile(&graph, &lib, CompileMode::NoCache, None).unwrap();
            let bus = Arc::new(EventBus::new(64));
            let mut ctx = Context::new(bus, "test").with_graph_info(GraphInfo::from_graph(&graph));
            ctx.set(
                somatize_core::keys::GRAPH_INPUT,
                Value::tensor(vec![1.0, 2.0, 3.0], vec![3]),
            );
            executor::execute(&plan, &mut ctx, &lib, &cache).unwrap();
            ctx.into_outputs()
        };

        let result = outputs.get("add").unwrap();
        let (data, _) = result.as_tensor().unwrap();
        assert_eq!(data, &[12.0, 14.0, 16.0]);
    }

    #[test]
    fn graph_run_diamond() {
        let mut graph = Graph::new();
        graph.nodes.push(Node::new("double", "Double", "double"));
        graph.nodes.push(Node::new("add", "Add", "add"));
        graph.nodes.push(Node::new("merge", "Merge", "merge"));
        graph.edges.push(Edge::data("e1", "double", "merge"));
        graph.edges.push(Edge::data("e2", "add", "merge"));

        let mut lib = NodeCatalog::new();
        lib.register("double", Box::new(DoublerFilter));
        lib.register("add", Box::new(AdderFilter(100.0)));

        struct MergeFilter;
        impl somatize_core::filter::Filter for MergeFilter {
            fn config_hash(&self) -> CacheKey {
                CacheKey::from_parts(&[b"Merge"])
            }
            fn fit(&self, _x: &Value, _y: Option<&Value>) -> Result<Value> {
                Ok(Value::Empty)
            }
            fn forward(&self, x: &Value, _state: &Value) -> Result<Value> {
                Ok(x.clone())
            }
            fn meta(&self) -> FilterMeta {
                FilterMeta {
                    name: "Merge".into(),
                    kind: FilterKind::Stateless,
                    cacheable: true,
                    differentiable: false,
                    deterministic: true,
                    stream_mode: StreamMode::FixedState,
                    distribution: somatize_core::filter::Distribution::Local,
                    input_schema: None,
                    output_schema: None,
                }
            }
        }
        lib.register("merge", Box::new(MergeFilter));

        let cache = MemoryCache::default();
        let CompileResult { plan, .. } = compile(&graph, &lib, CompileMode::NoCache, None).unwrap();

        let bus = Arc::new(EventBus::new(64));
        let mut ctx = Context::new(bus, "test").with_graph_info(GraphInfo::from_graph(&graph));
        ctx.set(
            somatize_core::keys::GRAPH_INPUT,
            Value::tensor(vec![5.0], vec![1]),
        );
        executor::execute(&plan, &mut ctx, &lib, &cache).unwrap();

        let merge_output = ctx.get("merge").unwrap();
        assert!(
            merge_output.as_json().is_some(),
            "merge should receive JSON from multiple predecessors"
        );
    }

    #[test]
    fn graph_fit_trainable() {
        let graph = linear_graph(&["mean", "double"]);
        let mut lib = NodeCatalog::new();
        lib.register("mean", Box::new(MeanFilter));
        lib.register("double", Box::new(DoublerFilter));

        let cache = Arc::new(MemoryCache::default());
        let x = Value::tensor(vec![10.0, 20.0, 30.0], vec![3]);

        let outputs = graph_fit(&graph, &lib, &x, None, cache.clone()).unwrap();

        let result = outputs.get("double").unwrap();
        let (data, _) = result.as_tensor().unwrap();
        assert_eq!(data, &[-20.0, 0.0, 20.0]);

        assert!(!cache.is_empty());
    }

    #[test]
    fn the_catalog_is_the_compiler_registry() {
        let mut lib = NodeCatalog::new();
        lib.register("a", Box::new(DoublerFilter));

        let registry: &dyn NodeRegistry = &lib;
        assert!(registry.meta("a").is_some());
        assert_eq!(registry.meta("a").unwrap().name, "Doubler");
        assert!(registry.config_hash("a").is_some());
        assert!(registry.meta("b").is_none());
    }

    fn session_of(graph: Graph) -> GraphSession {
        let mut lib = NodeCatalog::new();
        for node in &graph.nodes {
            lib.register(&node.id, Box::new(DoublerFilter));
        }
        GraphSession::new(graph, lib)
    }

    /// The persisted-state address follows the wiring, not just the names.
    /// It used to be `node_ids.join(",")`, so these two graphs shared one
    /// address and each would load back the other's trained states.
    #[test]
    fn state_address_separates_graphs_that_share_node_ids() {
        let chain = session_of(linear_graph(&["a", "b", "c"]));

        // Same three nodes, different wiring: a fan-out from `a`.
        let mut fan = Graph::new();
        for id in ["a", "b", "c"] {
            fan.nodes.push(Node::new(id, id, id));
        }
        fan.edges.push(Edge::data("e0", "a", "b"));
        fan.edges.push(Edge::data("e1", "a", "c"));
        let fan = session_of(fan);

        assert_ne!(
            chain.graph_config_hash().unwrap(),
            fan.graph_config_hash().unwrap(),
            "two differently wired graphs must not persist states to one address"
        );
    }

    /// The same graph built twice is the same address — otherwise nothing
    /// persisted could ever be loaded back.
    #[test]
    fn state_address_is_stable_for_the_same_graph() {
        assert_eq!(
            session_of(linear_graph(&["a", "b"]))
                .graph_config_hash()
                .unwrap(),
            session_of(linear_graph(&["a", "b"]))
                .graph_config_hash()
                .unwrap()
        );
    }
}
