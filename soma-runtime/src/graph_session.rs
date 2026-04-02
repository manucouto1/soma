//! Graph session — the primary orchestrator for Graph → Compile → Execute.
//!
//! [`GraphSession`] binds a [`Graph`] with its [`FilterLibrary`], cache,
//! event bus, and optional distributed components into a single object
//! that can compile, fit, and execute.

use crate::cache::MemoryCache;
use crate::event_bus::EventBus;
use crate::executor::{self, Context, GraphInfo, RemoteExecutor};
use crate::filter_library::FilterLibrary;
use somatize_compiler::{CompileMode, CompileResult, compile};
use somatize_core::cache::{CacheKey, CacheStore};
use somatize_core::error::{Result, SomaError};
use somatize_core::event::Event;
use somatize_core::filter::FilterKind;
use somatize_core::graph::Graph;
use somatize_core::store::{DataRef, DataStore};
use somatize_core::util::timestamp_id;
use somatize_core::value::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// The primary orchestrator: Graph + filters + cache + events.
///
/// ```ignore
/// let mut lib = FilterLibrary::new();
/// lib.register("scaler", Box::new(MyScaler::new()));
/// lib.register("model", Box::new(MyModel::new()));
///
/// let mut session = GraphSession::new(graph, lib);
/// session.fit(&train_x, Some(&train_y))?;
/// let output = session.forward(&test_x)?;
/// ```
pub struct GraphSession {
    graph: Graph,
    library: FilterLibrary,
    cache: Arc<dyn CacheStore>,
    event_bus: Arc<EventBus>,
    data_store: Option<Arc<dyn DataStore>>,
    remote_executor: Option<Arc<dyn RemoteExecutor>>,
    fitted: bool,
}

impl GraphSession {
    pub fn new(graph: Graph, library: FilterLibrary) -> Self {
        Self {
            graph,
            library,
            cache: Arc::new(MemoryCache::default()),
            event_bus: Arc::new(EventBus::new(256)),
            data_store: None,
            remote_executor: None,
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

    pub fn with_remote_executor(mut self, executor: Arc<dyn RemoteExecutor>) -> Self {
        self.remote_executor = Some(executor);
        self
    }

    // ── Core operations ──

    /// Compile the graph and return diagnostics without executing.
    pub fn compile(&self, mode: CompileMode) -> Result<CompileResult> {
        compile(&self.graph, &self.library, mode, Some(self.cache.as_ref()))
    }

    /// Compile and execute the graph, returning all node outputs.
    pub fn run(&mut self, mode: CompileMode) -> Result<HashMap<String, Value>> {
        let CompileResult { plan, diagnostics } =
            compile(&self.graph, &self.library, mode, Some(self.cache.as_ref()))?;

        for diag in &diagnostics {
            tracing::warn!("compile diagnostic: {:?}", diag);
        }

        let graph_info = GraphInfo::from_graph(&self.graph);
        let mut ctx = Context::new(self.event_bus.clone(), timestamp_id("graph_run"))
            .with_graph_info(graph_info);

        if let Some(store) = &self.data_store {
            ctx = ctx.with_data_store(store.clone());
        }
        if let Some(remote) = &self.remote_executor {
            ctx = ctx.with_remote_executor(remote.clone());
        }

        executor::execute(&plan, &mut ctx, &self.library, self.cache.as_ref())?;

        Ok(ctx
            .store
            .into_iter()
            .filter_map(|(k, vv)| vv.as_value().cloned().map(|v| (k, v)))
            .collect())
    }

    /// Fit all trainable filters in topological order.
    ///
    /// For each node in topo order:
    /// 1. Resolve input from predecessors
    /// 2. If trainable: call `fit(input, y)`, cache state, then `forward(input, state)`
    /// 3. If stateless: call `forward(input, Empty)`
    /// 4. Store output for downstream nodes
    pub fn fit(&mut self, x: &Value, y: Option<&Value>) -> Result<HashMap<String, Value>> {
        self.graph.validate()?;
        let sorted = self.graph.topological_sort()?;
        let graph_info = GraphInfo::from_graph(&self.graph);

        let run_id = timestamp_id("graph_fit");
        let mut outputs: HashMap<String, Value> = HashMap::new();

        // Set initial input for root nodes
        let roots = self.graph.roots();
        for root_id in &roots {
            outputs.insert(format!("__input_{root_id}"), x.clone());
        }

        for node_id in &sorted {
            let filter = self
                .library
                .get(node_id)
                .ok_or_else(|| SomaError::NodeNotFound(node_id.to_string()))?;

            self.event_bus.emit(Event::NodeStarted {
                run_id: run_id.clone(),
                node_id: node_id.to_string(),
                kind: filter.meta().kind,
            });

            // Resolve input from predecessors
            let preds = graph_info.predecessors(node_id);
            let input = match preds.len() {
                0 => x.clone(),
                1 => outputs.get(&preds[0]).cloned().unwrap_or_else(|| x.clone()),
                _ => {
                    let mut merged = serde_json::Map::new();
                    for pred_id in preds {
                        if let Some(val) = outputs.get(pred_id.as_str()) {
                            let json_val =
                                serde_json::to_value(val).unwrap_or(serde_json::Value::Null);
                            merged.insert(pred_id.clone(), json_val);
                        }
                    }
                    Value::Json(serde_json::Value::Object(merged))
                }
            };

            let meta = filter.meta();
            let start = std::time::Instant::now();

            // Fit trainable filters, forward all
            let (state, output) = if meta.kind == FilterKind::Trainable {
                let data_hash =
                    CacheKey::hash_data(&serde_json::to_vec(&input).unwrap_or_default());
                let state_key = CacheKey::for_state(&filter.config_hash(), &data_hash);

                let state = if let Some(cached) = self.cache.get(&state_key)? {
                    cached
                } else {
                    let s = filter.fit(&input, y)?;
                    self.cache.put(&state_key, &s)?;
                    s
                };

                let output = filter.forward(&input, &state)?;
                // Store state in library for later use by forward()
                self.library.set_state(node_id.to_string(), state.clone());
                (state, output)
            } else {
                let output = filter.forward(&input, &Value::Empty)?;
                (Value::Empty, output)
            };

            let _ = state; // state already cached above

            self.event_bus.emit(Event::NodeCompleted {
                run_id: run_id.clone(),
                node_id: node_id.to_string(),
                duration: start.elapsed(),
                output_summary: format!("{output}"),
            });

            outputs.insert(node_id.to_string(), output);
        }

        self.fitted = true;
        Ok(outputs)
    }

    /// Forward pass through the fitted graph (inference).
    ///
    /// Compiles in Inference mode and executes. Trainable filters
    /// use states cached during `fit()`.
    pub fn forward(&self, x: &Value) -> Result<Value> {
        let CompileResult { plan, .. } = compile(
            &self.graph,
            &self.library,
            CompileMode::Inference,
            Some(self.cache.as_ref()),
        )?;

        let graph_info = GraphInfo::from_graph(&self.graph);
        let mut ctx = Context::new(self.event_bus.clone(), timestamp_id("graph_forward"))
            .with_graph_info(graph_info);

        if let Some(store) = &self.data_store {
            ctx = ctx.with_data_store(store.clone());
        }
        if let Some(remote) = &self.remote_executor {
            ctx = ctx.with_remote_executor(remote.clone());
        }

        // Set input for root nodes
        let roots = self.graph.roots();
        if roots.len() == 1 {
            ctx.set(format!("__input_{}", roots[0]), x.clone());
        }
        ctx.set("__input__", x.clone());

        executor::execute(&plan, &mut ctx, &self.library, self.cache.as_ref())?;

        // Return the last leaf's output
        let leaves = self.graph.leaves();
        let mut extract = |id: &str| -> Option<Value> {
            ctx.store.remove(id).and_then(|vv| vv.as_value().cloned())
        };

        if let Some(leaf_id) = leaves.first() {
            extract(leaf_id).ok_or_else(|| {
                SomaError::Other(format!("leaf node '{leaf_id}' produced no output"))
            })
        } else {
            ctx.execution_order
                .last()
                .and_then(|id| extract(id))
                .ok_or_else(|| SomaError::Other("no output produced".into()))
        }
    }

    /// Forward pass in batches from a DataStore reference.
    pub fn forward_batched(&self, data_ref: &DataRef, batch_size: usize) -> Result<Value> {
        let store = self
            .data_store
            .as_ref()
            .ok_or_else(|| SomaError::Execution {
                node_id: "session".into(),
                message: "forward_batched requires a data store (use with_data_store)".into(),
            })?;

        let meta = store.meta(data_ref)?;
        let total_rows = meta.total_rows;
        if total_rows == 0 {
            return Ok(Value::Empty);
        }

        let mut all_values: Vec<f64> = Vec::new();
        let mut result_shape: Option<Vec<usize>> = None;
        let mut rows_processed = 0;

        while rows_processed < total_rows {
            let batch_len = batch_size.min(total_rows - rows_processed);
            let batch = store.get_rows(data_ref, rows_processed, batch_len)?;
            let output = self.forward(&batch)?;

            if let Value::Tensor { values, shape } = &output {
                if result_shape.is_none() {
                    result_shape = Some(shape.clone());
                }
                all_values.extend_from_slice(values);
            } else {
                return Ok(output);
            }

            rows_processed += batch_len;
        }

        match result_shape {
            Some(mut shape) => {
                shape[0] = total_rows;
                Ok(Value::tensor(all_values, shape))
            }
            None => Ok(Value::Empty),
        }
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
                let json = serde_json::to_value(state)
                    .map_err(|e| SomaError::Other(format!("state serialize: {e}")))?;
                states_map.insert(node_id.to_string(), json);
            }
        }

        let states_value = Value::Json(serde_json::Value::Object(states_map));
        let key = CacheKey::from_parts(&[b"graph_states", self.graph_config_hash().as_bytes()]);
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
            self.library.set_state(node_id.clone(), value);
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
    pub fn library(&self) -> &FilterLibrary {
        &self.library
    }

    /// Mutable access to the filter library (for registering filters after creation).
    pub fn library_mut(&mut self) -> &mut FilterLibrary {
        &mut self.library
    }

    // ── Private helpers ──

    fn graph_config_hash(&self) -> String {
        let node_ids: Vec<&str> = self.graph.nodes.iter().map(|n| n.id.as_str()).collect();
        node_ids.join(",")
    }
}

// ── Convenience free functions (backward compat) ──

/// Compile and execute a graph, returning all node outputs.
pub fn graph_run(
    graph: &Graph,
    library: &FilterLibrary,
    mode: CompileMode,
    cache: &dyn CacheStore,
) -> Result<HashMap<String, Value>> {
    let CompileResult { plan, diagnostics } = compile(graph, library, mode, Some(cache))?;

    for diag in &diagnostics {
        tracing::warn!("compile diagnostic: {:?}", diag);
    }

    let bus = Arc::new(EventBus::new(256));
    let graph_info = GraphInfo::from_graph(graph);

    let mut ctx = Context::new(bus, timestamp_id("graph_run")).with_graph_info(graph_info);

    executor::execute(&plan, &mut ctx, library, cache)?;

    Ok(ctx
        .store
        .into_iter()
        .filter_map(|(k, vv)| vv.as_value().cloned().map(|v| (k, v)))
        .collect())
}

/// Fit all trainable filters in topological order.
pub fn graph_fit(
    graph: &Graph,
    library: &FilterLibrary,
    x: &Value,
    y: Option<&Value>,
    cache: &dyn CacheStore,
) -> Result<HashMap<String, Value>> {
    graph.validate()?;
    let sorted = graph.topological_sort()?;
    let graph_info = GraphInfo::from_graph(graph);

    let bus = Arc::new(EventBus::new(256));
    let run_id = timestamp_id("graph_fit");

    let mut outputs: HashMap<String, Value> = HashMap::new();

    let roots = graph.roots();
    for root_id in &roots {
        outputs.insert(format!("__input_{root_id}"), x.clone());
    }

    for node_id in &sorted {
        let filter = library
            .get(node_id)
            .ok_or_else(|| SomaError::NodeNotFound(node_id.to_string()))?;

        bus.emit(Event::NodeStarted {
            run_id: run_id.clone(),
            node_id: node_id.to_string(),
            kind: filter.meta().kind,
        });

        let preds = graph_info.predecessors(node_id);
        let input = match preds.len() {
            0 => x.clone(),
            1 => outputs.get(&preds[0]).cloned().unwrap_or_else(|| x.clone()),
            _ => {
                let mut merged = serde_json::Map::new();
                for pred_id in preds {
                    if let Some(val) = outputs.get(pred_id.as_str()) {
                        let json_val = serde_json::to_value(val).unwrap_or(serde_json::Value::Null);
                        merged.insert(pred_id.clone(), json_val);
                    }
                }
                Value::Json(serde_json::Value::Object(merged))
            }
        };

        let meta = filter.meta();
        let start = std::time::Instant::now();

        let (state, output) = if meta.kind == FilterKind::Trainable {
            let data_hash = CacheKey::hash_data(&serde_json::to_vec(&input).unwrap_or_default());
            let state_key = CacheKey::for_state(&filter.config_hash(), &data_hash);

            let state = if let Some(cached) = cache.get(&state_key)? {
                cached
            } else {
                let s = filter.fit(&input, y)?;
                cache.put(&state_key, &s)?;
                s
            };

            let output = filter.forward(&input, &state)?;
            (state, output)
        } else {
            let output = filter.forward(&input, &Value::Empty)?;
            (Value::Empty, output)
        };

        let _ = state;

        bus.emit(Event::NodeCompleted {
            run_id: run_id.clone(),
            node_id: node_id.to_string(),
            duration: start.elapsed(),
            output_summary: format!("{output}"),
        });

        outputs.insert(node_id.to_string(), output);
    }

    Ok(outputs)
}

/// Compile in Inference mode and execute, returning the last leaf's output.
pub fn graph_predict(
    graph: &Graph,
    library: &FilterLibrary,
    x: &Value,
    cache: &dyn CacheStore,
) -> Result<Value> {
    let CompileResult { plan, .. } = compile(graph, library, CompileMode::Inference, Some(cache))?;

    let bus = Arc::new(EventBus::new(256));
    let graph_info = GraphInfo::from_graph(graph);
    let mut ctx = Context::new(bus, timestamp_id("graph_predict")).with_graph_info(graph_info);

    let roots = graph.roots();
    if roots.len() == 1 {
        ctx.set(format!("__input_{}", roots[0]), x.clone());
    }
    ctx.set("__input__", x.clone());

    executor::execute(&plan, &mut ctx, library, cache)?;

    let leaves = graph.leaves();
    let mut extract =
        |id: &str| -> Option<Value> { ctx.store.remove(id).and_then(|vv| vv.as_value().cloned()) };

    if let Some(leaf_id) = leaves.first() {
        extract(leaf_id)
            .ok_or_else(|| SomaError::Other(format!("leaf node '{leaf_id}' produced no output")))
    } else {
        ctx.execution_order
            .last()
            .and_then(|id| extract(id))
            .ok_or_else(|| SomaError::Other("no output produced".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::MemoryCache;
    use somatize_compiler::FilterRegistry;
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
        let mut lib = FilterLibrary::new();
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
        ctx.set("__input__", Value::tensor(vec![1.0, 2.0, 3.0], vec![3]));
        executor::execute(&plan, &mut ctx, session.library(), &MemoryCache::default()).unwrap();

        let outputs: HashMap<String, Value> = ctx
            .store
            .into_iter()
            .filter_map(|(k, vv)| vv.as_value().cloned().map(|v| (k, v)))
            .collect();

        let result = outputs.get("add").unwrap();
        let (data, _) = result.as_tensor().unwrap();
        assert_eq!(data, &[12.0, 14.0, 16.0]);
    }

    #[test]
    fn session_fit_and_forward() {
        let graph = linear_graph(&["mean", "double"]);
        let mut lib = FilterLibrary::new();
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
        let mut lib = FilterLibrary::new();
        lib.register("double", Box::new(DoublerFilter));

        let session = GraphSession::new(graph, lib);
        let result = session.compile(CompileMode::NoCache).unwrap();
        assert!(result.plan.node_count() > 0);
    }

    // ── Free function tests (backward compat) ──

    #[test]
    fn graph_run_linear() {
        let graph = linear_graph(&["double", "add"]);
        let mut lib = FilterLibrary::new();
        lib.register("double", Box::new(DoublerFilter));
        lib.register("add", Box::new(AdderFilter(10.0)));

        let cache = MemoryCache::default();

        let outputs = {
            let CompileResult { plan, .. } =
                compile(&graph, &lib, CompileMode::NoCache, None).unwrap();
            let bus = Arc::new(EventBus::new(64));
            let mut ctx = Context::new(bus, "test").with_graph_info(GraphInfo::from_graph(&graph));
            ctx.set("__input__", Value::tensor(vec![1.0, 2.0, 3.0], vec![3]));
            executor::execute(&plan, &mut ctx, &lib, &cache).unwrap();
            ctx.store
                .into_iter()
                .filter_map(|(k, vv)| vv.as_value().cloned().map(|v| (k, v)))
                .collect::<HashMap<String, Value>>()
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

        let mut lib = FilterLibrary::new();
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
        ctx.set("__input__", Value::tensor(vec![5.0], vec![1]));
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
        let mut lib = FilterLibrary::new();
        lib.register("mean", Box::new(MeanFilter));
        lib.register("double", Box::new(DoublerFilter));

        let cache = MemoryCache::default();
        let x = Value::tensor(vec![10.0, 20.0, 30.0], vec![3]);

        let outputs = graph_fit(&graph, &lib, &x, None, &cache).unwrap();

        let result = outputs.get("double").unwrap();
        let (data, _) = result.as_tensor().unwrap();
        assert_eq!(data, &[-20.0, 0.0, 20.0]);

        assert!(!cache.is_empty());
    }

    #[test]
    fn filter_library_registry_compat() {
        let mut lib = FilterLibrary::new();
        lib.register("a", Box::new(DoublerFilter));

        let registry: &dyn FilterRegistry = &lib;
        assert!(registry.meta("a").is_some());
        assert_eq!(registry.meta("a").unwrap().name, "Doubler");
        assert!(registry.config_hash("a").is_some());
        assert!(registry.meta("b").is_none());
    }
}
