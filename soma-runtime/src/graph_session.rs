//! Graph session — high-level orchestration of Graph → Compile → Execute.
//!
//! This is the primary execution path for Soma. Users build a [`Graph`],
//! register filters in a [`FilterLibrary`], then call [`graph_run`],
//! [`graph_fit`], or [`graph_predict`].

use crate::event_bus::EventBus;
use crate::executor::{self, Context, GraphInfo};
use crate::filter_library::FilterLibrary;
use soma_compiler::{CompileMode, CompileResult, compile};
use soma_core::cache::CacheStore;
use soma_core::error::{Result, SomaError};
use soma_core::filter::FilterKind;
use soma_core::graph::Graph;
use soma_core::value::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// Compile and execute a graph, returning all node outputs.
///
/// This is the fundamental operation: Graph → compile → ExecutionPlan → execute.
///
/// ```ignore
/// let mut lib = FilterLibrary::new();
/// lib.register("scaler", Box::new(MyScaler::new()));
/// lib.register("model", Box::new(MyModel::new()));
///
/// let outputs = graph_run(&graph, &lib, CompileMode::Inference, &cache)?;
/// let prediction = outputs.get("model").unwrap();
/// ```
pub fn graph_run(
    graph: &Graph,
    library: &FilterLibrary,
    mode: CompileMode,
    cache: &dyn CacheStore,
) -> Result<HashMap<String, Value>> {
    let CompileResult { plan, diagnostics } = compile(graph, library, mode, Some(cache))?;

    // Log diagnostics as warnings
    for diag in &diagnostics {
        tracing::warn!("compile diagnostic: {:?}", diag);
    }

    let bus = Arc::new(EventBus::new(256));
    let graph_info = GraphInfo::from_graph(graph);
    let filter_store = library.to_filter_store();

    let mut ctx = Context::new(bus, format!("graph_run_{}", timestamp_id()))
        .with_graph_info(graph_info);

    executor::execute(&plan, &mut ctx, &filter_store, cache)?;

    // Extract materialized values from VirtualValue store
    Ok(ctx
        .store
        .into_iter()
        .filter_map(|(k, vv)| vv.as_value().cloned().map(|v| (k, v)))
        .collect())
}

/// Fit all trainable filters in topological order, then return states + outputs.
///
/// For each node in topo order:
/// 1. Resolve input from predecessors
/// 2. If trainable: call `fit(input, y)`, store state, then `forward(input, state)`
/// 3. If stateless: call `forward(input, Empty)`
/// 4. Store output for downstream nodes
///
/// Returns a map of node_id → output value. States are stored in the cache.
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
    let run_id = format!("graph_fit_{}", timestamp_id());

    let mut outputs: HashMap<String, Value> = HashMap::new();

    // Set initial input for root nodes
    let roots = graph.roots();
    for root_id in &roots {
        outputs.insert(format!("__input_{root_id}"), x.clone());
    }

    for node_id in &sorted {
        let filter = library.get(node_id).ok_or_else(|| {
            SomaError::NodeNotFound(node_id.to_string())
        })?;

        bus.emit(soma_core::event::Event::NodeStarted {
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
                        let json_val = serde_json::to_value(val).unwrap_or(serde_json::Value::Null);
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
            // Check state cache
            let data_hash = soma_core::cache::CacheKey::hash_data(
                &serde_json::to_vec(&input).unwrap_or_default(),
            );
            let state_key = soma_core::cache::CacheKey::for_state(&filter.config_hash(), &data_hash);

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

        // Store state in cache for later use by graph_predict
        let _ = state; // state already cached above

        bus.emit(soma_core::event::Event::NodeCompleted {
            run_id: run_id.clone(),
            node_id: node_id.to_string(),
            duration: start.elapsed(),
            output_summary: format!("{output}"),
        });

        outputs.insert(node_id.to_string(), output);
    }

    Ok(outputs)
}

/// Compile in Inference mode and execute, returning the last node's output.
///
/// Assumes filters have already been fitted (states in cache).
/// For each trainable filter, the executor loads the cached state.
pub fn graph_predict(
    graph: &Graph,
    library: &FilterLibrary,
    x: &Value,
    cache: &dyn CacheStore,
) -> Result<Value> {
    let filter_store = library.to_filter_store();

    // Note: states for trainable filters should already be in the cache
    // from a prior graph_fit() call. The executor loads them via Cached plan nodes.
    // TODO: store a state→key mapping during fit for precise lookup in predict.

    let CompileResult { plan, .. } = compile(graph, library, CompileMode::Inference, Some(cache))?;

    let bus = Arc::new(EventBus::new(256));
    let graph_info = GraphInfo::from_graph(graph);
    let mut ctx = Context::new(bus, format!("graph_predict_{}", timestamp_id()))
        .with_graph_info(graph_info);

    // Set input for root nodes
    let roots = graph.roots();
    if roots.len() == 1 {
        ctx.set(format!("__input_{}", roots[0]), x.clone());
    }
    // Also set as fallback for nodes with no predecessors
    ctx.set("__input__", x.clone());

    executor::execute(&plan, &mut ctx, &filter_store, cache)?;

    // Return the last leaf's output (materialized)
    let leaves = graph.leaves();
    let mut extract = |id: &str| -> Option<Value> {
        ctx.store.remove(id).and_then(|vv| vv.as_value().cloned())
    };

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

fn timestamp_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{nanos:x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::MemoryCache;
    use soma_compiler::FilterRegistry;
    use soma_core::cache::CacheKey;
    use soma_core::error::Result;
    use soma_core::filter::{FilterKind, FilterMeta, StreamMode};
    use soma_core::graph::{Edge, Node};

    // ── Test filters ──

    struct DoublerFilter;
    impl soma_core::filter::Filter for DoublerFilter {
        fn config_hash(&self) -> CacheKey { CacheKey::from_parts(&[b"Doubler"]) }
        fn fit(&self, _x: &Value, _y: Option<&Value>) -> Result<Value> { Ok(Value::Empty) }
        fn forward(&self, x: &Value, _state: &Value) -> Result<Value> {
            let (data, shape) = x.as_tensor().ok_or(SomaError::Other("need tensor".into()))?;
            Ok(Value::tensor(data.iter().map(|v| v * 2.0).collect(), shape.to_vec()))
        }
        fn meta(&self) -> FilterMeta {
            FilterMeta {
                name: "Doubler".into(), kind: FilterKind::Stateless, cacheable: true,
                differentiable: true, stream_mode: StreamMode::FixedState,
                distribution: soma_core::filter::Distribution::Local,
                input_schema: None, output_schema: None,
            }
        }
    }

    struct AdderFilter(f64);
    impl soma_core::filter::Filter for AdderFilter {
        fn config_hash(&self) -> CacheKey { CacheKey::from_parts(&[b"Adder", &self.0.to_le_bytes()]) }
        fn fit(&self, _x: &Value, _y: Option<&Value>) -> Result<Value> { Ok(Value::Empty) }
        fn forward(&self, x: &Value, _state: &Value) -> Result<Value> {
            let (data, shape) = x.as_tensor().ok_or(SomaError::Other("need tensor".into()))?;
            Ok(Value::tensor(data.iter().map(|v| v + self.0).collect(), shape.to_vec()))
        }
        fn meta(&self) -> FilterMeta {
            FilterMeta {
                name: "Adder".into(), kind: FilterKind::Stateless, cacheable: true,
                differentiable: true, stream_mode: StreamMode::FixedState,
                distribution: soma_core::filter::Distribution::Local,
                input_schema: None, output_schema: None,
            }
        }
    }

    struct MeanFilter;
    impl soma_core::filter::Filter for MeanFilter {
        fn config_hash(&self) -> CacheKey { CacheKey::from_parts(&[b"Mean"]) }
        fn fit(&self, x: &Value, _y: Option<&Value>) -> Result<Value> {
            let (data, _) = x.as_tensor().ok_or(SomaError::Other("need tensor".into()))?;
            let mean = data.iter().sum::<f64>() / data.len() as f64;
            Ok(Value::json(serde_json::json!({ "mean": mean })))
        }
        fn forward(&self, x: &Value, state: &Value) -> Result<Value> {
            let (data, shape) = x.as_tensor().ok_or(SomaError::Other("need tensor".into()))?;
            let mean = state.as_json().and_then(|j| j["mean"].as_f64()).unwrap_or(0.0);
            Ok(Value::tensor(data.iter().map(|v| v - mean).collect(), shape.to_vec()))
        }
        fn meta(&self) -> FilterMeta {
            FilterMeta {
                name: "Mean".into(), kind: FilterKind::Trainable, cacheable: true,
                differentiable: true, stream_mode: StreamMode::FixedState,
                distribution: soma_core::filter::Distribution::Local,
                input_schema: None, output_schema: None,
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

    // ── Tests ──

    #[test]
    fn graph_run_linear() {
        let graph = linear_graph(&["double", "add"]);
        let mut lib = FilterLibrary::new();
        lib.register("double", Box::new(DoublerFilter));
        lib.register("add", Box::new(AdderFilter(10.0)));

        let cache = MemoryCache::default();

        // Set input for first node
        let outputs = {
            let CompileResult { plan, .. } = compile(&graph, &lib, CompileMode::NoCache, None).unwrap();
            let bus = Arc::new(EventBus::new(64));
            let mut ctx = Context::new(bus, "test")
                .with_graph_info(GraphInfo::from_graph(&graph));
            ctx.set("__input__", Value::tensor(vec![1.0, 2.0, 3.0], vec![3]));
            executor::execute(&plan, &mut ctx, &lib.to_filter_store(), &cache).unwrap();
            // Extract materialized values
            ctx.store.into_iter()
                .filter_map(|(k, vv)| vv.as_value().cloned().map(|v| (k, v)))
                .collect::<HashMap<String, Value>>()
        };

        // double: [1,2,3] → [2,4,6], add: [2,4,6] → [12,14,16]
        let result = outputs.get("add").unwrap();
        let (data, _) = result.as_tensor().unwrap();
        assert_eq!(data, &[12.0, 14.0, 16.0]);
    }

    #[test]
    fn graph_run_diamond() {
        // input → double → merge
        // input → add    → merge
        let mut graph = Graph::new();
        graph.nodes.push(Node::new("double", "Double", "double"));
        graph.nodes.push(Node::new("add", "Add", "add"));
        graph.nodes.push(Node::new("merge", "Merge", "merge"));
        graph.edges.push(Edge::data("e1", "double", "merge"));
        graph.edges.push(Edge::data("e2", "add", "merge"));

        let mut lib = FilterLibrary::new();
        lib.register("double", Box::new(DoublerFilter));
        lib.register("add", Box::new(AdderFilter(100.0)));

        // Merge filter: receives JSON { "double": [...], "add": [...] }
        struct MergeFilter;
        impl soma_core::filter::Filter for MergeFilter {
            fn config_hash(&self) -> CacheKey { CacheKey::from_parts(&[b"Merge"]) }
            fn fit(&self, _x: &Value, _y: Option<&Value>) -> Result<Value> { Ok(Value::Empty) }
            fn forward(&self, x: &Value, _state: &Value) -> Result<Value> { Ok(x.clone()) }
            fn meta(&self) -> FilterMeta {
                FilterMeta {
                    name: "Merge".into(), kind: FilterKind::Stateless, cacheable: true,
                    differentiable: false, stream_mode: StreamMode::FixedState,
                    distribution: soma_core::filter::Distribution::Local,
                    input_schema: None, output_schema: None,
                }
            }
        }
        lib.register("merge", Box::new(MergeFilter));

        let cache = MemoryCache::default();
        let CompileResult { plan, .. } = compile(&graph, &lib, CompileMode::NoCache, None).unwrap();

        let bus = Arc::new(EventBus::new(64));
        let mut ctx = Context::new(bus, "test")
            .with_graph_info(GraphInfo::from_graph(&graph));
        // Both double and add have no predecessors — they read from last executed or input
        ctx.set("__input__", Value::tensor(vec![5.0], vec![1]));
        executor::execute(&plan, &mut ctx, &lib.to_filter_store(), &cache).unwrap();

        // merge should exist with JSON input from both branches
        let merge_output = ctx.get("merge").unwrap();
        assert!(merge_output.as_json().is_some(), "merge should receive JSON from multiple predecessors");
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

        // mean: fit learns mean=20, forward: [10-20, 20-20, 30-20] = [-10, 0, 10]
        // double: [-10, 0, 10] → [-20, 0, 20]
        let result = outputs.get("double").unwrap();
        let (data, _) = result.as_tensor().unwrap();
        assert_eq!(data, &[-20.0, 0.0, 20.0]);

        // State should be cached
        assert!(!cache.is_empty());
    }

    #[test]
    fn filter_library_registry_compat() {
        let mut lib = FilterLibrary::new();
        lib.register("a", Box::new(DoublerFilter));

        // Use as FilterRegistry
        let registry: &dyn FilterRegistry = &lib;
        assert!(registry.meta("a").is_some());
        assert_eq!(registry.meta("a").unwrap().name, "Doubler");
        assert!(registry.config_hash("a").is_some());
        assert!(registry.meta("b").is_none());
    }
}
