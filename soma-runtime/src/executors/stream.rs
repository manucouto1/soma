//! The stream driver: chunked execution through `run_node`'s primitives.
//!
//! One execution site is the runtime's core invariant, and streaming is
//! no longer the exception: every chunk of every node goes through the
//! same three primitives the topological walk composes — `output_key`
//! (the memoization guard and the one key derivation), `compute_node`
//! (panic containment around the only filter-vs-step match) and
//! `store_output` (provenance on every write). What lives here is only
//! what is genuinely streaming's: chunk flow per [`StreamMode`], the
//! evolving state carried between chunks, barrier buffers and their
//! flush, and the per-node event bracket.
//!
//! **Events.** One `NodeStarted` when the first chunk reaches a node,
//! one `NodeCompleted` per started node at [`StreamRun::finish`] with an
//! aggregated summary (`stream: N chunks, H hits, M misses`), and a real
//! `NodeFailed` naming the chunk on error — so an upstream span left
//! open means exactly what it means everywhere else: the run died
//! mid-node. Per-chunk cache hit/miss events are deliberately not
//! emitted (hundreds of standalone spans would drown a reader); the
//! counts travel in the summary. A per-chunk `NodeStarted` under made-up
//! ids (`model#chunk_3`) was tried once and reverted.
//!
//! **Evolving.** The forward's output value doubles as the next chunk's
//! state — a documented conflation. Separating them needs a
//! `step(chunk, state) -> (out, state)` API on filters, which is a
//! user-facing change this driver deliberately does not smuggle in.
//!
//! The worker's remote streaming holds a [`StreamRun`] (plus its
//! `Context`) alive between WebSocket messages — which is why the type
//! is public and why the state that must survive between chunks lives
//! here rather than in the plan walk.

use crate::executor::{Context, compute_node, output_key, store_output};
use crate::node_catalog::{NodeCatalog, NodeImpl};
use somatize_core::cache::{CacheKey, CacheStore};
use somatize_core::error::{Result, SomaError};
use somatize_core::event::Event;
use somatize_core::filter::StreamMode;
use somatize_core::node::{NodeMeta, NodeOutcome};
use somatize_core::value::Value;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// One plan node with its chunk-flow state and event/statistics bookkeeping.
struct StreamNode {
    id: String,
    node: NodeImpl,
    meta: NodeMeta,
    /// From the filter's own meta — [`NodeMeta`] does not carry it, and
    /// only a filter can be here (steps are refused at construction).
    stream_mode: StreamMode,
    /// The catalog state, same escalation `run_node` uses (`Value::Empty`
    /// when nothing is fitted). Shadowed by `evolving` once set.
    base_state: Arc<Value>,
    /// Accumulated chunks awaiting the flush (Barrier mode).
    barrier: Vec<Value>,
    /// The last output, doubling as the next state (Evolving mode).
    evolving: Option<Value>,
    started: bool,
    chunks: u64,
    cache_hits: u64,
    cache_misses: u64,
    compute: Duration,
}

/// Drives one stream plan: chunks in, one concatenated output out.
///
/// Built from the [`NodeCatalog`] — a node the catalog does not know is
/// an error, never a silent skip. The chunk loop lives in the caller —
/// the plan executor locally, the worker's WS/DataStore loops remotely —
/// and this type owns the per-node flow, so the state that must survive
/// between chunks (and between RPC messages) has a single home.
pub struct StreamRun {
    nodes: Vec<StreamNode>,
    chunk_count: usize,
}

impl StreamRun {
    pub fn new(node_ids: &[String], catalog: &NodeCatalog) -> Result<Self> {
        let nodes = node_ids
            .iter()
            .map(|id| {
                let node = catalog
                    .node(id)
                    .ok_or_else(|| SomaError::NodeNotFound(id.clone()))?
                    .clone();
                // The compiler refuses steps in a stream plan; this is the
                // driver's own line of defense, not a user-facing path.
                let stream_mode = match &node {
                    NodeImpl::Filter(f) => f.meta().stream_mode,
                    NodeImpl::Step(_) => {
                        return Err(SomaError::Execution {
                            node_id: id.clone(),
                            message: "a step cannot run inside a stream plan".into(),
                        });
                    }
                };
                let meta = node.meta();
                let base_state = catalog
                    .get_state(id)
                    .unwrap_or_else(|| Arc::new(Value::Empty));
                Ok(StreamNode {
                    id: id.clone(),
                    node,
                    meta,
                    stream_mode,
                    base_state,
                    barrier: Vec::new(),
                    evolving: None,
                    started: false,
                    chunks: 0,
                    cache_hits: 0,
                    cache_misses: 0,
                    compute: Duration::ZERO,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            nodes,
            chunk_count: 0,
        })
    }

    /// Push one chunk through the chain. `None` means a barrier swallowed
    /// it — the nodes past the barrier see nothing until [`Self::flush`].
    pub fn process_chunk(
        &mut self,
        chunk: Value,
        ctx: &mut Context,
        cache: &dyn CacheStore,
    ) -> Result<Option<Value>> {
        let stage = format!("chunk {}", self.chunk_count);
        self.chunk_count += 1;
        let mut current = chunk;
        for i in 0..self.nodes.len() {
            if matches!(self.nodes[i].stream_mode, StreamMode::Barrier) {
                self.nodes[i].barrier.push(current);
                return Ok(None);
            }
            current = self.run_compute(i, current, &stage, ctx, cache)?;
        }
        Ok(Some(current))
    }

    /// Materialize every barrier buffer and cascade the result through the
    /// rest of the chain — a second barrier downstream receives the whole
    /// materialized value as one "chunk", which at flush time it is.
    pub fn flush(&mut self, ctx: &mut Context, cache: &dyn CacheStore) -> Result<Option<Value>> {
        let mut current: Option<Value> = None;
        for i in 0..self.nodes.len() {
            if !self.nodes[i].barrier.is_empty() {
                let buffer = std::mem::take(&mut self.nodes[i].barrier);
                let materialized = materialize_buffer(&buffer)?;
                current = Some(self.run_compute(i, materialized, "flush", ctx, cache)?);
            } else if let Some(v) = current.take() {
                current = Some(self.run_compute(i, v, "flush", ctx, cache)?);
            }
        }
        Ok(current)
    }

    /// Close each started node's event bracket with its aggregate.
    pub fn finish(&mut self, ctx: &Context) {
        for node in &mut self.nodes {
            if !node.started {
                continue;
            }
            node.started = false;
            ctx.event_bus.emit(Event::NodeCompleted {
                run_id: ctx.run_id.clone(),
                node_id: node.id.clone(),
                duration: node.compute,
                output_summary: format!(
                    "stream: {} chunks, {} hits, {} misses",
                    node.chunks, node.cache_hits, node.cache_misses
                ),
            });
        }
    }

    pub fn chunks_processed(&self) -> usize {
        self.chunk_count
    }

    /// One node, one value, through the shared primitives. Mode-agnostic:
    /// the caller decides whether the value is a live chunk or a
    /// materialized barrier buffer.
    fn run_compute(
        &mut self,
        i: usize,
        input: Value,
        stage: &str,
        ctx: &Context,
        cache: &dyn CacheStore,
    ) -> Result<Value> {
        let node = &mut self.nodes[i];
        if !node.started {
            node.started = true;
            ctx.event_bus.emit(Event::NodeStarted {
                run_id: ctx.run_id.clone(),
                node_id: node.id.clone(),
                kind: node.meta.kind,
                effectful: node.meta.effectful,
            });
        }

        let state_ref: &Value = match &node.evolving {
            Some(v) => v,
            None => node.base_state.as_ref(),
        };
        let input_key = CacheKey::for_value(&input);
        let key = output_key(&node.node, &node.meta, state_ref, &input_key, ctx.seed);

        if let Some(k) = &key {
            if let Ok(Some((cached, _tier))) = cache.get_located(k) {
                node.chunks += 1;
                node.cache_hits += 1;
                if matches!(node.stream_mode, StreamMode::Evolving) {
                    node.evolving = Some(cached.clone());
                }
                return Ok(cached);
            }
            node.cache_misses += 1;
        }

        let started_at = Instant::now();
        match compute_node(&node.node, &node.id, ctx, &input, state_ref) {
            Ok(NodeOutcome::Produced(out)) => {
                let duration = started_at.elapsed();
                node.compute += duration;
                node.chunks += 1;
                if let Some(k) = &key {
                    store_output(
                        cache,
                        k,
                        &out,
                        &node.id,
                        &ctx.run_id,
                        duration,
                        node.meta.deterministic,
                    );
                }
                if matches!(node.stream_mode, StreamMode::Evolving) {
                    node.evolving = Some(out.clone());
                }
                Ok(out)
            }
            // The compiler refuses steps in a stream plan, so reaching
            // this is a soma bug — but a silent pass-through would be a
            // worse answer than a clear refusal.
            Ok(NodeOutcome::HandOff { .. } | NodeOutcome::Paused { .. }) => {
                Err(SomaError::Execution {
                    node_id: node.id.clone(),
                    message: "a step cannot run inside a stream plan".into(),
                })
            }
            Err(e) => {
                ctx.event_bus.emit(Event::NodeFailed {
                    run_id: ctx.run_id.clone(),
                    node_id: node.id.clone(),
                    error: format!("{stage}: {e}"),
                });
                Err(e)
            }
        }
    }
}

/// Incremental concatenation of chunk outputs.
///
/// Each chunk's data is folded in and the chunk dropped, keeping peak
/// memory proportional to the final output rather than
/// `O(n_chunks x chunk_size)`. Non-tensor outputs do not concatenate;
/// the last one wins (a chain ending in an aggregate produces exactly
/// one).
#[derive(Default)]
pub struct StreamOutput {
    all_data: Vec<f64>,
    result_shape: Option<Vec<usize>>,
    non_tensor: Option<Value>,
}

impl StreamOutput {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fold one chunk's output in.
    pub fn push(&mut self, output: Value) {
        match output {
            Value::Tensor { values, shape } => {
                if self.result_shape.is_none() {
                    self.result_shape = Some(shape);
                }
                self.all_data.extend_from_slice(values.as_slice());
            }
            other => self.non_tensor = Some(other),
        }
    }

    /// The concatenated result, its first dimension corrected to the
    /// total number of rows. `Value::Empty` if nothing was pushed.
    pub fn finish(self) -> Value {
        if let Some(mut shape) = self.result_shape {
            let row_size: usize = shape.iter().skip(1).product::<usize>().max(1);
            shape[0] = self.all_data.len() / row_size;
            return Value::tensor(self.all_data, shape);
        }
        self.non_tensor.unwrap_or(Value::Empty)
    }
}

/// Concatenate tensor chunks along first dimension.
pub fn materialize_buffer(buffer: &[Value]) -> Result<Value> {
    if buffer.is_empty() {
        return Ok(Value::Empty);
    }
    let mut all_data = Vec::new();
    let mut total_rows = 0;
    let mut cols = 0;

    for chunk in buffer {
        match chunk {
            Value::Tensor { values, shape } => {
                all_data.extend(values.iter());
                if shape.len() == 1 {
                    total_rows += shape[0];
                    cols = 1;
                } else if shape.len() >= 2 {
                    total_rows += shape[0];
                    cols = shape[1];
                }
            }
            _ => {
                return Err(SomaError::Other(
                    "barrier buffer contains non-tensor values".into(),
                ));
            }
        }
    }

    if cols <= 1 {
        Ok(Value::tensor(all_data, vec![total_rows]))
    } else {
        Ok(Value::tensor(all_data, vec![total_rows, cols]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::memory::MemoryCache;
    use crate::event_bus::EventBus;
    use somatize_core::error::Result as SomaResult;
    use somatize_core::filter::{Distribution, Filter, FilterKind, FilterMeta};

    fn meta(name: &str, stream_mode: StreamMode, cacheable: bool) -> FilterMeta {
        FilterMeta {
            name: name.into(),
            kind: FilterKind::Stateless,
            cacheable,
            differentiable: false,
            deterministic: true,
            stream_mode,
            distribution: Distribution::Local,
            input_schema: None,
            output_schema: None,
        }
    }

    struct DoubleChunk;
    impl Filter for DoubleChunk {
        fn config_hash(&self) -> CacheKey {
            CacheKey::from_parts(&[b"DoubleChunk"])
        }
        fn fit(&self, _x: &Value, _y: Option<&Value>) -> SomaResult<Value> {
            Ok(Value::Empty)
        }
        fn forward(&self, x: &Value, _state: &Value) -> SomaResult<Value> {
            if let Value::Tensor { values, shape } = x {
                Ok(Value::tensor(
                    values.iter().map(|v| v * 2.0).collect(),
                    shape.clone(),
                ))
            } else {
                Ok(x.clone())
            }
        }
        fn meta(&self) -> FilterMeta {
            meta("DoubleChunk", StreamMode::FixedState, true)
        }
    }

    /// Identity, but declared uncacheable — the probe for the guard.
    struct UncachedDouble;
    impl Filter for UncachedDouble {
        fn config_hash(&self) -> CacheKey {
            CacheKey::from_parts(&[b"UncachedDouble"])
        }
        fn fit(&self, _x: &Value, _y: Option<&Value>) -> SomaResult<Value> {
            Ok(Value::Empty)
        }
        fn forward(&self, x: &Value, _state: &Value) -> SomaResult<Value> {
            DoubleChunk.forward(x, &Value::Empty)
        }
        fn meta(&self) -> FilterMeta {
            meta("UncachedDouble", StreamMode::FixedState, false)
        }
    }

    /// Barrier: forwards whatever it materialized.
    struct Accumulator;
    impl Filter for Accumulator {
        fn config_hash(&self) -> CacheKey {
            CacheKey::from_parts(&[b"Accumulator"])
        }
        fn fit(&self, _x: &Value, _y: Option<&Value>) -> SomaResult<Value> {
            Ok(Value::Empty)
        }
        fn forward(&self, x: &Value, _state: &Value) -> SomaResult<Value> {
            Ok(x.clone())
        }
        fn meta(&self) -> FilterMeta {
            meta("Accumulator", StreamMode::Barrier, true)
        }
    }

    /// Evolving: output = sum(chunk) + state, and the output IS the next
    /// state — the documented conflation.
    struct RunningSum;
    impl Filter for RunningSum {
        fn config_hash(&self) -> CacheKey {
            CacheKey::from_parts(&[b"RunningSum"])
        }
        fn fit(&self, _x: &Value, _y: Option<&Value>) -> SomaResult<Value> {
            Ok(Value::tensor(vec![0.0], vec![1]))
        }
        fn forward(&self, x: &Value, state: &Value) -> SomaResult<Value> {
            let x_sum: f64 = match x {
                Value::Tensor { values, .. } => values.iter().sum(),
                _ => 0.0,
            };
            let state_sum: f64 = match state {
                Value::Tensor { values, .. } => values.first().copied().unwrap_or(0.0),
                _ => 0.0,
            };
            Ok(Value::tensor(vec![x_sum + state_sum], vec![1]))
        }
        fn meta(&self) -> FilterMeta {
            let mut m = meta("RunningSum", StreamMode::Evolving, false);
            m.kind = FilterKind::Trainable;
            m
        }
    }

    struct Panicker;
    impl Filter for Panicker {
        fn config_hash(&self) -> CacheKey {
            CacheKey::from_parts(&[b"Panicker"])
        }
        fn fit(&self, _x: &Value, _y: Option<&Value>) -> SomaResult<Value> {
            Ok(Value::Empty)
        }
        fn forward(&self, _x: &Value, _state: &Value) -> SomaResult<Value> {
            panic!("chunk went sideways")
        }
        fn meta(&self) -> FilterMeta {
            meta("Panicker", StreamMode::FixedState, true)
        }
    }

    fn harness(nodes: Vec<(&str, Box<dyn Filter>)>) -> (StreamRun, Context, MemoryCache) {
        let mut catalog = NodeCatalog::new();
        let mut ids = Vec::new();
        for (id, filter) in nodes {
            catalog.register(id, filter);
            ids.push(id.to_string());
        }
        let run = StreamRun::new(&ids, &catalog).unwrap();
        let ctx = Context::new(Arc::new(EventBus::new(64)), "stream-test");
        (run, ctx, MemoryCache::default())
    }

    fn tensor(vals: &[f64]) -> Value {
        Value::tensor(vals.to_vec(), vec![vals.len()])
    }

    #[test]
    fn fixed_state_processes_each_chunk() {
        let (mut run, mut ctx, cache) = harness(vec![("double", Box::new(DoubleChunk))]);
        let out = run
            .process_chunk(tensor(&[1.0, 2.0]), &mut ctx, &cache)
            .unwrap();
        assert_eq!(out, Some(tensor(&[2.0, 4.0])));
        let out = run.process_chunk(tensor(&[3.0]), &mut ctx, &cache).unwrap();
        assert_eq!(out, Some(tensor(&[6.0])));
    }

    #[test]
    fn barrier_accumulates_then_flushes() {
        let (mut run, mut ctx, cache) = harness(vec![("acc", Box::new(Accumulator))]);
        assert_eq!(
            run.process_chunk(tensor(&[1.0, 2.0]), &mut ctx, &cache)
                .unwrap(),
            None
        );
        assert_eq!(
            run.process_chunk(tensor(&[3.0, 4.0]), &mut ctx, &cache)
                .unwrap(),
            None
        );
        let flushed = run.flush(&mut ctx, &cache).unwrap().unwrap();
        assert_eq!(flushed, tensor(&[1.0, 2.0, 3.0, 4.0]));
    }

    #[test]
    fn evolving_state_accumulates() {
        let (mut run, mut ctx, cache) = harness(vec![("sum", Box::new(RunningSum))]);
        let r1 = run
            .process_chunk(tensor(&[10.0]), &mut ctx, &cache)
            .unwrap()
            .unwrap();
        assert_eq!(r1, tensor(&[10.0]));
        let r2 = run
            .process_chunk(tensor(&[5.0]), &mut ctx, &cache)
            .unwrap()
            .unwrap();
        assert_eq!(r2, tensor(&[15.0]), "10 + 5: the output was the state");
    }

    #[test]
    fn mixed_pipeline_fixed_then_barrier() {
        let (mut run, mut ctx, cache) = harness(vec![
            ("double", Box::new(DoubleChunk)),
            ("acc", Box::new(Accumulator)),
        ]);
        assert_eq!(
            run.process_chunk(tensor(&[1.0]), &mut ctx, &cache).unwrap(),
            None
        );
        assert_eq!(
            run.process_chunk(tensor(&[2.0]), &mut ctx, &cache).unwrap(),
            None
        );
        let flushed = run.flush(&mut ctx, &cache).unwrap().unwrap();
        assert_eq!(flushed, tensor(&[2.0, 4.0]), "doubled then accumulated");
    }

    /// T4: `cacheable: false` means NOTHING is written per chunk. The old
    /// executor cached every filter it was handed a store for.
    #[test]
    fn uncacheable_chunks_are_not_cached() {
        let (mut run, mut ctx, cache) = harness(vec![("raw", Box::new(UncachedDouble))]);
        run.process_chunk(tensor(&[1.0]), &mut ctx, &cache).unwrap();
        run.process_chunk(tensor(&[2.0]), &mut ctx, &cache).unwrap();
        assert!(
            cache.is_empty(),
            "an uncacheable filter's chunks reached the store"
        );
    }

    /// A second pass over the same chunks is served from the store, and
    /// the stats say so.
    #[test]
    fn cached_chunks_are_served_and_counted() {
        let mut catalog = NodeCatalog::new();
        catalog.register("double", Box::new(DoubleChunk));
        let ids = vec!["double".to_string()];
        let cache = MemoryCache::default();
        let mut ctx = Context::new(Arc::new(EventBus::new(64)), "stream-test");

        let mut first = StreamRun::new(&ids, &catalog).unwrap();
        let a = first
            .process_chunk(tensor(&[5.0]), &mut ctx, &cache)
            .unwrap();
        assert!(!cache.is_empty(), "the chunk should have been cached");

        let mut second = StreamRun::new(&ids, &catalog).unwrap();
        let b = second
            .process_chunk(tensor(&[5.0]), &mut ctx, &cache)
            .unwrap();
        assert_eq!(a, b);
        assert_eq!(second.nodes[0].cache_hits, 1);
        assert_eq!(second.nodes[0].cache_misses, 0);
    }

    /// T7: two seeds must not share a chunk's cache line. `output_key`
    /// salts exactly as the standard path does — that is the point of
    /// sharing the derivation instead of spelling it out again.
    #[test]
    fn a_chunk_cache_key_follows_the_run_seed() {
        let mut catalog = NodeCatalog::new();
        catalog.register("double", Box::new(DoubleChunk));
        let ids = vec!["double".to_string()];
        let cache = MemoryCache::default();
        let bus = Arc::new(EventBus::new(64));

        for seed in [Some(1), Some(2), None] {
            let mut ctx = Context::new(bus.clone(), "stream-test").with_seed(seed);
            let mut run = StreamRun::new(&ids, &catalog).unwrap();
            run.process_chunk(tensor(&[1.0, 2.0]), &mut ctx, &cache)
                .unwrap();
        }
        assert_eq!(
            cache.len(),
            3,
            "each seed must own its own cache line for the same chunk"
        );
    }

    /// T8: JSON flattens NaN and +inf to the same bytes; the content key
    /// must not.
    #[test]
    fn non_finite_chunks_do_not_share_a_cache_key() {
        let nan = tensor(&[f64::NAN]);
        let inf = tensor(&[f64::INFINITY]);
        assert_eq!(
            serde_json::to_vec(&nan).unwrap(),
            serde_json::to_vec(&inf).unwrap(),
            "if this ever stops being true the bug is gone by other means"
        );

        let (mut run, mut ctx, cache) = harness(vec![("double", Box::new(DoubleChunk))]);
        let out_nan = run.process_chunk(nan, &mut ctx, &cache).unwrap().unwrap();
        let out_inf = run.process_chunk(inf, &mut ctx, &cache).unwrap().unwrap();

        let first = |v: &Value| match v {
            Value::Tensor { values, .. } => values[0],
            other => panic!("expected a tensor, got {other:?}"),
        };
        assert!(first(&out_nan).is_nan(), "NaN doubled is still NaN");
        assert_eq!(
            first(&out_inf),
            f64::INFINITY,
            "the infinite chunk was served the NaN chunk's cached output"
        );
    }

    /// T3: a panic in a filter is an error, not a dead process — the
    /// catch_unwind is inherited from `compute_node`, not reimplemented.
    #[test]
    fn a_panicking_chunk_is_contained() {
        let (mut run, mut ctx, cache) = harness(vec![("boom", Box::new(Panicker))]);
        let err = run
            .process_chunk(tensor(&[1.0]), &mut ctx, &cache)
            .unwrap_err();
        assert!(err.to_string().contains("panicked"), "{err}");
    }

    /// T9: the barrier's flush leg goes through the cache like any other
    /// execution — a second run's flush is a hit. (The old executor's
    /// flush ran bare `forward`s: no cache, no events.)
    #[test]
    fn barrier_flush_goes_through_the_cache() {
        let mut catalog = NodeCatalog::new();
        catalog.register("acc", Box::new(Accumulator));
        let ids = vec!["acc".to_string()];
        let cache = MemoryCache::default();
        let mut ctx = Context::new(Arc::new(EventBus::new(64)), "stream-test");

        let mut first = StreamRun::new(&ids, &catalog).unwrap();
        first
            .process_chunk(tensor(&[1.0]), &mut ctx, &cache)
            .unwrap();
        first
            .process_chunk(tensor(&[2.0]), &mut ctx, &cache)
            .unwrap();
        first.flush(&mut ctx, &cache).unwrap();
        assert!(!cache.is_empty(), "the flush output should be cached");

        let mut second = StreamRun::new(&ids, &catalog).unwrap();
        second
            .process_chunk(tensor(&[1.0]), &mut ctx, &cache)
            .unwrap();
        second
            .process_chunk(tensor(&[2.0]), &mut ctx, &cache)
            .unwrap();
        second.flush(&mut ctx, &cache).unwrap();
        assert_eq!(second.nodes[0].cache_hits, 1, "the flush should be a hit");
    }

    /// An unknown node id is an error at construction — the worker's old
    /// `filter_map` silently dropped it and streamed a shorter chain.
    #[test]
    fn an_unknown_node_is_an_error_not_a_skip() {
        let catalog = NodeCatalog::new();
        let Err(err) = StreamRun::new(&["ghost".to_string()], &catalog) else {
            panic!("an unknown node must not stream");
        };
        assert!(matches!(err, SomaError::NodeNotFound(id) if id == "ghost"));
    }
}
