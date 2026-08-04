//! Legacy chunk executor — the worker's remote streaming path, pending
//! unification with the runtime's `StreamRun`.
//!
//! The local plan-driven stream path now runs every chunk through
//! `run_node`'s primitives (one key derivation, the cacheable guard,
//! panic containment, per-node events). This executor predates that and
//! keeps the old behavior for the two remote entry points that hold an
//! executor alive *between* RPC messages (`ServerState.active_streams`)
//! or drive it straight from a `DataStore` — unifying them needs a
//! `Context` that survives across WS messages, which is its own piece
//! of work.
//!
//! Known gaps, inherited and documented rather than silently kept:
//! per-chunk caching ignores `cacheable`/`deterministic`, writes carry
//! no provenance, no per-node events are emitted, and the worker
//! protocol carries no run seed — so remote chunk caching shares lines
//! across seeds (`with_seed` exists but nothing calls it with a real
//! seed yet).

use somatize_core::cache::{CacheKey, CacheStore};
use somatize_core::error::Result;
use somatize_core::filter::{Filter, StreamMode};
use somatize_core::value::Value;
use somatize_runtime::executors::materialize_buffer;
use std::sync::Arc;

/// A fitted filter with its learned state, ready for streaming.
#[derive(Clone)]
pub struct FittedFilter {
    pub name: String,
    pub filter: Arc<dyn Filter>,
    pub state: Arc<Value>,
}

/// Per-filter streaming state — one per filter in the pipeline.
struct FilterStreamState {
    /// Accumulated chunks for Barrier mode.
    barrier_buffer: Vec<Value>,
    /// Evolving state (mutated per chunk).
    evolving_state: Option<Value>,
}

/// Processes a stream of chunks through a sequence of fitted filters.
///
/// Each filter's StreamMode defines its contract:
/// - FixedState: each chunk processed independently, cacheable per chunk
/// - Evolving: state mutates with each chunk (the output is the state)
/// - Barrier: accumulates all chunks, processes as batch on flush
pub struct StreamExecutor {
    filters: Vec<FittedFilter>,
    cache: Option<Arc<dyn CacheStore>>,
    states: Vec<FilterStreamState>,
    chunk_count: usize,
    /// The run's seed, folded into every chunk's cache key. The worker
    /// protocol does not carry one yet, so today this is always `None`
    /// remotely — see the module doc.
    seed: Option<i64>,
}

impl StreamExecutor {
    pub fn new(filters: Vec<FittedFilter>) -> Self {
        let n = filters.len();
        Self {
            filters,
            cache: None,
            states: (0..n)
                .map(|_| FilterStreamState {
                    barrier_buffer: Vec::new(),
                    evolving_state: None,
                })
                .collect(),
            chunk_count: 0,
            seed: None,
        }
    }

    /// Fold the run's seed into every cache key.
    pub fn with_seed(mut self, seed: Option<i64>) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_cache(mut self, cache: Arc<dyn CacheStore>) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Process a single chunk through the pipeline.
    /// Returns the output chunk, or None if a Barrier filter is still accumulating.
    pub fn process_chunk(&mut self, chunk: Value) -> Result<Option<Value>> {
        let cache = self.cache.clone();
        self.process_chunk_cached(chunk, cache.as_deref())
    }

    /// Process a chunk against a borrowed cache.
    pub fn process_chunk_cached(
        &mut self,
        chunk: Value,
        cache: Option<&dyn CacheStore>,
    ) -> Result<Option<Value>> {
        let mut current = chunk;
        self.chunk_count += 1;

        for i in 0..self.filters.len() {
            let mode = self.filters[i].filter.meta().stream_mode;
            match process_by_mode(
                &mode,
                &self.filters[i],
                &current,
                &mut self.states[i],
                cache,
                self.seed,
            )? {
                ChunkResult::Output(val) => current = val,
                ChunkResult::Buffered => return Ok(None),
            }
        }

        Ok(Some(current))
    }

    /// Flush barrier filters and process remaining data as batch.
    pub fn flush(&mut self) -> Result<Option<Value>> {
        let mut current: Option<Value> = None;

        for i in 0..self.filters.len() {
            let mode = self.filters[i].filter.meta().stream_mode;
            if let Some(val) = flush_by_mode(&mode, &self.filters[i], &mut self.states[i])? {
                current = Some(val);
            } else if let Some(val) = current.take() {
                current = Some(
                    self.filters[i]
                        .filter
                        .forward(&val, &self.filters[i].state)?,
                );
            }
        }

        Ok(current)
    }

    /// Number of chunks processed so far.
    pub fn chunks_processed(&self) -> usize {
        self.chunk_count
    }
}

/// Result of processing a chunk through one filter.
enum ChunkResult {
    /// Filter produced output — pass to next filter.
    Output(Value),
    /// Filter is buffering (Barrier) — no output yet.
    Buffered,
}

/// Process one chunk according to the stream mode.
fn process_by_mode(
    mode: &StreamMode,
    fitted: &FittedFilter,
    input: &Value,
    state: &mut FilterStreamState,
    cache: Option<&dyn CacheStore>,
    seed: Option<i64>,
) -> Result<ChunkResult> {
    match mode {
        StreamMode::FixedState => {
            let result = forward_cached(fitted, input, cache, seed)?;
            Ok(ChunkResult::Output(result))
        }
        StreamMode::Evolving => {
            let default_state: &Value = &fitted.state;
            let filter_state = state.evolving_state.as_ref().unwrap_or(default_state);
            let result = fitted.filter.forward(input, filter_state)?;
            state.evolving_state = Some(result.clone());
            Ok(ChunkResult::Output(result))
        }
        StreamMode::Barrier => {
            state.barrier_buffer.push(input.clone());
            Ok(ChunkResult::Buffered)
        }
    }
}

/// Flush a filter by mode. Only Barrier has work to do.
fn flush_by_mode(
    mode: &StreamMode,
    fitted: &FittedFilter,
    state: &mut FilterStreamState,
) -> Result<Option<Value>> {
    match mode {
        StreamMode::Barrier if !state.barrier_buffer.is_empty() => {
            let materialized = materialize_buffer(&state.barrier_buffer)?;
            state.barrier_buffer.clear();
            let result = fitted.filter.forward(&materialized, &fitted.state)?;
            Ok(Some(result))
        }
        _ => Ok(None),
    }
}

/// Forward with optional cache lookup — the pre-unification derivation,
/// kept byte-identical so remote streaming's cache lines do not move
/// before its own unification pass.
fn forward_cached(
    fitted: &FittedFilter,
    input: &Value,
    cache: Option<&dyn CacheStore>,
    seed: Option<i64>,
) -> Result<Value> {
    if let Some(c) = cache {
        let cache_key = salt_with_seed(
            CacheKey::for_output(
                &fitted.filter.config_hash(),
                &CacheKey::for_value(&fitted.state),
                &CacheKey::for_value(input),
            ),
            seed,
        );
        if let Some(cached) = c.get(&cache_key)? {
            return Ok(cached);
        }
        let result = fitted.filter.forward(input, &fitted.state)?;
        let _ = c.put(&cache_key, &result);
        return Ok(result);
    }
    fitted.filter.forward(input, &fitted.state)
}

/// The runtime's seed salt, replicated: same parts, same bytes. Local
/// because the runtime's `salt_with_seed` is crate-private, and this
/// whole module is scheduled to disappear into `StreamRun`.
fn salt_with_seed(key: CacheKey, seed: Option<i64>) -> CacheKey {
    match seed {
        Some(s) => CacheKey::from_parts(&[b"seed", &s.to_le_bytes(), &key.0]),
        None => key,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use somatize_core::error::Result as SomaResult;
    use somatize_core::filter::{Distribution, FilterKind, FilterMeta};
    use somatize_runtime::MemoryCache;

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
                let doubled: Vec<f64> = values.iter().map(|v| v * 2.0).collect();
                Ok(Value::tensor(doubled, shape.clone()))
            } else {
                Ok(x.clone())
            }
        }
        fn meta(&self) -> FilterMeta {
            FilterMeta {
                name: "DoubleChunk".into(),
                kind: FilterKind::Stateless,
                cacheable: true,
                differentiable: false,
                deterministic: true,
                stream_mode: StreamMode::FixedState,
                distribution: Distribution::Local,
                input_schema: None,
                output_schema: None,
            }
        }
    }

    fn fitted() -> FittedFilter {
        FittedFilter {
            name: "doubler".into(),
            filter: Arc::new(DoubleChunk),
            state: Arc::new(Value::Empty),
        }
    }

    #[test]
    fn processes_and_flushes_chunks() {
        let mut exec = StreamExecutor::new(vec![fitted()]);
        let out = exec
            .process_chunk(Value::tensor(vec![1.0, 2.0], vec![2]))
            .unwrap();
        assert_eq!(out, Some(Value::tensor(vec![2.0, 4.0], vec![2])));
        assert_eq!(exec.chunks_processed(), 1);
        assert_eq!(exec.flush().unwrap(), None);
    }

    /// Legacy key-derivation pin: two seeds and the unseeded case own
    /// three distinct cache lines for the same chunk.
    #[test]
    fn a_chunk_cache_key_follows_the_run_seed() {
        let cache = MemoryCache::default();
        let chunk = Value::tensor(vec![1.0, 2.0], vec![2]);
        let f = fitted();

        let a = forward_cached(&f, &chunk, Some(&cache), Some(1)).unwrap();
        let b = forward_cached(&f, &chunk, Some(&cache), Some(2)).unwrap();
        assert_eq!(a, b, "the computation itself does not depend on the seed");
        forward_cached(&f, &chunk, Some(&cache), None).unwrap();
        assert_eq!(cache.len(), 3);
    }

    /// Legacy content-key pin: JSON flattens NaN and +inf to the same
    /// bytes; the cache key must not.
    #[test]
    fn non_finite_chunks_do_not_share_a_cache_key() {
        let cache = MemoryCache::default();
        let nan = Value::tensor(vec![f64::NAN], vec![1]);
        let inf = Value::tensor(vec![f64::INFINITY], vec![1]);
        let f = fitted();

        let out_nan = forward_cached(&f, &nan, Some(&cache), None).unwrap();
        let out_inf = forward_cached(&f, &inf, Some(&cache), None).unwrap();

        let first = |v: &Value| match v {
            Value::Tensor { values, .. } => values[0],
            other => panic!("expected a tensor, got {other:?}"),
        };
        assert!(first(&out_nan).is_nan());
        assert_eq!(first(&out_inf), f64::INFINITY);
    }
}
