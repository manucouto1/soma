//! Streaming executor — processes data in chunks through fitted filters.
//!
//! Respects each filter's [`StreamMode`]: FixedState processes chunks
//! independently, Evolving updates state per chunk with checkpoints,
//! Barrier accumulates all chunks before processing.

use somatize_core::cache::{CacheKey, CacheStore};
use somatize_core::error::{Result, SomaError};
use somatize_core::filter::{Filter, StreamMode};
use somatize_core::value::Value;
use std::sync::Arc;

/// A fitted filter with its learned state, ready for streaming.
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
/// - Evolving: state mutates with each chunk, periodic checkpoints
/// - Barrier: accumulates all chunks, processes as batch on flush
pub struct StreamExecutor {
    filters: Vec<FittedFilter>,
    cache: Option<Arc<dyn CacheStore>>,
    states: Vec<FilterStreamState>,
    chunk_count: usize,
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
        }
    }

    pub fn with_cache(mut self, cache: Arc<dyn CacheStore>) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Process a single chunk through the pipeline.
    /// Returns the output chunk, or None if a Barrier filter is still accumulating.
    pub fn process_chunk(&mut self, chunk: Value) -> Result<Option<Value>> {
        let mut current = chunk;
        self.chunk_count += 1;

        for i in 0..self.filters.len() {
            let mode = self.filters[i].filter.meta().stream_mode;
            match process_by_mode(
                &mode,
                &self.filters[i],
                &current,
                &mut self.states[i],
                self.cache.as_deref(),
                self.chunk_count,
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

    /// Process multiple chunks and collect outputs.
    pub fn process_all(&mut self, chunks: Vec<Value>) -> Result<Vec<Value>> {
        let mut outputs = Vec::new();
        for chunk in chunks {
            if let Some(output) = self.process_chunk(chunk)? {
                outputs.push(output);
            }
        }
        if let Some(flushed) = self.flush()? {
            outputs.push(flushed);
        }
        Ok(outputs)
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

// ── StreamMode dispatch ──

/// Process one chunk according to the stream mode.
fn process_by_mode(
    mode: &StreamMode,
    fitted: &FittedFilter,
    input: &Value,
    state: &mut FilterStreamState,
    cache: Option<&dyn CacheStore>,
    chunk_count: usize,
) -> Result<ChunkResult> {
    match mode {
        StreamMode::FixedState => {
            let result = forward_cached(fitted, input, cache)?;
            Ok(ChunkResult::Output(result))
        }
        StreamMode::Evolving { checkpoint_every } => {
            let default_state: &Value = &fitted.state;
            let filter_state = state.evolving_state.as_ref().unwrap_or(default_state);
            let result = fitted.filter.forward(input, filter_state)?;
            state.evolving_state = Some(result.clone());

            if *checkpoint_every > 0
                && chunk_count.is_multiple_of(*checkpoint_every)
                && let Some(c) = cache
            {
                let key = CacheKey::from_parts(&[
                    b"checkpoint",
                    fitted.name.as_bytes(),
                    &(chunk_count as u64).to_le_bytes(),
                ]);
                let _ = c.put(&key, &result);
            }
            Ok(ChunkResult::Output(result))
        }
        StreamMode::Barrier => {
            state.barrier_buffer.push(input.clone());
            Ok(ChunkResult::Buffered)
        }
        _ => {
            // Default: treat as FixedState
            let result = forward_cached(fitted, input, cache)?;
            Ok(ChunkResult::Output(result))
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

/// Forward with optional cache lookup.
fn forward_cached(
    fitted: &FittedFilter,
    input: &Value,
    cache: Option<&dyn CacheStore>,
) -> Result<Value> {
    if let Some(c) = cache {
        let chunk_hash = CacheKey::hash_data(&serde_json::to_vec(input).unwrap_or_default());
        let state_hash =
            CacheKey::hash_data(&serde_json::to_vec(&fitted.state).unwrap_or_default());
        let cache_key =
            CacheKey::for_output(&fitted.filter.config_hash(), &state_hash, &chunk_hash);
        if let Some(cached) = c.get(&cache_key)? {
            return Ok(cached);
        }
        let result = fitted.filter.forward(input, &fitted.state)?;
        let _ = c.put(&cache_key, &result);
        return Ok(result);
    }
    fitted.filter.forward(input, &fitted.state)
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
    use somatize_core::cache::CacheKey;
    use somatize_core::error::Result as SomaResult;
    use somatize_core::filter::{Distribution, FilterKind, FilterMeta};

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
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

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
            FilterMeta {
                name: "Accumulator".into(),
                kind: FilterKind::Stateless,
                cacheable: false,
                differentiable: false,
                deterministic: true,
                stream_mode: StreamMode::Barrier,
                distribution: Distribution::Local,
                input_schema: None,
                output_schema: None,
            }
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

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
            FilterMeta {
                name: "RunningSum".into(),
                kind: FilterKind::Trainable,
                cacheable: false,
                differentiable: false,
                deterministic: true,
                stream_mode: StreamMode::Evolving {
                    checkpoint_every: 2,
                },
                distribution: Distribution::Local,
                input_schema: None,
                output_schema: None,
            }
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    fn make_fitted(filter: impl Filter + 'static, state: Value) -> FittedFilter {
        let name = filter.meta().name.clone();
        FittedFilter {
            name,
            filter: Arc::new(filter),
            state: Arc::new(state),
        }
    }

    #[test]
    fn fixed_state_processes_each_chunk() {
        let f = make_fitted(DoubleChunk, Value::Empty);
        let mut exec = StreamExecutor::new(vec![f]);

        let out1 = exec
            .process_chunk(Value::tensor(vec![1.0, 2.0], vec![2]))
            .unwrap();
        assert_eq!(out1, Some(Value::tensor(vec![2.0, 4.0], vec![2])));

        let out2 = exec
            .process_chunk(Value::tensor(vec![3.0], vec![1]))
            .unwrap();
        assert_eq!(out2, Some(Value::tensor(vec![6.0], vec![1])));
    }

    #[test]
    fn barrier_accumulates_then_flushes() {
        let f = make_fitted(Accumulator, Value::Empty);
        let mut exec = StreamExecutor::new(vec![f]);

        let r1 = exec
            .process_chunk(Value::tensor(vec![1.0, 2.0], vec![2]))
            .unwrap();
        assert_eq!(r1, None);

        let r2 = exec
            .process_chunk(Value::tensor(vec![3.0, 4.0], vec![2]))
            .unwrap();
        assert_eq!(r2, None);

        let flushed = exec.flush().unwrap().unwrap();
        let (data, shape) = flushed.as_tensor().unwrap();
        assert_eq!(data, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(shape, &[4]);
    }

    #[test]
    fn evolving_state_accumulates() {
        let f = make_fitted(RunningSum, Value::tensor(vec![0.0], vec![1]));
        let mut exec = StreamExecutor::new(vec![f]);

        let r1 = exec
            .process_chunk(Value::tensor(vec![10.0], vec![1]))
            .unwrap()
            .unwrap();
        let (d1, _) = r1.as_tensor().unwrap();
        assert_eq!(d1, &[10.0]);

        let r2 = exec
            .process_chunk(Value::tensor(vec![5.0], vec![1]))
            .unwrap()
            .unwrap();
        let (d2, _) = r2.as_tensor().unwrap();
        assert_eq!(d2, &[15.0]); // 10 + 5
    }

    #[test]
    fn mixed_pipeline_fixed_then_barrier() {
        let f1 = make_fitted(DoubleChunk, Value::Empty);
        let f2 = make_fitted(Accumulator, Value::Empty);
        let mut exec = StreamExecutor::new(vec![f1, f2]);

        let r1 = exec
            .process_chunk(Value::tensor(vec![1.0], vec![1]))
            .unwrap();
        assert_eq!(r1, None); // barrier

        let r2 = exec
            .process_chunk(Value::tensor(vec![2.0], vec![1]))
            .unwrap();
        assert_eq!(r2, None);

        let flushed = exec.flush().unwrap().unwrap();
        let (data, _) = flushed.as_tensor().unwrap();
        assert_eq!(data, &[2.0, 4.0]); // doubled then accumulated
    }

    #[test]
    fn process_all_combines_chunks() {
        let f = make_fitted(DoubleChunk, Value::Empty);
        let mut exec = StreamExecutor::new(vec![f]);

        let outputs = exec
            .process_all(vec![
                Value::tensor(vec![1.0], vec![1]),
                Value::tensor(vec![2.0], vec![1]),
                Value::tensor(vec![3.0], vec![1]),
            ])
            .unwrap();

        assert_eq!(outputs.len(), 3);
        let (d, _) = outputs[0].as_tensor().unwrap();
        assert_eq!(d, &[2.0]);
    }

    #[test]
    fn fixed_state_with_cache() {
        let f = make_fitted(DoubleChunk, Value::Empty);
        let cache = Arc::new(crate::MemoryCache::default());
        let mut exec = StreamExecutor::new(vec![f]).with_cache(cache);

        let r1 = exec
            .process_chunk(Value::tensor(vec![5.0], vec![1]))
            .unwrap()
            .unwrap();
        // Second call with same input should hit cache
        let r2 = exec
            .process_chunk(Value::tensor(vec![5.0], vec![1]))
            .unwrap()
            .unwrap();
        assert_eq!(r1, r2);
    }
}
