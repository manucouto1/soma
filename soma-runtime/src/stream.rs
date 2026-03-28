use soma_core::cache::{CacheKey, CacheStore};
use soma_core::error::{Result, SomaError};
use soma_core::filter::{Filter, StreamMode};
use soma_core::value::Value;
use std::sync::Arc;

/// A fitted filter with its learned state, ready for streaming.
pub struct FittedFilter {
    pub name: String,
    pub filter: Arc<dyn Filter>,
    pub state: Value,
}

/// Processes a stream of chunks through a sequence of fitted filters.
///
/// Respects each filter's StreamMode:
/// - FixedState: each chunk processed independently, cacheable per chunk
/// - Evolving: state mutates with each chunk, periodic checkpoints
/// - Barrier: accumulates all chunks, processes as batch
pub struct StreamExecutor {
    filters: Vec<FittedFilter>,
    cache: Option<Arc<dyn CacheStore>>,
    /// Accumulated chunks for Barrier filters (keyed by filter index).
    barrier_buffers: Vec<Vec<Value>>,
    /// Evolving states (keyed by filter index, mutated on each chunk).
    evolving_states: Vec<Option<Value>>,
    /// Chunk counter for checkpoint scheduling.
    chunk_count: usize,
}

impl StreamExecutor {
    pub fn new(filters: Vec<FittedFilter>) -> Self {
        let n = filters.len();
        Self {
            filters,
            cache: None,
            barrier_buffers: vec![Vec::new(); n],
            evolving_states: vec![None; n],
            chunk_count: 0,
        }
    }

    pub fn with_cache(mut self, cache: Arc<dyn CacheStore>) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Process a single chunk through the pipeline.
    ///
    /// Returns the output chunk, or None if a Barrier filter is still accumulating.
    pub fn process_chunk(&mut self, chunk: Value) -> Result<Option<Value>> {
        let mut current = chunk;
        self.chunk_count += 1;

        let n = self.filters.len();
        for i in 0..n {
            let mode = self.filters[i].filter.meta().stream_mode;

            match mode {
                StreamMode::FixedState => {
                    current = self.process_fixed_state(i, &current)?;
                }
                StreamMode::Evolving { checkpoint_every } => {
                    current = self.process_evolving(i, &current, checkpoint_every)?;
                }
                StreamMode::Barrier => {
                    self.barrier_buffers[i].push(current);
                    return Ok(None);
                }
                _ => {
                    current = self.process_fixed_state(i, &current)?;
                }
            }
        }

        Ok(Some(current))
    }

    /// Flush barrier filters and process remaining data as batch.
    ///
    /// Call this after the stream ends to materialize barrier outputs.
    pub fn flush(&mut self) -> Result<Option<Value>> {
        let mut current: Option<Value> = None;
        let n = self.filters.len();

        for i in 0..n {
            let mode = self.filters[i].filter.meta().stream_mode;

            if mode == StreamMode::Barrier && !self.barrier_buffers[i].is_empty() {
                let materialized = self.materialize_buffer(i)?;
                let result = self.filters[i]
                    .filter
                    .forward(&materialized, &self.filters[i].state)?;
                self.barrier_buffers[i].clear();
                current = Some(result);
            } else if let Some(val) = current.take() {
                let result = self.filters[i]
                    .filter
                    .forward(&val, &self.filters[i].state)?;
                current = Some(result);
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

        // Flush any barrier buffers
        if let Some(flushed) = self.flush()? {
            outputs.push(flushed);
        }

        Ok(outputs)
    }

    /// Number of chunks processed so far.
    pub fn chunks_processed(&self) -> usize {
        self.chunk_count
    }

    fn process_fixed_state(&self, filter_idx: usize, input: &Value) -> Result<Value> {
        let fitted = &self.filters[filter_idx];

        // Try cache
        if let Some(cache) = &self.cache {
            let chunk_hash = CacheKey::hash_data(&serde_json::to_vec(input).unwrap_or_default());
            let cache_key = CacheKey::for_output(
                &fitted.filter.config_hash(),
                &CacheKey::hash_data(&serde_json::to_vec(&fitted.state).unwrap_or_default()),
                &chunk_hash,
            );
            if let Some(cached) = cache.get(&cache_key)? {
                return Ok(cached);
            }
            let result = fitted.filter.forward(input, &fitted.state)?;
            let _ = cache.put(&cache_key, &result);
            return Ok(result);
        }

        fitted.filter.forward(input, &fitted.state)
    }

    fn process_evolving(
        &mut self,
        filter_idx: usize,
        input: &Value,
        checkpoint_every: usize,
    ) -> Result<Value> {
        let fitted = &self.filters[filter_idx];

        // Use evolving state if available, else initial state
        let state = self.evolving_states[filter_idx]
            .as_ref()
            .unwrap_or(&fitted.state);

        let result = fitted.filter.forward(input, state)?;

        // For evolving: the output becomes the new state for next chunk
        // (simplified model: state = last output)
        self.evolving_states[filter_idx] = Some(result.clone());

        // Checkpoint
        if checkpoint_every > 0
            && self.chunk_count.is_multiple_of(checkpoint_every)
            && let Some(cache) = &self.cache
        {
            let checkpoint_key = CacheKey::from_parts(&[
                b"checkpoint",
                fitted.name.as_bytes(),
                &(self.chunk_count as u64).to_le_bytes(),
            ]);
            let _ = cache.put(&checkpoint_key, &result);
        }

        Ok(result)
    }

    fn materialize_buffer(&self, filter_idx: usize) -> Result<Value> {
        let buffer = &self.barrier_buffers[filter_idx];
        if buffer.is_empty() {
            return Ok(Value::Empty);
        }

        // Concatenate tensor chunks along first dimension
        let mut all_data = Vec::new();
        let mut total_rows = 0;
        let mut cols = 0;

        for chunk in buffer {
            match chunk {
                Value::Tensor { values, shape } => {
                    all_data.extend(values);
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use soma_core::cache::CacheKey;
    use soma_core::filter::{FilterKind, FilterMeta};

    // ── Test filters ──

    struct DoubleChunk;
    impl Filter for DoubleChunk {
        fn config_hash(&self) -> CacheKey {
            CacheKey::from_parts(&[b"DoubleChunk"])
        }
        fn fit(&self, _: &Value, _: Option<&Value>) -> Result<Value> {
            Ok(Value::Empty)
        }
        fn forward(&self, x: &Value, _: &Value) -> Result<Value> {
            match x {
                Value::Tensor { values, shape } => Ok(Value::tensor(
                    values.iter().map(|v| v * 2.0).collect(),
                    shape.clone(),
                )),
                _ => Ok(x.clone()),
            }
        }
        fn meta(&self) -> FilterMeta {
            FilterMeta {
                name: "DoubleChunk".into(),
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

    struct Accumulator;
    impl Filter for Accumulator {
        fn config_hash(&self) -> CacheKey {
            CacheKey::from_parts(&[b"Accumulator"])
        }
        fn fit(&self, _: &Value, _: Option<&Value>) -> Result<Value> {
            Ok(Value::Empty)
        }
        fn forward(&self, x: &Value, _: &Value) -> Result<Value> {
            // For barrier: receives concatenated tensor, computes mean
            match x {
                Value::Tensor { values, shape } => {
                    let mean = values.iter().sum::<f64>() / values.len() as f64;
                    Ok(Value::tensor(vec![mean], vec![1]))
                }
                _ => Ok(x.clone()),
            }
        }
        fn meta(&self) -> FilterMeta {
            FilterMeta {
                name: "Accumulator".into(),
                kind: FilterKind::Trainable,
                cacheable: false,
                differentiable: false,
                stream_mode: StreamMode::Barrier,
                distribution: soma_core::filter::Distribution::Local,
                input_schema: None,
                output_schema: None,
            }
        }
    }

    struct RunningSum;
    impl Filter for RunningSum {
        fn config_hash(&self) -> CacheKey {
            CacheKey::from_parts(&[b"RunningSum"])
        }
        fn fit(&self, _: &Value, _: Option<&Value>) -> Result<Value> {
            Ok(Value::tensor(vec![0.0], vec![1]))
        }
        fn forward(&self, x: &Value, state: &Value) -> Result<Value> {
            let x_val = x.as_tensor().map(|(d, _)| d[0]).unwrap_or(0.0);
            let s_val = state.as_tensor().map(|(d, _)| d[0]).unwrap_or(0.0);
            Ok(Value::tensor(vec![x_val + s_val], vec![1]))
        }
        fn meta(&self) -> FilterMeta {
            FilterMeta {
                name: "RunningSum".into(),
                kind: FilterKind::Trainable,
                cacheable: false,
                differentiable: false,
                stream_mode: StreamMode::Evolving {
                    checkpoint_every: 3,
                },
                distribution: soma_core::filter::Distribution::Local,
                input_schema: None,
                output_schema: None,
            }
        }
    }

    // ── Tests ──

    #[test]
    fn fixed_state_processes_each_chunk() {
        let mut executor = StreamExecutor::new(vec![FittedFilter {
            name: "double".into(),
            filter: Arc::new(DoubleChunk),
            state: Value::Empty,
        }]);

        let chunks = vec![
            Value::tensor(vec![1.0, 2.0], vec![2]),
            Value::tensor(vec![3.0, 4.0], vec![2]),
            Value::tensor(vec![5.0], vec![1]),
        ];

        let outputs = executor.process_all(chunks).unwrap();
        assert_eq!(outputs.len(), 3);

        let (d0, _) = outputs[0].as_tensor().unwrap();
        assert_eq!(d0, &[2.0, 4.0]);
        let (d1, _) = outputs[1].as_tensor().unwrap();
        assert_eq!(d1, &[6.0, 8.0]);
        let (d2, _) = outputs[2].as_tensor().unwrap();
        assert_eq!(d2, &[10.0]);
    }

    #[test]
    fn barrier_accumulates_then_flushes() {
        let mut executor = StreamExecutor::new(vec![FittedFilter {
            name: "acc".into(),
            filter: Arc::new(Accumulator),
            state: Value::Empty,
        }]);

        // Process chunks: barrier should return None for each
        assert!(
            executor
                .process_chunk(Value::tensor(vec![1.0, 2.0], vec![2]))
                .unwrap()
                .is_none()
        );
        assert!(
            executor
                .process_chunk(Value::tensor(vec![3.0, 4.0], vec![2]))
                .unwrap()
                .is_none()
        );
        assert!(
            executor
                .process_chunk(Value::tensor(vec![5.0, 6.0], vec![2]))
                .unwrap()
                .is_none()
        );

        // Flush: should materialize and compute mean of [1,2,3,4,5,6]
        let result = executor.flush().unwrap().unwrap();
        let (data, _) = result.as_tensor().unwrap();
        assert!((data[0] - 3.5).abs() < 0.01); // mean of 1..6
    }

    #[test]
    fn evolving_state_accumulates() {
        let mut executor = StreamExecutor::new(vec![FittedFilter {
            name: "sum".into(),
            filter: Arc::new(RunningSum),
            state: Value::tensor(vec![0.0], vec![1]), // initial sum = 0
        }]);

        let r1 = executor
            .process_chunk(Value::tensor(vec![5.0], vec![1]))
            .unwrap()
            .unwrap();
        assert_eq!(r1.as_tensor().unwrap().0, &[5.0]); // 0+5=5

        let r2 = executor
            .process_chunk(Value::tensor(vec![3.0], vec![1]))
            .unwrap()
            .unwrap();
        assert_eq!(r2.as_tensor().unwrap().0, &[8.0]); // 5+3=8

        let r3 = executor
            .process_chunk(Value::tensor(vec![2.0], vec![1]))
            .unwrap()
            .unwrap();
        assert_eq!(r3.as_tensor().unwrap().0, &[10.0]); // 8+2=10
    }

    #[test]
    fn mixed_pipeline_fixed_then_barrier() {
        let mut executor = StreamExecutor::new(vec![
            FittedFilter {
                name: "double".into(),
                filter: Arc::new(DoubleChunk),
                state: Value::Empty,
            },
            FittedFilter {
                name: "acc".into(),
                filter: Arc::new(Accumulator),
                state: Value::Empty,
            },
        ]);

        let chunks = vec![
            Value::tensor(vec![1.0], vec![1]),
            Value::tensor(vec![2.0], vec![1]),
            Value::tensor(vec![3.0], vec![1]),
        ];

        let outputs = executor.process_all(chunks).unwrap();
        // DoubleChunk doubles: [2,4,6]. Accumulator sees barrier after double.
        // After flush: mean of [2,4,6] = 4.0
        assert_eq!(outputs.len(), 1);
        let (data, _) = outputs[0].as_tensor().unwrap();
        assert!((data[0] - 4.0).abs() < 0.01);
    }

    #[test]
    fn fixed_state_with_cache() {
        let cache = Arc::new(crate::MemoryCache::default());
        let mut executor = StreamExecutor::new(vec![FittedFilter {
            name: "double".into(),
            filter: Arc::new(DoubleChunk),
            state: Value::Empty,
        }])
        .with_cache(cache.clone());

        let chunk = Value::tensor(vec![7.0], vec![1]);

        // First call: cache miss
        let r1 = executor.process_chunk(chunk.clone()).unwrap().unwrap();
        assert_eq!(r1.as_tensor().unwrap().0, &[14.0]);
        assert!(!cache.is_empty()); // cached

        // Second call with same chunk: cache hit
        let r2 = executor.process_chunk(chunk).unwrap().unwrap();
        assert_eq!(r2.as_tensor().unwrap().0, &[14.0]);
    }

    #[test]
    fn chunks_processed_counter() {
        let mut executor = StreamExecutor::new(vec![FittedFilter {
            name: "double".into(),
            filter: Arc::new(DoubleChunk),
            state: Value::Empty,
        }]);

        assert_eq!(executor.chunks_processed(), 0);
        executor
            .process_chunk(Value::tensor(vec![1.0], vec![1]))
            .unwrap();
        assert_eq!(executor.chunks_processed(), 1);
        executor
            .process_chunk(Value::tensor(vec![2.0], vec![1]))
            .unwrap();
        assert_eq!(executor.chunks_processed(), 2);
    }

    #[test]
    fn empty_stream() {
        let mut executor = StreamExecutor::new(vec![FittedFilter {
            name: "double".into(),
            filter: Arc::new(DoubleChunk),
            state: Value::Empty,
        }]);

        let outputs = executor.process_all(vec![]).unwrap();
        assert!(outputs.is_empty());
    }
}
