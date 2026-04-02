---
title: Streaming
description: How filters operate on continuous data streams with caching and state management.
---

## Batch vs Stream Unification

Soma eliminates the traditional separation between batch and streaming processing. The same filter definition works in both modes. The difference is in **how data arrives** and **how state is managed**.

```
Batch:   all data available upfront → fit once, forward once
Stream:  data arrives in chunks     → state may evolve, forward per chunk
```

## Stream Modes

Each filter declares how it behaves when processing a stream:

```rust
pub enum StreamMode {
    /// State is fixed (pre-trained). Each chunk processed independently.
    /// Example: scaler with pre-computed mean/std
    /// Cache: each chunk cached as independent K/V entry
    FixedState,

    /// State evolves with each chunk (online learning).
    /// Example: running mean, online classifier
    /// Cache: periodic state checkpoints
    Evolving { checkpoint_every: usize },

    /// Must see all data before producing output.
    /// Example: PCA, global sort, SVD
    /// The runtime accumulates all chunks and runs as batch.
    Barrier,
}
```

### FixedState (Most Common)

The filter has been trained (via `fit()`) and its state is frozen. Each chunk is processed independently with the same state.

```
Stream: chunk_1, chunk_2, chunk_3, ...

Scaler (state = { mean: 0.5, std: 0.2 })

chunk_1 → forward(chunk_1, state) → out_1
chunk_2 → forward(chunk_2, state) → out_2
chunk_3 → forward(chunk_3, state) → out_3

Cache entries:
  hash(config + state + chunk_1) → out_1
  hash(config + state + chunk_2) → out_2
  hash(config + state + chunk_3) → out_3

If the same stream is reprocessed: all cache hits.
```

### Evolving

The filter's state changes with each chunk. Periodic checkpoints are saved.

```
RunningMean (checkpoint_every = 5)

chunk_1 → step(chunk_1, state_0) → out_1, state_1
chunk_2 → step(chunk_2, state_1) → out_2, state_2
chunk_3 → step(chunk_3, state_2) → out_3, state_3
chunk_4 → step(chunk_4, state_3) → out_4, state_4
chunk_5 → step(chunk_5, state_4) → out_5, state_5
                                          ↓
                               checkpoint! cache[hash(config + "cp_5")] = state_5

chunk_6 → step(chunk_6, state_5) → out_6, state_6
...

If processing fails at chunk_8:
  → Find latest checkpoint: state_5
  → Resume from chunk_6
  → Only 2 chunks recomputed instead of 8
```

```rust
/// Trait extension for evolving filters
pub trait EvolvingFilter: Filter {
    /// Process one chunk, mutate state in place
    fn step(&self, chunk: &Tensor, state: &mut Self::State) -> Result<Tensor>;
}
```

### Barrier

The filter cannot operate on partial data. The runtime detects this and switches to batch mode for that section of the graph.

```
Graph: [Scaler] → [PCA] → [Classifier]
           stream ✓   barrier   stream ✓

Compiled plan:
  StreamSection([Scaler])           ← processes chunks as they arrive
  → Materialize                     ← accumulates all scaled chunks
  → Execute(PCA.fit + PCA.forward)  ← runs as batch on full data
  → StreamSection([Classifier])     ← back to streaming
```

The compiler inserts materialization points automatically when it encounters a Barrier filter in a streaming context.

## Stream Caching

### FixedState Caching

Each chunk is cached independently in the K/V store. The key includes the chunk's content hash:

```
Key   = hash(filter_config + state_hash + chunk_content_hash)
Value = transformed chunk
```

This means identical chunks always produce cache hits, regardless of their position in the stream.

### Evolving State Caching

Checkpoints are cached with a position-based key:

```
Key   = hash(filter_config + "checkpoint" + position)
Value = serialized state at that position
```

Recovery flow:
1. Find the latest checkpoint before the failure point
2. Load the state from that checkpoint
3. Replay chunks from checkpoint to failure point
4. Resume normal processing

### ChronosVector Integration

For streams, ChronosVector adds temporal awareness to the cache:

```rust
pub struct StreamCacheEntry {
    pub key: CacheKey,
    pub window: TimeRange,            // [t0, t1]
    pub process_config_hash: u64,
    pub state_hash: Option<u64>,
    pub value: Value,
    pub embedding: Vec<f32>,          // for semantic cache
}
```

This enables temporal queries on cached stream data:

- "Do I have cached results for MyScaler between 14:00 and 14:05?"
- "What's the most recent checkpoint for MyOnlineModel before 15:00?"
- "How did this filter's output distribution change over the last 2 hours?"

## Graph Streaming

When a graph runs in streaming mode:

```rust
impl Graph {
    pub async fn process_stream(
        &self,
        stream: impl Stream<Item = Tensor>,
    ) -> impl Stream<Item = Result<Tensor>> {
        stream.map(|chunk| {
            let mut current = chunk;
            for (filter, state, mode) in &self.fitted_filters {
                match mode {
                    StreamMode::FixedState => {
                        let key = CacheKey::for_chunk(filter, state, &current);
                        current = match self.cache.get(&key).await? {
                            Some(cached) => cached,
                            None => {
                                let result = filter.forward(&current, state)?;
                                self.cache.put(&key, &result).await?;
                                result
                            }
                        };
                    }
                    StreamMode::Evolving { .. } => {
                        current = filter.step(&current, state)?;
                        // checkpoint logic handled by runtime
                    }
                    StreamMode::Barrier => {
                        unreachable!("Compiler inserts materialization before barriers")
                    }
                }
            }
            Ok(current)
        })
    }
}
```

## Backpressure and Flow Control

Soma streams support backpressure natively via Rust's async Stream trait:

- Fast producer + slow consumer: chunks buffer up to a configurable limit
- When the buffer is full, the producer is suspended (async await)
- Each filter in the pipeline processes at its own speed
- Parallel branches have independent backpressure
