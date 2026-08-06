---
title: Streaming
description: Chunked execution through the same primitives as everything else — one execution site, stream included.
---

## Batch vs Stream Unification

Soma eliminates the traditional separation between batch and streaming
processing. The same filter definition works in both modes; the
difference is in **how data arrives** and **how state is managed**.

```
Batch:   all data available upfront → fit once, forward once
Stream:  data arrives in chunks     → state may evolve, forward per chunk
```

The unification is not just at the API surface. Every chunk of every
node runs through the same three primitives `run_node` composes for
the topological walk: `output_key` (the `cacheable && deterministic`
guard and the one key derivation, salted with the run seed),
`compute_node` (panic containment around the only filter-vs-step
match), and `store_output` (provenance on every cache write). The
stream driver (`StreamRun`, in `soma-runtime/src/execution/stream.rs`)
owns only what is genuinely streaming's: chunk flow per `StreamMode`,
the state carried between chunks, barrier buffers and their flush, and
the per-node event bracket.

A practical consequence worth knowing: for a `FixedState` filter, a
single-chunk stream and a plain forward of the same input share **one
cache line** — the second path is a hit on the first path's entry.

## What streams

Streaming executes a **single linear chain of filters**. `compile_stream`
validates and refuses, by name, anything the chunk loop cannot honour:

- a node with more than one predecessor or successor (a diamond used to
  be silently executed as a chain — the wrong answer);
- a step (the effect journal keys by `(run, node, turn)`, so chunk 2
  would replay chunk 1's effects — there is no defensible semantics);
- `chunk_size == 0`, and any node the registry does not know.

A stream plan also refuses to run in fit mode: streaming has no
training semantics. Fit the graph first, then stream the forward.

From Python the entry is the same `forward` as always —
`g.forward(x, stream=True, chunk_size=...)` — and it is literally the
same code path as the plain local forward (driver, transport, resume
semantics and output resolution included); only the compiler entry
differs.

## Stream Modes

Each filter declares how it behaves when processing a stream. The enum
is deliberately exhaustive — a stream driver must decide chunk flow for
every mode, and a wildcard arm there once silently treated unknown
modes as `FixedState`:

```rust
pub enum StreamMode {
    /// State is fixed (pre-trained). Each chunk processed independently.
    /// Example: scaler with pre-computed mean/std.
    FixedState,

    /// State evolves with each chunk (online learning): the forward's
    /// output value doubles as the next chunk's state.
    Evolving,

    /// Must see all data before producing output.
    /// Example: PCA, global sort, SVD.
    Barrier,
}
```

### FixedState (most common)

The filter has been trained (via `fit()`) and its state is frozen. Each
chunk is processed independently with the same state, and — because the
key derivation is `run_node`'s — cached independently:

```
Key   = hash(config + state + chunk), salted with the run seed
Value = transformed chunk
```

Identical chunks produce cache hits regardless of their position in the
stream, and different seeds never share a line. Filters that declare
`cacheable: false` or `deterministic: false` are not cached per chunk,
by the same guard the batch path applies.

### Evolving

The filter's state changes with each chunk: **the output of chunk N is
the state for chunk N+1**. This conflation is the current contract —
`RunningMean`-style filters return the value that is both their output
and their carried state. Separating the two needs a
`step(chunk, state) -> (out, state)` API on filters, which is future
work, not a streaming-internal detail.

```
chunk_1 → forward(chunk_1, state_0) → out_1  (= state_1)
chunk_2 → forward(chunk_2, out_1)   → out_2  (= state_2)
...
```

### Barrier

The filter cannot operate on partial data. Chunks accumulate in the
driver's buffer; nodes downstream of the barrier see nothing until the
flush, which materializes the buffer (tensors concatenated along the
first dimension) and runs the barrier node — and then the rest of the
chain — through the same primitives as any chunk. The flush leg is
cached and reported like any other execution.

```
Graph: [Scaler] → [PCA(Barrier)] → [Classifier]

chunks flow through Scaler, buffer at PCA;
flush: materialize → PCA → Classifier, one pass over the whole set.
```

## Events

A stream run reads back like any other run:

- one `NodeStarted` when the first chunk reaches a node;
- one `NodeCompleted` per started node after the flush, whose summary
  aggregates the chunk work: `stream: 12 chunks, 9 hits, 3 misses`;
- a real `NodeFailed` naming the failing chunk (`chunk 7: ...`) — the
  upstream nodes' spans stay open, which means exactly what an open
  span means everywhere else: the run died mid-node.

Per-chunk cache hit/miss events are deliberately **not** emitted —
hundreds of standalone spans would drown the reader; the counts travel
in the completion summary. (A per-chunk `NodeStarted` under made-up ids
like `model#chunk_3` was tried once and reverted.)

## Memory

Outputs are concatenated incrementally: each chunk's data is folded
into the result and dropped, keeping peak memory proportional to the
final output rather than `O(n_chunks × chunk_size)`. Pinned by the
tests in `soma-runtime/tests/memory_usage.rs`.

## Remote streaming

The worker's two remote entry points drive the **same `StreamRun`** the
local path uses — the driver and its execution context are held alive
between WebSocket messages (`StreamBegin` builds the session,
`ChunkData` advances it, `StreamEnd` flushes and closes the event
brackets), and the DataStore auto-stream path (inputs above 1024 rows)
loops over `get_rows` into the same driver, concatenating the output
incrementally. A node the worker cannot resolve fails the plan by name.
The client compiles remote streams with `compile_stream`, so a diamond
or a step is refused before anything crosses the wire.

`SerializedPlan` carries the run's `seed`, and the worker folds it into
every cache key — remote runs, streamed or not, no longer share cache
lines across a sweep's seeds. Plans from older senders simply arrive
unseeded.

There is no checkpoint/recovery mechanism and no async backpressure:
earlier versions of this page described both, but neither existed —
checkpoints were written under a colliding key that nothing ever read,
and the parameter that configured them (`checkpoint_every`) was only
ever constructed with a hardcoded literal. Both were removed rather
than kept as promises.
