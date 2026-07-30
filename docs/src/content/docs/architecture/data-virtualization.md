---
title: Data Virtualization
description: How Soma treats data as virtual references materialized on demand.
---

## Concept

In Soma, data is not treated as a static entity. Instead, every intermediate value is a **virtual reference** -- a pointer to a result that *can* be materialized but hasn't been yet. This is analogous to how Denodo virtualizes SQL data sources, but applied to computation.

```
Traditional pipeline:
  Input → compute A → store result → compute B → store result → done
  (every intermediate result is materialized immediately)

Soma pipeline:
  Input → ref(A) → ref(B) → ref(C) → ... → materialize(what you need)
  (only the final or requested result is materialized)
```

## VirtualValue

The core type that enables this:

```rust
pub enum VirtualValue {
    /// Already computed and in memory
    Materialized(Tensor),

    /// Stored in the cache K/V store, loadable on demand
    Cached {
        key: CacheKey,
        store: StoreRef,
        schema: Schema,    // shape and dtype, available without loading
    },

    /// Not computed yet. Carries the "recipe" to produce it.
    Deferred {
        producer: NodeId,
        inputs: Vec<VirtualValue>,
        cache_key: CacheKey,
    },

    /// Materializes incrementally as chunks arrive
    Stream {
        source: StreamSource,
        buffer: VecDeque<Tensor>,
        state: Option<CacheKey>,  // checkpoint reference
    },
}
```

### Querying Without Materializing

You can inspect a VirtualValue without triggering computation:

```rust
impl VirtualValue {
    /// Shape, dtype, dimensions -- without loading data
    pub fn schema(&self) -> Schema { .. }

    /// Is it computed? Cached? Deferred? Streaming?
    pub fn status(&self) -> ValueStatus { .. }

    /// Estimated cost to materialize (time, compute)
    pub fn estimated_cost(&self) -> Cost { .. }

    /// Actually load/compute the data
    pub async fn materialize(&self, runtime: &Runtime) -> Result<Tensor> { .. }
}
```

### Materialization is Recursive and Cached

When you call `materialize()` on a `Deferred` value:

1. Check if the result exists in cache (by `cache_key`)
2. If cached, load it (promoting from cold to hot tiers)
3. If not cached, materialize all inputs recursively
4. Execute the producer filter
5. Store the result in cache
6. Return the materialized tensor

This means the system computes exactly what's needed and nothing more.

## The CacheStore as K/V

All cached data lives in a unified K/V store with tiered access:

```
┌──────────────────────────────────────────┐
│            CacheStore (K/V)              │
│                                          │
│   get(key) searches in order:            │
│                                          │
│   1. Memory     HashMap         <1ms     │
│      ↓ miss                              │
│   2. Local      FsActionStore  ~1ms     │
│      ↓ miss                              │
│   3. Remote     S3/shared       ~50ms    │
│      ↓ miss                              │
│   4. Not found  → must compute           │
│                                          │
│   put(key, value) writes by policy:      │
│   - Hot data → memory + local            │
│   - Evicted  → local only                │
│   - Shared   → remote (for workers)      │
└──────────────────────────────────────────┘
```

### Cache Interface

```rust
#[async_trait]
pub trait CacheStore: Send + Sync {
    async fn get(&self, key: &CacheKey) -> Result<Option<Tensor>>;
    async fn put(&self, key: &CacheKey, value: &Tensor) -> Result<()>;
    async fn exists(&self, key: &CacheKey) -> Result<bool>;
    async fn remove(&self, key: &CacheKey) -> Result<()>;
    async fn metadata(&self, key: &CacheKey) -> Result<Option<EntryMeta>>;
}
```

### Entry Metadata

Each cached entry carries metadata that enables cost estimation and debugging without loading the actual data:

```rust
pub struct EntryMeta {
    pub key: CacheKey,
    pub schema: Schema,           // dimensions, dtype
    pub size_bytes: u64,
    pub created_at: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
    pub ttl: Option<Duration>,
    pub origin: Origin,           // who produced it
}

pub enum Origin {
    Computed { node_id: NodeId, run_id: RunId },
    Ingested { source: String },
    Streamed { window: TimeRange },
}
```

## Comparison with Denodo

| Aspect | Denodo | Soma |
|---|---|---|
| **What's virtualized** | SQL data sources | Computation results |
| **Cache layer** | Query result cache | Content-addressable K/V |
| **Virtualization unit** | Table / View | `VirtualValue` (any tensor, dataframe, JSON) |
| **Materialization trigger** | SQL query | `.materialize()` call or pipeline execution |
| **Query optimizer** | SQL query planner | Soma compiler (cache-aware plan) |
| **Tiered storage** | Not built-in | Memory LRU → filesystem CAS |
| **Identity** | Table name / query hash | `hash(filter_config + input_hash)` -- content-addressable |
| **Cascade invalidation** | Manual refresh | Automatic: if input changes, downstream keys change |

## How the Compiler Uses Virtualization

At execution time each node's output is a VirtualValue, and the
**executor** resolves reuse per node with the materialized input in hand:

```
Graph: [A] → [B] → [C]

Executor resolves (runtime):
  A: key = hash(config_A + state + input)   → cache HIT  → skip, load
  B: key = hash(config_B + state + A.out)   → miss       → execute
  C: key = hash(config_C + state + B.out)   → miss       → execute

Only B and C actually run. A's result is loaded from the persistent cache.
```

Keys derive from input *content*, so an upstream re-run that produces
identical bytes leaves downstream keys unchanged (early cutoff).
