---
title: DataStore
description: Abstraction for moving data between workers — Local, S3, Cached, Stream
---

# DataStore

The DataStore trait abstracts WHERE data lives from HOW it's processed. Workers exchange `DataRef` references instead of raw data.

## DataRef

A reference to data that may live in different places:

| Variant | Description |
|---------|-------------|
| `Local { path }` | Data in local filesystem |
| `S3 { bucket, key, region }` | Data in S3-compatible storage |
| `Cached { cache_key }` | Data in Soma cache (content-addressable) |
| `Stream { endpoint, format }` | Real-time stream endpoint |
| `Inline { value }` | Small values embedded in the reference |

## DataStore Trait

```rust
pub trait DataStore: Send + Sync {
    fn put(&self, key: &CacheKey, data: &Value) -> Result<DataRef>;
    fn get(&self, data_ref: &DataRef) -> Result<Value>;
    fn exists(&self, data_ref: &DataRef) -> Result<bool>;
    fn remove(&self, data_ref: &DataRef) -> Result<()>;
    fn config(&self) -> &StorageConfig;
}
```

## Implementations

### LocalDataStore
Stores data as JSON files in a local directory. Used in development and single-machine setups.

### S3DataStore (feature: `s3`)
Stores data in S3-compatible storage (AWS S3, MinIO). Includes a local cache layer for frequently accessed data. Enable with:
```toml
soma-core = { features = ["s3"] }
```

## StreamCache

Optimizes inference pipelines by caching:
1. **Filter states** — loaded once from training, reused on every stream chunk
2. **Chunk results** — keyed by `hash(config + state + chunk_data)`

If the same chunk passes through the same filter with the same trained state, the result is returned from cache instantly (zero computation).
