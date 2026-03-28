---
title: Caching System
description: Content-addressable caching with cascade invalidation, resolved at compile time.
---

## Overview

Soma uses **content-addressable caching** inspired by LabChain. Every computation is identified by a deterministic hash of its configuration and inputs. If the same computation has been done before, the result is reused automatically.

What makes Soma's caching unique is that it is **resolved at compile time**. The compiler checks cache availability and produces an execution plan that already knows what to skip.

## Cache Keys

A cache key is a SHA-256 hash that uniquely identifies a computation:

```rust
pub struct CacheKey(pub [u8; 32]);

impl CacheKey {
    /// Hash for a filter's trained state
    pub fn for_state(filter: &dyn Filter, train_data_hash: &DataHash) -> CacheKey {
        sha256(&[filter.config_hash(), train_data_hash])
    }

    /// Hash for a filter's output
    pub fn for_output(
        filter: &dyn Filter,
        state_hash: &CacheKey,
        input_data_hash: &DataHash,
    ) -> CacheKey {
        sha256(&[filter.config_hash(), state_hash, input_data_hash])
    }
}
```

### What Contributes to the Hash

| Component | Contributes to State Key | Contributes to Output Key |
|---|---|---|
| Filter class name | Yes | Yes |
| Public constructor params | Yes | Yes |
| `#[soma(skip_hash)]` fields | No | No |
| Training data content | Yes | No |
| Learned state | No (it IS the state) | Yes |
| Input data content | No | Yes |

## Two Independent Caches

Each filter has two cacheable outputs, stored independently:

### State Cache

Stores the result of `fit()` -- the learned parameters, weights, or statistics.

```
Key   = hash(filter_config + training_data_hash)
Value = serialized State (e.g., ScalerState { mean, std })
```

**When to use**: When you change test data but keep the same filter and training data. The model doesn't need to be retrained.

### Output Cache

Stores the result of `forward()` -- the transformed data.

```
Key   = hash(filter_config + state_hash + input_data_hash)
Value = serialized Tensor (the output)
```

**When to use**: When you run the same filter with the same state on the same input data. No computation needed at all.

## Cascade Invalidation

When a filter's configuration changes, everything downstream is automatically invalidated because the cache keys change:

```
Pipeline: [A] → [B] → [C]

Scenario: Change B's configuration

A.state_key  = hash(A.config + train_data)     → unchanged → HIT
A.output_key = hash(A.config + A.state + data)  → unchanged → HIT
B.state_key  = hash(B'.config + A.output)        → CHANGED   → MISS
B.output_key = hash(B'.config + B'.state + data) → CHANGED   → MISS
C.state_key  = hash(C.config + B'.output)        → CHANGED   → MISS (input changed)
C.output_key = hash(C.config + C.state + data)   → CHANGED   → MISS

Compiled plan: Cached(A) → Execute(B') → Execute(C)
```

This happens automatically because keys are computed recursively. No manual invalidation needed.

## Compile-Time Resolution

The compiler resolves caching **before any filter executes**:

```rust
impl Compiler {
    fn resolve_cache(&self, plan: &ExecutionPlan, cache: &dyn CacheStore) -> ExecutionPlan {
        match plan {
            ExecutionPlan::Sequence(steps) => {
                let mut resolved = vec![];
                let mut upstream_changed = false;

                for step in steps {
                    if let ExecutionPlan::Execute { id, filter } = step {
                        let key = filter.cache_key(&self.input_hashes[id]);

                        if !upstream_changed && cache.exists(&key) {
                            resolved.push(ExecutionPlan::Cached { id, key });
                        } else {
                            upstream_changed = true; // everything downstream must re-execute
                            resolved.push(step.clone());
                        }
                    }
                }
                ExecutionPlan::Sequence(resolved)
            }
            ExecutionPlan::Parallel(branches) => {
                // Each branch resolved independently
                ExecutionPlan::Parallel(
                    branches.iter().map(|b| self.resolve_cache(b, cache)).collect()
                )
            }
            _ => plan.clone(),
        }
    }
}
```

## Tiered Storage

The cache store is a K/V interface with multiple tiers:

```rust
pub struct TieredCache {
    memory: MemoryCache,         // HashMap, <1ms
    local: RocksDbCache,         // on-disk, ~1ms
    remote: Option<S3Cache>,     // shared across workers, ~50ms
    policy: EvictionPolicy,
}
```

### Promotion and Eviction

```
GET flow:
  Memory HIT  → return immediately
  Memory MISS → check Local
    Local HIT  → promote to Memory, return
    Local MISS → check Remote
      Remote HIT  → promote to Local + Memory, return
      Remote MISS → cache miss, must compute

PUT flow:
  Always write to Memory + Local
  Optionally write to Remote (for sharing with workers)

EVICTION:
  Memory full → evict LRU entries (still in Local)
  Local full  → evict LRU entries (still in Remote if shared)
  Remote      → TTL-based or size-based policies
```

### Configuration

```rust
pub struct CacheConfig {
    pub memory_max_bytes: usize,       // default: 1GB
    pub local_path: PathBuf,           // default: ~/.soma/cache
    pub local_max_bytes: usize,        // default: 50GB
    pub remote: Option<RemoteConfig>,  // S3 bucket config
    pub default_ttl: Option<Duration>, // default: None (forever)
}

pub enum EvictionPolicy {
    LRU,
    LFU,
    SizeBased { max_entry_bytes: usize },
    TTL { default: Duration },
}
```

## Cache-Aware Events

The event system reports cache activity:

```rust
Event::NodeCacheHit {
    run_id: RunId,
    node_id: NodeId,
    key: CacheKey,
    tier: CacheTier,      // Memory, Local, or Remote
    load_time: Duration,  // how long it took to load
}
```

This enables monitoring dashboards to show cache hit rates, tier distribution, and load times.
