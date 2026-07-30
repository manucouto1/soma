---
title: TDD Strategy
description: Test-driven development approach for Soma.
---

## Philosophy

Soma follows **Test-Driven Development (TDD)** with the Red-Green-Refactor cycle:

1. **Red**: Write a failing test that defines the expected behavior
2. **Green**: Write the minimum code to make it pass
3. **Refactor**: Clean up without changing behavior

Every public API, every trait implementation, and every compiler behavior is driven by tests first.

## Test Levels

### Unit Tests (per crate)

Each crate has unit tests co-located with the source code:

```
soma-core/src/
├── value.rs          # + #[cfg(test)] mod tests { .. }
├── graph.rs          # + #[cfg(test)] mod tests { .. }
├── cache.rs          # + #[cfg(test)] mod tests { .. }
└── ...
```

Unit tests cover:

- **soma-core**: Value conversions, cache key computation, search space validation, graph construction, schema compatibility
- **soma-compiler**: Topological sort, pattern detection, validation, gradient flow analysis, worker scheduling
- **soma-runtime**: Executor correctness (sequence, parallel, loop, branch), event emission, cache tier promotion, sampler distributions, pruner decisions
- **soma-python**: PyO3 binding correctness, Python Filter class behavior

### Integration Tests (cross-crate)

Integration tests live in `tests/` at the workspace root or in each crate's `tests/` directory:

```
soma-compiler/tests/
├── compile_linear_pipeline.rs
├── compile_parallel_branches.rs
├── compile_with_cache.rs
└── gradient_flow_analysis.rs

soma-runtime/tests/
├── execute_simple_pipeline.rs
├── execute_with_caching.rs
├── execute_parallel.rs
├── study_bayesian.rs
└── stream_processing.rs
```

Integration tests verify end-to-end behavior: compile a graph, execute the plan, verify results and events.

### Property-Based Tests

For algorithmic correctness, use `proptest` or `quickcheck`:

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn cache_key_deterministic(
        scale in 0.1f64..100.0,
        method in prop::sample::select(vec!["standard", "robust"]),
    ) {
        let filter = MyScaler { scale, method: method.to_string() };
        let key1 = filter.config_hash();
        let key2 = filter.config_hash();
        assert_eq!(key1, key2, "Same config must produce same hash");
    }

    #[test]
    fn cache_key_sensitive_to_config(
        scale1 in 0.1f64..100.0,
        scale2 in 0.1f64..100.0,
    ) {
        prop_assume!(scale1 != scale2);
        let f1 = MyScaler { scale: scale1, ..Default::default() };
        let f2 = MyScaler { scale: scale2, ..Default::default() };
        assert_ne!(f1.config_hash(), f2.config_hash(),
            "Different configs must produce different hashes");
    }
}
```

## Test Fixtures

### Mock Filters

A set of test filters used across the test suite:

```rust
/// A simple passthrough filter (stateless, differentiable)
#[derive(SomaFilter)]
#[soma(kind = "Stateless", cacheable = true, differentiable = true)]
pub struct Identity;

impl Filter for Identity {
    type State = ();
    fn fit(&self, _x: &Tensor, _y: Option<&Tensor>) -> Result<()> { Ok(()) }
    fn forward(&self, x: &Tensor, _: &()) -> Result<Tensor> { Ok(x.clone()) }
}

/// A filter that doubles its input (stateless, differentiable)
#[derive(SomaFilter)]
#[soma(kind = "Stateless", cacheable = true, differentiable = true)]
pub struct Doubler;

/// A filter that always fails (for error handling tests)
#[derive(SomaFilter)]
#[soma(kind = "Stateless")]
pub struct FailFilter;

/// A filter that counts how many times it's called (for caching tests)
#[derive(SomaFilter)]
#[soma(kind = "Trainable", cacheable = true)]
pub struct CountingFilter {
    pub call_count: Arc<AtomicUsize>,
}
```

### Mock Cache Store

```rust
pub struct MockCacheStore {
    store: HashMap<CacheKey, Vec<u8>>,
    get_count: AtomicUsize,
    put_count: AtomicUsize,
}

impl MockCacheStore {
    pub fn get_count(&self) -> usize { self.get_count.load(Ordering::Relaxed) }
    pub fn put_count(&self) -> usize { self.put_count.load(Ordering::Relaxed) }
}
```

## TDD Examples

### Example: Implementing Cache Key Computation

```rust
// Step 1: RED - Write the failing test
#[test]
fn filter_config_hash_includes_public_fields() {
    let f1 = MyScaler { scale: 1.0, method: "standard".into() };
    let f2 = MyScaler { scale: 2.0, method: "standard".into() };

    assert_ne!(f1.config_hash(), f2.config_hash());
}

#[test]
fn filter_config_hash_excludes_skip_hash_fields() {
    let f1 = MyScaler { scale: 1.0, method: "standard".into(), verbose: true };
    let f2 = MyScaler { scale: 1.0, method: "standard".into(), verbose: false };

    assert_eq!(f1.config_hash(), f2.config_hash());
}

// Step 2: GREEN - Implement config_hash()
// Step 3: REFACTOR - Clean up
```

### Example: Implementing Cascade Invalidation

```rust
#[test]
fn changing_middle_filter_invalidates_downstream() {
    let cache = MockCacheStore::new();
    let graph = pipeline_graph(vec![
        scaler_filter(1.0),
        pca_filter(50),      // ← will change this
        svm_filter(1.0),
    ]);

    // First run: everything executes
    let plan1 = compile(&graph, &cache);
    assert!(matches!(plan1, Sequence([Execute, Execute, Execute])));

    // Execute to populate cache
    execute(&plan1, &cache).await;

    // Change PCA config
    let graph2 = pipeline_graph(vec![
        scaler_filter(1.0),
        pca_filter(100),     // ← changed
        svm_filter(1.0),
    ]);

    // Second compile: scaler cached, rest re-executes
    let plan2 = compile(&graph2, &cache);
    assert!(matches!(plan2, Sequence([Cached, Execute, Execute])));
}
```

## Coverage

Target: **80%+ line coverage** for soma-core and soma-compiler, **70%+** for soma-runtime.

```bash
# Run with coverage
cargo tarpaulin --workspace --out html

# View report
open tarpaulin-report.html
```

## What to Test vs What Not to Test

### Always Test

- Public trait implementations
- Cache key computation (determinism, sensitivity)
- Compiler output for known graph patterns
- Event emission sequence for known execution plans
- Error conditions and edge cases
- Sampler distributions (statistical properties)
- Pruner decisions at boundary conditions

### Don't Test

- Private helper functions (tested through public API)
- Serde serialization (trust the derive)
- Third-party library behavior
- Exact log output or formatting
