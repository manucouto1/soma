---
title: Architecture Review
description: A record of the first architecture review and the reasoning behind what changed.
---

This document captures the findings from a comprehensive architecture
review performed after the initial implementation of all crates. It is
kept for the reasoning, not as a description of the tree.

## The tree as it was

:::caution[Historical document]
Everything in this section describes the workspace at the time of the
review, including the test counts and the crate list — there are thirteen
crates now, and `soma-store` and `soma-llm` did not exist. Several
findings below have since been addressed: `Pipeline` was removed (`Graph`
is the only user-facing API), cache resolution moved from compile time to
runtime, filters and steps were unified under one registry and one
execution site, and errors became typed at the crate edges.

For decisions taken since, and what each was chosen *over*, see
[Design Decisions](/soma/design/decisions/). For the current tree, see
[Crates](/soma/architecture/crates/).

Every **Status** line below was re-checked against the code on 2026-08-05,
because several of them had drifted in the direction that matters least to
a reader and most to an evaluator: they described as missing things that
had shipped.
:::

```
907 tests (582 Rust + 325 Python), all passing, clippy clean.

soma-core        Foundation types, tracking schema, graph rendering
soma-macros      #[derive(SomaFilter)] proc macro
soma-compiler    Graph → ExecutionPlan compiler, scheduler
soma-runtime     Executor, caches, samplers, StudyRunner, run tracking
soma-coordinator Worker registry, routing, heartbeat monitoring
soma-worker      Worker protocol and daemon
soma-memory      KnowledgeBase (+ ChronosVector)
soma-agent       Research agent loop
soma-mcp         MCP server
soma-python      PyO3 bindings
soma             Facade crate re-exporting the workspace
```

## Identified Issues

### Priority 1: Critical Design Issues

#### 1.1 Filter Trait Mixes Concerns

**Current**: The `Filter` trait owns both computation (`fit`/`forward`) and caching (`config_hash`).

```rust
pub trait Filter: Send + Sync {
    fn config_hash(&self) -> CacheKey;  // caching concern
    fn fit(&self, x: &Value, ...) -> Result<Value>;  // computation
    fn forward(&self, x: &Value, state: &Value) -> Result<Value>;  // computation
    fn meta(&self) -> FilterMeta;  // metadata
}
```

**Problem**: Cache key computation is not a computation concern. It forces all filter implementations to know about `CacheKey`, even if they don't use caching.

**Proposed Fix**: Split into composable traits:

```rust
trait Compute: Send + Sync {
    fn fit(&self, x: &Value, y: Option<&Value>) -> Result<Value>;
    fn forward(&self, x: &Value, state: &Value) -> Result<Value>;
}

trait Describable {
    fn meta(&self) -> FilterMeta;
}

trait Cacheable {
    fn config_hash(&self) -> CacheKey;
}

// Filter = Compute + Describable + Cacheable (default blanket impl)
trait Filter: Compute + Describable + Cacheable {}
impl<T: Compute + Describable + Cacheable> Filter for T {}
```

**Impact**: Medium. Requires changing trait bounds everywhere but improves extensibility.

**Status**: Deferred to next iteration. Current monolithic trait works for MVP.

---

#### 1.2 SomaError Is Too Broad

**Current**: One error enum for all crates with a catch-all `Other(String)`.

```rust
pub enum SomaError {
    RequiresLabels,
    Cache(String),         // vague
    Compilation(String),   // vague
    Execution { .. },
    Pruned { .. },         // control flow as error!
    Other(String),         // catch-all
    ...
}
```

**Problems**:
- `Other(String)` used in 12+ places as a dumping ground
- `Pruned` is control flow disguised as an error
- No error context chain

**Proposed Fix**:

```rust
// Control flow separated from errors
enum TrialOutcome {
    Completed(Vec<MetricRecord>),
    Pruned { step: usize, reason: String },
}

// Per-concern errors
enum CacheError { NotFound(CacheKey), Corrupt(String), Io(io::Error) }
enum CompileError { CycleDetected, SchemaMismatch { .. }, NodeNotFound(String) }
enum RuntimeError { FilterFailed { node_id: String, source: Box<dyn Error> }, ... }
```

**Impact**: High. Touches every error site. Should be done before 1.0.

**Status**: Half done. `TrialOutcome` exists (`soma-runtime/src/executors/study.rs`) and separates completion from pruning, and `soma-llm` and `soma-worker` carry their own error types. `SomaError::Pruned` still exists alongside it, so the smell is narrowed, not removed.

---

#### 1.3 Stringly-Typed Node IDs

**Current**: `pub type NodeId = String;` used everywhere.

**Problem**: No compile-time guarantee that a node ID exists in the graph. Typos cause runtime errors.

**Proposed Fix**:

```rust
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(String);

impl NodeId {
    pub fn new(id: impl Into<String>) -> Self { Self(id.into()) }
    pub fn as_str(&self) -> &str { &self.0 }
}
```

**Impact**: Low-medium. Mostly mechanical replacement. Improves safety.

**Status**: Deferred. String-based IDs work for current scale.

---

### Priority 2: Scalability Issues

#### 2.1 Graph Uses Linear Scans

**Current**: `predecessors()` and `successors()` iterate all edges: O(edges) per call.

**Problem**: For graphs with 10k+ nodes, this is quadratic in the compiler.

**Proposed Fix**: Maintain adjacency lists.

```rust
pub struct Graph {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    // Precomputed indices
    preds: HashMap<String, Vec<String>>,
    succs: HashMap<String, Vec<String>>,
}
```

**Status**: Deferred. Current graphs are small (<100 nodes).

---

#### 2.2 MemoryCache Has No Eviction

**Current**: `max_bytes` field exists but is never enforced (`#[expect(dead_code)]`).

**Problem**: Cache grows unbounded. Long-running studies will OOM.

**Proposed Fix**: Implement LRU eviction.

**Status**: Done. `soma-runtime/src/cache/memory.rs` enforces `max_bytes` with `evict_until_fits`, evicting least-recently-accessed entries.

---

#### 2.3 GridSampler Builds Full Cartesian Product

**Current**: On first call, builds all combinations in memory.

**Problem**: 10 dimensions with 10 points each = 10 billion combinations.

**Proposed Fix**: Lazy index-based generation.

```rust
fn sample(&self, space: &SearchSpace, trial_index: usize) -> Option<Params> {
    // Convert trial_index to multi-dimensional index
    // without building the full grid
}
```

**Status**: Done. `GridSampler` generates combinations from the trial index; it never materializes the product.

---

#### 2.4 Serial Trial Execution

**Current**: StudyRunner runs trials one at a time in a `while` loop.

**Problem**: Can't leverage multiple workers or CPU cores.

**Proposed Fix**: Async trial execution with worker pool.

**Status**: Deferred to Phase 3 (agents and workers).

---

### Priority 3: Design Smells

#### 3.1 soma-core Does Too Much

**Current**: soma-core owns 7 unrelated domains (graph, filter, cache, value, study, search, event).

**Problem**: Can't import Filter without pulling in Study, Event, SearchSpace.

**Proposed Fix**: Eventually split into:
- `soma-types` (Value, Schema, Error)
- `soma-graph` (Graph, Node, Edge)
- `soma-filter` (Filter trait, FilterMeta)
- `soma-study` (Study, Trial, SearchSpace)
- `soma-event` (Event, EventBus)

**Status**: Deferred. Single crate is simpler for now. Split when compile times become an issue.

---

#### 3.2 Pipeline Has Too Many Responsibilities

**Current**: Pipeline manages filter composition, state storage, caching, events, and fit status.

**Problem**: Adding features (streaming, distribution) will bloat the struct.

**Proposed Fix**: Separate concerns:
- `FilterChain` — composition only
- `FittedPipeline` — holds trained states
- Pipeline wraps both with caching and events

**Status**: Moot. `Pipeline` was removed — `Graph` is the only user-facing API, and the responsibilities listed here now sit in `GraphSession`, `NodeCatalog` and the cache.

---

#### 3.3 ExecutionPlan Variants Are Public

**Current**: All plan variants (Sequence, Parallel, Execute, Cached, Remote, Loop, Branch, Empty) are public.

**Problem**: Runtime consumers must exhaustively match, making extension breaking.

**Proposed Fix**: Keep enum public but add `#[non_exhaustive]` attribute.

**Status**: Done, and deliberately not everywhere: `ExecutionPlan`, `Effect`, `Value` and `Event` are `#[non_exhaustive]`; `NodeOutcome` and `Transition` are not, because every consumer must decide over them and a wildcard arm there is a silent wrong answer.

---

### Priority 4: Missing Abstractions

| Abstraction | Purpose | Status |
|---|---|---|
| `DataFlow` trait | Abstract input resolution from graph topology | `GraphInfo` resolves by predecessors, not by "last executed" |
| `CachePolicy` | LRU, TTL, size-based eviction | LRU with `max_bytes` (memory), plus the tiered store's `soma cache gc`/`pin` |
| `StreamingFilter` | Chunk processing with stream modes | Done. `StreamRun` composes `run_node`'s primitives, locally and on the worker |
| `Scheduler` | Distribute trials/plans across workers | Done. `soma-compiler/src/scheduler.rs` produces a `DistributionPlan` |
| `MetricsCollector` | Pluggable observability backends | EventBus, plus the run-dir readers behind `soma.runs()` and `soma report`. A Python-implementable `EventSink` is still open |
| `DataSchema` | Type-safe input/output validation | Done. `Schema` (dtype + shape) is checked between connected nodes at `compile()` |

---

## Test Coverage Gaps

:::note[Superseded]
This section describes coverage at 907 tests. The suite is now 1030 Rust
`#[test]` functions plus 699 Python tests, and most of the "missing"
column below has since been written — cache invalidation, tiered
promotion, Bayesian sampling and pruning, filter panics, concurrency,
and end-to-end workflows all have tests today. Property tests live in
`soma-core/tests/proptests.rs`. The table is kept for the shape of the
reasoning, not as a gap list.
:::

### Currently Tested vs Missing

| Category | Tested | Missing |
|---|---|---|
| **Pipeline** | Linear sequential | Nested, branching, conditional, empty, dependent state |
| **Caching** | Basic put/get/exists | Invalidation on data change, cross-run, concurrent, tiered promotion |
| **Optimization** | Grid, Random, basic objectives | Hyperband, Bayesian, pruning, resumption, multi-objective |
| **Error handling** | Predict-before-fit, missing filter | Filter panic, type mismatch, corrupt cache, invalid config |
| **Concurrency** | None | Shared cache, parallel events, interleaving |
| **ML edge cases** | None | Empty/single sample, NaN/Inf, high-dimensional |
| **Integration** | Individual components | End-to-end workflows, study+pipeline, cache warm-up |

### Highest Priority Missing Tests

1. **Full end-to-end workflow**: define filters → build pipeline → fit → predict → cache hit on re-run
2. **Cache invalidation**: same pipeline, different data → different results
3. **Study + Pipeline integration**: study samples params → pipeline executes with those params
4. **Multi-objective optimization**: two objectives with different directions
5. **Error resilience**: filter that fails mid-execution, study continues with remaining trials
6. **Empty/single sample**: boundary conditions in real ML workflows

---

## Improvement Roadmap

### Before 1.0

1. ~~Apply `#[non_exhaustive]` to public enums~~ — done, and deliberately
   not to the control-flow enums
2. ~~Implement LRU eviction in MemoryCache~~ — done
3. ~~Fix GridSampler for large spaces (lazy generation)~~ — done
4. ~~Separate `Pruned` from `SomaError` into `TrialOutcome`~~ — `TrialOutcome`
   exists; the `SomaError::Pruned` variant still does too
5. ~~Add integration tests for full workflows~~ — done
6. Add `#[must_use]` annotations where appropriate — open

What actually blocks a 1.0, as of 2026-08-05, is none of the above: it is
that publishing has never been verified end to end. See
[Roadmap](/soma/development/roadmap/).

### Next Iteration

7. Split Filter trait into Compute + Describable + Cacheable
8. Per-crate error types
9. Typed NodeId (newtype over String)
10. Graph adjacency list precomputation
11. Async trial execution in StudyRunner

### Future

12. Split soma-core into focused crates
13. Pluggable hash algorithm for CacheKey
14. Streaming execution with checkpoint support
15. Worker scheduler and task queue
