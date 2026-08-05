---
title: Implementation Roadmap
description: MVP phases, priorities, and implementation order.
---

:::caution[Written before implementation]
This roadmap was drafted at the start of the project. Phases 1 and 2 have
largely shipped — the crates, the persistent cache, tracking, the
visualization layer and the experiment pool all exist — so treat the task
tables as a record of what was planned, not as a description of what is
left. [Crates](/soma/architecture/crates/) describes the tree as it is.
:::

## What is actually left

Re-derived from the code on 2026-08-05, because the phase tables below
answer a question nobody is asking any more.

### Blocking a release

**Publishing has never been verified end to end.** This is the one item
that stands between the workspace and a version a stranger can install.
`release.yml` publishes on a `v*` tag; until a tag runs green with the
crates and the wheels actually landing, "it works" is a claim about a
local checkout. Trusted publishing (OIDC) now replaces the stored tokens
on both crates.io and PyPI, which means the publisher must be configured
registry-side before the next tag.

### Public surface that exists but refuses

| What | Where | Behaviour |
|---|---|---|
| **The whole `TrainingStrategy` layer** | `soma-runtime/src/strategy.rs` | `impl StrategyExecutor for TrainingStrategy` **has no caller anywhere in the workspace**, and `soma-compiler/src/scheduler.rs` never mentions the strategy either. Setting one records it on the graph and changes nothing. `Local`, `DataParallel` and `Federated` are written and unreachable; `ModelParallel` and `PopulationBased` additionally return "not yet implemented". `PbtRunner` works but answers to a different trait (`PbtExecutor`), so connecting it is an adapter, not a rename |
| `run_pipeline`, `run_study` | `soma-mcp/src/context.rs` | Declared as MCP tools, not implemented: the server cannot load user code. Their own descriptions say so |
| Filter serialization over WS | `soma-worker/src/ws_transport.rs` | Sends an empty filter list where the catalog's would go |

The strategy layer is the largest single gap, and it is a wiring gap
rather than a blank page: the sharding, aggregation and federated round
loops exist. What is missing is the caller — something in `fit` that
looks at the graph's strategy and hands execution to it — plus the two
unwritten variants and a Python `set_strategy`.

### Deferred on purpose, with the seam in place

Documented where each belongs, not here:
[`soma ui`](/soma/design/visualization/) and the rest of the visualization
deferrals (fANOVA importances, `NodeProgress`/`ParetoUpdated` emitters,
a Python-implementable `EventSink`, parquet compaction), and the
[experiment-pool](/soma/design/experiment-pool/) ones (warm-starting a
study from the pool, dedup by cache key, ChronosVector as a real vector
index once an `Embedder` exists).

### Documentation

Notebooks 06–09 are in Spanish while the other eleven are in English.
Translating them is outstanding.


## Phase Overview

```
Phase 1: LabChain in Rust                    ← MVP
Phase 2: Distribution & Remote Execution
Phase 3: Memory, Knowledge Base & Agents
```

Each phase produces a **usable, releasable product**. Later phases build on earlier ones without rewriting.

## Phase 1: LabChain in Rust (MVP)

**Goal**: A functional replacement for LabChain, written in Rust, usable from Python. Graphs with caching, optimization, and events.

### 1.1 soma-core: Foundation Types

| Task | Description | Priority |
|---|---|---|
| `Value` enum | Tensor, Json, DataFrame, Bytes, Virtual | P0 |
| `Filter` trait | fit/forward lifecycle with associated State type | P0 |
| `FilterMeta` | kind, cacheable, differentiable, stream_mode | P0 |
| `Graph`, `Node`, `Edge` | Graph construction and validation | P0 |
| `CacheKey` | Content-addressable hash computation | P0 |
| `CacheStore` trait | K/V interface for cache backends | P0 |
| `Event` enum | All three levels (Run, Trial, Study) | P0 |
| `SearchDimension` | Float, Int, Categorical, Conditional | P0 |
| `SearchSpace` | Aggregation, merge, freeze | P0 |
| `Study`, `Trial` | Optimization types | P0 |
| `Schema` | Input/output type descriptions | P1 |
| `VirtualValue` | Lazy references (Materialized, Cached, Deferred) | P1 |
| `SomaError` | Error types | P0 |
| Derive macros | `#[derive(SomaFilter)]` | P1 |

### 1.2 soma-compiler: Graph to Plan

| Task | Description | Priority |
|---|---|---|
| Topological sort | Kahn's algorithm, cycle detection | P0 |
| Linear graph compilation | Sequence of Execute nodes | P0 |
| Cache resolution | Per-node, at runtime, with materialized input in hand | P0 |
| Cascade invalidation | Upstream change invalidates downstream | P0 |
| Parallel branch detection | Fork-join pattern recognition | P1 |
| Gradient flow analysis | Warn on non-differentiable interruptions | P1 |
| Schema validation | Type compatibility between filters | P1 |
| Loop compilation | Loop body extraction | P2 |
| Branch compilation | Conditional arms | P2 |
| Cost estimation | From cache metadata | P2 |

### 1.3 soma-runtime: Execution Engine

| Task | Description | Priority |
|---|---|---|
| Sequential executor | Walk Sequence plans | P0 |
| Event bus | Async broadcast, subscribe | P0 |
| Memory cache | In-memory HashMap | P0 |
| Local cache | Filesystem action store + BLAKE3 CAS | P0 |
| Tiered cache | Multi-level with promotion | P1 |
| Parallel executor | Tokio JoinSet for Parallel plans | P1 |
| Context | Store + event emitter + metric reporter | P0 |
| Graph struct | fit/forward with caching | P0 |
| Study runner | Sample → build → execute → record loop | P1 |
| Grid sampler | Exhaustive search | P1 |
| Random sampler | Random search | P1 |
| Bayesian sampler | TPE implementation | P2 |
| Median pruner | Median stopping rule | P2 |
| Hyperband | Successive halving | P2 |
| Stream driver | Chunk processing with modes, through run_node's primitives | Done |

### 1.4 soma-python: Python Bindings

| Task | Description | Priority |
|---|---|---|
| PyO3 module setup | maturin build, basic imports | P0 |
| Filter base class | Python class with search() descriptors | P0 |
| Graph class | fit/forward | P0 |
| Value wrappers | Tensor ↔ numpy, DataFrame ↔ polars | P0 |
| Study class | Run optimization from Python | P1 |
| Event subscription | Python callbacks for events | P1 |
| Search space display | Pretty-print search spaces | P1 |

### Phase 1 Deliverable

```python
from soma import Graph, Filter, Study, Bayesian, search

class MyScaler(Filter):
    scale: float = search(0.1, 10.0, scale="log")

    def fit(self, x, y=None):
        return {"mean": x.mean(0), "std": x.std(0)}

    def forward(self, x, state):
        return (x - state["mean"]) / state["std"] * self.scale

g = Graph.somatize(MyScaler(scale=2.0) >> MyClassifier(C=1.0))
g.fit(x_train, y_train)
result = g.forward(x_test)  # with automatic caching

study = Study(graph=g, strategy=Bayesian(n_trials=50))
study.run(x_train, y_train, x_val, y_val)
print(study.best_trial.params)
```

## Phase 2: Distribution & Remote Execution

**Goal**: Execute graphs on remote workers. Shared caching across a lab.

### 2.1 soma-worker

| Task | Description |
|---|---|
| Worker daemon | Register, heartbeat, receive plans |
| Protocol | Message types, serialization |
| Python loader | PyO3 dynamic loading of user filters |
| Capabilities | GPU detection, resource reporting |

### 2.2 Compiler Extensions

| Task | Description |
|---|---|
| Distribution planner | Assign nodes to local/remote targets |
| Remote plan wrapping | ExecutionPlan::Remote variant |
| Serialized plan | Full plan + filter serialization |

### 2.3 Runtime Extensions

| Task | Description |
|---|---|
| Remote cache | S3 backend for CacheStore |
| Event relay | WebSocket event streaming from workers |
| Plan coordinator | Schedule plans across workers |

### 2.4 Python Extensions

| Task | Description |
|---|---|
| `lab.connect()` | Connect to a Soma lab |
| `lab.run()` | Submit graphs for remote execution |
| `lab.workers()` | List available workers |

### Phase 2 Deliverable

```python
lab = soma.connect("https://my-lab.soma.dev")
lab.run(study, data=train_data)  # executes on remote workers
```

## Phase 3: Memory, Knowledge Base & Agents

**Goal**: Temporal experiment tracking and autonomous research agents.

### 3.1 soma-memory

| Task | Description |
|---|---|
| ChronosVector integration | Import as dependency or subcrate |
| ExperimentRecord | Indexing, embedding generation |
| Semantic search | Query by natural language |
| Trajectory analysis | Metric evolution over time |
| Change point detection | Breakthrough identification |
| Promising lines | Trend analysis and recommendations |

### 3.2 soma-agent

| Task | Description |
|---|---|
| Agent struct | Soul, skills, hands, memory |
| Research loop | Hypothesize → build → execute → analyze → iterate |
| Graph generation | LLM-driven graph construction |
| Report generation | Automatic documentation of findings |

### 3.3 Platform Integration

| Task | Description |
|---|---|
| Graph publishing | `lab.publish(graph)` |
| Graph editor integration | Graphs as platform nodes |
| Visual graph editor | Drag-and-drop filter composition |

### Phase 3 Deliverable

```python
from soma.agent import Researcher

agent = Researcher(lab=lab, plan="Investigate normalization for TS classification")
report = agent.investigate(max_iterations=20)

kb = lab.knowledge_base()
kb.promising_lines()
kb.trajectory("rocket_znorm", metric="f1")
```

## Implementation Order (Phase 1 Detail)

The recommended order for Phase 1, following TDD:

```
Week 1-2: soma-core types
  1. SomaError
  2. Value enum (without Tensor, just structure)
  3. CacheKey (hash computation)
  4. FilterMeta, FilterKind, StreamMode
  5. Filter trait
  6. Graph, Node, Edge
  7. Event enum
  8. SearchDimension, SearchSpace
  9. CacheStore trait
  10. Study, Trial, Objective

Week 3-4: soma-compiler
  11. Topological sort
  12. ExecutionPlan enum
  13. Linear graph compilation
  14. Cache key computation for graph
  15. Cache resolution (runtime, per node)
  16. Cascade invalidation
  17. Parallel branch detection

Week 5-6: soma-runtime
  18. Event bus
  19. Context
  20. Memory cache (HashMap)
  21. Sequential executor
  22. Graph (fit/forward)
  23. Local cache (FsActionStore + BLAKE3 CAS)
  24. Tiered cache
  25. Parallel executor

Week 7-8: soma-runtime optimization
  26. Grid sampler
  27. Random sampler
  28. Study runner
  29. Metric reporting + pruning
  30. Median pruner

Week 9-10: soma-python
  31. PyO3 module setup
  32. Filter base class
  33. Graph class
  34. Value wrappers (numpy interop)
  35. Study class
  36. search() descriptor

Week 11-12: Polish & release
  37. Derive macros (#[derive(SomaFilter)])
  38. Documentation site
  39. Examples and tutorials
  40. CI/CD pipeline
  41. Publish to crates.io + PyPI
```

## Technology Decisions

| Area | Choice | Rationale |
|---|---|---|
| Tensor backend | Candle or Burn | Pure Rust, GPU support, autograd |
| DataFrame | Polars | Fast, lazy evaluation, Rust-native |
| Async runtime | Tokio | Industry standard, JoinSet for parallelism |
| Serialization | serde + bincode | Fast binary serialization for plans |
| Local cache | `FsActionStore` | Bazel-style action cache + content-addressed blobs, no embedded DB to corrupt |
| Remote cache | S3-compatible | Universal, works with MinIO locally |
| Python bindings | PyO3 + maturin | Standard Rust-Python bridge |
| HTTP framework | Axum | For worker daemon and coordinator |
| Hashing | SHA-256 | Deterministic, collision-resistant |
| Testing | Built-in + proptest | Property-based testing for algorithms |
| Coverage | cargo-tarpaulin | Standard Rust coverage tool |
| Docs | Starlight (Astro) | This site |
