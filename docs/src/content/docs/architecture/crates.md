---
title: Crate Structure
description: The Rust workspace organization and crate responsibilities.
---

## Workspace Layout

```
soma/
├── Cargo.toml              # workspace definition
├── soma/                   # facade crate (`somatize`) re-exporting the workspace
├── soma-core/              # types, traits, serialization, tracking schema,
│                           #   graph rendering (mermaid/dot/SVG + overlays)
├── soma-macros/            # #[derive(SomaFilter)] proc macro
├── soma-compiler/          # graph → execution plan, scheduler
├── soma-runtime/           # plan executor, events, cache, optimization,
│                           #   run directories (LocalTracker + RunReader)
├── soma-coordinator/       # worker registry, routing, health monitoring
├── soma-worker/            # remote execution daemon
├── soma-memory/            # KnowledgeBase + ChronosVector integration
├── soma-agent/             # autonomous agent loop
├── soma-mcp/               # MCP server for agent integration
├── soma-python/            # PyO3 bindings (pip install somatize)
├── notebooks/              # nine executed tutorial notebooks
└── docs/                   # Starlight documentation (this site)
```

Eleven crates. Note the published names are prefixed `somatize-`
(`somatize-core`, `somatize-runtime`, …) — the directory names drop the
prefix.

## Dependency Graph

```
soma-python (PyO3)
    ├── soma-agent
    │     ├── soma-compiler
    │     │     └── soma-core
    │     ├── soma-runtime
    │     │     └── soma-core
    │     └── soma-memory
    │           └── chronos-vector (external)
    └── soma-worker
          ├── soma-runtime
          └── soma-memory

soma-coordinator
    └── soma-worker (protocol types)
```

## Crate Responsibilities

### `soma-core`

The foundation. Defines all shared types, traits, and enums. Has no heavy dependencies.

**Key exports:**

| Item | Kind | Purpose |
|---|---|---|
| `Filter` | trait | The fundamental computation unit (fit/forward) |
| `Searchable` | trait | Auto-derived search space introspection |
| `Value` | enum | Typed values flowing between filters |
| `VirtualValue` | enum | Lazy references to values (Materialized, Cached, Deferred, Stream) |
| `Graph` | struct | Collection of nodes and edges |
| `Node` / `Edge` | structs | Graph building blocks |
| `Event` | enum | Structured execution events (3 levels) |
| `SearchDimension` | enum | Float, Int, Categorical search parameters |
| `SearchSpace` | struct | Aggregation of search dimensions |
| `Study` / `Trial` | structs | Optimization orchestration |
| `CacheKey` | newtype | Content-addressable hash |
| `CacheStore` | trait | K/V store interface |
| `FilterMeta` | struct | Metadata: kind, differentiable, cacheable, stream mode |
| `Schema` | struct | Input/output type descriptions |
| `SomaError` | enum | Error types |

**Derive macros:**

| Macro | Generates |
|---|---|
| `#[derive(Filter)]` | `Filter` + `Searchable` impls from struct fields |
| `#[derive(SomaEnum)]` | Categorical choices from enum variants |

### `soma-compiler`

Converts `Graph` into `ExecutionPlan`. Pure logic, no I/O (except cache existence checks).

**Key modules:**

| Module | Purpose |
|---|---|
| `compiler.rs` | Main entry: `compile(graph, cache) -> ExecutionPlan` |
| `plan.rs` | `ExecutionPlan` enum definition |
| `cache_resolver.rs` | Compute cache keys, replace cached nodes in plan |
| `gradient_checker.rs` | Analyze gradient flow, emit diagnostics |
| `validator.rs` | Cycle detection, schema compatibility |
| `cost_estimator.rs` | Estimate execution cost from cache metadata |

### `soma-runtime`

Executes plans. This is where computation happens.

**Key modules:**

| Module | Purpose |
|---|---|
| `executor.rs` | Tree-walk plan executor (recursive async) |
| `context.rs` | `Context` passed to filters (store + event emitter + metrics reporter) |
| `event_bus.rs` | Async broadcast of `Event` to subscribers |
| `cache/local.rs` | RocksDB/sled local cache |
| `cache/memory.rs` | In-memory HashMap cache |
| `cache/tiered.rs` | Multi-level cache with promotion/eviction |
| `optimizer.rs` | Study runner: samples params, runs trials, tracks results |
| `samplers/grid.rs` | Grid search sampler |
| `samplers/random.rs` | Random search sampler |
| `samplers/bayesian.rs` | TPE (Tree-Parzen Estimator) sampler |
| `pruners/median.rs` | Median stopping rule |
| `pruners/percentile.rs` | Percentile stopping rule |
| `pruners/hyperband.rs` | Hyperband bracket-based pruning |
| `stream.rs` | Chunked stream processing with state management |

### `soma-worker`

A daemon process that runs on lab machines. Receives serialized plans and executes them.

**Key modules:**

| Module | Purpose |
|---|---|
| `daemon.rs` | Main loop: register, heartbeat, accept plans |
| `protocol.rs` | Message types for coordinator/worker communication |
| `python_loader.rs` | PyO3 dynamic loading of user Python filters |
| `capabilities.rs` | Worker self-description (GPU, RAM, Python envs) |

### `soma-memory`

The [experiment pool](/design/experiment-pool/): what has been tried,
what it descended from, and what came of it.

**Key modules:**

| Module | Purpose |
|---|---|
| `record.rs` | `ExperimentRecord` — the `experiments.jsonl` line format, plus its back-compat contract |
| `derivation.rs` | `DerivationMove` / `Change` — the edge between a parent run and its child |
| `retrieval.rs` | BM25 + structural + recency + importance ranking; the `Embedder` seam |
| `knowledge_base.rs` | The `KnowledgeBase` trait, its analytics defaults, and `MemoryKnowledgeBase` |
| `file_kb.rs` | Append-only JSONL backend with offset-based `refresh()` |
| `chronos_kb.rs` | ChronosVector-backed vector index (feature `chronos`) |

### `soma-agent`

Autonomous agent loop for research automation.

**Key modules:**

| Module | Purpose |
|---|---|
| `agent.rs` | Agent struct (soul, skills, hands, memory) |
| `planner.rs` | Hypothesis-to-graph generation |
| `analyzer.rs` | Result analysis and next-step decision |
| `reporter.rs` | Automatic report and documentation generation |

### `soma-python`

PyO3 bindings. Exposes the full API to Python.

**Key structure:**

```
soma-python/
├── src/lib.rs              # PyO3 module definition (PyGraph, PyStudy, PyFilterBridge)
├── python/soma/
│   ├── __init__.py         # re-exports, Graph.somatize() classmethod
│   ├── filter.py           # Filter base class with search(), >>, | operators
│   ├── chain.py            # Chain/Fork lazy builder types
│   ├── builder.py          # Graph materialization (_walk algorithm)
│   ├── search.py           # search() descriptor for Python filters
│   └── lab.py              # Remote connection
├── pyproject.toml          # maturin build config
└── Cargo.toml
```
