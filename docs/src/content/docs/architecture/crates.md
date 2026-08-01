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
├── soma-agent/             # ResearchStep: the research loop as a Step
├── soma-llm/               # providers (OpenAI-compatible), tools, MCP client
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
| `#[derive(SomaFilter)]` | `Filter` + `Searchable` impls from struct fields |

### `soma-compiler`

Converts `Graph` into `ExecutionPlan`. Pure logic, no I/O (except cache existence checks).

**Key modules:**

| Module | Purpose |
|---|---|
| `compiler.rs` | Main entry: `compile(graph, …) -> ExecutionPlan`; validation and gradient diagnostics |
| `plan.rs` | `ExecutionPlan` enum definition |
| `scheduler.rs` | `schedule(plan, workers) -> DistributionPlan` — assigns nodes to workers |

Cache keys are **not** resolved here: they depend on materialized input, so
resolution happens per node at runtime. See [Caching](/soma/design/caching/).

### `soma-runtime`

Executes plans. This is where computation happens.

**Key modules:**

| Module | Purpose |
|---|---|
| `graph_session.rs` | `GraphSession` — the primary orchestrator (graph + library + cache + events) |
| `executor.rs` | Tree-walk plan executor, and the per-node runtime cache resolution |
| `forward.rs` | Forward-pass helpers shared by the executors |
| `node_catalog.rs` | `NodeCatalog` — every node (filter or step), their states, and the compiler's `NodeRegistry` |
| `event_bus.rs` | Async broadcast of `Event` to subscribers, plus lossless sinks |
| `cache/fs_store.rs` | `FsActionStore` — action records + BLAKE3 content-addressed blobs |
| `cache/gc.rs` | Value-density eviction down to a size budget (`soma cache gc`) |
| `cache/memory.rs` | In-memory LRU with a byte budget |
| `cache/local.rs` | Filesystem cache tier |
| `cache/tiered.rs` | Memory → filesystem, with promotion |
| `executors/study.rs` | `StudyRunner`: samples params, runs trials, replays completed ones |
| `executors/pbt.rs` | `PbtRunner`: population-based train → evaluate → exploit/explore |
| `executors/stream.rs` | Chunked stream processing with state management |
| `executors/simple.rs` | Single-graph execution |
| `sampler/mod.rs` | `Sampler` trait, grid and random samplers |
| `sampler/bayesian.rs` | TPE (Tree-Parzen Estimator) sampler |
| `pruner.rs` | `MedianPruner` and `PercentilePruner` |
| `runner/` | `LocalRunner` and `RemoteRunner` behind the `Runner` trait |
| `tracking/local_tracker.rs` | Writes a run directory: manifest, status, artifacts, event sink |
| `tracking/jsonl_sink.rs` | The lossless `events.jsonl` sink |
| `tracking/reader.rs` | `RunReader` — every aggregate a chart or report needs |
| `tracking/summary.rs` | `summarize()` — one run directory folded into a `RunSummary` |
| `tracking/head.rs` | `.soma/HEAD`: parent resolution, `checkout`, `advance_head` |

### `soma-worker`

A daemon process that runs on lab machines. Receives serialized plans and executes them.

**Key modules:**

| Module | Purpose |
|---|---|
| `worker.rs` | The `Worker` itself: cache, filter library, data stores, env manager |
| `server.rs` | Axum HTTP/WS server: register, heartbeat, accept plans |
| `protocol.rs` | Message types for coordinator/worker communication |
| `ws_transport.rs` | WebSocket framing for plans, chunks and event streams |
| `python_process.rs` | Runs user Python filters in an isolated interpreter process |
| `env_manager.rs` | Per-pipeline venv/conda with incremental dependency updates |
| `detect.rs` | Worker self-description (GPU, RAM, Python envs) |

### `soma-memory`

The [experiment pool](/soma/design/experiment-pool/): what has been tried,
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

The research loop, as a `Step`. Proposes an experiment, runs it via
`Effect::Graph`, reads the metrics, decides whether to continue.

**Key modules:**

| Module | Purpose |
|---|---|
| `research.rs` | `ResearchStep`: the loop, its prompt, and its record-keeping |
| `action.rs` | `Action` — `RunExperiment` or `Conclude`, and nothing else |

It owns no loop, no journal and no record type: those are the effect
driver's, the journal's and `somatize-memory`'s. See
[Agents & Memory](/soma/platform/agents/).

### `soma-python`

PyO3 bindings. Exposes the full API to Python.

**Key structure:**

```
soma-python/
├── src/lib.rs              # the whole PyO3 module: PyGraph, PyStudy, PyRun, PyWorker,
│                           #   the filter bridge, and the cache/runs/kb pyfunctions
├── python/soma/
│   ├── __init__.py         # re-exports; imports the modules below for their side effects
│   ├── filter.py           # Filter base class, >> and | operators
│   ├── search.py           # search() descriptor and FilterMeta
│   ├── chain.py            # Chain/Fork lazy builder types
│   ├── builder.py          # Graph materialization (_walk algorithm)
│   ├── _orchestrator.py    # installs train/eval/forward/backward/step/materialize on Graph
│   ├── _composite.py       # DifferentiableFilter (torch)
│   ├── _audit.py           # gradient_audit(), AuditScope, ChannelConfig, the flags
│   ├── _study.py           # installs search_space/apply_params/study on Graph
│   ├── _tracking.py        # installs track_run on Graph
│   ├── _checkpoint.py      # state/load_state/save/load, the .somack bundle
│   ├── _compile.py         # CompileInfo (dict + notebook repr)
│   ├── _identity.py        # cache identity: the code-fingerprint ladder
│   ├── _runs.py            # RunView / RunList over run directories
│   ├── _experiments.py     # reads experiments.jsonl
│   ├── _lineage.py         # checkout/head/detach/reindex over .soma/HEAD
│   ├── _cache_cli.py       # the `soma` CLI: cache, runs, graph, report, kb
│   ├── cli.py              # the `somatize-worker` CLI
│   ├── lab.py              # remote connection (connect/health/info/workers)
│   └── viz/                # optional plotly/pandas figures — the somatize[viz] extra
├── pyproject.toml          # maturin build config
└── Cargo.toml
```

Most of the Python layer works by monkeypatching the Rust `Graph` at import
time, which is why `soma/__init__.py` imports several modules purely for their
side effects.
