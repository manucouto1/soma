---
title: Crate Structure
description: The Rust workspace organization and crate responsibilities.
---

## Workspace Layout

```
soma/
├── Cargo.toml              # workspace definition
├── soma/                   # facade crate (`somatize`) re-exporting the workspace
├── soma/                   # facade crate (`somatize`), re-exports the rest
├── soma-core/              # types, traits, serialization, tracking schema,
│                           #   graph rendering (mermaid/dot/SVG + overlays)
├── soma-store/             # remote DataStore backends (S3, Zarr), off by default
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
├── notebooks/              # fourteen executed tutorial notebooks
└── docs/                   # Starlight documentation (this site)
```

Thirteen crates. Note the published names are prefixed `somatize-`
(`somatize-core`, `somatize-runtime`, …) — the directory names drop the
prefix.

## Dependency Graph

Acyclic, and read top to bottom — nothing below depends on anything
above it.

```
soma-macros                        proc macros; no internal dependencies
    │
soma-core                          types, traits, serialization
    ├── soma-store                 S3 / Zarr backends (feature-gated)
    ├── soma-compiler              graph → execution plan
    │     └── soma-runtime         the executor, cache, effects
    │           ├── soma-llm       providers, tools, MCP client
    │           ├── soma-worker    remote execution daemon
    │           │     └── soma-coordinator
    │           ├── soma-agent     ┐ both also on soma-memory
    │           └── soma-mcp       ┘
    └── soma-memory                experiment pool, KnowledgeBase

soma-python   → core, compiler, runtime, llm, memory, store, worker
soma          → the facade; re-exports all of the above
```

`soma-macros` takes `soma-core` as a *dev*-dependency, so its `trybuild`
cases have the traits to derive against. That is not a cycle: no normal
edge points back up.

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

### `soma-macros`

The proc macros: `#[derive(SomaFilter)]` and `#[derive(SomaStep)]`. The
latter is what gives every step its journal key, so two structurally
identical steps with different configuration do not share journal
entries.

A misspelled attribute is a compile error rather than a silent no-op —
`#[soma(serach(...))]` used to compile and do nothing. Errors carry spans,
so the message points at the attribute rather than the whole derive.

### `soma-store`

Remote `DataStore` backends: S3 and Zarr, each behind a feature and off by
default.

Split out of `soma-core` because each owns a `tokio::runtime::Runtime` and
`block_on`s network I/O, so anything depending on `soma-core` inherited a
runtime it never asked for. `soma-core` keeps `LocalDataStore` and its
`std::fs`, which costs a caller nothing. See
[Design Decisions](/soma/design/decisions/).

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
| `runner/` | `LocalRunner` behind the `Runner` trait, and the `Transport` seam |
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

### `soma-llm`

Model providers and tools. One OpenAI-compatible client serves ollama,
HuggingFace, NVIDIA, Kimi, GLM, DeepSeek, Groq, vLLM and the rest; the
catalog is TOML data, not code, so adding a provider is not a patch. Each
entry carries its `RetryPolicy` and its `Quirks`, which is why there is no
`if id == "openai"` anywhere.

Retries live in the client rather than in the step: a 429 is transport,
not domain. `Retry-After` is honoured in both RFC forms, the wall-clock
budget is checked *before* sleeping, and giving up reports the last
failure plus the first when they differ. Retries never reach the EventBus.

Also holds `Toolbox`, the MCP client, `ReactStep` and `JudgeStep`. The
crate is entirely blocking by decision — the effect driver runs it on
threads — which is why no lock is ever held across an await in the core.

### `soma-mcp`

An MCP server exposing the project to an agent: 20 tools over code,
knowledge, project state and the experiment pool. The rendered text *is*
the API, so every result ends with a `next:` line and a `run_dir:`.

`run_pipeline` and `run_study` **execute**: a model describes a graph out
of the project's own filters — the ones `list_filters` lists and
`read_filter_source` reads — and `soma-mcp/src/exec.rs` runs it in a
Python subprocess rooted at the project directory. A config value written
as `{"__search__": {...}}` becomes a search dimension, so the only
difference between running a graph and searching it is which values were
marked. Both say in their own descriptions that they execute project
code.

### `soma-coordinator`

Worker registry and placement, with a `soma-coordinator` binary. Workers
heartbeat every 10 seconds and the coordinator reaps whoever goes quiet.

`POST /submit` **places**: it returns a worker and takes a lease rather
than proxying the plan, so tensor payloads travel client→worker directly
instead of through the coordinator twice. `/complete` releases the lease.
Authentication is a bearer header compared in constant time.

### `soma` (the facade)

Published as `somatize`. Re-exports the workspace so a Rust caller adds
one dependency instead of eight, and carries the prelude — which reaches
the effectful half too: `Step`, `Transition`, `Effect`, `NodeOutcome`,
`SomaStep`.

### `soma-python`

PyO3 bindings. Exposes the full API to Python.

**Key structure:**

```
soma-python/
├── src/                    # one module per area, not one file
│   ├── lib.rs              # the module definition and its exports
│   ├── graph.rs            # PyGraph — the bulk of the surface
│   ├── agentic.rs          # Agent, Judge, Tool, StepCtx, the step bridge
│   ├── study.rs            # PyStudy, PyTrial
│   ├── readers.rs          # run directories and the experiment pool
│   ├── bridge.rs           # the Python filter as a Rust Filter
│   ├── convert.rs          # Value ↔ Python, natively (no JSON round-trip)
│   ├── run.rs, cache.rs, worker.rs
├── python/soma/
│   ├── __init__.py         # re-exports
│   ├── _soma.pyi           # the extension's surface, checked against the build
│   ├── py.typed            # the package means what it says about itself
│   ├── _graph.py           # class Graph(_RustGraph) — where its methods are declared
│   ├── filter.py           # Filter base class, >> and | operators
│   ├── search.py           # search() descriptor and FilterMeta
│   ├── chain.py            # Chain/Fork lazy builder types
│   ├── builder.py          # Graph materialization (_walk algorithm)
│   ├── agentic.py          # patterns as functions returning a Graph
│   ├── library.py          # Eval, Accumulator, Retriever, Compact
│   ├── _orchestrator.py    # train/eval/forward/backward/step/materialize
│   ├── _composite.py       # DifferentiableFilter (torch)
│   ├── _audit.py           # gradient_audit(), AuditScope, ChannelConfig, the flags
│   ├── _study.py           # class Study(_Study); search_space/apply_params
│   ├── _tracking.py        # track_run
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

`soma.Graph` and `soma.Study` are Python subclasses of the extension
classes, and their methods are declared in the class body — the
implementations still live in `_orchestrator`, `_checkpoint` and the rest,
but which methods exist is one list you can read. They used to be assigned
onto the Rust class at import time from seven modules, which meant the
surface of a graph depended on what had been imported.

The package ships `py.typed` and a hand-written `_soma.pyi` for the
extension; the Python layer above it is annotated in place. Because a
hand-written stub rots silently, `tests/test_stubs.py` compares it against
the module that was actually compiled.
