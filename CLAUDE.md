# Soma - Development Guide

## Project

Soma is a computational graph runtime for research pipelines, agent orchestration, and data virtualization. Written in Rust with Python bindings (PyO3). Part of the Nous-Soma-Chronos ecosystem.

- **Soma** (this project): Executes, materializes — pipelines, optimization, caching
- **[Nous](https://github.com/manucouto1/nous)**: Understands, reasons — agent graphs, evaluation, knowledge
- **[ChronosVector](https://github.com/manucouto1/chronos-vector)**: Remembers — temporal vector database

## Quick Commands

```bash
# Full check (what CI runs)
cargo fmt --all -- --check && cargo clippy --workspace -- -D warnings && cargo test --workspace

# With ChronosVector feature
cargo test -p soma-memory --features chronos

# Python
cd soma-python && maturin develop && pytest tests/ -v

# MCP server
cargo run -p soma-mcp -- /path/to/project

# Docs
cd docs && npm run dev     # dev server
cd docs && npm run build   # production build
cargo doc --workspace --open  # Rust API docs
```

## Workspace (10 crates)

```
soma-macros     → proc macro (#[derive(SomaFilter)])
soma-macros     → proc macro (#[derive(SomaFilter)])
soma-core       → types, traits, serialization ONLY (no execution logic):
                  Filter, Value, Graph, Event, Schema, VirtualValue, Search, Study,
                  TrainingStrategy, DataStore (Local/S3/Zarr), StreamCache
soma-compiler   → Graph → ExecutionPlan (cache resolution, schema validation, distribution)
                  Scheduler (distribute plan across workers), ExecutionPlan visualization
soma-runtime    → GraphSession (primary orchestrator), parallel executor (threads),
                  FilterLibrary (unified registry), stream executor,
                  LRU/local/tiered cache, Grid/Random/Bayesian samplers,
                  Median/Percentile pruners, StudyRunner, PbtRunner
soma-memory     → KnowledgeBase trait + MemoryKB + ChronosKB (feature-gated)
soma-worker     → Protocol (Rust plans + Python jobs), Worker, EnvManager
                  (isolated venv/conda per pipeline), Axum HTTP/WS server
soma-agent      → Agent loop, Action, ResearchPlan trait
soma-mcp        → MCP server (13 tools: code, execution, knowledge, project)
soma-python     → PyO3 bindings: Graph (primary API), Filter, Study, Lab
docs/           → 24 Starlight pages
```

## Tests

```bash
# 340+ total: 342 Rust + 17 Python
cargo test --workspace                              # Rust tests
cd soma-python && maturin develop && pytest tests/  # 17 Python tests
cargo test -p soma-memory --features chronos        # +8 ChronosVector tests
```

## Conventions

- **Commits**: Conventional Commits with crate scope: `feat(core): add Schema type`
- **Branches**: Gitflow — `main`, `develop`, `feature/<crate>-<desc>`
- **Tests**: TDD. All public APIs must have tests.
- **Clippy**: `cargo clippy --workspace -- -D warnings` must pass
- **Enums**: Public enums use `#[non_exhaustive]`
- **License**: Elastic License 2.0

## Key Design Decisions

- **Filter trait**: `fit()` learns state, `forward()` transforms. Both independently cacheable.
- **CacheKey**: SHA-256 content-addressable. `hash(config + input_hash)` with cascade invalidation.
- **GraphSession**: Primary orchestrator — binds Graph + FilterLibrary + cache + events. Methods: fit, forward, compile, run.
- **FilterLibrary**: Unified registry — implements FilterRegistry (compiler) + holds filters + states (executor). Replaces old FilterStore.
- **ExecutionPlan**: Compiled from Graph. Variants: Sequence, Parallel, Execute, Cached, Loop, Branch, Remote.
- **VirtualValue**: Lazy references (Materialized | Cached | Deferred | Stream).
- **Schema**: dtype + shape for compile-time type checking between connected filters.
- **TrialOutcome**: Separates control flow (Completed | Pruned) from errors.
- **StreamMode**: FixedState | Evolving (checkpoints) | Barrier (materializes).
- **Distribution**: Local | Remote(WorkerId | Tag) — compiler wraps in ExecutionPlan::Remote.
- **GraphInfo**: Topology-aware input resolution (predecessors, not "last executed").
- **LRU Cache**: Enforces max_bytes with eviction. No unbounded growth.
- **DataStore**: Abstraction for data movement between workers (Local, S3, Cached, Stream, Inline).
- **StreamCache**: Inference optimization — caches filter states and chunk results by content hash.
- **Scheduler**: Analyzes ExecutionPlan topology, assigns to workers (sequential→same worker,
  parallel→distribute, differentiable→group together). Produces DistributionPlan.
- **TrainingStrategy**: Graph-level attribute (inherited by subgraphs): Local, DataParallel, ModelParallel, Federated, PopulationBased, Custom.
- **Partition**: Maps arbitrary node subsets to RemoteTargets for model parallelism.
- **PbtRunner**: Population-Based Training — cyclic train→evaluate→exploit/explore per generation.
- **Graph visualization**: `to_mermaid()`, `to_graphviz()`, `to_text()` — pure data→string, no runtime deps.
- **EnvManager**: Isolated Python environments per pipeline with incremental dependency updates.
  Hashes requirements to detect changes, only installs/upgrades/removes what changed.
- **Pipeline removed**: Graph is the ONLY user-facing API. No Pipeline class.

## MCP Server

```bash
# Run the MCP server
soma-mcp /path/to/project

# Or via env var
SOMA_PROJECT_DIR=/path/to/project soma-mcp
```

13 tools: list_filters, read_filter_source, write_filter_source, run_graph,
run_study, record_experiment, query_knowledge_base, get_trajectory,
get_change_points, list_research_lines, promising_lines,
create_research_line, generate_report.

## Feature Flags

- `soma-memory/chronos`: Enables ChronosVector-backed KnowledgeBase

## Release

```bash
# Bump version in workspace Cargo.toml, then:
cargo release patch  # or minor/major
# GitHub Actions publishes to crates.io and PyPI on tag push.
```
