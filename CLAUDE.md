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

## Workspace (11 crates)

```
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
soma-memory     → Experiment pool: ExperimentRecord (experiments.jsonl), DerivationMove,
                  BM25+structural retrieval, KnowledgeBase trait + MemoryKB/FileKB/ChronosKB
soma-worker     → Protocol (Rust plans + Python jobs), Worker, EnvManager
                  (isolated venv/conda per pipeline), Axum HTTP/WS server
soma-agent      → Agent loop, Action, ResearchPlan trait
soma-mcp        → MCP server (20 tools: code, knowledge, project, 7 experiment-pool kb_*)
soma-coordinator→ worker registry, routing, heartbeat monitoring
soma-python     → PyO3 bindings: Graph (primary API), Filter, Study, Run, RunView, soma.viz
soma/           → facade crate (`somatize`) re-exporting the workspace
docs/           → 34 Starlight pages (sidebar guard: `cd docs && npm run check`)
notebooks/      → 12 executed tutorial notebooks (10-12 are one campaign, sharing
                  campaign.py); re-run with `python notebooks/execute.py`
```

## Tests

```bash
# 1043 total: 698 Rust + 345 Python (incl. property tests and 4 robustness tests)
cargo test --workspace                              # Rust tests
cd soma-python && maturin develop && pytest tests/  # Python tests (fast set)
cd soma-python && pytest tests/ -m slow             # robustness: SIGKILL crash-sim, statistical TPE
cargo test -p somatize-memory --features chronos    # +ChronosVector tests

# Coverage (informational, no gate)
cd soma-python && pytest tests/ --cov=soma --cov-report=term-missing
cargo llvm-cov --workspace --summary-only           # needs cargo-llvm-cov
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
- **CacheKey**: SHA-256, resolved at RUNTIME per node: state = `hash(config + x + y)`, output = `hash(config + state + input)`, seed salted in when set. Downstream keys use input *content* hashes → early cutoff.
- **Persistent cache**: `Graph()` defaults to tiered(memory LRU → `FsActionStore` at `$SOMA_CACHE_DIR` || `~/.soma/cache`). Two-table store: action records (kept forever) + BLAKE3 CAS blobs (evictable). `soma cache stats|gc|pin|verify|purge-v1`.
- **Filter identity**: Rust = canonical CBOR of fields (+`#[soma(cache_version)]`); Python = qualname + canonical config + source-hash ladder (`_cache_version` → `inspect.getsource` → cloudpickle+warning). Unhashable ⇒ `CacheConfigError`, never a silent key.
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
- **Graph visualization**: `to_mermaid()`, `to_graphviz()`, `to_text()`, `to_svg()` — pure data→string,
  no runtime deps. `to_svg` (soma-core/src/svg.rs, longest-path layering) exists because notebooks
  sanitize `<script>`: it backs `Graph._repr_html_` (evaluate `g` → diagram),
  `DifferentiableFilter._repr_html_` (inner layers + θ counts), `RunView.to_svg(node=...)` and the
  --inline report diagrams.
  `to_mermaid_with(&GraphOverlay)` / `to_graphviz_with` fold per-node status/duration/cache/health-flag
  annotations in (soma-core/src/viz.rs); empty overlay ⇒ byte-identical plain output.
- **Visualization (3 layers, GUI reuse at the DATA layer)**: `RunReader` (soma-runtime/src/tracking/reader.rs)
  aggregates run dirs into chart-ready serde structs (node_timings, cache_activity, metric_series,
  health_flags, trial_timeline, overlay); Python `soma.runs()`/`RunView`; `soma.viz` = optional
  `somatize[viz]` extra (plotly+pandas) with Optuna-named `study.plot_*`, `run.plot_*` and dataframes;
  `soma runs|graph|report` CLI. `soma report` emits one self-contained HTML whose
  `<script type="application/json" id="soma-data-*">` blobs are the future GUI's contract
  (see docs design/visualization.md). Local fit/run paths emit RunStarted/Completed/Failed brackets
  sharing the node events' run_id. `soma ui` live server: deliberately deferred.
- **Intra-node audit (`gradient_audit(inside=...)`)**: submodule hooks under hierarchical ids
  `"<node>/<module.path>"` (opaque strings end-to-end; hierarchy travels in Audit._children, never
  parsed). Progressive disclosure: `inside=True` auto / dict duck-typing (int=depth, list=fnmatch) /
  class attr `_audit_scope` / `AuditScope(depth, patterns, sample_every, max_modules)` — precedence
  call-site > class > auto, mirroring channels=True|ChannelConfig. Persists
  `diagnostics/modules/<node>.json` (soma-core Graph schema + execution order); child HealthFlags
  roll up to one parent flag per family. Viz: `run.plot_module_flow`, `run.to_mermaid(node=...)`,
  `plot_audit(node=...)` (default hides submodule series), report "Module flow" section +
  `soma-data-module-trees` blob.
- **Terminal/notebook UX**: `soma.runs()` → `RunList` with `_repr_html_` (status-color chips shared
  with the report CSS); `soma runs` CLI uses a rich table when available (`--plain` for pipes);
  `Study.run(progress=True)` = tqdm bar via lossy StudyProgress events (finalized from n_trials).
  rich/tqdm live in the `somatize[viz]` extra, lazy imports, plain fallbacks everywhere.
  Notebook filters MUST declare `_cache_version` (getsource unavailable under headless kernels);
  notebooks ship executed (PNG figs via kaleido, scale 2) — re-execute with the warning-gated
  runner in the job tmp dir (`execute_notebooks.py` pattern: temp cwd + fresh SOMA_CACHE_DIR).
- **EnvManager**: Isolated Python environments per pipeline with incremental dependency updates.
  Hashes requirements to detect changes, only installs/upgrades/removes what changed.
- **Experiment pool** (`design/experiment-pool.md`): every tracked run appends an
  `ExperimentRecord` to `.soma/experiments.jsonl` with a templated deterministic
  `RunConclusion`, an `ArchitectureFingerprint` and the `DerivationMove` from its parent
  (VisTrails-style: nodes are runs, edges are the changes applied to the parent).
  `begin_run` is the SINGLE writer of graph.json/graph.mmd/fingerprint.json.
  Parent resolution: `parent=` → `$SOMA_PARENT_RUN` → `.soma/HEAD` → none; HEAD advances
  only on success; NEVER inferred from timestamps. `soma.checkout/head/detach/reindex`
  and `soma kb reindex|head|checkout|detach`. Retrieval is additive
  `0.40·BM25 + 0.25·structural + 0.15·recency + 0.20·importance`, with importance floored
  at 0.6 for failures that carry a conclusion (dead ends must stay retrievable).
  `soma-mcp/render.rs` holds the pure text renderers — the MCP text IS the API, every
  result ends with a `next:` line and a `run_dir:`.
- **Pipeline removed**: Graph is the ONLY user-facing API. No Pipeline class.

## MCP Server

```bash
# Run the MCP server
soma-mcp /path/to/project

# Or via env var
SOMA_PROJECT_DIR=/path/to/project soma-mcp
```

20 tools: list_filters, read_filter_source, write_filter_source, run_pipeline*,
run_study*, record_experiment, query_knowledge_base, get_trajectory,
get_change_points, list_research_lines, promising_lines,
create_research_line, generate_report, plus the experiment pool:
kb_find_similar, kb_lineage, kb_diff, kb_record_conclusion, kb_branch_from,
kb_summarize_run, kb_stats.  (* declared but NOT implemented — they cannot
load user code; their descriptions say so.)

## Feature Flags

- `soma-memory/chronos`: Enables ChronosVector-backed KnowledgeBase

## Release

```bash
# Bump version in workspace Cargo.toml, then:
cargo release patch  # or minor/major
# GitHub Actions publishes to crates.io and PyPI on tag push.
```
