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

## Workspace (13 crates)

```
soma-macros     → proc macros: #[derive(SomaFilter)] and #[derive(SomaStep)]
                  (the latter is what gives every step its journal key)
soma-core       → types, traits, serialization. The rule is no runtime, no
                  network, no optional heavy dep — NOT "no I/O": LocalDataStore
                  and its std::fs stay, because they cost a caller nothing.
                  Verify with `cargo tree -p somatize-core | grep tokio` (empty).
                  Filter, Step, Value, Graph, Event, Schema, VirtualValue, Search, Study,
                  Effect/Transition, Message/ContentBlock, ToolSpec, LoopCondition,
                  TrainingStrategy (the type; running one is in soma-runtime),
                  DataStore trait + LocalDataStore, StreamCache
soma-store      → remote DataStore backends (S3, Zarr), feature-gated and off
                  by default. Split out of soma-core because each owns a tokio
                  runtime; see docs design/decisions.
soma-compiler   → Graph → ExecutionPlan (cache resolution, schema validation, distribution)
                  Scheduler (distribute plan across workers), ExecutionPlan visualization
soma-runtime    → GraphSession (primary orchestrator), parallel executor (threads),
                  NodeCatalog (filters AND steps), EffectDriver + EffectJournal + GraphHandler,
                  stream executor,
                  LRU/local/tiered cache, Grid/Random/Bayesian samplers,
                  Median/Percentile pruners, StudyRunner, PbtRunner
soma-memory     → Experiment pool: ExperimentRecord (experiments.jsonl), DerivationMove,
                  BM25+structural retrieval, KnowledgeBase trait + MemoryKB/FileKB/ChronosKB
soma-worker     → Protocol (Rust plans + Python jobs), Worker, EnvManager
                  (isolated venv/conda per pipeline), Axum HTTP/WS server
soma-llm        → LlmProvider + OpenAI-compatible client (ollama/hf/nvidia/kimi/glm/
                  deepseek/groq/vllm...), provider catalog as TOML data (incl.
                  RetryPolicy + Quirks), Toolbox, MCP client, ReactStep/JudgeStep
soma-agent      → ResearchStep: the research loop as a Step (propose → Effect::Graph →
                  read metrics → conclude). Action = RunExperiment | Conclude
soma-mcp        → MCP server (20 tools: code, knowledge, project, 7 experiment-pool kb_*)
soma-coordinator→ worker registry + placement, with a `soma-coordinator` binary.
                  Workers beat every 10s; the coordinator reaps whoever goes
                  quiet. `/submit` PLACES (returns a worker, takes a lease) —
                  it does not proxy the plan, so tensor payloads go client→worker
                  direct. `/complete` releases the lease.
soma-python     → PyO3 bindings: Graph (primary API), Filter, Agent, Judge, Tool,
                  Study, Run, RunView, soma.viz, soma.agentic, soma.library
soma/           → facade crate (`somatize`) re-exporting the workspace
docs/           → 35 Starlight pages (sidebar guard: `cd docs && npm run check`)
notebooks/      → 15 executed tutorial notebooks (10-12 are one campaign, sharing
                  campaign.py; 13 is agentic, with an embedded mock provider so it
                  runs with no key; 14 replicates Du et al. multi-agent debate on
                  GSM8K and NEEDS a real model — no mock can measure an accuracy
                  gap; 15 is the seam both ways: Agent in a pipeline, RunGraph
                  from a step, schemas, journal replay, Suspend/resume — mock,
                  no key); re-run with `python notebooks/execute.py`
```

## Tests

```bash
# 984 Rust + 699 Python (14 deselected by default: slow + live)
# Property tests are Rust-side (soma-core/tests/proptests.rs); the Python
# suite does not use hypothesis.
cargo test --workspace                              # Rust tests
cd soma-python && maturin develop && pytest tests/  # Python tests (fast set)
cd soma-python && pytest tests/ -m slow             # robustness: SIGKILL crash-sim, statistical TPE
cd soma-python && SOMA_LIVE=1 pytest tests/ -m live # real endpoints: needs OLLAMA_HOST / NVIDIA_API_KEY
cargo test -p somatize-memory --features chronos    # +ChronosVector tests
cd soma-python && mypy                              # the package ships py.typed

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
- **GraphSession**: Primary orchestrator — binds Graph + NodeCatalog + cache + events. Methods: fit, forward, compile, run.
- **NodeCatalog**: THE registry — every node, filter or step, plus the trained states.
  Implements `NodeRegistry` (the compiler's port, two required methods: `node_meta` +
  `config_hash`) and is what the
  executor reads. Filters and steps used to live in two registries joined by an adapter a
  caller had to remember to build, which is how `.compile()` came to skip every step's
  schema while `.run()` checked them.
- **NodeMeta**: one metadata type for both kinds, with `effectful`. `From<StepMeta>` sets
  `cacheable: false, deterministic: false`, so "a step is not output-cacheable" is data the
  executor's existing guard reads — there is no `if is_step` anywhere.
- **NodeOutcome** (`Produced` | `HandOff` | `Paused`): how a node finished, whichever kind
  it was. Deliberately NOT `#[non_exhaustive]` — every consumer decides control flow, and a
  wildcard arm there is a silent wrong answer.
- **run_node**: the one execution site. Input resolution, the output cache, `catch_unwind`,
  and the start/complete/fail events happen once for filters and steps alike;
  `run_node_inner` is the only `match` on the execution hot path that tells them apart
  (`fit_state_if_needed` and `composite_fit` also discriminate, on the fit side).
- **RunContext**: what a runner needs besides the plan — catalog, cache, events, run id, and
  the *real* `GraphInfo`. `RunContext::linear` is the explicit fallback for a caller that
  has only a plan (the worker); the runner no longer invents a topology.
- **ExecutionPlan**: Compiled from Graph. Variants: Sequence, Parallel, Execute,
  Step (with handoffs), Composite, Loop (with `until` + `carry_from`), Branch, Stream, Remote, Empty.
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
- **Agentic layer** (`design/agentic.md`): a flow is a graph whose nodes are effectful.
  `Step` (sync `poll` → `Transition`; a driver performs the effects) sits beside `Filter`.
  A filter memoizes by content; a step *journals* — pure effects keyed by content, impure
  ones by `(run, node, turn, index)`: record once, replay on resume, never re-run.
  Five structural NodeKinds; every behaviour (LLM, tool, judge, router) is library, and
  every pattern is a function in `soma.agentic` returning a Graph — `react`, `route`,
  `refine`, `debate`, `board`, `parallel_vote`, `orchestrate`. `board` is Du et al.
  multi-agent debate (ICML 2024): `brief → members → chair`, the chair also reads the
  brief (or round 2 forgets the question), `MajorityVote` chair is a filter not a model,
  and `done` = unanimity, so a converged panel stops early. Notebook 14 replicates it.
  **Steps CAN be written in Python**: any object with `poll(ctx)` (duck-typed like a
  filter's `forward`). Returns dict transitions — `Done/Await/Spawn/Goto/Suspend` from
  `soma.agentic`. `Spawn` gives real dynamic fan-out, so `orchestrate` sizes its pool
  from the plan (`planner → fanout → synthesize`). `g.register_step(id, step)` registers
  a spawn target WITHOUT adding a node (a node with no edges is a root and would also
  run on the graph input); `g.handoff(a, b)` declares the control edge `Goto` needs.
  `py_to_value` also accepts bare int/float and non-numeric lists (both were errors —
  numeric lists stay tensors, so no cache key moves).
  Control flow: compiler claims loop bodies / branch arms by **dominance**, resolves
  `BodyTerminal` → `WhenSignaled(node)`, and `ExecutionPlan::Loop` carries `carry_from`
  separately from `until` (what a loop carries ≠ what stops it). A branch passes its
  *input* to the chosen arm — the selector is control, not data.
  `Effect::Graph` + `GraphHandler` make a pipeline a first-class tool for an agent —
  from Python: `soma.agentic.RunGraph(sub, input=, mode=)` + `g.register_graph(sub)`
  (structure travels in the effect; implementations are merged once, same id +
  different config = error). Sub-graphs may contain steps (`GraphHandler::
  with_step_runtime`, nesting capped at `MAX_GRAPH_DEPTH = 8`); `Effect::Graph`
  is pure ONLY for filter-only Forward graphs — a step-containing or Fit sub-graph
  is journaled by site, never reused by content (`Graph::contains_steps`).
  Rust callers: `GraphSession::with_driver(driver)` makes the session drive steps
  (tests: soma-runtime/tests/session_steps.rs); the driver carries its own catalog
  via `EffectDriver::with_catalog` — Context/RunContext::with_driver only store.
  Python nodes declare edge contracts with `_input_schema`/`_output_schema`
  ("text"|"json"|"messages"|"bytes" or a {dtype, shape} mapping); a typo fails at
  registration, an impossible edge at compile. A step holding a live Graph stores
  it underscored — a Graph cannot enter the JSON identity, and need not.
  Searchable: `soma.Agent(model=search(...), system=search(...))` and `g.optional(a, b)`
  (topology as a dimension) fold into the same `search_space()` a filter graph produces.
- **Provider resilience**: retries live in the HTTP client, not the step — a 429 is
  transport, not domain. `RetryPolicy` is a `ProviderConfig` field (TOML-overridable):
  408/425/429/5xx + transport errors retry, everything else is fatal. `Retry-After`
  honoured in both RFC forms, capped by `max_ms`; exponential backoff with full jitter
  otherwise; wall-clock `budget_secs` checked BEFORE sleeping. Giving up reports the
  last failure plus the first when they differ. Retries do NOT reach the EventBus.
- **Structured output**: `LlmRequest.schema` + `Quirks::supports_json_schema` (which was
  declared and never read until now) → `response_format` when the endpoint can enforce
  it, system-prompt append when it cannot. `soma.Agent(schema=..., max_repairs=1)`:
  one violation buys one correction, quoted back. Validation is STRUCTURAL and
  permissive on purpose (root type, `required`, property types) — an invented violation
  costs a real model call to "fix" a correct answer. `soma.agentic.Validate` mirrors it
  in Python (uses `jsonschema` if installed), returning `{"ok","errors","value","branch"}`.
- **`soma.library`**: Eval (accuracy/exact_match/token-F1/top_k; scoring nothing is an
  ERROR, not a 0.0), Accumulator (stateful + `_deterministic=False`, the documented
  exception), Retriever (over the pool), Compact (sliding window; enabling it
  invalidates replay of earlier runs), `agentic.self_consistency` (one agent sampled N
  times; `MajorityVote(mode="number"|"text")`).
- **Not answers**: `finish_reason: length` and `content_filter` are ERRORS in ReactStep,
  not empty replies. `forward` picks the leaf that actually ran, not `leaves.first()`.
- **Typing**: the package ships `py.typed`, so what it says about itself is public
  API. The extension has a hand-written `soma/_soma.pyi`; the Python layer on top is
  annotated in place and needs no stub. A hand-written stub can lie, so
  `tests/test_stubs.py` checks it against the module that was *built* — same classes,
  methods, attributes, parameter names and defaults, and no constructor for the three
  classes that have no `#[new]`. What no test can check is whether a type is right.
  Two consequences worth knowing: PyO3 puts a `#[new]`'s signature on the *type*
  (`cls.__text_signature__`), not on `__new__`; and a method bound dynamically in a
  class body is `Any` to a checker, which is why the `soma.viz` methods on `Study` and
  `RunView` are written out one by one instead of built by a loop.
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
