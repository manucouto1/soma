# Changelog

Notable changes per release. Written for someone deciding whether to
upgrade and what it will cost them, so the breaking changes come first
and each one says what to do about it.

Versions follow [semantic versioning](https://semver.org/). Pre-1.0, a
minor bump may break API — and this one does, extensively.

## [0.4.0] — unreleased

The release where the agentic layer arrived, the runtime grew one
execution site instead of four, and the Python bindings stopped lying
about what they contain.

### Breaking

**`soma.Graph` and `soma.Study` are Python classes now.** They subclass
the extension classes rather than being them. `isinstance(g, soma.Graph)`
and `isinstance(g, soma._soma.Graph)` both still hold; `type(g) is
soma._soma.Graph` does not. Nothing else about them changed — the surface
is the same one, only declared where a reader and a type checker can see
it instead of assigned at import time from seven modules.

**Persisted caches and trained states from 0.3.x will not be found.**
Three separate fixes moved the addresses:

- Trained states were keyed by `node_ids.join(",")`, so two graphs with
  the same node names but different edges shared a slot. They are keyed by
  the graph's shape now.
- `fit` consults the output cache, like `run` already did. A cold fit now
  reports misses where it previously reported nothing.
- Effect journal keys went through raw `serde_json` bytes with
  `unwrap_or_default()`, which gave every effect that failed to serialize
  the *same* key. They go through the canonical encoder.

There is no migration: delete `$SOMA_CACHE_DIR` (or `~/.soma/cache`) and
re-run. `soma cache purge-v1` removes pre-CAS entries specifically.

**The worker wire protocol is versioned and its binary frames changed.**
Workers and clients from 0.3.x cannot talk to 0.4.0 — a mismatch is now a
typed error instead of a corrupt decode. Upgrade both sides together. The
frames changed because msgpack was encoding structs positionally while
`Value` could only be read from named maps, so every frame carrying a
tensor failed to decode; both receivers swallowed the error, which is why
this was invisible rather than loud.

**`soma-core` no longer carries a tokio runtime.** The S3 and Zarr stores
moved to a new `soma-store` crate, feature-gated and off by default. Add
it explicitly if you used them. Execution of a `TrainingStrategy` moved to
`soma-runtime`, and `Study::save`/`load` with it; the *types* stayed.

**Errors are typed at the edges.** `soma-worker` and `soma-llm` have their
own error enums instead of producing `SomaError::Other(String)` for
everything — 51 of them in the worker alone, so nothing downstream could
tell a dropped socket from a dead Python interpreter. They still convert
to `SomaError` at the trait seams, keeping the prefix that says which
layer failed.

**A misspelled `#[soma(...)]` attribute is a compile error.** It used to
be a silent no-op, so `#[soma(serach(...))]` compiled and did nothing.

**Assorted renames**, each because the old name described something that
had stopped being true: `NodeCatalog`/`NodeRegistry` (the two registries
that filters and steps used to live in are one), `RunMode` instead of a
`fit: bool` flag on the remote path, and `KnowledgeBase` returning owned
values.

**Streaming — local AND remote — runs through the one execution site.**
Consequences, in decreasing order of how likely they are to reach you:

- *Stream runs' events changed shape*: one `NodeStarted`/`NodeCompleted`
  bracket per node (the completion summary aggregates chunk and
  hit/miss counts) and a real `NodeFailed` naming the chunk — instead of
  a single bracket for the whole plan under the last node's id. Anything
  parsing stream summaries of the form "N chunks through M filters"
  must read "stream: N chunks, H hits, M misses" per node now.
- *Streaming a non-linear graph or a step is a compile error* that
  names the node. It used to run a DAG as a chain silently — for a
  diamond, a wrong answer with no warning.
- *Filters declaring `cacheable: false` or `deterministic: false` are
  no longer cached per chunk* (they always should not have been). Their
  stale entries become dead weight; `soma cache gc` reclaims them.
- *`StreamMode::Evolving` lost its `checkpoint_every` field* — it wrote
  checkpoints under a colliding key that nothing ever read, and only
  two hardcoded constructors ever set it. `StreamMode` is also
  exhaustive now (no `#[non_exhaustive]`): a new mode should break
  every stream driver, not fall through to `FixedState`.
- *`StreamCache` and `StreamExecutor` left the public API* — the former
  was dead code; the latter is gone entirely. The worker's remote
  streaming (WS sessions and the DataStore auto-stream) drives the same
  `StreamRun` as the local path, holding the driver and its context
  alive between messages. Two remote behavior changes ride along: the
  DataStore auto-stream returns the CONCATENATED output (it used to
  return only the last chunk's), and remote streams are compiled with
  `compile_stream`, so a non-linear graph or a step is refused
  client-side by name.
- *`SerializedPlan` gains `seed`*, and the worker folds it into every
  cache key — remote runs (streamed or not) no longer share cache lines
  across a sweep's seeds. Additive field: older peers' plans arrive
  unseeded and behave as before.

`FixedState` chunk cache keys did **not** move: a single-chunk stream
and a plain forward of the same input share one cache line, pinned by
test.

### Added

- **An agentic layer.** `Step` sits beside `Filter`: sync `poll` returning
  a `Transition`, with a driver performing the effects. A filter memoizes
  by content; a step *journals*, so an effect is recorded once and
  replayed on resume rather than re-run. Every behaviour (LLM, tool,
  judge, router) is library code, and every pattern is a function in
  `soma.agentic` returning a `Graph`: `react`, `route`, `refine`,
  `debate`, `board`, `parallel_vote`, `orchestrate`, `self_consistency`.
  Steps can be written in Python — any object with `poll(ctx)`.
- **Suspend and resume.** A run can stop to wait for a human or an
  external event and pick up where it left off, replaying its journal.
- **Agent telemetry reaches the reader.** `RunView.agentic_activity()`
  (per-node turns, tokens, effects by label, tools, replays,
  suspensions), `RunView.agentic_timeline()` and `plot_agentic()` (an
  effect gantt), an "Agent activity" report section with a
  `soma-data-agentic` blob, and `RunSummary.conclusion.agent_cost` in
  the headline — so the experiment pool sees what a run spent. The
  emitters grew to match: `Suspended` carries cost-so-far,
  `AgentStepCompleted` fires on error exits too (marked `failed`), and
  `Spawn` emits `AgentSpawned` with its hierarchical child ids.
- **`soma-llm`**: one OpenAI-compatible client and a provider catalog as
  TOML data, covering ollama, HuggingFace, NVIDIA, Kimi, GLM, DeepSeek,
  Groq, vLLM and others. Retries live in the client, not the step — a 429
  is transport, not domain — with `Retry-After` honoured in both RFC forms
  and a wall-clock budget checked before sleeping. Structured output via
  `response_format` where the endpoint enforces it and a system-prompt
  append where it cannot, with one repair when the answer comes back
  malformed.
- **An experiment pool.** Every tracked run appends a record to
  `.soma/experiments.jsonl` with a deterministic conclusion, an
  architecture fingerprint, and the derivation move from its parent —
  runs are nodes, edges are the changes applied. BM25 + structural
  retrieval, `soma.checkout/head/detach/reindex`, and seven `kb_*` MCP
  tools.
- **A persistent cache, on by default.** Tiered memory LRU over a
  content-addressed store at `$SOMA_CACHE_DIR` or `~/.soma/cache`. Action
  records are kept forever; BLAKE3 blobs are evictable. `soma cache
  stats|gc|pin|verify|purge-v1`.
- **A visualization layer.** `RunReader` aggregates run directories into
  chart-ready structs; `soma.runs()`/`RunView` in Python; `soma.viz` (the
  `somatize[viz]` extra) with Optuna-named `study.plot_*` and `run.plot_*`;
  `soma runs|graph|report` on the CLI. `soma report` emits one
  self-contained HTML. Diagrams via `to_mermaid`/`to_graphviz`/`to_svg`,
  the SVG existing because notebooks strip `<script>`.
- **Intra-node gradient auditing** (`gradient_audit(inside=...)`):
  submodule hooks under hierarchical ids, with progressive disclosure from
  `inside=True` through `AuditScope(depth, patterns, ...)`. Plus
  channel-level diagnostics with CKA between declared channel groups.
- **A coordinator that tracks a real cluster.** Workers heartbeat every
  10s and the coordinator reaps whoever goes quiet; `/submit` *places*
  (returns a worker and takes a lease) rather than proxying, so tensor
  payloads go client→worker directly. Ships a `soma-coordinator` binary.
- **`soma.library`**: `Eval` (accuracy, exact match, token F1, top-k —
  scoring nothing is an error, not a 0.0), `Accumulator`, `Retriever`,
  `Compact`.
- **Type information.** The package ships `py.typed` and a hand-written
  `soma/_soma.pyi` for the extension; the Python layer above it is
  annotated in place. Because a hand-written stub can rot silently, a test
  compares it against the module that was actually built.
- **`soma-macros` has tests**, including compile-fail cases.

### Fixed

Bugs that were live, and invisible for a reason worth recording:

- **Streaming was broken for any frame carrying a tensor** (see the wire
  protocol note above). Both receivers swallowed the decode error.
- **A worker built a fresh venv per plan.** `ensure_env` was keyed on
  `plan_id`, a new timestamp every time, so nothing was ever reused and
  nothing cleaned up — one test pass left 17 GB behind. Keyed on the hash
  of the requirements now, which is what the lockfile inside it was
  already written for. Invisible while `requirements` was always empty,
  which was itself a bug: the binding read the list from `__main__`
  instead of the dict it had written it to, and swallowed the `NameError`.
- **`fit` could not run a graph containing a step** — it had its own
  execution walk that flattened the plan and knew only about filters.
  There is one execution site now, so `Parallel`, `Loop` and `Branch` work
  in `fit` the same way they work in `run`.
- **A forward pass followed the plan's node order rather than the graph.**
- **`soma.Worker.serve()` called `std::process::exit(0)` on shutdown**,
  killing the host Python interpreter. The worker also ran plans inline in
  an async handler while holding a std `Mutex`, serializing every plan and
  blocking the reactor.
- **The streaming cache ignored the run seed**, so every seed in a study
  shared a cache line whenever chunking was on.
- **`EventBus::emit` held a read guard while calling user code**, so a
  reentrant sink deadlocked.
- **`FileKnowledgeBase` had no `fsync` and no `flock`**, which is what
  corrupts a journal mid-file.
- **`JudgeStep` and `ResearchStep` ignored `stop_reason`**, turning a
  truncated answer into a silent zero.
- **An experiment record's `metrics` and `params` were `HashMap`s**, so the
  same run serialized to `experiments.jsonl` with a different key order in
  every process. `summary.rs` already used `BTreeMap` for the same thing.
  Now they agree, and a record round-trips byte-identically.
- **Compiler wildcards planned unknown node kinds as ordinary filters.**
  They fail loudly now; the two matches that could be made exhaustive
  were.

[0.4.0]: https://github.com/manucouto1/soma/compare/v0.3.1...HEAD
