---
title: Capabilities
description: One row per thing Soma can do, from the name a user types to the code that does it, the types to look up, and the debt hanging off it.
---

Every other page in this section cuts **horizontally**. The crate pages answer
"what is in `soma-runtime`?"; the [Codebase Map](/soma/internals/map/) and the
domain folders answer "where does caching live?". Neither answers the two
questions you usually arrive with:

> *What actually happens when I call `g.fit()`?*
>
> *What is known to be wrong with the cache?*

This page cuts **vertically**. One row per capability, crossing five things
that already exist separately and were never lined up: the entry points from
the [Surface Census](/soma/internals/surface/), the execution trace from
[Call Paths](/soma/internals/paths/), the types from the
[Symbol Index](/soma/internals/symbols/), the entries in
[Known Debt](/soma/internals/debt/), and the tests.

## How to use it

Find the row for what you are about to change. **Hops** is where to set a
breakpoint; **types** is what you will have to understand first; **debt** is
what someone already found and did not fix. If the debt column is empty that
means nothing is *recorded*, not that nothing is wrong.

## How it is kept honest

The page is generated from `docs/data/capabilities.json` by
[`gen-capabilities.mjs`](https://github.com/manucouto1/soma/blob/main/docs/scripts/gen-capabilities.mjs),
which refuses to render if any cross-reference is broken: an entry the surface
census does not list, a trace `data/traces.json` does not contain, a debt id
that is not a heading in the register, or a test path that matches nothing on
disk. The `file:line` hops are expanded into full repo paths so
`check-anchors.mjs` verifies them like every other anchor in this section.

A table of hand-typed cross-references rots in a week; this one fails the build
instead.

## What "trace" means, and when it is a dash

Execution traces are written down, by hand, in
[Execution](/soma/internals/execution/#execution-traces). A row links one when
it has one. **A dash means no trace has been written for that capability yet** —
not that it has no path. The gap is left visible on purpose: **6 of the 15**
rows below are untraced, and a plausible-looking trace nobody walked would be
worse than an honest hole.

That count is checked, not typed. It said "eight of the fifteen" for exactly
as long as it took to write one more trace, on a page whose whole argument is
that hand-typed cross-references rot — so `gen-capabilities.mjs` now fails if
this sentence and the table disagree.

## The capabilities

15 rows. **Entry** is what a user writes; **trace** links the
execution path when one is written down; **hops** are where the work actually
happens; **types** are what to look up in the
[Symbol Index](/soma/internals/symbols/); **debt** is what is known to be wrong
with it.

### Build a graph

Nodes, edges, and the two shapes that are not edges — a loop and a branch. Adding a node registers *what it is* separately from *where it sits*, which is why an `Agent` and a `Filter` are added by the same method.

| | |
|---|---|
| **Entry** | `Graph.node` · `Graph.edge` · `Graph.branch` · `Graph.loop` · `Graph.handoff` |
| **Trace** | — |
| **Hops** | `soma-python/src/graph/topology.rs:68` → `soma-python/src/graph/registry.rs:116` → `soma-python/src/graph/topology.rs:56` |
| **Types** | `Graph` · `Node` · `Edge` · `NodeKind` · `LoopCondition` |
| **Debt** | [D-56](/soma/internals/debt/#d-56--nodeid-is-a-string-and-so-is-everything-else) |
| **Tests** | `soma-python/tests/test_graph.py` · `soma-compiler/tests/control_flow.rs` |

### Run a graph

Compile to a plan, walk the plan, answer with the leaf that actually ran. One execution site for filters and steps alike; the only thing `fit` adds is that a trainable node learns before it computes.

| | |
|---|---|
| **Entry** | `Graph.forward` · `Graph.compile` |
| **Trace** | [(b)](/soma/internals/execution/#b-execute--run_node--the-four-primitives) |
| **Hops** | `soma-python/src/graph/execution.rs:220` → `soma-compiler/src/compiler.rs:1009` → `soma-runtime/src/execution/executor.rs:323` → `soma-runtime/src/execution/executor.rs:811` → `soma-runtime/src/execution/executor.rs:648` |
| **Types** | `ExecutionPlan` · `NodeCatalog` · `NodeOutcome` · `Runner` · `GraphInfo` |
| **Debt** | [D-03](/soma/internals/debt/#d-03--context-carries-five-unrelated-concerns) · [D-44](/soma/internals/debt/#d-44--resolve_input-falls-back-to-whatever-ran-last) · [D-96](/soma/internals/debt/#d-96--forwards-payload-type-still-depends-on-the-mode) |
| **Tests** | `soma-python/tests/test_integration.py` · `soma-runtime/tests/topology.rs` |

### Learn state from data

The same walk as a forward, in `RunMode::Fit`. What comes back says which half is which: what each node computed, and what each trainable node learned.

| | |
|---|---|
| **Entry** | `Graph.fit` · `Filter` · `Graph.get_node_state` |
| **Trace** | [(b)](/soma/internals/execution/#b-execute--run_node--the-four-primitives) |
| **Hops** | `soma-python/src/graph/execution.rs:32` → `soma-runtime/src/execution/runner/local.rs:25` → `soma-runtime/src/execution/executor.rs:995` → `soma-runtime/src/execution/runner/mod.rs:134` |
| **Types** | `Filter` · `FilterKind` · `Fitted` · `RunMode` · `StateStore` |
| **Debt** | [D-48](/soma/internals/debt/#d-48--valueempty-for-state-and-evolvings-valuestate-conflation) · [D-91](/soma/internals/debt/#d-91--the-filter-trait-mixes-computation-with-cache-identity) |
| **Tests** | `soma-runtime/tests/fit_answer.rs` · `soma-runtime/tests/fit_through_run_node.rs` · `soma-python/tests/test_composite_fit.py` |

### Memoize by content

The key is derived at runtime, per node, from the config, the state and the *content* of the input — so an unchanged upstream output cuts the run off early. A memory LRU in front of a persistent store, by default.

| | |
|---|---|
| **Entry** | `Graph(cache=)` · `Graph(cache_max_bytes=)` · `cache_stats` · `cache_gc` |
| **Trace** | [(b)](/soma/internals/execution/#b-execute--run_node--the-four-primitives) |
| **Hops** | `soma-runtime/src/execution/executor.rs:626` → `soma-runtime/src/execution/executor.rs:696` → `soma-runtime/src/execution/executor.rs:712` → `soma-runtime/src/cache/tiered.rs:46` → `soma-runtime/src/cache/fs_store.rs:310` |
| **Types** | `CacheKey` · `CacheStore` · `TieredCache` · `FsActionStore` · `MemoryCache` · `Origin` · `CacheTier` |
| **Debt** | [D-46](/soma/internals/debt/#d-46--tieredcache-promotion-destroys-provenance) · [D-62](/soma/internals/debt/#d-62--memorycaches-lru-touch-is-on-on-every-read) |
| **Tests** | `soma-python/tests/test_cache_resume.py` · `soma-python/tests/test_identity.py` · `soma-runtime/tests/integration.rs` |

### Run without materializing

Chunks through the same four primitives, one node's worth at a time. A single-chunk stream and a plain forward share one cache line, which is what makes the two paths the same path.

| | |
|---|---|
| **Entry** | `Graph.forward` |
| **Trace** | [(d)](/soma/internals/execution/#d-streamrun) |
| **Hops** | `soma-compiler/src/compiler.rs:1034` → `soma-runtime/src/execution/stream.rs:132` → `soma-runtime/src/execution/stream.rs:206` → `soma-runtime/src/execution/stream.rs:169` |
| **Types** | `StreamMode` · `StreamRun` · `StreamOutput` · `VirtualValue` |
| **Debt** | [D-48](/soma/internals/debt/#d-48--valueempty-for-state-and-evolvings-valuestate-conflation) |
| **Tests** | `soma-python/tests/test_worker_e2e.py` · `soma-runtime/tests/memory_usage.rs` |

### Loop, branch, hand over

The compiler claims a loop body and a branch arm by dominance, so each is compiled once — inside the construct — and never again as a top-level step. What a loop *carries* is separate from what tells it to *stop*.

| | |
|---|---|
| **Entry** | `Graph.loop` · `Graph.branch` · `Graph.handoff` · `Goto` |
| **Trace** | — |
| **Hops** | `soma-compiler/src/compiler.rs:727` → `soma-compiler/src/plan.rs:204` → `soma-runtime/src/execution/executor.rs:416` → `soma-runtime/src/execution/executor.rs:487` |
| **Types** | `LoopCondition` · `ExecutionPlan` · `Transition` · `NodeOutcome` |
| **Debt** | [D-42](/soma/internals/debt/#d-42--executionplanremote-discards-its-routing-target) |
| **Tests** | `soma-compiler/tests/control_flow.rs` · `soma-python/tests/test_python_steps.py` |

### Run effectful nodes

A step polls and returns a transition; a driver performs the effects and journals them. A pure effect is keyed by content, an impure one by `(run, node, turn, index)` — which is what makes a resumed run replay instead of re-calling a model.

| | |
|---|---|
| **Entry** | `Agent` · `Judge` · `Tool` · `agentic` · `Graph.register_step` |
| **Trace** | [(c)](/soma/internals/execution/#c-the-effectdriver-turn-loop) |
| **Hops** | `soma-python/src/graph/agentic.rs:15` → `soma-runtime/src/agentic/mod.rs:110` → `soma-runtime/src/agentic/mod.rs:522` → `soma-runtime/src/agentic/graph_handler.rs:141` |
| **Types** | `Step` · `StepCtx` · `Transition` · `Effect` · `EffectHandler` · `EffectJournal` · `EffectDriver` |
| **Debt** | [D-22](/soma/internals/debt/#d-22--a-suspension-reason-that-fails-to-serialize-collides-with-every-other-one) · [D-54](/soma/internals/debt/#d-54--nine-string-match-dispatch-sites-across-the-ffi) · [D-57](/soma/internals/debt/#d-57--prose-parsing-as-control-flow) |
| **Tests** | `soma-python/tests/test_agentic.py` · `soma-python/tests/test_python_steps.py` · `soma-runtime/tests/agentic_step.rs` |

### Pause and resume

A step that needs an answer nobody can compute suspends. The journal is the checkpoint: running the graph again replays every prior effect from the record, reaches the pause, and finds the answer waiting.

| | |
|---|---|
| **Entry** | `Suspend` · `Graph.resume` · `SomaSuspended` |
| **Trace** | [(c)](/soma/internals/execution/#c-the-effectdriver-turn-loop) |
| **Hops** | `soma-runtime/src/agentic/mod.rs:190` → `soma-python/src/graph/agentic.rs:116` → `soma-runtime/src/execution/executor.rs:765` |
| **Types** | `SuspendReason` · `NodeOutcome` · `EffectJournal` · `ActionCache` |
| **Debt** | [D-22](/soma/internals/debt/#d-22--a-suspension-reason-that-fails-to-serialize-collides-with-every-other-one) |
| **Tests** | `soma-python/tests/test_suspend_resume.py` · `soma-python/tests/test_crash_replay.py` |

### Search hyperparameters

A study samples a space, runs a trial, and reads what the trial reported. The space is collected from the graph — a filter's `search(...)` config, an agent's prompt, and an optional edge, which makes topology a dimension.

| | |
|---|---|
| **Entry** | `Study` · `search` · `Trial` · `Graph.study` · `Graph.search_space` |
| **Trace** | [(e)](/soma/internals/execution/#e-studyrunner-and-pbtrunner) |
| **Hops** | `soma-runtime/src/optimizer/study.rs:187` → `soma-core/src/optimizer/search.rs:172` → `soma-runtime/src/optimizer/study.rs:281` |
| **Types** | `Study` · `Trial` · `SearchSpace` · `SearchDimension` · `Sampler` · `Pruner` · `TrialOutcome` · `Searchable` |
| **Debt** | [D-14](/soma/internals/debt/#d-14--two-pruners-two-samplers-one-algorithm-each) · [D-64](/soma/internals/debt/#d-64--studyrunnerrun-is-otrials-in-four-places) · [D-93](/soma/internals/debt/#d-93--trials-run-one-at-a-time) |
| **Tests** | `soma-python/tests/test_study.py` · `soma-python/tests/test_hpo_ux.py` · `soma-python/tests/test_agentic_search.py` |

### Record what a run did

Every event goes to a lossless sink before it goes anywhere else. `begin_run` is the single writer of the topology snapshot — the one place where the graph and the catalog are both in scope, so the only place that can stamp per-node config hashes.

| | |
|---|---|
| **Entry** | `Graph.track_run` · `Graph.begin_run` · `Run` · `runs` · `RunView` |
| **Trace** | [(f)](/soma/internals/execution/#f-a-tracked-run-track_run--the-run-directory--experimentsjsonl) |
| **Hops** | `soma-python/src/graph/tracking.rs:10` → `soma-runtime/src/tracking/local_tracker.rs:36` → `soma-runtime/src/tracking/event_bus.rs:59` → `soma-runtime/src/tracking/jsonl_sink.rs:125` → `soma-runtime/src/tracking/reader.rs:266` |
| **Types** | `Event` · `EventSink` · `EventBus` · `Tracker` · `LocalTracker` · `JsonlEventSink` · `RunReader` · `RunManifest` |
| **Debt** | [D-05](/soma/internals/debt/#d-05--event-is-30-variants-across-six-unrelated-concerns) · [D-07](/soma/internals/debt/#d-07--runreader-is-17-methods-over-one-pathbuf) · [D-63](/soma/internals/debt/#d-63--runreader-re-parses-eventsjsonl-once-per-accessor) |
| **Tests** | `soma-runtime/tests/tracking.rs` · `soma-python/tests/test_runs_reader.py` · `soma-python/tests/test_run_api.py` |

### Save and restore

A checkpoint is the learned state of every node plus the manifest needed to rebuild the topology — not a pickle of the graph. `save`/`load` round-trip through that, so a reloaded graph is the same graph and not a lookalike.

| | |
|---|---|
| **Entry** | `Graph.save` · `Graph.load` · `Graph.state` · `Graph.load_state` · `Graph.restore_optimizer` |
| **Trace** | [(g)](/soma/internals/execution/#g-a-checkpoint-save-the-topology-and-the-weights-load-them-back) |
| **Hops** | `soma-python/python/soma/_checkpoint.py:60` → `soma-python/python/soma/_checkpoint.py:166` → `soma-python/python/soma/_checkpoint.py:224` → `soma-python/python/soma/_checkpoint.py:75` → `soma-python/src/graph/registry.rs:333` |
| **Types** | `StateStore` · `NodeCatalog` · `ArchitectureFingerprint` |
| **Debt** | none recorded |
| **Tests** | `soma-python/tests/test_checkpoint.py` |

### See the graph and the run

Pure data → string: no runtime, no I/O. SVG exists because a notebook sanitizes `<script>`, so the Mermaid a terminal reader is happy with cannot be what `_repr_html_` returns. An overlay folds a run's status onto the same renderer.

| | |
|---|---|
| **Entry** | `Graph.to_mermaid` · `Graph.to_svg` · `Graph.to_text` · `Graph.architecture` · `RunView` |
| **Trace** | — |
| **Hops** | `soma-python/src/graph/viz.rs:12` → `soma-core/src/graph/mod.rs:547` → `soma-core/src/viz/svg.rs:65` → `soma-runtime/src/tracking/reader.rs:763` → `soma-python/python/soma/viz/_report.py:178` |
| **Types** | `GraphOverlay` · `NodeStatus` · `RunReader` |
| **Debt** | [D-17](/soma/internals/debt/#d-17--four-renderers-four-independent-match-nodekind) · [D-15](/soma/internals/debt/#d-15--five-formatters-for-a-duration-two-for-a-truncation) |
| **Tests** | `soma-python/tests/test_viz.py` · `soma-python/tests/test_viz_report.py` · `soma-python/tests/test_viz_health.py` |

### Watch the gradients

Hooks under hierarchical ids (`node/module.path`), rolled up into one health flag per family. The ids are opaque strings end to end — the hierarchy travels in the report, never parsed back out of a name.

| | |
|---|---|
| **Entry** | `Graph.gradient_audit` · `audit_modules` · `Audit` · `AuditScope` · `Thresholds` |
| **Trace** | — |
| **Hops** | `soma-python/python/soma/_audit.py:1239` → `soma-python/python/soma/_audit.py:1016` → `soma-python/src/graph/tracking.rs:122` → `soma-runtime/src/tracking/reader.rs:518` |
| **Types** | `HealthFlagRecord` · `RunReader` · `Event` |
| **Debt** | [D-09](/soma/internals/debt/#d-09--audit-is-a-30-method-class-in-a-1-338-line-module) |
| **Tests** | `soma-python/tests/test_gradient_audit.py` · `soma-python/tests/test_diagnostics.py` · `soma-python/tests/test_pathologies.py` |

### Run it on another machine

A plan crosses the wire with the cloudpickle bytes of its filters, because a catalog holds live filters and never the pickle. A strategy is the other shape: the coordinator drives rounds and the workers keep their catalogs between messages.

| | |
|---|---|
| **Entry** | `Graph.add_worker` · `Graph.set_strategy` · `Graph.set_data_store` · `Worker` |
| **Trace** | — |
| **Hops** | `soma-python/src/graph/distributed.rs:171` → `soma-worker/src/worker.rs:275` → `soma-python/src/graph/distributed.rs:114` → `soma-runtime/src/distributed.rs:146` |
| **Types** | `TrainingStrategy` · `Partition` · `RemoteTarget` · `Transport` · `DataStore` · `DataRef` · `StrategyExecutor` |
| **Debt** | [D-02](/soma/internals/debt/#d-02--worker-and-workerexecute_plan) · [D-41](/soma/internals/debt/#d-41--transportexecute_node-runs-remotes-with-an-empty-catalog) · [D-42](/soma/internals/debt/#d-42--executionplanremote-discards-its-routing-target) · [D-55](/soma/internals/debt/#d-55--set_strategy--strategy-is-a-lossy-round-trip) |
| **Tests** | `soma-python/tests/test_worker_e2e.py` · `soma-runtime/src/distributed.rs` |

### Remember what was tried

Every tracked run appends a record with its conclusion, its architecture fingerprint and the move that derived it from its parent. Nodes are runs, edges are the changes — so retrieval can answer "what did I already try that looked like this?"

| | |
|---|---|
| **Entry** | `experiments` · `find_similar` · `lineage` · `diff` · `checkout` · `record_conclusion` |
| **Trace** | — |
| **Hops** | `soma-memory/src/record.rs:169` → `soma-memory/src/file_kb.rs:128` → `soma-memory/src/retrieval.rs:200` |
| **Types** | `ExperimentRecord` · `DerivationMove` · `RetrievalQuery` · `ScoreComponents` · `KnowledgeBase` · `ArchitectureFingerprint` |
| **Debt** | [D-16](/soma/internals/debt/#d-16--two-knowledge-base-front-ends-already-divergent) |
| **Tests** | `soma-python/tests/test_experiment_pool.py` · `soma-python/tests/test_experiments.py` · `soma-memory/src/retrieval.rs` |
