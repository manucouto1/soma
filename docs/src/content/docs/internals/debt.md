---
title: Known Debt
description: A register of code smells and antipatterns in the Soma workspace, with file:line evidence and the shape of each fix.
---

## Why this page exists

Every finding below was read out of the source, not inferred from a design
document. Each entry names a file and a line, says what it costs, and sketches
the shape of a fix. Nothing here is a plan — it is the evidence a plan would be
built from.

The register is grouped **by smell class**, not by crate, because that is how
the work gets done: you fix all four copies of `write_atomic` in one sitting,
not one per crate over four months. Within a class, entries are ranked by
consequence.

Read [Design Patterns](/soma/internals/patterns) for what the codebase does
*well* — and read the [closing section](#what-is-already-healthy) of this page
before drawing conclusions. A debt list read alone always reads worse than the
code is.

### Severity

| | Meaning |
|---|---|
| **High** | Produces wrong results, panics, or makes a whole subsystem unreachable. Or: blocks the refactor everything else waits on. |
| **Medium** | Costs real time on every change to the area, or degrades silently in a way a user would not notice. |
| **Low** | Friction, inconsistency, or rot. Worth fixing when already in the file. |

### Resolved entries

An entry that has been fixed keeps its heading and its number, and gains a
**Resolved** line saying what changed. Nothing is renumbered and nothing is
deleted: eleven pages link into this register by anchor, and a closed finding
still answers "was this ever considered?" — which is most of what a register is
for. [`check-debt-refs.mjs`](https://github.com/manucouto1/soma/blob/main/docs/scripts/check-debt-refs.mjs)
is what keeps those anchors honest.

### The ten that matter most

| ID | Finding | Severity |
|---|---|---|
| [D-31](#d-31--zarrstores-chunk-cache-is-write-only) | `ZarrStore`'s local chunk cache is never read — every `get` goes back to S3 | High |
| [D-01](#d-01--pygraph-is-the-workspaces-god-object) | `PyGraph`: 2 458 lines, 19 fields, ~47 public methods | High |
| [D-11](#d-11--the-stream-path-re-implements-run_node-and-has-drifted) | Stream execution re-implements `run_node` and has drifted — no cache events | High |
| [D-21](#d-21--mean_by_key-panics-on-an-empty-slice) | `mean_by_key` panics on an empty slice, reachable from Python | High |
| [D-32](#d-32--the-compiler-never-descends-into-loop-or-branch) | Compiler never descends into `Loop`/`Branch` for distribution or fusion | High |
| [D-41](#d-41--transportexecute_node-runs-remotes-with-an-empty-catalog) | `Transport::execute_node` runs every remote node with an empty catalog and an unsalted key | High |
| [D-02](#d-02--worker-and-worker-execute_plan) | `Worker::execute_plan`: 324 lines across nine responsibilities | High |
| [D-33](#d-33--value-to_plain_json-contradicts-its-own-contract) | `Value::to_plain_json` emits the tagged encoding it promises never to emit | Medium |
| [D-12](#d-12--four-write_atomic-implementations-two-of-them-unsafe) | Four `write_atomic` implementations, two without fsync or unique temp names | Medium |
| [D-61](#d-61--contextsnapshot-deep-clones-the-value-store-per-branch) | `Context::snapshot` deep-clones the whole value store per parallel branch | Medium |

---

## God objects

A type is on this list when it owns concerns that do not need each other, so
that touching one forces you to read all of them.

### D-01 · `PyGraph` is the workspace's god object

**Class** God object · **Severity** High · **Crate** `soma-python`

**Evidence** `soma-python/src/graph.rs:27` — 2 458 lines, **19 fields**,
~47 public `#[pymethods]` plus 22 private helpers. It is simultaneously the
graph builder, the filter registry, the cache owner, the event-bus owner, the
worker-pool manager, the coordinator client, the data-store owner, the renderer,
the run tracker and the executor.

Five of the 19 fields are **parallel maps keyed by node id** — `pickled_filters`,
`filter_sources`, `filter_trainable`, `live_filters`, `live_steps` — all written
together in `register_behaviour` (`soma-python/src/graph.rs:256`) and never
removed from. There is no node-removal API, which is the only reason they cannot
drift apart.

`PyGraph::fit` (`soma-python/src/graph.rs:1371`) is **262 lines** with five
distinct execution paths, and the tail

```rust
for (node_id, state) in states { self.library.try_set_state(node_id, state)?; }
self.fitted = true;
```

appears **five times** in it (`:1460`, `:1498`, `:1518`, `:1534`, `:1616`).

**Consequence** Any change to how a graph is built, run or distributed lands in
this one file. It is the single biggest obstacle to understanding the Python
API from the Rust side.

**Fix shape** Extract the node registry (the five maps) into one `NodeRecord`
struct in its own module; extract the worker/coordinator fields into a
`Distribution` struct; give `fit` one outcome-handling tail instead of five.
None of these is behaviour-changing.

### D-02 · `Worker` and `Worker::execute_plan`

**Class** God object · **Severity** High · **Crate** `soma-worker`

**Evidence** `soma-worker/src/worker.rs:16` — 11 fields covering identity,
capabilities, event bus, cache, node catalog, *two* data stores, env manager and
interpreter path. `Worker::execute_plan` (`soma-worker/src/worker.rs:275`) is
**324 lines** performing: protocol version check → requirement collection → venv
provisioning → subprocess spawn → state loading → filter registration → input
resolution → mode dispatch → streaming decision → state harvesting → result
wrapping.

**Consequence** Nine reasons to change one function. The two silent-degradation
bugs below ([D-24](#d-24--venv-provisioning-fails-into-the-system-interpreter),
[D-25](#d-25--state-load-failure-silently-restarts-from-random-init)) both live
inside it, which is not a coincidence — a 324-line function is where a `warn!`
and a fallback look reasonable.

**Fix shape** The stages are already sequential and independent. Extract each to
a named private function taking and returning explicit state; the function body
becomes the pipeline it already is.

### D-03 · `Context` carries five unrelated concerns

**Class** God object · **Severity** Medium · **Crate** `soma-runtime`

**Evidence** `soma-runtime/src/executor.rs:124` — 12 fields: run mode; the value
store + write order + input-hash memo; event bus + run id; graph topology;
distributed transport; data store + spill threshold; effect driver. Eight
consuming builders at `:195`–`:245`. It is a parameter of `execute`,
`execute_loop`, `execute_branch`, `execute_remote`, `execute_parallel`,
`execute_stream`, `run_node`, `fit_state_if_needed`, `composite_fit`,
`run_node_inner`, `resolve_input` and `StreamRun::run_compute`.

Worse, it overlaps `RunContext` (`soma-runtime/src/runner/mod.rs:32`) — five
fields are copied across field-by-field at `soma-runtime/src/runner/local.rs:33`
— and `ForwardEnv` (`soma-runtime/src/forward.rs:25`) is a third struct with the
same "avoid six parameters" justification.

**Consequence** Three context types with overlapping contents; a reader has to
learn which one carries what before reading any execution code.

**Fix shape** Split `Context` into an immutable `RunEnv` (mode, ids, bus,
topology, transport, store, driver) and a mutable `ValueStore` (store,
execution order, hash memo). `RunContext` and `ForwardEnv` then become views of
`RunEnv` rather than parallel structs.

### D-04 · `GraphSession` has two unrelated transport fields

**Class** God object · **Severity** Medium · **Crate** `soma-runtime`

**Evidence** `soma-runtime/src/graph_session.rs:38` — 10 fields, 20 methods, and
both `transport: Option<Arc<dyn Transport>>` (`:44`, used by `ExecutionPlan::Remote`
nodes) and `transports: Vec<Arc<dyn Transport>>` (`:50`, used by training
strategies). No invariant ties them together; a caller can set one and not the
other and get a plan that half-executes remotely.

**Fix shape** One `Vec` with the single-transport case as `len() == 1`, or an
explicit enum naming the two modes.

### D-05 · `Event` is 30 variants across six unrelated concerns

**Class** God object · **Severity** Medium · **Crate** `soma-core`

**Evidence** `soma-core/src/tracking/event.rs:54` — 30 variants spanning pipeline
execution, trials, studies, PBT generations, training telemetry and the agentic
loop. Every consumer either matches all 30 or writes `_ => {}`; the readers do
the latter (`soma-runtime/src/tracking/reader.rs:520`,
`soma-runtime/src/tracking/jsonl_sink.rs:79`). The naming collision it has already
had to work around is documented at `soma-core/src/tracking/event.rs:376` —
`StepCompleted` is an optimizer step, `AgentStepCompleted` is an agent step.

**Consequence** The one enum is also the JSONL wire format, so splitting it is a
schema change, not a refactor. That is why it is Medium and not High: the cost
is real but the fix is expensive.

**Fix shape** If it is ever split, split it into `RunEvent` / `StudyEvent` /
`AgentEvent` with a wrapper enum for the sink, and bump `RUN_SCHEMA_VERSION`
(`soma-core/src/tracking/mod.rs:18`).

### D-06 · Wide data structs with no builder

**Class** God object · **Severity** Low · **Crate** `soma-core`

**Evidence** `RunManifest` (`soma-core/src/tracking/mod.rs:97`) — 20 fields, one
3-argument constructor; the other 17 are set by field assignment.
`RunSummary` (`soma-core/src/tracking/summary.rs:331`) — 17 fields.
`Study` (`soma-core/src/optimizer/study.rs:319`) — 15 fields mixing definition, results and
provenance, with builders covering 2 of them.
`ExperimentRecord` (`soma-memory/src/record.rs:53`) — 26 fields, but this one
*does* have 12 `with_*` builders and is the model the others should follow.

### D-07 · `RunReader` is 17 methods over one `PathBuf`

**Class** God object · **Severity** Low · **Crate** `soma-runtime`

**Evidence** `soma-runtime/src/tracking/reader.rs:36` — one field, 17 public
accessors, 10 distinct DTO shapes. See also [D-63](#d-63--runreader-re-parses-eventsjsonl-once-per-accessor),
which is the performance consequence of the same design.

### D-08 · `SomaContext` mixes three subsystems

**Class** God object · **Severity** Low · **Crate** `soma-mcp`

**Evidence** `soma-mcp/src/context.rs:9` — 13 handlers spanning knowledge-base
access, filesystem CRUD on user source files, and subprocess execution.
`generate_report` alone is 73 lines (`:464`).

### D-09 · `Audit` is a 30-method class in a 1 338-line module

**Class** God object · **Severity** Medium · **Crate** `soma-python` (Python layer)

**Evidence** `soma-python/python/soma/_audit.py:356` — installs torch hooks, computes
channel statistics and CKA, persists JSON to disk (`_persist_step` `:680`,
`_persist_module_trees` `:694` — 88 lines, `_persist_snapshot` `:782`), aggregates
and reports. Instrumentation, statistics and I/O in one class.

**Fix shape** The three concerns already have clean seams: hooks produce
`StepRecord`s, statistics consume them, persistence writes them. Split along
those lines.

---

## Duplicated logic

### D-11 · The stream path re-implements `run_node` and has drifted

**Class** Duplication · **Severity** High · **Crate** `soma-runtime`

**Evidence** `soma-runtime/src/executors/stream.rs:1` opens by claiming the three
shared primitives (`output_key` / `compute_node` / `store_output`) eliminate the
duplication between batch and stream execution. They eliminate the *middle* of
it. `StreamRun::run_compute` (`soma-runtime/src/executors/stream.rs:194`)
re-implements everything around them that `run_node`
(`soma-runtime/src/executor.rs:816`) also does:

| Step | `run_node` | `StreamRun::run_compute` |
|---|---|---|
| emit `NodeStarted` | `executor.rs:884` | `stream.rs:203` |
| resolve state | `executor.rs:842` | `stream.rs:213` |
| input hash | `ctx.input_hash(..)` — memoized | `CacheKey::for_value(..)` — never memoized |
| cache hit | `executor.rs:862` — **emits `NodeCacheHit`** | `stream.rs:220` — **counts only** |
| cache miss | `executor.rs:876` — **emits `NodeCacheMiss`** | `stream.rs:229` — counter only |
| `NodeCompleted` | per node, `executor.rs:921` | aggregated in `finish()`, `stream.rs:167` |
| `HandOff` / `Paused` | handled, `executor.rs:931` | hard error, `stream.rs:257` |

**Consequence** This is behavioural drift, not just duplication. **A streamed run
emits no cache events at all**, so `RunReader::cache_activity`
(`soma-runtime/src/tracking/reader.rs:403`) reports zero cache activity for every
streamed run while `node_timings` still produces spans. The diamond-sharing
input-hash memoization at `soma-runtime/src/executor.rs:322` also simply does not
exist on the stream path.

**Fix shape** Push the event emission and the hash memo down into the primitives
so both callers get them, or lift a fourth primitive (`observe_node`) that both
paths call. The remote worker drives the same `StreamRun`
(`soma-worker/src/worker.rs`), so the fix repairs remote streaming too.

### D-12 · Four `write_atomic` implementations, two of them unsafe

**Class** Duplication · **Severity** Medium · **Crate** `soma-runtime`

**Evidence**

| Site | Temp name | fsync |
|---|---|---|
| `soma-runtime/src/cache/local.rs:69` | pid + `AtomicU64` seq | yes |
| `soma-runtime/src/cache/fs_store.rs:232` | pid + `AtomicU64` seq | yes |
| `soma-runtime/src/tracking/local_tracker.rs:214` | fixed `.json.tmp` | **no** |
| `soma-runtime/src/tracking/head.rs:46` | fixed `.tmp` | **no** |

The first two are character-for-character identical, including the `static
WRITE_SEQ` and the comment. The third and fourth are weaker variants: a fixed
temp name means two concurrent writers to the same run directory collide, and
without `sync_all` a crash can leave a truncated file that reads as valid JSON
right up to the tear.

**Fix shape** One `soma-runtime/src/fsutil.rs` with the strong version; four call
sites.

### D-13 · `GraphSession` emits the run bracket four times

**Class** Duplication · **Severity** Medium · **Crate** `soma-runtime`

**Evidence** `RunStarted` / `RunCompleted` / `RunFailed` is written by hand at
`soma-runtime/src/graph_session.rs:186`, `:192`, `:198` (in `run`), then again at
`:222`, `:248`, `:255` (the strategy branch of `fit`), then again at `:278`,
`:285` (the local branch of `fit`).

The two `fit` branches also diverge in a way that looks accidental: the strategy
path calls `try_set_state` for every returned state (`:245`) and returns them
all; the local path filters `__state_` keys into the catalog and then **strips
them from the return value** (`:294`). So `fit()` returns node outputs locally
and trained states remotely.

**Fix shape** One `with_run_bracket(run_id, || …)` helper, and one decision about
what `fit` returns.

### D-14 · Two pruners, two samplers, one algorithm each

**Class** Duplication · **Severity** Low · **Crate** `soma-runtime`

**Evidence** `MedianPruner::should_prune` (`soma-runtime/src/pruner.rs:59`) and
`PercentilePruner::should_prune` (`soma-runtime/src/pruner.rs:128`) are the same
20 lines differing only in the final statistic; their `values_at_step` collection
loops are byte-identical. `RandomSampler::sample_dim`
(`soma-runtime/src/sampler/mod.rs:190`) and `BayesianSampler::sample_uniform`
(`soma-runtime/src/sampler/bayesian.rs:99`) are the same uniform-sampling switch
with slightly different clamping.

### D-15 · Five formatters for a duration, two for a truncation

**Class** Duplication · **Severity** Low · **Crate** cross-cutting

**Evidence** `soma-core/src/viz/mod.rs:140` `format_duration_ms` and
`soma-core/src/tracking/summary.rs:393` `human_duration` **disagree on the same input**:
187 000 ms renders as `"3.1m"` from one and `"3m 07s"` from the other. Python adds
three more: `soma-python/python/soma/_runs.py:372` `_fmt_ms_html`,
`soma-python/python/soma/viz/_report.py:123` `_fmt_ms`,
`soma-python/python/soma/_cache_cli.py:44` `_fmt_duration`.

Truncation is written twice: `soma-core/src/util.rs:62` `truncate` and
`soma-core/src/tracking/summary.rs:422` `one_line`.

### D-16 · Two knowledge-base front-ends, already divergent

**Class** Duplication · **Severity** Medium · **Crates** `soma-mcp`, `soma-python`

**Evidence** `soma-mcp/src/tools/knowledge.rs:24` (`find_similar`) and
`soma-python/src/readers.rs:85` (`kb_find_similar_json`) build the same
`RetrievalQuery` from the same parameter names — and have already drifted. MCP
clamps `limit` to `1..=50` (`soma-mcp/src/tools/knowledge.rs:41`); Python clamps
to `1..=100` (`soma-python/src/readers.rs:103`). The error texts differ too. The
same pairing exists for `kb_lineage`, `kb_diff` and `kb_record_conclusion`.

**Fix shape** One query-building function in `soma-memory`, called by both.

### D-17 · Four renderers, four independent `match &node.kind`

**Class** Duplication · **Severity** Low · **Crate** `soma-core`

**Evidence** `Graph::to_mermaid_with` (`soma-core/src/graph/mod.rs:553`),
`to_text` (`:628`) and `svg::to_svg_with` (`soma-core/src/viz/svg.rs:65`) each
map the five `NodeKind`s to a shape independently. `ExecutionPlan` repeats the pattern: `mermaid_nodes`
(`soma-compiler/src/plan.rs:271`) and `graph_nodes` (`:407`) duplicate a whole
recursive walk, and the 10-line comment at `soma-compiler/src/plan.rs:264`
acknowledges it and declines to unify.

### D-18 · Two worker-capability models in one workspace

**Class** Duplication · **Severity** Low · **Crates** `soma-compiler`, `soma-coordinator`

**Evidence** `WorkerInfo` with `has_capacity` / `matches_tag`
(`soma-compiler/src/scheduler.rs:15`) and `WorkerStatus` with `has_capacity` /
`matches_tags` (`soma-coordinator/src/registry.rs:22`). Two independent
placement models; only one of them is wired to anything
([D-65](#d-65--the-schedulers-capability-model-is-unimplemented)).

### D-19 · Two embedded Python interpreters as Rust string constants

**Class** Duplication · **Severity** Medium · **Crates** `soma-worker`, `soma-mcp`

**Evidence** `soma-worker/src/python_process.rs:19` — `const DAEMON_SCRIPT: &str`,
~515 lines of Python. `soma-mcp/src/exec.rs:26` — `const DRIVER: &str`, ~225 lines.
Neither is syntax-checked, linted, type-checked or unit-tested; both dispatch on
strings (`action == "LOAD" | "FIT" | "FORWARD" | …`, else `unknown command` at
`soma-worker/src/python_process.rs:527`).

`DRIVER`'s `_searchable` (`soma-mcp/src/exec.rs:96`) is a **third** encoding of
"a constructor argument may be a search dimension", after
`soma-python/src/agentic.rs:212` in Rust and
`soma-python/python/soma/search.py:4` in Python.

**Fix shape** Move both scripts to real `.py` files in the crate, `include_str!`
them, and point the test suite at them. That alone buys syntax checking and
linting for 740 lines that currently have neither.

---

## Latent panics and silent failures

### D-21 · `mean_by_key` panics on an empty slice

**Class** Latent panic · **Severity** High · **Crate** `soma-runtime`

**Evidence** `soma-runtime/src/strategy.rs:439`:

```rust
fn mean_by_key(what: &str, entries: &[HashMap<String, Value>]) -> Result<...> {
    let mut out = HashMap::new();
    for key in entries[0].keys() {          // panics on an empty slice
```

`StateAggregator for FederatedAggregation` guards it
(`soma-runtime/src/strategy.rs:488`, "aggregation over zero clients").
`GradientAggregator for GradientAggregation` does **not**
(`soma-runtime/src/strategy.rs:461`): `len() == 1` short-circuits and `len() == 0`
falls straight through.

Reachable path: `TrainingStrategy::DataParallel { num_replicas: 0, .. }` →
`soma-runtime/src/strategy.rs:163` computes `n = 0` → the fit and gradient loops
never run → `aggregate(&[])` panics. `num_replicas` is user-supplied from Python
(`soma-python/src/graph.rs:2094`, `num_replicas.unwrap_or(1)` — an explicit `0`
passes straight through) and is validated nowhere.

**Fix shape** Reject `num_replicas == 0` at the Python boundary *and* return an
error from `mean_by_key` on an empty slice. Both, not either.

### D-22 · A suspension reason that fails to serialize collides with every other one

**Class** Silent failure · **Severity** Medium · **Crate** `soma-runtime`

**Evidence** `soma-runtime/src/effects/mod.rs:47` —
`serde_json::to_value(reason).unwrap_or(Null)` builds the journal key for a
suspension. Two different unserializable reasons produce the same `Null` payload,
hence the same key, so a resume replays the wrong answer into the wrong
suspension.

### D-23 · A serialization failure gives every failing value the same cache key

**Class** Silent failure · **Severity** Medium · **Crate** `soma-core`

**Evidence** `soma-core/src/cache/mod.rs:111` —
`serde_json::to_vec(json).unwrap_or_default()` inside `CacheKey::absorb`. Any
JSON value that fails to serialize contributes the empty byte string to the hash,
so all of them collide. The adjacent comment argues this is unreachable; it is
the one silent fallback left on the key path, and the key path is where a silent
fallback is least acceptable.

### D-24 · Venv provisioning fails into the system interpreter

**Class** Silent failure · **Severity** Medium · **Crate** `soma-worker`

**Evidence** `soma-worker/src/worker.rs:328` — on failure, `tracing::warn!` and
fall back to `self.python.clone()`. The plan then runs with the wrong
dependencies and fails later, inside user code, with an error that points
nowhere near the cause.

### D-25 · State-load failure silently restarts from random init

**Class** Silent failure · **Severity** High · **Crate** `soma-worker`

**Evidence** `soma-worker/src/worker.rs:385` — `tracing::warn!("Failed to load
state (will use fresh weights)")`. A resumed epoch quietly restarts from random
initialization. Nothing in the returned metrics distinguishes this from a
genuinely bad run.

**Fix shape** Fail. A resume that cannot resume is not a resume.

### D-26 · `JudgeStep` scores an unparseable reply as 0.0

**Class** Silent failure · **Severity** Medium · **Crate** `soma-llm`

**Evidence** `soma-llm/src/steps.rs:562` — a missing or non-numeric `score`
becomes `0.0`. `reject_non_answers` (`:614`) catches truncation and refusals, but
a well-formed reply whose score is prose still scores zero with the raw text as
the reason, which reads exactly like a genuine rejection.

### D-27 · `unwrap` inside a detached thread makes bind failures unreportable

**Class** Latent panic · **Severity** Medium · **Crate** `soma-python`

**Evidence** `soma-python/src/worker.rs:202` `Runtime::new().unwrap()`, `:262` and
`:266` `serve_worker*(…).await.unwrap()` — all inside `std::thread::spawn`. A port
already in use surfaces at `:290` as `"worker thread panicked"` with no cause.

### D-28 · `unwrap` on the JSON-RPC hot path

**Class** Latent panic · **Severity** Low · **Crate** `soma-mcp`

**Evidence** `soma-mcp/src/server.rs:57` and `:80` —
`serde_json::to_value(result).unwrap()`. A non-serializable `ToolCallResult`
aborts the server loop rather than returning an error to the model.

### D-29 · Macro-generated code panics inside the cache-key path

**Class** Latent panic · **Severity** Medium · **Crate** `soma-macros`

**Evidence** `soma-macros/src/lib.rs:85` and `:580` — the derived `config_hash`
calls `canonical_bytes(&self.field).unwrap_or_else(|e| panic!(…))`. A user's
filter or step with a field that is not CBOR-serializable panics inside
`config_hash()`, which the executor calls on every node.

**Fix shape** The panic is arguably right (an unhashable config is a programming
error), but it should be a compile-time bound or a clear `CacheConfigError`, not
a panic from generated code the user never wrote.

---

## Dead and unreachable API

Verified by grepping the whole workspace, including tests and the Python
bindings.

### D-31 · `ZarrStore`'s chunk cache is write-only

**Class** Dead code / correctness · **Severity** High · **Crate** `soma-store`

**Evidence** `soma-store/src/zarr.rs:515`:

```rust
fn key_from_path(&self, array_path: &str) -> CacheKey {
    let hex = array_path.strip_prefix(&self.prefix).unwrap_or(array_path);
    CacheKey::hash_data(hex.as_bytes())     // hashes the hex, does not decode it
}
```

`put_tensor` writes chunks to `local_cache/<real hex>/c_i`
(`soma-store/src/zarr.rs:411`). Every read path — `get` (`:669`), `get_rows`
(`:685`), `meta` (`:698`), `append` (`:548`) — derives the directory through
`key_from_path`, producing `local_cache/<sha256-of-the-hex>/…`. The two never
coincide.

**Consequence** The LRU-tracked chunk cache is written and never read: every
`get` goes back to S3, while the LRU accounts bytes nobody will ever hit. And
`remove` (`:736`) deletes the *real*-hex directory, so the derived-key entries
are never cleaned up either — the cache grows without bound.

**Fix shape** Decode the hex back into a `CacheKey` (the function's own doc
comment at `:514` says "parsing the hex suffix", which is what it should do),
or key the cache by the path string directly and stop pretending it is a
`CacheKey`.

### D-32 · The compiler never descends into `Loop` or `Branch`

**Class** Correctness · **Severity** High · **Crate** `soma-compiler`

**Evidence** `resolve_distribution` (`soma-compiler/src/compiler.rs:721`) matches
`Execute | Step`, `Sequence`, `Parallel`, `Composite`, then `other => other`
(`:778`). `ExecutionPlan::Loop { body, .. }` and `Branch { arms, .. }` are returned
unchanged. `collapse_differentiable` (`:786`) has the identical omission at `:830`.

**Consequence** A node declaring `Distribution::Remote(..)` inside a loop body or
a branch arm is never wrapped in `ExecutionPlan::Remote` — it silently runs
locally. Differentiable nodes inside a loop body are never fused into a
`Composite`, so gradients do not flow through them.

This is exactly the class of bug `ExecutionPlan::children()`
(`soma-compiler/src/plan.rs:142`) was written to prevent — and these two
functions do not use it.

**Fix shape** Rewrite both as walks over `children()`.

### D-33 · `Value::to_plain_json` contradicts its own contract

**Class** Correctness · **Severity** Medium · **Crate** `soma-core`

**Evidence** `soma-core/src/data/value.rs:111` documents "This is what a
multi-predecessor node receives per upstream branch — never the internal
serde-tagged encoding". At `:139`:

```rust
other => serde_json::to_value(other).unwrap_or(serde_json::Value::Null),
```

`Bytes` and `Object` fall into `other`, so they come out as
`{"type":"Bytes","data":[…]}` — precisely the encoding the doc promises never to
emit. The `unwrap_or(Null)` also turns any failure into `null`.

### D-34 · `RemoteRunner` is never constructed

**Class** Dead code · **Severity** Medium · **Crate** `soma-runtime`

**Evidence** `soma-runtime/src/runner/remote.rs`, lines 73–114 as they stood —
every reference in the workspace was its own definition (73, 77, 91), the
re-export chain (`soma-runtime/src/runner/mod.rs`, `soma-runtime/src/lib.rs`) or
a doc comment. The entire remote `Runner` implementation was unreachable. Line
numbers are given plainly here because the code they pointed at is gone.

Relatedly, the `Runner` trait itself (`soma-runtime/src/runner/mod.rs:124`) is
**never used as `dyn Runner`** — all call sites name `LocalRunner` concretely
(`soma-runtime/src/forward.rs:94`, `soma-runtime/src/graph_session.rs:264`). The
polymorphism its doc claims at `:122` is not exercised by anything.

**Fix shape** Delete `RemoteRunner`, or wire it up. Keeping a second
implementation of the central execution interface that nothing constructs is
worse than either.

**Resolved** Deleted. `soma-runtime/src/runner/remote.rs` now holds the
`Transport` trait alone; the compiler's `ExecutionPlan::Remote` arm is what
sends work out. The `Runner` trait stays — it has one implementation and one
caller shape, which is a trait carrying its own documentation, not a strategy
pattern.

### D-35 · Enum variants that exist only to be refused or ignored

**Class** Dead code · **Severity** Low · **Crate** `soma-core`

| Item | Evidence |
|---|---|
| `DataRef::Stream` | `soma-core/src/data/store.rs:119` — zero constructions; only reachable through `_ =>` arms |
| `StreamFormat` (whole enum) | `soma-core/src/data/store.rs:145` — referenced only by the unused `DataRef::Stream` |
| `CacheTier::Remote` | `soma-core/src/cache/mod.rs:159` — never produced |
| `Origin::Streamed` | `soma-core/src/cache/mod.rs:177` — never produced |
| `SearchStrategy::Hyperband` | `soma-core/src/optimizer/study.rs:144` — self-documented "no sampler implements it yet" |
| `SearchStrategy::MultiObjective` | `soma-core/src/optimizer/study.rs:155` — same |
| `PruningStrategy::Hyperband` | `soma-core/src/optimizer/study.rs:204` — "behaves like `None`" |
| `TrainingStrategy::Custom` | `soma-core/src/distributed.rs:74` — the executor refuses it |
| `ExploitStrategy::Binary.threshold` | `soma-core/src/distributed.rs:188` — "the current `PbtRunner` does not read this field yet" |
| `ExploitStrategy::Binary.threshold` | `soma-core/src/distributed.rs:188` — "the current `PbtRunner` does not read this field yet" |

`TrainingStrategy::PopulationBased` (`soma-runtime/src/strategy.rs:246`) is a
deliberate permanent error arm — that one is [documented as a design
decision](/soma/design/decisions) and is not debt.

**Resolved** All nine deleted. Two consequences worth naming: `SearchStrategy`
and `PruningStrategy` are now exhaustive at every Python call site, so the
`_ => "Unsupported strategy"` arm in `soma-python/src/study.rs` is gone and a
new variant would fail to compile rather than fail at runtime; and
`soma.Pbt(threshold=…)` no longer exists, because the argument reached a field
nothing read. `ExploitStrategy::Binary` is now a unit variant.

### D-36 · Unreached methods and a pluggable seam with no injection site

**Class** Dead code · **Severity** Low · **Crate** `soma-runtime`

**Evidence** Never called anywhere: `NodeCatalog::with_state_store`
(`soma-runtime/src/node_catalog.rs:94`), `NodeCatalog::clear_states` (`:208`),
`NodeCatalog::state_store` (`:214`), `MedianPruner::with_min_trials`
(`soma-runtime/src/pruner.rs:52`), `EventBus::subscriber_count`
(`soma-runtime/src/event_bus.rs:72`), `TrialContext::metrics`
(`soma-runtime/src/executors/study.rs:107`).

The `StateStore` seam the module docs advertise
(`soma-runtime/src/node_catalog.rs:13`) has one implementation and no injection
site. The whole spill path (`Context::with_spill_threshold`,
`soma-runtime/src/executor.rs:245`; `maybe_spill`, `:252`) defaults to disabled
and is set by nothing outside `tests/coverage_boost.rs`.

**Resolved, and two of its six claims were wrong.** Deleted:
`NodeCatalog::clear_states`, `NodeCatalog::state_store`,
`MedianPruner::with_min_trials` (a second path to a `pub` field — a test now
writes `MedianPruner { min_trials: 5, .. }` and loses nothing),
`EventBus::subscriber_count`, and the spill path entire — which takes
`data_store` and `spill_threshold` off `Context`, leaving it eleven fields, and
collapses `resolve_value` to its materialized arm.

Kept, because "never called anywhere" was not true of either:
`TrialContext::metrics` is called from `soma-python/src/study.rs:586`, on the
path that turns a trial's reported metrics into a Python dict; and
`NodeCatalog::with_state_store` is the only way to inject the failing store in
`a_failing_state_store_is_reported_not_fatal`, the test that holds the line
against a full disk aborting the host process. A method whose only caller is a
test is not automatically dead — it depends on whether the test's subject is
the method or something the method makes reachable.

### D-37 · Dead helper in the Python bindings

**Class** Dead code · **Severity** Low · **Crate** `soma-python`

**Evidence** `soma-python/src/graph.rs:466` — `#[allow(dead_code)] fn
split_value_into_batches`, 52 lines, zero callers. `chunk_value` at `:821` is the
live one. This is the only `#[allow(dead_code)]` in the workspace.

**Resolved** Deleted. The workspace now has no `#[allow(dead_code)]` at all,
which makes the attribute's next appearance meaningful.

### D-38 · `Phase::Trial` and the scheduler's capability model

See [D-65](#d-65--the-schedulers-capability-model-is-unimplemented).

---

## Leaky abstractions

### D-41 · `Transport::execute_node` runs remotes with an empty catalog

**Class** Leaky abstraction · **Severity** High · **Crate** `soma-runtime`

**Evidence** `soma-runtime/src/runner/remote.rs:61` — the default implementation
builds a throwaway `NodeCatalog::new()` and calls `execute` with `seed: None`.
Its own doc comment (`:56`) says: "Unseeded, and it has to be… Callers that have
a `RunContext` should go through `Transport::execute` with `ctx.seed` instead of
reaching for this."

`soma-runtime/src/executor.rs:616` reaches for it anyway.

**Consequence** Every `ExecutionPlan::Remote` node executes with an empty filter
catalog and an unsalted cache key. Seeded runs are not reproducible across the
remote boundary, and two runs with different seeds share cache lines.

### D-42 · `ExecutionPlan::Remote` discards its routing target

**Class** Leaky abstraction · **Severity** Medium · **Crate** `soma-runtime`

**Evidence** `soma-runtime/src/executor.rs:415` destructures
`ExecutionPlan::Remote { node_id, target: _, plan }` — the `RemoteTarget` is
thrown away. `execute_remote` (`:601`) sends to `ctx.transport` (the single one)
or falls back to running locally (`:608`). The target-resolution machinery exists
(`soma-runtime/src/strategy.rs:634`) but is wired only into `ModelParallel`.

### D-43 · `StrategyContext::execute_on_worker` has a dead JSON parameter

**Class** Leaky abstraction · **Severity** Medium · **Crate** `soma-runtime`

**Evidence** `soma-runtime/src/strategy.rs:38` declares
`fn execute_on_worker(&self, worker_idx, plan: &serde_json::Value, input, y)`.
All three call sites pass `&serde_json::json!({})`
(`soma-runtime/src/strategy.rs:156`, `:169`, `:204`), and the sole implementor
ignores it — `_plan: &serde_json::Value` with the comment at `:597` naming it "a
JSON placeholder every strategy passes as `{}`".

**Fix shape** Delete the parameter.

### D-44 · `resolve_input` falls back to "whatever ran last"

**Class** Leaky abstraction · **Severity** Medium · **Crate** `soma-runtime`

**Evidence** `soma-runtime/src/executor.rs:1183` — a node with zero predecessors
takes `ctx.execution_order.last()`. Combined with `GraphInfo::predecessors`
returning `&[]` for an *unknown* node id (`soma-runtime/src/executor.rs:74`), a
node missing from `GraphInfo` silently receives the previous node's output
instead of producing an error.

That is the exact bug class the `RunContext::graph_info` doc
(`soma-runtime/src/runner/mod.rs:14`) says the field exists to prevent.

### D-45 · `RunMode` — an executor-internal enum — is a wire parameter

**Class** Leaky abstraction · **Severity** Low · **Crate** `soma-runtime`

**Evidence** `soma-runtime/src/runner/remote.rs:34` — `Transport::execute` takes
`&RunMode`, which carries `Fit { y: Option<Value> }`, i.e. the entire label
tensor, by value.

### D-46 · `TieredCache` promotion destroys provenance

**Class** Leaky abstraction · **Severity** Medium · **Crate** `soma-runtime`

**Evidence** `soma-runtime/src/cache/tiered.rs:51` — `let _ = faster_store.put(key,
&value);`. `put` (not `put_with_origin`) gives the promoted copy
`Origin::Ingested { source: "unknown" }` (`soma-runtime/src/cache/memory.rs:146`),
destroying the `Origin::Computed { node_id, run_id }` that `store_output`
(`soma-runtime/src/executor.rs:726`) went out of its way to record. The error is
discarded too.

### D-47 · A cross-crate contract carried by an environment variable

**Class** Leaky abstraction · **Severity** Low · **Crates** `soma-python`, `soma-worker`

**Evidence** `soma-python/src/worker.rs:42` — `unsafe { std::env::set_var(
"SOMA_LOCAL_PACKAGE", parent) }`, called from `PyWorker::new`, so that
`soma-worker/src/env_manager.rs:117` `link_local_package` can `pip install -e`
the caller's own build. A process-global written by a constructor is how two
crates agree on something.

### D-48 · `Value::Empty` for state and `Evolving`'s value/state conflation

**Class** Leaky abstraction · **Severity** Low · **Crate** `soma-runtime`

**Evidence** `soma-runtime/src/executors/stream.rs:249` — in `StreamMode::Evolving`
the forward output *is* the next chunk's state. Documented at
`soma-runtime/src/executors/stream.rs:23` as "a documented conflation", which is
honest but does not make it typed.

### D-49 · `maybe_spill` mis-estimates bytes and swallows failure

**Class** Leaky abstraction · **Severity** Low · **Crate** `soma-runtime`

**Evidence** `soma-runtime/src/executor.rs:256` estimates size as
`value.size() * 8` ("approximate bytes, f64 = 8 bytes") — wrong for `Text`,
`Json` and `Bytes`. At `:264` a failed `store.put` is swallowed by `if let Ok(_)`
with no log, silently keeping in memory a value the caller asked to spill.

**Resolved by deletion** `maybe_spill` is gone with the rest of the spill path
([D-36](#d-36--unreached-methods-and-a-pluggable-seam-with-no-injection-site)).
Neither bug was ever reachable: nothing outside a test set a threshold. If
spilling comes back it should come back measured — this entry is what it has to
answer.

---

## Stringly-typed APIs and primitive obsession

### D-51 · Four style tables keyed by the same magic strings

**Class** Stringly-typed · **Severity** Medium · **Crate** `soma-core`

**Evidence** `NodeOverlay::style_class()` (`soma-core/src/viz/mod.rs:100`) returns one
of five `&'static str` values. Two separate functions then re-match those
strings, each with a silent `_` fallback meaning "flagged":
`viz::mermaid_class_style` (`soma-core/src/viz/mod.rs:116`) and `svg::class_colors`
(`soma-core/src/viz/svg.rs:27`). It was three until `viz::dot_class_style` went with
`to_graphviz`; the heading's "four" counts `style_class` itself.

**Consequence** A new status requires four coordinated edits, and a typo in any
of them falls through to the flagged colour rather than failing to compile — with
`NodeStatus` (`soma-core/src/viz/mod.rs:22`) sitting right there as the enum that
would make it a compile error.

### D-52 · Node placement has one typed mechanism and one stringly one

**Class** Primitive obsession · **Severity** Medium · **Crate** `soma-core`

**Evidence** `Node.target: Option<String>` with the magic value `"local"`
(`soma-core/src/graph/mod.rs:82`, tested by `is_local()` at `:205`) coexists with the
typed `Distribution` / `RemoteTarget` on `FilterMeta`
(`soma-core/src/graph/filter.rs:53`, `:64`).

### D-53 · Typed enums shadowed by their own string forms

**Class** Primitive obsession · **Severity** Low · **Crate** cross-cutting

| String field | Enum it shadows |
|---|---|
| `StoreMeta.dtype: String` (`soma-core/src/data/store.rs:26`) | `DataType` |
| `EdgeRef.kind: String` (`soma-core/src/tracking/fingerprint.rs:66`) | `EdgeKind` |
| `RunSummary.kind: String` (`soma-core/src/tracking/summary.rs:338`) | `RunKind` |
| `NodeOverlay.cache_tier: Option<String>` (`soma-core/src/viz/mod.rs:44`) | `CacheTier` |
| `DataTransfer.transfer_type: String` (`soma-compiler/src/scheduler.rs:133`) | — (only `"s3"` is ever written, `:278`) |

`Event::HealthFlag.flag: String` (`soma-core/src/tracking/event.rs:363`) goes further and
encodes a count inside the string: `"DEAD_CHANNELS(3)"`.

`soma-runtime/src/tracking/reader.rs:345` and `:411` stringify an enum with
`format!("{tier:?}").to_lowercase()`, in two places.

### D-54 · Nine string-match dispatch sites across the FFI

**Class** Stringly-typed · **Severity** Medium · **Crate** `soma-python`

**Evidence** `soma-python/src/graph.rs:2093` (`kind` — 5 strategies, with nested
`aggregation` matches at `:2095` and `:2109`); `:1465` / `:1470`
(`"differentiable"` / `"inference"`); `soma-python/src/store.rs:30`
(`store_type`); `soma-python/src/study.rs:44` (`dtype`) and `:52` (`scale`) and
`:184` (`parse_pruning`); `soma-python/src/agentic.rs:610` (5 transitions) and
`:660` (3 join policies) and `:766` (5 effects) and `:850` (`mode`) and `:729`
(schema shorthands).

**Consequence** None of these strings is a constant. They are literals repeated
across three places that must agree: the Rust match, the `_soma.pyi` stub, and
the Python constructor in `soma-python/python/soma/agentic.py`. Nothing checks
that they do.

**Fix shape** One `const` table in Rust, generated into the stub, imported by
name in Python.

### D-55 · `set_strategy` / `strategy()` is a lossy round trip

**Class** Stringly-typed · **Severity** Low · **Crate** `soma-python`

**Evidence** `soma-python/src/graph.rs:2079` returns only the discriminant name,
so `set_strategy("data_parallel", num_replicas=8)` followed by `strategy()`
yields `"data_parallel"` with the parameters gone. The `_ => "unknown"`
catch-all remains, because `TrainingStrategy` is `#[non_exhaustive]` and lives
in another crate.

### D-56 · `NodeId` is a `String`, and so is everything else

**Class** Primitive obsession · **Severity** Low · **Crate** `soma-core`

**Evidence** `NodeId` (`soma-core/src/graph/mod.rs:23`), `EdgeId` (`:20`), `RunId` /
`StudyId` / `TrialId` (`soma-core/src/tracking/event.rs:14`, `:17`, `:20`) are all
`String` aliases, hence mutually assignable. The `NodeId` case is a knowing
deferral, argued at `soma-core/src/graph/mod.rs:20`.

### D-57 · Prose parsing as control flow

**Class** Stringly-typed · **Severity** Medium · **Crate** `soma-python` (Python layer)

**Evidence** `soma-python/python/soma/agentic.py:196` — `PANEL_MARKER`, a sentinel
string parsed back out of prompts so a brief built from a brief does not nest.
`MajorityVote.extract` (`:288`) regex-extracts a number from model prose.
`Fanout.tasks` (`:759`) splits free prose into work items by stripping bullets and
numbering. All three are documented; none is schema-enforced, and structured
output is available (`soma-llm/src/steps.rs:107`).

---

## Performance

### D-61 · `Context::snapshot` deep-clones the value store per branch

**Class** Performance · **Severity** Medium · **Crate** `soma-runtime`

**Evidence** `soma-runtime/src/executor.rs:344`, called from `execute_parallel` at
`:1107` — one full clone of the value store per parallel branch. A 4-way
`Parallel` over a graph holding a 1 GB intermediate materializes 4 GB. The
comment at `:1090` explains the write-set merge but not the snapshot cost.

**Mitigating** `Value` payloads are all `Arc`-backed (`soma-core/src/data/value.rs:15`),
so the clone is refcount bumps, not byte copies — but `VirtualValue::Materialized`
holds the `Value` and the `HashMap` itself is rebuilt per branch.

### D-62 · `MemoryCache`'s LRU touch is O(n) on every read

**Class** Performance · **Severity** Medium · **Crate** `soma-runtime`

**Evidence** `soma-runtime/src/cache/memory.rs:45` —
`access_order.retain(|k| k != key)` on a `VecDeque`, called from `get` (`:133`),
`insert` (`:65`) and `remove` (`:79`). Every cache read is a linear scan of every
key, under the single `Mutex<LruStore>` (`:17`) that all parallel branches share.

`estimate_size` (`:208`) also serializes `Value::Json` in full
(`v.to_string().len()`) on every `put`.

### D-63 · `RunReader` re-parses `events.jsonl` once per accessor

**Class** Performance · **Severity** Medium · **Crate** `soma-runtime`

**Evidence** `RunReader::events()` (`soma-runtime/src/tracking/reader.rs:291`)
opens and parses the whole file. It is called independently by `node_timings`
(`:316`), `cache_activity` (`:405`), `metric_series` (`:440`), `health_flags`
(`:478`), `agentic_activity` (`:519`), `agentic_timeline` (`:640`) and `overlay`
(`:724` and `:746` — twice).

`summarize` (`soma-runtime/src/tracking/summary.rs:33`) calls five of them, so a
single summary is **five full reads and five full JSON parses of the same file**.
`to_mermaid` / `to_svg` add two more each via `overlay`.

**Fix shape** Parse once into a `Vec<EventEnvelope>` and have the accessors take
a slice.

### D-64 · `StudyRunner::run` is O(trials²) in four places

**Class** Performance · **Severity** Medium · **Crate** `soma-runtime`

**Evidence** All four inside the per-trial loop:
`normalized_histories(study, direction)` rebuilds every completed trial's history
(`soma-runtime/src/executors/study.rs:275` → `:425`); `study.best_trial()` (`:340`),
`study.best_value()` (`:342`, `:357`) and the terminal-trial count (`:352`) each
scan all trials; `save_study` rewrites the entire `study.json` (`:361`).

### D-65 · The scheduler's capability model is unimplemented

**Class** Performance / dead code · **Severity** Low · **Crate** `soma-compiler`

**Evidence** The module doc's rule 2 promises placement "by capability", but
`schedule_plan` places purely round-robin: `i % workers.len()`
(`soma-compiler/src/scheduler.rs:258`). `WorkerInfo::matches_tag` (`:50`), `.gpu`
(`:25`) and `.cpu_cores` (`:27`) are never read by `schedule`. Rule 4 ("study
trials round-robin") is likewise unimplemented — `Phase::Trial`
(`soma-compiler/src/scheduler.rs:83`) is never constructed.

### D-66 · Cheap booleans that cost a filesystem walk

**Class** Performance · **Severity** Low · **Crate** `soma-runtime`

**Evidence** `LocalCache::is_empty()` (`soma-runtime/src/cache/local.rs:50`) calls
`len()`, which is a full recursive directory walk (`walkdir_count`, `:173`).
`LocalCache` also never evicts and has no size bound at all — only
`FsActionStore` has a GC (`soma-runtime/src/cache/gc.rs`).

### D-67 · Four to five process spawns on the run-creation path

**Class** Performance · **Severity** Low · **Crate** `soma-runtime`

**Evidence** `collect_git_info` (`soma-runtime/src/tracking/local_tracker.rs:156`)
spawns `git` three times (`rev-parse HEAD`, `rev-parse --abbrev-ref HEAD`,
`status --porcelain`), plus a fourth in the dirty fallback at `:172`, plus a
possible `hostname` subprocess at `:188` — all synchronously, per
`LocalTracker::create`.

### D-68 · Unbounded growth inside long loops

**Class** Performance · **Severity** Low · **Crate** `soma-runtime`

**Evidence** `Context::execution_order` grows on every `set` / `set_virtual`
(`soma-runtime/src/executor.rs:305`, `:313`) with no dedup, so a 100-iteration
loop appends 100 × body-size entries — and `last_output`
(`soma-runtime/src/runner/local.rs:54`) reverse-scans it.
`EffectDriver::run` accumulates `history: Vec<Vec<EffectResult>>` for `max_turns`
turns (`soma-runtime/src/effects/mod.rs:124`), which for an LLM agent is the
whole conversation held in memory and re-borrowed into every `StepCtx`.

### D-69 · One MCP server serializes every tool call

**Class** Performance · **Severity** Low · **Crate** `soma-llm`

**Evidence** `soma-llm/src/mcp_client.rs:31` — a `Mutex<Pipe>` per server, so
fan-out across tools on one MCP server is effectively single-threaded even though
`EffectDriver::perform_all` (`soma-runtime/src/effects/mod.rs:459`) runs each
effect on its own thread.

### D-70 · `JoinPolicy::First` waits for everyone

**Class** Performance · **Severity** Low · **Crate** `soma-runtime`

**Evidence** `soma-runtime/src/effects/mod.rs:417` returns the first `Ok`, but
only after `thread::scope` has joined every sibling. The code says so at `:415`:
"Everything ran — these are threads, not cancellable tasks."

---

## Concurrency

### D-71 · Four policies for a poisoned mutex, three of them silent

**Class** Concurrency · **Severity** Medium · **Crate** `soma-runtime`

**Evidence**

| Site | Policy |
|---|---|
| `soma-runtime/src/cache/memory.rs:96` and 7 more | `unwrap_or_else(\|e\| e.into_inner())` — recover |
| `soma-runtime/src/event_bus.rs:39`, `:49`, `:90` | recover |
| `soma-runtime/src/strategy.rs:617` | **error** — `Other("state cache poisoned")` |
| `soma-runtime/src/strategy.rs:611`, `:684`, `:699`, `:720` | **silently skip** — `if let Ok(mut cache) = self.states.lock()` |

**Consequence** The last group is the problem. `execute_on_worker` (`:611`) drops
the just-returned worker states on a poisoned lock and `set_state` (`:720`) drops
the redistributed aggregate — after which `get_state` (`:617`) *errors* on the
same lock. A poisoned `states` mutex turns a federated round into "the closing
`get_state(0)` returns worker 0's own last fit instead of the aggregate", which
is exactly what the comment at `:716` says the field exists to prevent.

**Fix shape** One policy, written once. Recovery is defensible; silently
discarding written state is not.

### D-72 · Every event emission clones the sink vector

**Class** Concurrency · **Severity** Low · **Crate** `soma-runtime`

**Evidence** `soma-runtime/src/event_bus.rs:89` — `snapshot_sinks` clones the whole
`Vec<Arc<dyn EventSink>>` on every emission to avoid a reentrancy deadlock
(documented at `:83`). Correct, but an atomic refcount bump per sink per event on
the executor's hot path, and the sinks are then invoked **synchronously on the
emitting thread** (`:60`) — so `JsonlEventSink`'s buffered write and its flush
every 20 events sit inside `run_node`.

### D-73 · Catalog clones per run and per nesting level

**Class** Concurrency · **Severity** Low · **Crate** `soma-runtime`

**Evidence** `GraphSession::run_driver` clones the whole `NodeCatalog` per run
(`soma-runtime/src/graph_session.rs:148`); `GraphHandler::child_driver` clones it
**twice per nesting level** (`soma-runtime/src/effects/graph_handler.rs:120` and
`:122`), plus `(**graph).clone()` per `Effect::Graph` at `:148`.

### D-74 · The coordinator's reaper is a side effect of building a router

**Class** Concurrency · **Severity** Low · **Crate** `soma-coordinator`

**Evidence** `soma-coordinator/src/server.rs:42` — `coordinator_router` spawns the
10-second stale-worker reaper. Constructing a router in a test therefore spawns a
task that outlives the test unless the runtime is torn down; the function is not
idempotent in that sense.

---

## Long functions

Not a smell on its own, but the reliable predictor of the ones above. Production
code only — several of these files are 60% inline tests, which is why the raw
line counts mislead.

| Function | Location | Lines |
|---|---|---|
| `Worker::execute_plan` | `soma-worker/src/worker.rs:275` | 324 |
| `PyGraph::fit` | `soma-python/src/graph.rs:1371` | 262 |
| `handle_ws` | `soma-worker/src/server.rs:353` | 215 |
| `PyStudy::run` | `soma-python/src/study.rs:433` | 213 |
| `StudyRunner::run` | `soma-runtime/src/executors/study.rs:187` | 194 |
| `Graph::to_svg_with` | `soma-core/src/viz/svg.rs:65` | 206 |
| `schedule_plan` | `soma-compiler/src/scheduler.rs:197` | 185 |
| `handle_stream_message` | `soma-worker/src/server.rs:568` | 163 |
| `derive_soma_filter_impl` | `soma-macros/src/lib.rs:40` | 158 |
| `EffectDriver::run` | `soma-runtime/src/effects/mod.rs:107` | 143 |
| `execute_python_job_with_progress` | `soma-worker/src/server.rs:731` | 141 |
| `parse_transition` | `soma-python/src/agentic.rs:587` | 140 |
| `TrainingStrategy::fit` | `soma-runtime/src/strategy.rs:146` | 134 |
| `run_node` | `soma-runtime/src/executor.rs:816` | 131 |
| `parse_effect` | `soma-python/src/agentic.rs:755` | 129 |
| `PyGraph::forward_local` | `soma-python/src/graph.rs:99` | 127 |
| `agentic_activity` | `soma-runtime/src/tracking/reader.rs:507` | 125 |
| `EffectDriver::spawn_all` | `soma-runtime/src/effects/mod.rs:316` | 121 |
| `all_tools` | `soma-mcp/src/tools/mod.rs:13` | 349 (inline JSON schemas) |

`schedule_plan` is worth singling out: its 9-arm match repeats the same
`forced_worker.and_then(…).unwrap_or_else(least_loaded)` preamble five times
(`soma-compiler/src/scheduler.rs:204`, `:228`, `:297`, `:312`, `:329`).

`EffectDriver::run`'s `Transition` match repeats `finish(..) + return Err(..)`
five times (`soma-runtime/src/effects/mod.rs:141`, `:149`, `:161`, `:220`, `:232`)
— an invariant nothing enforces.

---

## Documentation rot

### D-81 · Doc comments that narrate history

**Class** Doc rot · **Severity** Low · **Crate** `soma-runtime`

**Evidence** Roughly 20 `///` comments describe what the code *used to be* rather
than what it does, and ship to docs.rs:

- `soma-runtime/src/runner/mod.rs:14` — "Both runner methods **used to** build `GraphInfo::for_linear(…)` … On a diamond that is simply wrong"
- `soma-runtime/src/executor.rs:82` — "They **used to be** two whole execution loops"
- `soma-runtime/src/graph_session.rs:441` — "`graph_fit` **was the worst of them**: a topological loop written from scratch…"
- `soma-runtime/src/effects/sleep_handler.rs:9` — "handled by nobody… The variant existed; the four lines that make it work did not"
- plus `soma-runtime/src/executor.rs:129`, `:440`, `:1090`, `:1121`, `:1316`; `graph_session.rs:228`, `:429`; `strategy.rs:282`, `:520`, `:831`; `runner/local.rs:22`; `forward.rs:70`; `node_catalog.rs:3`; `cache/memory.rs:152`; `cache/tiered.rs:43`; `event_bus.rs:83`; `study_io.rs:3`

**Consequence** This rationale is genuinely load-bearing — it is *why* the code
looks like it does, and losing it would cost more than keeping it. But as `///`
docs it is published API documentation that goes stale the moment the referenced
history stops being relevant to anyone.

**Fix shape** Move the load-bearing ones to
[Architecture Decisions](/soma/design/decisions) and leave a one-line pointer;
demote the rest to `//` comments, which do not ship.

### D-82 · Stale instructions in feature docs

**Class** Doc rot · **Severity** Low · **Crate** `soma-store`

**Evidence** `soma-store/src/s3.rs:3` and `soma-store/src/zarr.rs:16` still tell
the reader to write `soma-core = { path = "../soma-core", features = ["s3"] }`.
Those features moved to `somatize-store`.

### D-83 · The facade covers 10 of 13 crates

**Class** Doc rot / API gap · **Severity** Low · **Crate** `soma`

**Evidence** `soma/src/lib.rs:77` hand-rolls `any(s3, zarr)`:

```rust
#[cfg(feature = "s3")]                              pub use somatize_store as store;
#[cfg(all(feature = "zarr", not(feature = "s3")))]  pub use somatize_store as store;
```

`somatize-mcp` and `somatize-python` are workspace members that the facade does
not depend on or re-export. The comments at `soma/src/lib.rs:64` and `:82`
document two previous instances of exactly this gap being found and fixed.

### D-84 · `soma-core`'s re-export surface is asymmetric

**Class** API consistency · **Severity** Low · **Crate** `soma-core`

**Evidence** `soma-core/src/lib.rs:59` flat-re-exports ~60 symbols but omits whole
public modules: `action::*`, `codec::*`, `canon::*`, `any::AsAny`, `keys::*`,
`node::{NodeMeta, NodeOutcome}`, `util::*`, parts of `summary`. Downstream crates
consequently mix both styles — `soma-runtime/src/cache/fs_store.rs:26` reaches for
`somatize_core::codec::…` while everything else uses flat names — and
`soma::prelude` re-exports `node::{NodeMeta, NodeOutcome}` that `soma-core`'s own
`lib.rs` does not.

---

## Smaller observations

Individually trivial; listed because they are cheap to fix while already in the
file.

- `soma-core/src/graph/mod.rs:456` — `topological_sort` calls `queue.sort()` then `queue.pop()`, so roots come out in *descending* id order. The comment at `:462` describes an insertion that does not happen. Deterministic, just not what it says.
- `soma-compiler/src/scheduler.rs:341` — `let worker_id = worker.id.clone();` … `drop(worker_id);` with no use in between.
- `soma-compiler/src/compiler.rs:552`, `:582`, `:604`, and `:290` — the same whole-graph `HashSet<&str>` is rebuilt four times; it never changes.
- `soma-compiler/src/compiler.rs:534` — `plan_for_node` invents `ExecutionPlan::Execute` for an unknown node id, while the `other =>` arm 90 lines below (`:627`) argues at length that guessing is unacceptable.
- `soma-compiler/src/compiler.rs:573` — sub-graph compilation drops the inner `diagnostics` entirely; every warning the inner graph raises is computed and thrown away.
- `soma-core/src/viz/svg.rs` declares no public type — it exists only to hang two methods on `Graph` from another module, which a reader of `graph.rs` will not find.
- `soma-core/src/store/` is a directory containing exactly one file.
- `soma-runtime/src/executor.rs:596` — the branch selector's value is overwritten by the branch's input, so a downstream node can never see which arm was chosen. Deliberate (`:590`), but it means the branch node's stored value is not the branch node's output.
- `soma-runtime/src/executors/pbt.rs:117` — a training failure is warned and the member keeps its stale state; only *evaluation* failures are counted (`:133`). A member whose training always fails competes on stale weights forever.
- `soma-runtime/src/executors/pbt.rs:103` — `let mut rng_state: u64 = 42;` with no way to configure it; `PbtConfig` has no seed field.
- `soma-mcp/src/server.rs:60` — `notifications/initialized` returns a `JsonRpcResponse`. JSON-RPC notifications must not be answered; the comment at `:61` acknowledges it.
- `soma-worker/src/server.rs:390` — `CancelPlan` exists in the protocol (`soma-worker/src/protocol.rs:503`) and is answered with "not implemented".
- `soma-worker/src/lib.rs:29` — `pub use protocol::*` puts 15 wire types in the crate root, so the public API grows silently with the protocol.
- `soma-memory/src/knowledge_base.rs:76` and eight sibling defaults — every defaulted query calls `self.all()?`, cloning the entire record vector per call.
- `soma-llm/src/openai_compat.rs:76` — a hand-rolled 43-line HTTP-date parser, correct as written but untested against malformed input.
- `soma-llm/src/steps.rs:491` — `LlmStep` re-hashes `ReactStep`'s `config_hash` with a `b"LlmStep"` salt and rebuilds `meta()` by struct update. Two `Step` names for one behaviour.
- `soma-mcp/src/exec.rs:16` — `run_pipeline` / `run_study` execute project code with `sys.path.insert(0, os.getcwd())` (`:31`). The file argues this is no worse than the pre-existing `write_filter_source`, which is true, and it is still an unsandboxed exec reachable from a model.

---

## Adopted from the first architecture review

[Architecture Review](/soma/development/architecture-review/) is a historical
document — it describes the workspace before `soma-store` and `soma-llm` existed
and before `Pipeline` was removed. Most of its findings have shipped. The ones
still open are restated here, re-verified, so that this page is the single live
register and the older one can stay historical.

### D-91 · The `Filter` trait mixes computation with cache identity

**Class** God object · **Severity** Low · **Crate** `soma-core`

**Evidence** `soma-core/src/graph/filter.rs:120` — `Filter` requires `fit`, `forward`
(computation), `config_hash` (cache identity) and `meta` (description). Every
implementor must know about `CacheKey` whether or not it is ever cached.

**Still true.** The review proposed splitting into `Compute` + `Describable` +
`Cacheable`. Deferred then, deferred now — but note that `config_hash` is
derived by macro for Rust filters (`soma-macros/src/lib.rs:30`) and computed
in Python for Python ones (`soma-python/src/bridge.rs:27`), so in practice no
hand-written implementor writes it. That weakens the case for splitting.

### D-92 · `Graph::predecessors` / `successors` are linear scans

**Class** Performance · **Severity** Low · **Crate** `soma-core`

**Evidence** `soma-core/src/graph/mod.rs:396` and `:399` — both filter the entire
edge vector on every call, O(edges) each. The compiler calls them per node, so
compilation is O(nodes × edges).

**Mitigating** `GraphInfo::from_graph` (`soma-runtime/src/executor.rs:46`)
precomputes the predecessor map once per run, so the *executor* does not pay
this — only the compiler does, and only for graphs under ~100 nodes today.

### D-93 · Trials run one at a time

**Class** Performance · **Severity** Low · **Crate** `soma-runtime`

**Evidence** `soma-runtime/src/executors/study.rs:228` — a single `loop` running
one trial per iteration. Nothing distributes trials across cores or workers,
though `Phase::Trial` (`soma-compiler/src/scheduler.rs:83`) was defined for
exactly that and is never constructed ([D-65](#d-65--the-schedulers-capability-model-is-unimplemented)).

### D-94 · `soma-core` owns seven domains

**Class** God object · **Severity** Low · **Crate** `soma-core`

**Evidence** 26 modules covering graph, filter, step, cache, value, study,
search, event, tracking, summary and rendering. Importing `Filter` pulls in
`Study`, `Event` and `SearchSpace`.

**Partly addressed.** `soma-store` was split out precisely on this argument (see
`soma-store/src/lib.rs:5`), because each remote backend owns a tokio runtime.
The remaining domains share no such hard dependency, so the split would buy
compile time and little else.

### D-95 · `SomaError::Pruned` coexists with `TrialOutcome::Pruned`

**Class** API inconsistency · **Severity** Low · **Crate** `soma-core`

**Evidence** `soma-core/src/error.rs:43` still carries `Pruned { step, reason }`
as an error variant, while `TrialOutcome` (`soma-runtime/src/executors/study.rs:24`)
models pruning as a *non-error* outcome — which is the correct modelling, and
the reason the enum exists. `StudyRunner` handles both
(`soma-runtime/src/executors/study.rs:287`). The smell is narrowed, not removed.

Stringly-typed node ids, the review's finding 1.3, is
[D-56](#d-56--nodeid-is-a-string-and-so-is-everything-else) here.

---

## What is already healthy

Read in isolation, the list above suggests a codebase in trouble. It is not, and
the following are load-bearing reasons why — each of them a decision that a
codebase this size usually gets wrong.

**Error handling is genuinely clean.** There are **exactly three error enums in
the entire workspace**: `SomaError` (`soma-core/src/error.rs:14`), `WorkerError`
(`soma-worker/src/error.rs:24`) and `LlmError` (`soma-llm/src/error.rs:29`). Both
domain enums are `#[non_exhaustive]`, use `thiserror`, carry `#[from]` for `Io`
and `Core`, and convert at the seam with a `From` impl that preserves the
prefix — with a unit test asserting exactly that
(`soma-worker/src/error.rs:84`, `soma-llm/src/error.rs:107`). No error-type
sprawl, no string formatting at call sites. The one soft spot is that crossing
into `SomaError::Other(String)` erases the variant.

**Nothing is async, on purpose, everywhere.** `rg async_trait` returns zero hits.
Every trait is synchronous and the runtime uses `std::thread::scope`
(`soma-runtime/src/executor.rs:1103`, `soma-runtime/src/effects/mod.rs:459`), with
the rationale written down at `soma-runtime/src/effects/mod.rs:12`. Where async
is unavoidable — the axum server — it is correctly isolated with `spawn_blocking`
(`soma-worker/src/server.rs:366`, `:396`, `:450`, `:535`) and
`on_own_runtime` (`soma-worker/src/ws_transport.rs:42`), which explicitly refuses
to assume it is outside a runtime.

**No panic crosses the FFI.** `soma-python/src/study.rs:9` documents the
deliberate de-`unwrap`ing of `parse_py_search_dim`, because PyO3's
`PanicException` does not inherit `Exception` and would be uncatchable from
Python. The three remaining `unwrap`s in `soma-python/src/worker.rs` are in a
detached thread ([D-27](#d-27--unwrap-inside-a-detached-thread-makes-bind-failures-unreportable)),
not on a call path.

**Unknown variants refuse rather than guess.** `soma-runtime/src/executor.rs:445`,
`soma-runtime/src/strategy.rs:274`,
`soma-runtime/src/effects/graph_handler.rs:184` and
`soma-runtime/src/tracking/reader.rs:844` all return an error naming the
situation instead of falling through to a default. That is applied consistently
and it is the right call every time.

**The `#[non_exhaustive]` policy is deliberate and documented.** Data enums get
it; control-flow enums every consumer must decide over — `NodeOutcome`
(`soma-core/src/graph/node.rs:37`), `Transition` (`soma-core/src/graph/step.rs:38`),
`StreamMode` (`soma-core/src/graph/filter.rs:32`) — deliberately do not, with the reason
in the doc comment, so adding a variant breaks every match.

**Suppressions are almost absent.** Ten `#[allow(...)]` in ~70 000 lines, nine of
them `clippy::too_many_arguments` on PyO3 keyword constructors (a structural
consequence, not laziness) and one genuine dead-code marker
([D-37](#d-37--dead-helper-in-the-python-bindings)). No `todo!`,
`unimplemented!`, `unreachable!`, `FIXME`, `HACK` or `#[deprecated]` anywhere in
`soma-core`, `soma-compiler`, `soma-store`, `soma-macros` or `soma-runtime`'s
`src/`. Three `unsafe` blocks, all `env::set_var`, two of them in tests.

**Every crate opts into `#![warn(missing_docs)]`.**

**Tests outnumber code where it matters.** `soma-runtime` is ~9 268 lines of
production code against ~13 600 lines of tests — a 1.47:1 ratio — including a
regression suite named after the bug it prevents
(`soma-runtime/tests/fit_through_run_node.rs`,
`soma-runtime/tests/topology.rs`) and a memory-accounting test that asserts
streaming does not grow the heap (`soma-runtime/tests/memory_usage.rs`).

**Several of the silent-degradation bugs on this page have already been fixed
elsewhere in the same code.** `InputSource::resolve`
(`soma-worker/src/protocol.rs:116`) now hard-errors where it used to return
`Value::Empty`; `encode_frame` (`:249`) switched to `to_vec_named` after two
receivers swallowed a decode error. Each carries a comment naming the bug. This
is a live theme in the codebase, not a blind spot — which is the best available
evidence that the remaining instances
([D-24](#d-24--venv-provisioning-fails-into-the-system-interpreter),
[D-25](#d-25--state-load-failure-silently-restarts-from-random-init),
[D-26](#d-26--judgestep-scores-an-unparseable-reply-as-00)) are oversights rather
than a philosophy.
