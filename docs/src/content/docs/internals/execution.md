---
title: Execution — compiler and runtime
description: Trait, type and ownership reference for soma-compiler and soma-runtime, with annotated traces through every execution path.
---

These two crates are half the codebase and the part that is hardest to hold in
your head: the compiler turns a `Graph` into an `ExecutionPlan`, and the runtime
walks that plan. Everything else in the workspace is either vocabulary
([Foundation](/soma/internals/foundation/)) or a caller.

The [notation](/soma/internals/map/) legend applies throughout. `(!)` marks a
documented deviation, with the entry in the [Debt Register](/soma/internals/debt/).

---

## soma-compiler (`somatize-compiler`)

### Mandate

Turn a `Graph` into an `ExecutionPlan` and say what is wrong with it while doing
so. It resolves cache state, validates schemas between connected nodes, claims
loop bodies and branch arms by dominance, and wraps remote nodes. It performs no
I/O and executes nothing — it reads a `&dyn NodeRegistry` and a `&dyn CacheStore`
and returns a plan plus diagnostics.

`3 118 lines across 4 files · 1 trait · 10 structs · 4 enums · deps: somatize-core`

### Modules

| File | Lines | Owns |
|---|---|---|
| `soma-compiler/src/compiler.rs` | 1 624 | `CompileMode`, `Diagnostic`, `DiagnosticLevel`, `CompileResult`, `NodeRegistry`, `SimpleNodeRegistry`, `Compiler<'a>`, private `PlanCtx<'b>`, free `compile` / `compile_stream` |
| `soma-compiler/src/plan.rs` | 936 | `ExecutionPlan` and its structural walks (`own_node_ids`, `children`, `node_count`, `simplify`) plus three renderers |
| `soma-compiler/src/scheduler.rs` | 533 | `WorkerInfo`, `Assignment`, `Phase`, `DistributionPlan`, `PlanPhase`, `DataTransfer`, `schedule` |
| `soma-compiler/src/lib.rs` | 25 | 11 re-exports |

### Public contracts

#### `NodeRegistry` — `soma-compiler/src/compiler.rs:65`

```rust
pub trait NodeRegistry: Send + Sync {
    fn node_meta(&self, node_id: &str) -> Option<NodeMeta>;   // required
    fn config_hash(&self, node_id: &str) -> Option<CacheKey>; // required
    fn meta(&self, node_id: &str) -> Option<FilterMeta> { … } // provided :83
}
```

The compiler's only port into the outside world. `meta` filters out effectful
nodes and calls `as_filter_meta()`.

| Implementor | Crate | Distinguishing behaviour |
|---|---|---|
| `SimpleNodeRegistry` | `soma-compiler/src/compiler.rs:146` | A `HashMap` populated by hand; used by tests and by callers that have no runtime |
| `NodeCatalog` | `soma-runtime/src/execution/node_catalog.rs:230` | The real one — the same registry the executor reads, holding both filters and steps |

Object-safe, used as `&'a dyn NodeRegistry` (`soma-compiler/src/compiler.rs:204`).

The doc comment at `soma-compiler/src/compiler.rs:60` records why the shape is
what it is: an earlier version had a required `meta` and an *optional*
`step_meta`, which meant `.compile()` skipped every step's schema validation
while `.run()` checked them. Making both methods required over the unified
`NodeMeta` is what closed that.

### Types — structs

| Name | Role | Key fields | Owns | file:line |
|---|---|---|---|---|
| `Compiler<'a>` | The one-shot compile object | `graph`, `registry`, `mode`, `diagnostics` | `&'a Graph` ──▷, `&'a dyn NodeRegistry` ──▷ | `soma-compiler/src/compiler.rs:202` |
| `CompileResult` | What a compile returns | `plan`, `diagnostics` | `ExecutionPlan` ──◆ | `soma-compiler/src/compiler.rs:50` |
| `Diagnostic` | One warning or note | `node_id`, `level`, `message` | — | `soma-compiler/src/compiler.rs:28` |
| `SimpleNodeRegistry` | Hand-built registry | `entries: HashMap<String, (NodeMeta, CacheKey)>` | — | `soma-compiler/src/compiler.rs:91` |
| `PlanCtx<'b>` *(private)* | Dominance + level analysis | `levels`, `dominators` | — | `soma-compiler/src/compiler.rs:162` |
| `WorkerInfo` | A placement candidate | `id`, `name`, `tags`, `gpu`, `cpu_cores`, `active_jobs`, `max_concurrent` | — | `soma-compiler/src/scheduler.rs:15` |
| `Assignment` | One node → one worker | `node_id`, `worker_id`, `worker_name`, `phase`, `reason` | — | `soma-compiler/src/scheduler.rs:57` |
| `DistributionPlan` | The scheduler's output | `assignments`, `phases`, `data_transfers`, `warnings` | — | `soma-compiler/src/scheduler.rs:93` |
| `PlanPhase` | One barrier-delimited stage | `phase_index`, `phase_type`, `node_ids`, `worker_ids` | — | `soma-compiler/src/scheduler.rs:109` |
| `DataTransfer` | An edge that crosses workers | `from_node`, `to_node`, `from_worker`, `to_worker`, `transfer_type` | — | `soma-compiler/src/scheduler.rs:124` |

`Compiler::compile(self, …)` consumes `self` — it is a one-shot object, not a
reusable service.

### Types — enums

| Name | Variants | `!` | Why | file:line |
|---|---|---|---|---|
| `ExecutionPlan` | `Sequence(Vec)`, `Parallel(Vec)`, `Execute{node_id}`, `Step{node_id, handoffs}`, `Loop{node_id, body, max_iterations, until, carry_from}`, `Branch{node_id, arms}`, `Remote{node_id, target, plan}`, `Composite{node_ids}`, `Stream{node_ids, chunk_size}`, `Empty` | yes | A data enum crossing the wire (`SerializedPlan.plan`); an old worker must tolerate a new variant rather than fail to deserialize | `soma-compiler/src/plan.rs:19` |
| `CompileMode` | `Inference`, `Differentiable`, `NoCache` | no | Internal, three callers | `soma-compiler/src/compiler.rs:17` |
| `DiagnosticLevel` | `Warning`, `Info` | no | Internal | `soma-compiler/src/compiler.rs:40` |
| `Phase` | `Sequential`, `Parallel`, `Trial{trial_index, total}` | no | Internal — `Trial` is never constructed `(!)` [D-65](/soma/internals/debt/#d-65--the-schedulers-capability-model-is-unimplemented) | `soma-compiler/src/scheduler.rs:76` |

`ExecutionPlan` is recursive in four different shapes, which is why it needs a
single traversal rather than eight ad-hoc ones:

```
ExecutionPlan
  ├──◆ Vec<ExecutionPlan>            Sequence, Parallel
  ├──◆ Box<ExecutionPlan>            Loop.body, Remote.plan
  ├──◆ Vec<(NodeId, ExecutionPlan)>  Step.handoffs
  └──◆ Vec<(String, ExecutionPlan)>  Branch.arms
```

`ExecutionPlan::children()` (`soma-compiler/src/plan.rs:142`) is that traversal,
written out per variant on purpose so a new variant fails to compile — see the
comment at `:143`. Two functions in the same crate do **not** use it and are
wrong as a direct result: `(!)` [D-32](/soma/internals/debt/#d-32--the-compiler-never-descends-into-loop-or-branch).

### Ownership and relationships

```
compile(graph, registry, mode, cache)                soma-compiler/src/compiler.rs:1022
  └──▷ Compiler<'a>
        ├──▷ &'a Graph                   (borrowed, never mutated)
        ├──▷ &'a dyn NodeRegistry        « the only port out »
        ├──◆ CompileMode
        └──◆ Vec<Diagnostic>             (accumulated, returned)
             ↓
        CompileResult ──◆ ExecutionPlan
                      └──◆ Vec<Diagnostic> ──◆ DiagnosticLevel

schedule(plan, workers)                              soma-compiler/src/scheduler.rs
  └──◆ DistributionPlan ──◆ Vec<Assignment> ──◆ Phase
                        ├──◆ Vec<PlanPhase>
                        └──◆ Vec<DataTransfer>
```

### Entry points

| Function | file:line | What it does |
|---|---|---|
| `compile` | `soma-compiler/src/compiler.rs:1022` | The whole thing. Breakpoint here first. |
| `compile_stream` | `soma-compiler/src/compiler.rs:1047` | Produces `ExecutionPlan::Stream`; refuses DAGs, steps, and chunk size 0 |
| `Compiler::plan_for_node` | `soma-compiler/src/compiler.rs:531` | 105 lines — the per-`NodeKind` dispatch, including the dominance-based body/arm claiming |
| `Compiler::validate_control_flow` | `soma-compiler/src/compiler.rs:287` | 96 lines, 5 levels of nesting — where loop and branch structure is checked |
| `Compiler::validate_schemas` | `soma-compiler/src/compiler.rs:890` | Edge-by-edge dtype/shape compatibility |
| `schedule` | `soma-compiler/src/scheduler.rs:164` | Worker placement (round-robin today) |

### Patterns in use

- **Composite** — `ExecutionPlan` is a recursive tree over which `execute` and `children` both recurse. → [Patterns](/soma/internals/patterns/#composite)
- **Visitor-ish exhaustive walk** — `children()` written per variant so a new variant breaks the build rather than being silently skipped.
- **Strategy via `dyn`** — the registry port. → [Patterns](/soma/internals/patterns/#strategy)
- **Null Object** — `ExecutionPlan::Empty`.
- **Collecting parameter** — `Vec<Diagnostic>` accumulated through the compile and returned alongside the result rather than logged.

### Debt

- [D-32](/soma/internals/debt/#d-32--the-compiler-never-descends-into-loop-or-branch) — `resolve_distribution` / `collapse_differentiable` skip `Loop` and `Branch` bodies **(High)**
- [D-65](/soma/internals/debt/#d-65--the-schedulers-capability-model-is-unimplemented) — the scheduler's capability model is defined and unused
- [D-17](/soma/internals/debt/#d-17--four-renderers-four-independent-match-nodekind) — `mermaid_nodes` and `graph_nodes` duplicate a whole recursive walk
- [D-18](/soma/internals/debt/#d-18--two-worker-capability-models-in-one-workspace) — two worker-capability models in one workspace
- Sub-graph compilation drops its diagnostics; `plan_for_node` guesses for an unknown id — see [smaller observations](/soma/internals/debt/#smaller-observations)

---

## soma-runtime (`somatize-runtime`)

### Mandate

Execute a compiled plan. It owns the executor, the node catalog, the caches, the
effect driver and journal, the study and PBT loops, the samplers and pruners, and
the run-directory tracking. It is deliberately **not async**: `tokio` is pulled
in with `default-features = false, features = ["sync"]`
(`soma-runtime/Cargo.toml:14`) purely for the broadcast channel, and every
concurrency site uses `std::thread::scope`.

`17 290 lines in src/ across 35 files in 6 domains · but ~8 181 of those are inline #[cfg(test)],
so ~9 268 lines of production code · 12 traits · 3 enums · deps: somatize-core, somatize-compiler`

That test ratio is worth internalizing before reading anything here: `executor.rs`
is 2 472 lines of which 1 184 are tests (they start at `:1289`);
`executors/study.rs` is 1 915 lines of which 1 470 are tests (`:446`).

### Modules

Six domain folders, named as they are in `soma-core` and `soma-python`.
Each folder's `mod.rs` opens by saying what the domain is; the two written
for this crate are `execution/mod.rs`, which describes the stack from
`GraphSession` down to `run_node`'s three primitives, and `optimizer/mod.rs`,
which says what this crate adds to the search space `soma-core` defines.

**`execution/` — turning a plan into results.**

| File | Lines (code) | Owns |
|---|---|---|
| `soma-runtime/src/execution/executor.rs` | 2 371 (1 187) | The plan walker: `GraphInfo`, `RunMode`, `Context`, `execute`, the three primitives, `run_node`, per-variant handlers |
| `soma-runtime/src/execution/graph_session.rs` | 835 (485) | `GraphSession` — the top orchestrator; `run` / `fit` / `forward` |
| `soma-runtime/src/execution/stream.rs` | 699 (355) | `StreamRun`, `StreamOutput`, `materialize_buffer` |
| `soma-runtime/src/execution/node_catalog.rs` | 446 (228) | `NodeImpl`, `NodeCatalog` — the one registry |
| `soma-runtime/src/execution/forward.rs` | 260 (153) | `ForwardEnv`, `ForwardStrategy` + `Standard` / `Stream` / `Batched` |
| `soma-runtime/src/execution/runner/mod.rs` | 140 | `RunContext<'a>`, `Runner` trait |
| `soma-runtime/src/execution/runner/local.rs` | 181 (86) | `LocalRunner` — the only real runner |
| `soma-runtime/src/execution/runner/remote.rs` | 70 | `Transport` trait |

**`optimizer/` — the half of hyperparameter search that runs.**

| File | Lines (code) | Owns |
|---|---|---|
| `soma-runtime/src/optimizer/study.rs` | 1 915 (445) | `TrialOutcome`, `TrialContext`, `TrialExecutor`, `StudyRunner` |
| `soma-runtime/src/optimizer/sampler/mod.rs` | 543 (309) | `Sampler`, `GridSampler`, `RandomSampler`, RNG helpers |
| `soma-runtime/src/optimizer/pbt.rs` | 465 (349) | `PbtConfig`, `PopulationMember`, `PbtExecutor`, `PbtRunner` |
| `soma-runtime/src/optimizer/sampler/bayesian.rs` | 361 (196) | `BayesianSampler` — simplified TPE |
| `soma-runtime/src/optimizer/pruner.rs` | 260 (165) | `Pruner`, `MedianPruner`, `PercentilePruner` |
| `soma-runtime/src/optimizer/study_io.rs` | 96 (41) | `StudyIo` extension trait — `Study::save` / `load` |

**`agentic/` — performing what steps ask for.**

| File | Lines (code) | Owns |
|---|---|---|
| `soma-runtime/src/agentic/mod.rs` | 1 469 (554) | `EffectDriver` — the turn loop |
| `soma-runtime/src/agentic/graph_handler.rs` | 723 (244) | `GraphHandler`, `MAX_GRAPH_DEPTH` |
| `soma-runtime/src/agentic/journal.rs` | 503 (185) | `EffectSite`, `EffectJournal` |
| `soma-runtime/src/agentic/sleep_handler.rs` | 54 (35) | `SleepHandler` |

**`cache/` — the three tiers and the store beneath them.**

| File | Lines (code) | Owns |
|---|---|---|
| `soma-runtime/src/cache/fs_store.rs` | 539 (376) | `FsActionStore` — action records + CAS blobs + pins |
| `soma-runtime/src/cache/memory.rs` | 449 (220) | `MemoryCache` + byte-bounded LRU |
| `soma-runtime/src/cache/local.rs` | 353 (196) | `LocalCache` — sharded filesystem cache |
| `soma-runtime/src/cache/gc.rs` | 275 (130) | `GcPolicy`, `GcReport`, value-density eviction |
| `soma-runtime/src/cache/tiered.rs` | 228 (104) | `TieredCache` — ordered tiers, promotion on hit |

**`tracking/` — the bus, the run directory, and reading one back.**

| File | Lines (code) | Owns |
|---|---|---|
| `soma-runtime/src/tracking/reader.rs` | 926 (882) | `RunReader` + 10 chart-ready DTOs |
| `soma-runtime/src/tracking/summary.rs` | 646 (313) | `summarize(&RunReader) -> RunSummary` |
| `soma-runtime/src/tracking/event_bus.rs` | 336 (95) | `EventBus` — broadcast + synchronous sinks |
| `soma-runtime/src/tracking/local_tracker.rs` | 245 | `LocalTracker` |
| `soma-runtime/src/tracking/head.rs` | 224 (126) | `.soma/HEAD` lineage |
| `soma-runtime/src/tracking/jsonl_sink.rs` | 161 | `JsonlEventSink` |

**`distributed/` — one file, as in `soma-core`.**

| File | Lines (code) | Owns |
|---|---|---|
| `soma-runtime/src/distributed.rs` | 1 400 (899) | Running a `TrainingStrategy`: data-parallel, federated, model-parallel |

### Public contracts

Twelve traits are defined here. **None uses an associated type or a generic
method**, so all but `StudyIo` are object-safe by construction — a deliberate
uniformity that makes every seam swappable at runtime.

#### `Runner` — `soma-runtime/src/execution/runner/mod.rs:153`

```rust
pub trait Runner: Send + Sync {
    fn fit(&self, plan: &ExecutionPlan, ctx: &RunContext<'_>, input: &Value, y: Option<&Value>)
        -> Result<Fitted>;
    fn forward(&self, plan: &ExecutionPlan, ctx: &RunContext<'_>, input: &Value) -> Result<Value>;
}
```

`Fitted` (`soma-runtime/src/execution/runner/mod.rs:134`) is `{ last, outputs,
states }`. It used to be one `HashMap` holding both, told apart by a
`__state_` prefix on the key — and all four callers separated them again in
their own three lines, one of them wrongly. The prefix is a key inside the
runner's value store, and it stops at `LocalRunner::fit`.

| Implementor | file:line | Note |
|---|---|---|
| `LocalRunner` | `soma-runtime/src/execution/runner/local.rs:59` | The only one used. `walk()` builds a `Context` and calls `executor::execute` |

`(!)` Never used as `dyn Runner` anywhere. Both call sites name `LocalRunner`
concretely.

#### `Transport` — `soma-runtime/src/execution/runner/remote.rs:18`

```rust
pub trait Transport: Send + Sync {
    fn execute(&self, plan: &ExecutionPlan, filters: &NodeCatalog, input: &Value,
               mode: &RunMode, seed: Option<i64>) -> Result<(Value, HashMap<String, Value>)>;
    fn get_state(&self, node_ids: &[String]) -> Result<HashMap<String, Value>>;
    fn set_state(&self, states: &HashMap<String, Value>) -> Result<()>;
    fn get_gradients(&self, node_ids: &[String]) -> Result<HashMap<String, Value>>;
    fn apply_gradients(&self, gradients: &HashMap<String, Value>) -> Result<()>;
    fn execute_node(&self, node_id: &str, input: Option<&Value>) -> Result<Value>; // provided :61 (!)
}
```

The wire seam. No implementor lives in this crate — `WsTransport` is in
`soma-worker/src/ws_transport.rs:404`. Heavily dyn-dispatched:
`Arc<dyn Transport>` in `Context` (`soma-runtime/src/execution/executor.rs:144`),
`GraphSession` (`:44`, `:50`) and `TransportContext` (`soma-runtime/src/distributed.rs:531`).

`(!)` The provided `execute_node` builds a throwaway empty catalog and passes
`seed: None` — [D-41](/soma/internals/debt/#d-41--transportexecute_node-runs-remotes-with-an-empty-catalog).

#### `ForwardStrategy` — `soma-runtime/src/execution/forward.rs:40`

`fn forward(&self, graph: &Graph, env: &ForwardEnv<'_>, x: &Value) -> Result<Value>`

| Implementor | file:line | Difference |
|---|---|---|
| `Standard` | `soma-runtime/src/execution/forward.rs:48` | `compile` then `run_forward` |
| `Stream` | `soma-runtime/src/execution/forward.rs:63` | `compile_stream(chunk_size)` instead |
| `Batched<'a>` | `soma-runtime/src/execution/forward.rs:106` | Loops `store.get_rows` and calls `run_forward` per batch |

#### `Sampler` — `soma-runtime/src/optimizer/sampler/mod.rs:22`

```rust
fn prepare(&mut self, _space: &SearchSpace) {}                                   // provided
fn sample(&mut self, space, trial_index) -> Result<Option<HashMap<String, Value>>>; // required
fn n_trials(&self) -> Option<usize>;                                              // required
fn record_result(&mut self, _params, _value: f64) {}                              // provided
```

| Implementor | file:line | Overrides |
|---|---|---|
| `GridSampler` | `soma-runtime/src/optimizer/sampler/mod.rs:151` | `prepare` — lazy mixed-radix index, never materializes the product |
| `RandomSampler` | `soma-runtime/src/optimizer/sampler/mod.rs:216` | — |
| `BayesianSampler` | `soma-runtime/src/optimizer/sampler/bayesian.rs:162` | `record_result` — simplified TPE, γ = 0.25 |

#### `Pruner` — `soma-runtime/src/optimizer/pruner.rs:10`

`fn should_prune(&self, metric_name, current_value, step, history) -> Option<String>`
— returning the *reason* rather than a bool, so the event carries it.
Implementors: `MedianPruner` (`:58`), `PercentilePruner` (`:127`), which are the
same 20 lines with a different statistic `(!)` [D-14](/soma/internals/debt/#d-14--two-pruners-two-samplers-one-algorithm-each).

#### `TrialExecutor` / `PbtExecutor`

`TrialExecutor::execute_trial(&self, params, ctx) -> Result<TrialOutcome>`
(`soma-runtime/src/optimizer/study.rs:133`), implemented by
`FnTrialExecutor<F>` (`:144`). `PbtExecutor` has `train` + `evaluate`
(`soma-runtime/src/optimizer/pbt.rs:55`), implemented by `FnPbtExecutor<T, E>`
(`:63`). Both are the callback-adapter pattern: the only implementor wraps a
closure, so the trait exists to be `dyn`-able, not to be subclassed.

#### `StrategyContext` / `StrategyExecutor` / `GradientAggregator` / `StateAggregator`

The distributed-training seam, and the crate's clearest use of **extension
traits** — the traits are defined here, but implemented on *foreign* types from
`soma-core`, keeping "core holds contracts, runtime holds execution"
(`soma-runtime/src/distributed.rs:1`).

| Trait | file:line | Implemented on |
|---|---|---|
| `StrategyContext` | `soma-runtime/src/distributed.rs:33` | `TransportContext<'_>` (`:589`) — 6 required, 3 provided, two of which default to *refusing* |
| `StrategyExecutor` | `soma-runtime/src/distributed.rs:120` | `TrainingStrategy` (foreign, `:145`) |
| `GradientAggregator` | `soma-runtime/src/distributed.rs:132` | `GradientAggregation` (foreign, `:461`) `(!)` [D-21](/soma/internals/debt/#d-21--mean_by_key-panics-on-an-empty-slice) |
| `StateAggregator` | `soma-runtime/src/distributed.rs:139` | `FederatedAggregation` (foreign, `:486`) |

`(!)` `StrategyContext::execute_on_worker` carries a dead `plan: &serde_json::Value`
parameter — [D-43](/soma/internals/debt/#d-43--strategycontextexecute_on_worker-has-a-dead-json-parameter).

#### `StudyIo` — `soma-runtime/src/optimizer/study_io.rs:19`

The one **non-object-safe** trait (`Sized` + `impl Trait` args + a static
`load`). Implemented on `Study` (foreign, `:28`) so that a `soma-core` type gets
filesystem persistence without `soma-core` gaining a filesystem.

#### Foreign traits realized here

| Trait (from) | Implementors | file:line |
|---|---|---|
| `CacheStore` (core) | `MemoryCache`, `LocalCache`, `TieredCache`, `FsActionStore` | `cache/memory.rs:129`, `cache/local.rs:115`, `cache/tiered.rs:30`, `cache/fs_store.rs:304` |
| `BlobStore` + `ActionCache` (core) | `FsActionStore` (both) | `cache/fs_store.rs:255`, `:281` |
| `NodeRegistry` (compiler) | `NodeCatalog` | `node_catalog.rs:230` |
| `EventSink` (core) | `JsonlEventSink` | `tracking/jsonl_sink.rs:124` |
| `Tracker` (core) | `LocalTracker` | `tracking/local_tracker.rs:90` |
| `EffectHandler` (core) | `GraphHandler`, `SleepHandler` | `effects/graph_handler.rs:136`, `effects/sleep_handler.rs:20` |

### Types — structs

| Name | Role | Key fields | Owns | file:line |
|---|---|---|---|---|
| `GraphSession` | The top orchestrator | graph, catalog, cache, bus, 2× transport, identities, driver, `fitted` | `Graph` ──◆, `NodeCatalog` ──◆, `Arc<dyn CacheStore>` ──◇ | `soma-runtime/src/execution/graph_session.rs:38` |
| `NodeCatalog` | THE registry — filters *and* steps | `nodes: HashMap<String, NodeImpl>`, `states: Arc<dyn StateStore>` | `Arc<dyn Filter\|Step>` ──◇; **clones share the state store** (`:75`) | `soma-runtime/src/execution/node_catalog.rs:79` |
| `Context` | The executor's mutable run state | 12 fields `(!)` | value store, execution order, hash memo (all private) | `soma-runtime/src/execution/executor.rs:124` |
| `RunContext<'a>` | A runner's borrowed view | catalog, cache, events, run id, `GraphInfo`, seed, driver | all borrowed except `GraphInfo` | `soma-runtime/src/execution/runner/mod.rs:32` |
| `ForwardEnv<'a>` | A forward strategy's borrowed view | catalog, cache, bus, store, driver | all borrowed | `soma-runtime/src/execution/forward.rs:25` |
| `GraphInfo` | Topology, not order | `predecessors: HashMap<String, Vec<String>>` | — | `soma-runtime/src/execution/executor.rs:28` |
| `EffectDriver` | The turn loop | handlers, journal, bus, catalog | `Vec<Arc<dyn EffectHandler>>` ──◇ | `soma-runtime/src/agentic/mod.rs:57` |
| `EffectJournal` | Record once, replay forever | `actions`, `blobs`, `enabled` | `Arc<dyn ActionCache>` ──◇, `Arc<dyn BlobStore>` ──◇ | `soma-runtime/src/agentic/journal.rs:51` |
| `EffectSite<'a>` | The impure-effect key | `run_id`, `node_id`, `turn`, `index` | — (`Copy`) | `soma-runtime/src/agentic/journal.rs:36` |
| `GraphHandler` | A graph as a tool for an agent | `library`, `cache`, `step_runtime` | `NodeCatalog` ──◆; recursion capped at 8 | `soma-runtime/src/agentic/graph_handler.rs:47` |
| `EventBus` | Dual-path pub/sub | `broadcast::Sender<Event>`, `RwLock<Vec<Arc<dyn EventSink>>>` | sinks ──◇ | `soma-runtime/src/tracking/event_bus.rs:22` |
| `StreamRun` | The chunk driver | `nodes: Vec<StreamNode>`, `chunk_count` | per-node base state, barrier buffer, evolving state | `soma-runtime/src/execution/stream.rs:73` |
| `StreamOutput` | Chunk accumulator | `all_data`, `result_shape`, `non_tensor` | — | `soma-runtime/src/execution/stream.rs:283` |
| `StudyRunner` | The trial loop | `event_bus`, `tracker` | — | `soma-runtime/src/optimizer/study.rs:160` |
| `TrialContext` | What user trial code sees | objective, pruner, history, bus, `Arc<Mutex<TrialShared>>` | — | `soma-runtime/src/optimizer/study.rs:48` |
| `PbtRunner` / `PbtConfig` / `PopulationMember` | Population-based training | — | — | `soma-runtime/src/optimizer/pbt.rs:84`, `:22`, `:41` |
| `TransportContext<'a>` | The `StrategyContext` impl | transports, plan, catalog, seed, `Mutex<Vec<states>>`, identities | `Vec<Arc<dyn Transport>>` ──◇ | `soma-runtime/src/distributed.rs:530` |
| `MemoryCache` / `LocalCache` / `TieredCache` / `FsActionStore` | The cache tiers | see [D3](#d3--the-cache-and-journal-stack) | — | `cache/memory.rs:16`, `local.rs:15`, `tiered.rs:11`, `fs_store.rs:40` |
| `RunReader` | Run dir → chart-ready DTOs | `dir: PathBuf` | — | `soma-runtime/src/tracking/reader.rs:36` |
| `LocalTracker` / `JsonlEventSink` | Run dir writers | — | — | `tracking/local_tracker.rs:27`, `jsonl_sink.rs:19` |

### Types — enums

Only three, and **none is `#[non_exhaustive]`** — all three are internal
control-flow enums every consumer must decide over.

| Name | Variants | Why exhaustive | file:line |
|---|---|---|---|
| `RunMode` | `Forward`, `Fit { y: Option<Value> }` | Two whole execution loops collapsed into one parameter (`:82`); `(!)` it also crosses the wire — [D-45](/soma/internals/debt/#d-45--runmode--an-executor-internal-enum--is-a-wire-parameter) | `soma-runtime/src/execution/executor.rs:92` |
| `NodeImpl` | `Filter(Arc<dyn Filter>)`, `Step(Arc<dyn Step>)` | "the only place in the workspace that names the two kinds" (`:31`) | `soma-runtime/src/execution/node_catalog.rs:37` |
| `TrialOutcome` | `Completed(Vec<MetricRecord>)`, `Pruned { step, reason }` | Separates control flow from error — pruning is not a failure | `soma-runtime/src/optimizer/study.rs:24` |

---

## D1 · The node seam

The single most important structure in the workspace. One registry, one
execution site, and exactly one `match` that tells a filter from a step.

```
      «trait» Filter                    «trait» Step
      (soma-core/src/graph/filter.rs:120)     (soma-core/src/graph/step.rs:250)
         fit / forward                     poll -> Transition
              ▲                                 ▲
              │                                 │
        ┌─────┴──────┐                    ┌─────┴─────────────┐
   PyFilterBridge  SubprocessFilter   ReactStep  PyStepBridge  ResearchStep
                                      LlmStep    JudgeStep
              │                                 │
              └────────────┐     ┌──────────────┘
                           ▼     ▼
                  [enum] NodeImpl { Filter | Step }
                     soma-runtime/src/execution/node_catalog.rs:37
                           │
                           ◆
                  NodeCatalog ──◇ Arc<dyn StateStore>   « shared across clones »
                     soma-runtime/src/execution/node_catalog.rs:79
                           │
             ┌─────────────┴──────────────┐
             ▼                            ▼
   «trait» NodeRegistry            executor::run_node
   (the compiler's port)           soma-runtime/src/execution/executor.rs:816
   soma-compiler/…:65                        │
             │                               ▼
             ▼                       run_node_inner   « THE match »
     NodeMeta ◁── From<FilterMeta>   soma-runtime/src/execution/executor.rs:1058
              ◁── From<StepMeta>       Filter → forward()
     soma-core/src/graph/node.rs:72          Step   → driver.run()
```

`From<StepMeta> for NodeMeta` (`soma-core/src/graph/node.rs:132`) sets
`cacheable: false, deterministic: false`, which is why "a step is not
output-cacheable" needs no `if is_step` anywhere — the executor's existing
cacheability guard (`soma-runtime/src/execution/executor.rs:677`) reads it as data.

---

## D2 · The execution pipeline

```
Graph ──▷ compile() ──▷ ExecutionPlan ──▷ LocalRunner::walk ──▷ Context
                                                                  │
                                            executor::execute(plan, ctx, catalog, cache)
                                                soma-runtime/src/execution/executor.rs:367
                                                                  │
        ┌────────────┬──────────┬─────────┬──────────┬────────────┼──────────┐
        ▼            ▼          ▼         ▼          ▼            ▼          ▼
   Sequence     Parallel      Loop     Branch     Remote      Composite    Stream
   (recurse)  thread::scope  :460      :531       :601      composite_fit  :1208
                  :1084                                          :953        │
        └────────────┴──────────┴─────────┴──────────┴────────────┘          │
                                    │                                        │
                                    ▼                                        ▼
                              run_node  :816                          StreamRun::run_compute
                                    │                                 stream.rs:194
                    ┌───────────────┼───────────────┐                        │
                    ▼               ▼               ▼                        │
             output_key :670  compute_node :692  store_output :717 ◁─────────┘
             guard+derive+    catch_unwind       provenance         « the three shared
             seed salt              │                                 primitives »
                                    ▼
                            run_node_inner :1058
                            Filter → forward | Step → EffectDriver::run
```

The three primitives are the whole point: `run_node` composes them once for the
batch path, `StreamRun` composes them per chunk. `(!)` But everything *around*
them is written twice, and the two copies have drifted —
[D-11](/soma/internals/debt/#d-11--the-stream-path-re-implements-run_node-and-has-drifted).

---

## D3 · The cache and journal stack

Two content-addressed systems that look similar and are not. A filter
**memoizes by content**; a step **journals by site**.

```
                    «trait» CacheStore              soma-core/src/cache/mod.rs:204
                     get/put/exists/remove/metadata
                     + put_computed, get_located, tier   « defaulted »
                              ▲
        ┌─────────────┬───────┴───────┬────────────────┐
   MemoryCache    LocalCache     TieredCache      FsActionStore
   LRU, max_bytes  aa/bb/hex.json  ordered tiers,   two tables + pins
   memory.rs:16    local.rs:15     promotes on hit   fs_store.rs:40
                   (!) no bound    tiered.rs:11            │
                                   (!) promotion loses     ├──▷ «trait» ActionCache
                                       Origin                │    action records, kept forever
                                                             └──▷ «trait» BlobStore
                                                                  BLAKE3 CAS, evictable by gc.rs

  CacheKey derivation                       soma-core/src/cache/mod.rs:18
     state  = hash(config ‖ x ‖ y)                for_state
     output = hash(config ‖ state ‖ input_hash)   for_output
     + salt_with_seed(seed)                       executor.rs:634
     → downstream keys use input *content* hashes, so an unchanged
       intermediate cuts off the rest of the graph early

  EffectJournal                             soma-runtime/src/agentic/journal.rs:51
     pure effect   → key = content only                    « reusable across runs »
     impure effect → key = b"sited" ‖ run ‖ node ‖ turn ‖ index
     lookup() replays; record() writes; Failed results are never recorded (:157)
```

---

## Execution traces

Annotated call chains. These are the fastest way back into the code: pick one,
set a breakpoint at the top, and step.

They read as five separate chains and they are not — they are five entry points
into one graph, and the six hops they share are where the architecture's
load-bearing claims live. [Call Paths](/soma/internals/paths/) draws that
overlap, which two blocks three hundred lines apart cannot.

:::note[Generated]
The blocks below come from `docs/data/traces.json` via
`node scripts/gen-traces.mjs`, which also feeds the Call Paths page. Edit the
data, not the blocks — two hand-maintained copies of the same call chains is the
[duplication smell](/soma/internals/debt/#duplicated-logic) this section
documents. `npm run check` verifies all 99 of their source anchors.
:::

<!-- traces:begin -->

### (a) `GraphSession::forward`

```
GraphSession::forward(x)                                      graph_session.rs:333
└─ forward_with(x, &Standard)                                 graph_session.rs:334
   ├─ run_driver()  → driver.clone().with_catalog(…)          graph_session.rs:145
   └─ Standard::forward(graph, &ForwardEnv{…}, x)             forward.rs:49
      ├─ compile(graph, catalog, Inference, Some(cache))      forward.rs:51
      └─ run_forward(graph, &plan, env, x)                    forward.rs:77
         ├─ timestamp_id("forward")                           forward.rs:83
         ├─ RunContext::new(…, GraphInfo::from_graph(graph))  forward.rs:84
         └─ LocalRunner.forward(plan, &ctx, x)                forward.rs:94
            └─ walk(plan, ctx, input, RunMode::Forward)       runner/local.rs:26
               ├─ Context::new(…).with_graph_info(…).with_seed(…)  runner/local.rs:33
               ├─ exec.set(input_key(first), input)           runner/local.rs:40
               ├─ executor::execute(…)  → (b)                 runner/local.rs:44
               └─ last_output(&exec)                          runner/local.rs:51
                     execution_order().rev().find(!reserved)
```

`Stream` diverges only at `soma-runtime/src/execution/forward.rs:65` (`compile_stream`); `Batched` at
`soma-runtime/src/execution/forward.rs:107` (its own `get_rows` loop calling `run_forward` per batch).

### (b) `execute` → `run_node` → the three primitives

```
execute(plan, ctx, catalog, cache)                              executor.rs:367
├─ Empty                → Ok(())                                :374
├─ Execute{id}          → execute_node(id, &[], …)              :377
├─ Step{id, handoffs}   → execute_node(id, handoffs, …)         :379
├─ Sequence(v)          → for each: execute(…)                  :383
├─ Parallel(b)          → execute_parallel                      :1084
├─ Loop{…}              → execute_loop                          :460
├─ Branch{…}            → execute_branch                        :531
├─ Remote{…, target: _} → execute_remote  (! target discarded)  :601
├─ Composite{ids}       → composite_fit (fit) | per-node (fwd)  :953
├─ Stream{ids, size}    → execute_stream  → (d)                 :1208
└─ other                → Err("newer compiler")                 :445

run_node(node_id, ctx, catalog, cache)                            executor.rs:816
├─ node = catalog.node(id)?.clone(); meta = node.meta()           :824
├─ input = resolve_input(node_id, ctx)                            :1173
│     0 preds → execution_order.last()  (!)  D-44
│     1 pred  → that pred
│     n preds → merged JSON object, keyed by predecessor
├─ fitted = fit_state_if_needed(…)                                :1002
│     guard: mode.is_fit() && meta.trainable()
│     key = salt_with_seed(CacheKey::for_state(config, x, y), seed)
│     hit → reuse | miss → catch_unwind(filter.fit) → put_computed
├─ ▸ PRIMITIVE 1  output_key(node, meta, state, input_hash, seed) :670
│     guard: !(meta.cacheable && meta.deterministic) → None
├─ cache.get_located(key) hit → emit NodeCacheHit, return Produced  :862
├─ miss → emit NodeCacheMiss, emit NodeStarted                    :876
├─ ▸ PRIMITIVE 2  compute_node(…) = catch_unwind(run_node_inner)  :692
│  └─ run_node_inner                                              :1058
│     ├─ NodeImpl::Filter(f) → f.forward(input, state) → Produced :1066
│     └─ NodeImpl::Step(s)   → driver.run(s, run_id, node_id, input)  → (c)  :1069
└─ match outcome                                                  :905
   ├─ Produced(out) → ▸ PRIMITIVE 3  store_output(…)              :717
   │     → maybe_spill → set_virtual → NodeCompleted
   ├─ HandOff{target, carry} → ctx.set(node, carry); NodeCompleted  :931
   └─ Paused{turn, reason}   → nothing stored                     :942

back in execute_node                              executor.rs:748
├─ Produced → Ok(())
├─ HandOff  → select_handoff(:770) → execute(that plan)
└─ Paused   → Err(SomaError::Suspended{…})
```

`execute_parallel` (`soma-runtime/src/execution/executor.rs:1084`) is worth reading in full: it marks
`execution_order.len()`, opens a `std::thread::scope`, gives each branch a
`ctx.snapshot()` `(!)` [D-61](/soma/internals/debt/#d-61--contextsnapshot-deep-clones-the-value-store-per-branch),
then merges back only the *write set* — the entries each branch appended past
the mark.

### (c) The `EffectDriver` turn loop

```
EffectDriver::run(step, run_id, node_id, input)                   effects/mod.rs:107
├─ journal = self.journal.with_enabled(enabled && meta.journal)   :116
└─ for turn in 0..meta.max_turns                                  :127
   ├─ emit AgentTurnStarted
   ├─ ctx = StepCtx::new(…).with_history(&history)                :134
   ├─ transition = step.poll(&ctx)?                               :138
   └─ match transition                                            :146
      ├─ Await(effects)  → perform_all(…)                         :440
      │  ├─ emit EffectRequested per effect
      │  ├─ thread::scope: one thread per effect                  :459
      │  │  └─ perform_one(journal, EffectSite{run,node,turn,i})  :519
      │  │     ├─ journal.lookup(site, effect)? → replayed        :527
      │  │     ├─ handlers.iter().find(|h| h.handles(effect))     :531
      │  │     ├─ handler.perform(effect)?                        :543
      │  │     └─ journal.record(…)   (Failed is never recorded)  :545
      │  └─ usage += …; emit ToolCalled / EffectCompleted  → history.push(results)  :159
      ├─ Done(v)             → Ok(NodeOutcome::Produced(v))       :167
      ├─ Goto{target, carry} → Ok(NodeOutcome::HandOff{…})        :172
      ├─ Suspend{reason}                                          :187
      │  ├─ journal.lookup(site, suspension_effect(reason))?
      │  │     Some → emit Resumed, continue   « the resume path »
      │  └─ None → emit Suspended → Ok(NodeOutcome::Paused{…})    :206
      └─ Spawn{specs, join} → spawn_all(…)                        :316
            child ids "{node_id}/{label|turn.index}"; thread::scope;
            RECURSES into self.run per child                         :365
            JoinPolicy: All | AllSettled | First  (! First still joins all)
loop exhausted → Err("did not finish within N turns")             :240
```

`resume_with(run_id, node_id, turn, reason, answer)`
(`soma-runtime/src/agentic/mod.rs:279`) writes the answer at the *same site*, so
the next run replays into it. That is the whole resume mechanism — there is no
separate checkpoint format.

### (d) `StreamRun`

Two entry points into one object. Locally, `execute_stream`
(`soma-runtime/src/execution/executor.rs:1208`) chunks the input and drives it. Remotely,
`soma-worker` holds the `StreamRun` *and its `Context`* alive in `active_streams`
between WebSocket messages and calls the same three methods itself.

```
execute_stream                                              executor.rs:1208
├─ refuse if mode is Fit                                    :1220
├─ chunks = chunk_value(input, chunk_size)                  :1264
├─ run = StreamRun::new(node_ids, catalog)   (steps → Err)  stream.rs:83
├─ per chunk: run.process_chunk(chunk, ctx, cache)          stream.rs:130
│  └─ per node i:
│     ├─ Barrier  → buffer the value, stop the cascade      :140
│     └─ else     → current = run_compute(i, current, …)    :194
│        ├─ first touch → emit NodeStarted                  :203
│        ├─ state = evolving.or(base_state)                 :213
│        ├─ ▸ output_key(…)                                 :217
│        ├─ cache hit → counters only  (!) no event         :220
│        ├─ ▸ compute_node(…)                               :233
│        └─ ▸ store_output(…); evolving update              :239
├─ run.flush(ctx, cache)  → materialize_buffer per barrier node  stream.rs:152
├─ run.finish(ctx)  → one NodeCompleted per node,           stream.rs:167
│     "stream: N chunks, H hits, M misses"
└─ ctx.set(last_id, output.finish())                        stream.rs:310
```

`FixedState` keys are **identical** to the batch path's, so a single-chunk stream
and a plain `forward` share one cache line. That invariant is what makes the
three primitives worth having.

### (e) `StudyRunner` and `PbtRunner`

```
StudyRunner::run(study, sampler, executor)                        executors/study.rs:187
├─ sampler.prepare(&study.search_space)                           :193
├─ RESUME: replay completed trials into sampler.record_result     :205
├─ pruner = build_pruner(&study.pruning)                          :407
├─ trial_index = study.trials.len()          « the resume point » :218
└─ loop                                                           :228
   ├─ config_index = i / n_seeds ; seed_slot = i % n_seeds        :229
   │     seed_slot > 0 → reuse the previous trial's params minus "seed"
   │     else          → sampler.sample(space, config_index)?
   ├─ params += {"seed": …} += study.frozen                       :250
   ├─ ctx = TrialContext{objective, pruner, history, bus, shared} :270
   ├─ outcome = executor.execute_trial(&params, &ctx)             :281
   │     user code calls ctx.report(name, value, step)             :70
   │       → push metric, emit TrialMetric, ask the pruner
   ├─ match (outcome, pruned)                                     :287
   │     (Ok(_), Some(..)) | (Ok(Pruned{..}), None) → Pruned
   │     (Ok(Completed(m)), None)                   → Completed
   │     (Err(e), _)                                → Failed
   ├─ sampler.record_result(…); best-trial check; StudyProgress   :335
   └─ save_study(study)  (! rewrites the whole file per trial)    :361

PbtRunner::run(config, executor)                                executors/pbt.rs:97
├─ rng_state = 42  (! hardcoded, no seed field)                 :103
├─ initialize_population                                        :195
└─ for generation in 0..generations                             :108
   ├─ TRAIN each member  (! failure → warn, keeps stale state)  :116
   ├─ EVAL each member  (failure → NEG_INFINITY, counted)       :135
   ├─ sort by fitness desc                                      :156
   └─ evolve: exploit (truncation | binary tournament)          :215
         then explore (perturbation | resample)
```

<!-- traces:end -->

### Ownership spine

```
GraphSession                                     graph_session.rs:38
 ├──◆ Graph
 ├──◆ NodeCatalog ──◇ Arc<dyn StateStore>        « shared across clones »
 │     └──◆ HashMap<String, NodeImpl> ──◇ Arc<dyn Filter | Step>
 ├──◇ Arc<dyn CacheStore>
 ├──◇ Arc<EventBus>
 │     ├──◆ broadcast::Sender<Event>             « lossy subscribers »
 │     └──◆ RwLock<Vec<Arc<dyn EventSink>>>      « lossless sinks »
 ├──? Option<Arc<dyn DataStore>>
 ├──? Option<Arc<dyn Transport>>    ┐ (!) two independent transport fields
 ├──◆ Vec<Arc<dyn Transport>>       ┘     D-04
 └──? Option<EffectDriver>
       ├──◆ Vec<Arc<dyn EffectHandler>>   → GraphHandler, SleepHandler, llm, tools
       ├──◆ EffectJournal ──◇ Arc<dyn ActionCache> + Arc<dyn BlobStore>
       └──? Option<Arc<NodeCatalog>>      « needed only for Transition::Spawn »

RunContext<'a> ──▷ builds ──▷ Context           runner/local.rs:33
   (borrowed view)              (owned, &mut through the walk)

GraphHandler ──◆ NodeCatalog
             └──? StepRuntime ──▷ child_driver() ──▷ EffectDriver ──▷ GraphHandler
                  « a real cycle, capped at MAX_GRAPH_DEPTH = 8 »
                  effects/graph_handler.rs:28
```

**There are zero `impl From` blocks in this crate.** Every conversion is foreign
(`FilterMeta → NodeMeta` in `soma-core`) or implicit (`?` on `io::Error`).

### Patterns in use

- **Strategy** — `ForwardStrategy`, `Sampler`, `Pruner`, `StrategyExecutor`. → [Patterns](/soma/internals/patterns/#strategy)
- **Template method** — `LocalRunner::walk` shared by `fit` and `forward`, differing only by `RunMode`; documented at `soma-runtime/src/execution/runner/local.rs:22` as the fix for two divergent loops.
- **Extension trait** — `StudyIo for Study`, `StrategyExecutor for TrainingStrategy`. → [Patterns](/soma/internals/patterns/#extension-trait)
- **Chain of responsibility** — `EffectDriver::perform_one`, first handler whose `handles()` claims the effect.
- **Event sourcing / durable execution** — `EffectJournal`; suspension modelled as an effect so resume needs no separate format.
- **Observer** — `EventBus`, dual lossy/lossless path.
- **Memento** — `Context::snapshot` for parallel branches, merged by write-set diff.
- **Decorator** — `TieredCache` is a `CacheStore` of `CacheStore`s that promotes on read.
- **Callback adapter** — `FnTrialExecutor<F>`, `FnPbtExecutor<T, E>`.
- **Scoped-thread fan-out** — three sites, no async runtime, rationale at `soma-runtime/src/agentic/mod.rs:12`.
- **Two-phase commit** — temp + fsync + rename in four places (two of them weaker `(!)`).
- **Forward-compatible refusal** — unknown variants return an error naming the situation rather than guessing. Applied consistently; a genuine strength.

### Debt

**High**
- [D-11](/soma/internals/debt/#d-11--the-stream-path-re-implements-run_node-and-has-drifted) — stream path re-implements `run_node`; emits no cache events
- [D-21](/soma/internals/debt/#d-21--mean_by_key-panics-on-an-empty-slice) — `mean_by_key` panics on an empty slice, reachable from Python
- [D-41](/soma/internals/debt/#d-41--transportexecute_node-runs-remotes-with-an-empty-catalog) — remotes run with an empty catalog and unsalted keys

**Medium**
- [D-03](/soma/internals/debt/#d-03--context-carries-five-unrelated-concerns) `Context` god object · [D-04](/soma/internals/debt/#d-04--graphsession-has-two-unrelated-transport-fields) two transport fields · [D-07](/soma/internals/debt/#d-07--runreader-is-17-methods-over-one-pathbuf) `RunReader`
- [D-12](/soma/internals/debt/#d-12--four-write_atomic-implementations-two-of-them-unsafe) four `write_atomic`s · ~~[D-13](/soma/internals/debt/#d-13--graphsession-emits-the-run-bracket-four-times) the run bracket ×4~~ resolved
- [D-22](/soma/internals/debt/#d-22--a-suspension-reason-that-fails-to-serialize-collides-with-every-other-one) suspension key collision
- [D-42](/soma/internals/debt/#d-42--executionplanremote-discards-its-routing-target) `Remote` target discarded · [D-43](/soma/internals/debt/#d-43--strategycontextexecute_on_worker-has-a-dead-json-parameter) dead JSON param · [D-44](/soma/internals/debt/#d-44--resolve_input-falls-back-to-whatever-ran-last) `resolve_input` fallback · [D-46](/soma/internals/debt/#d-46--tieredcache-promotion-destroys-provenance) promotion loses provenance
- [D-61](/soma/internals/debt/#d-61--contextsnapshot-deep-clones-the-value-store-per-branch) snapshot cost · [D-62](/soma/internals/debt/#d-62--memorycaches-lru-touch-is-on-on-every-read) O(n) LRU · [D-63](/soma/internals/debt/#d-63--runreader-re-parses-eventsjsonl-once-per-accessor) reader re-parses · [D-64](/soma/internals/debt/#d-64--studyrunnerrun-is-otrials-in-four-places) O(trials²)
- [D-71](/soma/internals/debt/#d-71--four-policies-for-a-poisoned-mutex-three-of-them-silent) four mutex-poison policies

**Low** — [D-14](/soma/internals/debt/#d-14--two-pruners-two-samplers-one-algorithm-each), [D-45](/soma/internals/debt/#d-45--runmode--an-executor-internal-enum--is-a-wire-parameter), [D-48](/soma/internals/debt/#d-48--valueempty-for-state-and-evolvings-valuestate-conflation), [D-49](/soma/internals/debt/#d-49--maybe_spill-mis-estimates-bytes-and-swallows-failure), [D-66](/soma/internals/debt/#d-66--cheap-booleans-that-cost-a-filesystem-walk)–[D-70](/soma/internals/debt/#d-70--joinpolicyfirst-waits-for-everyone), [D-72](/soma/internals/debt/#d-72--every-event-emission-clones-the-sink-vector)–[D-73](/soma/internals/debt/#d-73--catalog-clones-per-run-and-per-nesting-level), [D-81](/soma/internals/debt/#d-81--doc-comments-that-narrate-history), [D-93](/soma/internals/debt/#d-93--trials-run-one-at-a-time)

### Test coverage

10 integration files, 5 443 lines. Several are named after the bug they prevent,
which is the most useful thing a test file name can do.

| File | Lines | Covers |
|---|---|---|
| `soma-runtime/tests/agentic_step.rs` | 1 271 | The step-as-node seam: compile, `run_node`, handoffs, suspend/resume, journal replay, spawn fan-out |
| `soma-runtime/tests/tracking.rs` | 1 175 | `JsonlEventSink` (seq, append, torn-tail repair), `LocalTracker`, `RunReader`, `summarize`, HEAD lineage |
| `soma-runtime/tests/coverage_boost.rs` | 845 | Explicitly path-coverage-driven: `get_virtual`, state persistence, remote fallback |
| `soma-runtime/tests/integration.rs` | 529 | End-to-end fit → forward → cache hit → invalidation |
| `soma-runtime/tests/memory_usage.rs` | 462 | A tracking allocator asserting `Batched` and `Stream` do not grow the heap with batch count |
| `soma-runtime/tests/fit_through_run_node.rs` | 450 | Regression suite for the fit/forward unification |
| `soma-runtime/tests/pbt_integration.rs` | 251 | `PbtRunner` against real trainable filters |
| `soma-runtime/tests/session_steps.rs` | 183 | `GraphSession::with_driver` reaching steps from `run` / `fit` / `forward` |
| `soma-runtime/tests/topology.rs` | 168 | Forward follows graph topology, not plan order — the diamond regression |
| `soma-runtime/tests/fit_determinism.rs` | 87 | `fit` is reproducible |

**Not covered by anything:** `TieredCache` promotion
provenance, `ModelParallel` against a real transport (unit tests use a mock
`StrategyContext` at `soma-runtime/src/distributed.rs:996`), and `MemoryCache`
eviction under concurrent parallel branches.
