---
title: Design Patterns in Use
description: The patterns this codebase actually uses, where each one lives, and the Rust idioms that stand in for the object-oriented ones.
---

Rust has no classes, so the object-oriented pattern catalogue does not map
one-to-one. Some patterns survive unchanged (strategy, composite, chain of
responsibility). Some are absorbed by the language and stop being patterns
(iterator, singleton). And a few Rust idioms have no GoF name at all but do the
same structural work — the extension trait, the newtype, `#[non_exhaustive]`,
the consuming builder.

This page is the index. Each entry says what the pattern buys *here*, not what
it means in general, and points at the code.

---

## The reader's map

If you are trying to answer "how is this codebase organized", four patterns
carry most of the weight:

| Pattern | What it structures |
|---|---|
| [Strategy](#strategy) | Every swappable backend — caches, stores, providers, samplers, pruners |
| [Adapter / bridge](#adapter--bridge) | Every language and process boundary — Python, subprocess, socket |
| [Template method](#template-method) | Every trait where a minimal backend should be cheap to write |
| [Chain of responsibility](#chain-of-responsibility) | The effect system, which is how an agent does anything |

The rest are local decisions.

---

## Structural patterns

### Strategy

Behaviour selected at runtime through a `dyn` trait object. The dominant pattern
in the workspace, and the reason none of the 29 public traits declares an
associated type or a generic parameter — every one of them stays object-safe.

| Site | Trait | Implementations |
|---|---|---|
| Caching | `CacheStore` (`soma-core/src/cache/mod.rs:207`) | `MemoryCache`, `LocalCache`, `TieredCache`, `FsActionStore` |
| Data movement | `DataStore` (`soma-core/src/data/store.rs:208`) | `LocalDataStore`, `S3DataStore`, `ZarrStore` |
| Model access | `LlmProvider` (`soma-llm/src/lib.rs:72`) | `OpenAiCompatible` + a `Router` over `Arc<dyn LlmProvider>` |
| Forward execution | `ForwardStrategy` (`soma-runtime/src/execution/forward.rs:40`) | `Standard`, `Stream`, `Batched` |
| Search | `Sampler` (`soma-runtime/src/optimizer/sampler/mod.rs:22`) | `GridSampler`, `RandomSampler`, `BayesianSampler` |
| Pruning | `Pruner` (`soma-runtime/src/optimizer/pruner.rs:10`) | `MedianPruner`, `PercentilePruner` |
| Distributed training | `StrategyExecutor` (`soma-runtime/src/distributed.rs:120`) | `TrainingStrategy` |
| Compilation input | `NodeRegistry` (`soma-compiler/src/compiler.rs:65`) | `SimpleNodeRegistry`, `NodeCatalog` |
| Knowledge storage | `KnowledgeBase` (`soma-memory/src/knowledge_base.rs:50`) | `MemoryKnowledgeBase`, `FileKnowledgeBase`, `ChronosKnowledgeBase` |

The dyn census in `soma-runtime` alone: `CacheStore` 33 sites, `Transport` 13,
`DataStore` 7, `EffectHandler` 6, `Step` 5, `EventSink` 9, `Filter` 5.

`(!)` One trait is defined for this and never used polymorphically: `Runner`
(`soma-runtime/src/execution/runner/mod.rs:124`). Its second implementation went with
[D-34](/soma/internals/debt/#d-34--remoterunner-is-never-constructed); the trait
stayed, now documenting one shape rather than abstracting two.

### Adapter / bridge

Making a foreign thing satisfy a local contract. This is how every boundary in
the system is crossed, and there are more instances than of any other pattern.

| Adapter | file:line | Adapts |
|---|---|---|
| `NodeMeta` | `soma-core/src/graph/node.rs:72` | `FilterMeta` **and** `StepMeta` into one shape — the workspace's central adapter |
| `PyFilterBridge` | `soma-python/src/graph/bridge.rs:224` | A Python object → `Filter` |
| `PyStepBridge` | `soma-python/src/agentic.rs:883` | Anything with `poll(ctx)` → `Step` |
| `PyToolAdapter` | `soma-python/src/agentic.rs:198` | A Python callable → `Tool` |
| `PyPbtExecutor` | `soma-python/src/optimizer/pbt.rs:40` | Python callables → `PbtExecutor` |
| `SubprocessFilter` | `soma-worker/src/python_process.rs:1025` | A pipe to another interpreter → `Filter` |
| `WsTransport` | `soma-worker/src/ws_transport.rs:404` | A WebSocket → `Transport` |
| `McpTool` | `soma-llm/src/tools.rs:212` | An MCP server → `Tool` |
| `FnTool<F>` | `soma-llm/src/tools.rs:62` | A closure → `Tool` |
| `FnTrialExecutor<F>` / `FnPbtExecutor<T,E>` | `soma-runtime/src/optimizer/study.rs:144`, `pbt.rs:63` | Closures → trait objects |

`NodeMeta` deserves the emphasis. Filters and steps used to live in two
registries joined by an adapter that a caller had to remember to build — which
is how `.compile()` came to skip every step's schema validation while `.run()`
checked them. Collapsing both into one metadata type with an `effectful` flag
means the executor's existing cacheability guard reads "a step is not
output-cacheable" as **data**. There is no `if is_step` anywhere in the executor.

### Composite

A recursive tree walked uniformly.

- `ExecutionPlan` (`soma-compiler/src/plan.rs:19`) — recursive in four shapes; `children()` (`:142`) is the single traversal, written per variant so a new variant breaks the build. `(!)` Two functions in the same crate do not use it and are wrong as a result — [D-32](/soma/internals/debt/#d-32--the-compiler-never-descends-into-loop-or-branch).
- `NodeKind::SubGraph` (`soma-core/src/graph/mod.rs:32`) — a graph inside a node.
- `SearchDimension::Conditional` (`soma-core/src/optimizer/search.rs:36`) — a dimension gated on another.
- `TieredCache` (`soma-runtime/src/cache/tiered.rs:11`) — a `CacheStore` made of `CacheStore`s.

### Facade

- `GraphSession` (`soma-runtime/src/execution/graph_session.rs:38`) over compile + execute + cache + events, plus the free functions `graph_run` / `graph_fit` / `graph_predict` (`:450`).
- `soma` (`soma/src/lib.rs`) — the crate facade. `(!)` [D-83](/soma/internals/debt/#d-83--the-facade-covers-10-of-13-crates).
- `SomaContext` (`soma-mcp/src/context.rs:9`) over memory + filesystem + subprocess.
- `RunView` (`soma-python/python/soma/_runs.py:30`) over the run-directory readers.

### Decorator and proxy

- `TieredCache` — decorates by promoting on read. `(!)` The promotion loses provenance ([D-46](/soma/internals/debt/#d-46--tieredcache-promotion-destroys-provenance)).
- `FileKnowledgeBase` (`soma-memory/src/file_kb.rs:25`) — decorates `MemoryKnowledgeBase` with a durable JSONL log and byte-offset incremental refresh.
- `SubprocessFilter`, `WsTransport` — remote proxies (see above).

### Registry

- `NodeCatalog` (`soma-runtime/src/execution/node_catalog.rs:79`) — **the** registry, holding both node kinds and doubling as the compiler's `NodeRegistry`.
- `Router` (`soma-llm/src/lib.rs:93`), `Toolbox` (`soma-llm/src/tools.rs:92`), `WorkerRegistry` (`soma-coordinator/src/registry.rs:73`).
- `FilterMeta` metaclass (`soma-python/python/soma/filter.py:9`) — collects `SearchDescriptor`s into `_soma_search_space` at class-definition time.

---

## Behavioural patterns

### Chain of responsibility

`EffectHandler` (`soma-core/src/agentic/effect.rs:262`) is two methods — `handles` and
`perform` — and the contract is in the doc at `:256`: "Handlers are tried in
order; the first that claims an effect wins."

Dispatch happens in `EffectDriver::perform_one`
(`soma-runtime/src/agentic/mod.rs:531`). Handlers: `LlmHandler`, `Toolbox`,
`GraphHandler`, `SleepHandler`.

This is what makes `Effect::Custom { kind, payload }` work as an extension point
with no registration step — a new handler that claims a `kind` is the whole
feature.

### Template method

A trait with a few required methods and many defaulted ones, so a minimal
implementation is cheap and a sophisticated one can specialize.

| Trait | Required | Provided | Payoff |
|---|---|---|---|
| `KnowledgeBase` (`soma-memory/src/knowledge_base.rs:50`) | 3 | **11** | A backend implements storage and inherits every analytic |
| `CacheStore` (`soma-core/src/cache/mod.rs:207`) | 5 | 4 | A simple store ignores origin and timing; a rich one records them |
| `DataStore` (`soma-core/src/data/store.rs:208`) | 5 | 2 | The defaults download everything and slice locally; `ZarrStore` overrides both to serve a row range remotely |
| `Sampler` (`soma-runtime/src/optimizer/sampler/mod.rs:22`) | 2 | 2 | `prepare` and `record_result` are no-ops unless the sampler learns |
| `StrategyContext` (`soma-runtime/src/distributed.rs:33`) | 6 | 3 | Two of the provided methods default to *refusing* — an honest "not supported" |
| `LocalRunner::walk` (`soma-runtime/src/execution/runner/local.rs:26`) | — | — | Not a trait: one method shared by `fit` and `forward`, differing only by `RunMode`. Documented at `:22` as the fix for two divergent loops |

### Extension trait

Defining a trait in one crate and implementing it on a *foreign* type from
another. This is how Soma keeps "core holds contracts, runtime holds execution"
without either crate depending on the other in the wrong direction.

| Trait | Defined in | Implemented on | From |
|---|---|---|---|
| `StudyIo` | `soma-runtime/src/optimizer/study_io.rs:19` | `Study` | `soma-core` |
| `StrategyExecutor` | `soma-runtime/src/distributed.rs:120` | `TrainingStrategy` | `soma-core` |
| `GradientAggregator` | `soma-runtime/src/distributed.rs:132` | `GradientAggregation` | `soma-core` |
| `StateAggregator` | `soma-runtime/src/distributed.rs:139` | `FederatedAggregation` | `soma-core` |

`StudyIo` is the clearest example of what it buys: `Study` gains `save` and
`load` without `soma-core` gaining a filesystem.

### State machine / trampoline

`Step::poll(&ctx) -> Transition` (`soma-core/src/graph/step.rs:250`). A step never
blocks and never awaits — it returns a description of what it wants, and a driver
performs the effects and calls it again with the results.

The consequence is the whole agentic design: a step holds **no hidden state
between turns**. Everything it knows arrives through `StepCtx::history`
(`soma-core/src/graph/step.rs:128`), which is what makes journal replay exact rather
than approximate, and what makes `async fn` in a trait unnecessary.

### Observer

`EventBus` (`soma-runtime/src/tracking/event_bus.rs:22`) with two deliberately different
paths:

- **lossy** — a tokio broadcast channel for live subscribers who may miss events
- **lossless** — a synchronous `Vec<Arc<dyn EventSink>>` for anything that must persist every one

`(!)` The sinks run on the emitting thread, so a JSONL write sits inside
`run_node` — [D-72](/soma/internals/debt/#d-72--every-event-emission-clones-the-sink-vector).

### Command / interpreter

`Effect` (`soma-core/src/agentic/effect.rs:35`) describes work as data; the runtime
performs it. `Effect::label`, `is_pure` and `cache_key` (`:80`–`:127`) are what
let the journal treat "what was asked" as a value.

### Memento

- `Context::snapshot` (`soma-runtime/src/execution/executor.rs:344`) — each parallel branch gets a copy, and only the write set (the entries appended past a mark) is merged back.
- `EffectJournal` — every effect result recorded so a resumed run replays rather than re-runs.

### Callback adapter

`FnTrialExecutor<F>` and `FnPbtExecutor<T, E>` — the only generic public structs
in `soma-runtime`. In both cases the trait exists to be `dyn`-able, not to be
implemented by users; the only implementor wraps a closure.

---

## Persistence and identity patterns

### Content-addressed memoization

The caching model in one line, from `soma-core/src/cache/mod.rs:21`:

```
state  = hash(config ‖ x ‖ y)
output = hash(config ‖ state ‖ input_content_hash)   + seed salt
```

Because a downstream key uses the *content* hash of its input rather than the
identity of its producer, an unchanged intermediate cuts off the rest of the
graph early — the whole point of the design.

Filter identity is derived, not written: Rust from canonical CBOR of the field
list plus `#[soma(cache_version)]` (`soma-macros/src/lib.rs:30`), Python from
qualname + canonical config + a source-hash ladder
(`soma-python/python/soma/_identity.py:124`). An unhashable config raises
`CacheConfigError` — never a silent key.

### Event sourcing and journaling

Three independent instances, which is how you know it is the codebase's actual
philosophy rather than one clever file:

- **`EffectJournal`** (`soma-runtime/src/agentic/journal.rs:51`) — pure effects keyed by content, impure ones by `(run, node, turn, index)`. Record once, replay forever. Suspension is modelled as an effect, so resume needs no separate checkpoint format.
- **`ResearchStep::completed`** (`soma-agent/src/research.rs:87`) — reconstructs its record list from `ctx.history` rather than holding it in a field.
- **The experiment pool** (`soma-memory/src/record.rs:26`) — append-only, with `RecordKind::Amendment` and an `amends` field. Nothing is ever rewritten.

### Two-phase / atomic commit

Temp file → `write_all` → `sync_all` → `rename`, once, at
`soma-runtime/src/fsutil.rs:31`; the four callers are the cache's blob and
action stores, the run manifest and HEAD. `FsActionStore` also commits
blob-first, record-last (`soma-runtime/src/cache/fs_store.rs:195`), so a crash
can leave an unreferenced blob but never a record pointing at nothing.

It was four implementations, two of them missing the unique temp name and the
fsync — [D-12](/soma/internals/debt/#d-12--four-write_atomic-implementations-two-of-them-unsafe),
now resolved.

### Content-addressed pooling

`EnvManager::env_id_for` (`soma-worker/src/env_manager.rs:362`) keys a Python
environment by the hash of its requirements rather than by plan id. Same idea as
the cache, applied to venvs.

### Explicit wire versioning

`PROTOCOL_VERSION` + `check_version` (`soma-worker/src/protocol.rs:33`, `:311`),
`RECORD_SCHEMA_VERSION` (`soma-memory/src/record.rs:20`),
`RUN_SCHEMA_VERSION` (`soma-core/src/tracking/mod.rs`), `FORMAT_VERSION`
(`soma-runtime/src/cache/fs_store.rs`). Every persisted or transmitted format
carries a version and refuses a mismatch rather than guessing.

---

## Rust idioms doing pattern work

These have no GoF name, and they are where most of the design lives.

### `#[non_exhaustive]` as a deliberate policy

Not applied uniformly, and the non-uniformity **is** the design:

| Applied to | Reason |
|---|---|
| Data enums — `Value`, `Effect`, `Event`, `SomaError`, `NodeKind`, `ExecutionPlan`, `DataRef` | A consumer need not have an opinion about a new variant, and an old worker must tolerate a new one on the wire |
| **Not** applied to — `NodeOutcome` (`soma-core/src/graph/node.rs:37`), `Transition` (`soma-core/src/graph/step.rs:38`), `StreamMode` (`soma-core/src/graph/filter.rs:32`) | Every consumer must decide over them. A wildcard arm there is a silent wrong answer, and adding a variant *should* break every match |

The reason is written into each doc comment either way. This is the single most
transferable convention in the codebase.

### Newtype

`CacheKey([u8; 32])` (`soma-core/src/cache/mod.rs:21`), `Messages(Vec<Message>)`
(`soma-core/src/agentic/message.rs:189`), `ContentHash` (`soma-core/src/cache/action.rs:52`),
`ShutdownSignal` (`soma-worker/src/server.rs:33`).

`(!)` The idiom is *not* applied to `NodeId`, `EdgeId`, `RunId`, `StudyId` or
`TrialId`, which are all `String` aliases and therefore mutually assignable —
[D-56](/soma/internals/debt/#d-56--nodeid-is-a-string-and-so-is-everything-else).

### Consuming builder

`fn with_x(mut self, x: X) -> Self`. The crate's dominant construction idiom:
`Context` has 8 (`soma-runtime/src/execution/executor.rs:195`), `GraphSession` 7
(`graph_session.rs:82`), `LlmRequest` 5 (`soma-core/src/agentic/effect.rs:201`),
`ExperimentRecord` 12 (`soma-memory/src/record.rs`), plus `StepMeta`, `Node`,
`Edge`, `Graph`, `StepCtx`, `Study`, `ArchitectureFingerprint`, `NodeSpec`,
`EffectDriver`, `GraphHandler`, `Worker`, `ProviderConfig`, `RetryPolicy`,
`SerializedPlan`, `RetrievalQuery`, `WorkerRegistry`.

`(!)` Three wide structs skipped it: `RunManifest` (20 fields, 3-argument
constructor), `RunSummary` (17), `Study` (15 fields, 2 builders) —
[D-06](/soma/internals/debt/#d-06--wide-data-structs-with-no-builder).

### Flyweight via `Arc`

Every `Value` payload is `Arc`-wrapped (`soma-core/src/data/value.rs:15`), so `Clone`
is a refcount bump rather than a tensor copy. `Arc<dyn Filter>` and
`Arc<dyn Step>` in the catalog; `Arc<Value>` for trained state; and a
`NodeCatalog` clone deliberately **shares** its `StateStore`
(`soma-runtime/src/execution/node_catalog.rs:75`).

### Blanket impl as a marker

`AsAny` (`soma-core/src/graph/any.rs:13`) with `impl<T: Any> AsAny for T` — a supertrait
of `Filter` and `Step` that costs implementors nothing and buys three downcast
sites.

### Scoped threads instead of async

Three fan-out sites, all `std::thread::scope`, no runtime:
`execute_parallel` (`soma-runtime/src/execution/executor.rs:1103`), `perform_all`
(`soma-runtime/src/agentic/mod.rs:459`), `spawn_all` (`:355`). The rationale is
at `soma-runtime/src/agentic/mod.rs:12`.

`rg async_trait` over the workspace returns **zero hits**. The only async code is
the axum servers, and both isolate the boundary properly — `spawn_blocking` in
`soma-worker/src/server.rs`, and `on_own_runtime`
(`soma-worker/src/ws_transport.rs:42`) which refuses to assume whether it is
inside a runtime.

`(!)` The cost is real and acknowledged: `JoinPolicy::First`
(`soma-runtime/src/agentic/mod.rs:417`) returns the first success only after every
sibling has joined, because "these are threads, not cancellable tasks" (`:415`).

### Forward-compatible refusal

An unknown variant returns an error naming the situation rather than falling
through to a default:

```rust
other => Err(SomaError::Execution { … })   // executor.rs:445
                                           // strategy.rs:274
                                           // graph_handler.rs:184
                                           // reader.rs:844
```

Applied consistently, and one of the codebase's genuine strengths.

### Error-as-data

Where the consumer is a *model* rather than a program, a failure is a message
rather than a `Result::Err`: `ToolOutcome` (`soma-llm/src/tools.rs:182`),
`EffectResult::Failed` (`soma-core/src/agentic/effect.rs:278`), and every `soma-mcp`
handler returning `ToolCallResult::error(…)` instead of a JSON-RPC error
(`soma-mcp/src/context.rs:100`). An agent that can read the failure can retry;
one that gets a transport error cannot.

### Out-of-process isolation

Two anti-corruption layers, same shape: `soma-worker`'s `DAEMON_SCRIPT`
(`soma-worker/src/python_process.rs:19`) and `soma-mcp`'s `DRIVER`
(`soma-mcp/src/exec.rs:26`). Both swap `sys.stdout` away from the protocol
channel so a user's `print` cannot corrupt it. `(!)` Both live in Rust string
constants — [D-19](/soma/internals/debt/#d-19--two-embedded-python-interpreters-as-rust-string-constants).

### Recursive composition with a depth cap

`GraphHandler` → `EffectDriver` → `GraphHandler` is a real cycle — a graph can be
a tool for an agent that is itself a node in a graph. It terminates because
`MAX_GRAPH_DEPTH = 8` (`soma-runtime/src/agentic/graph_handler.rs:28`) and
`child_driver` (`:112`) refuses past it.

---

## Python-side patterns

### Mixin assembly in a class body

`soma-python/python/soma/_graph.py:35` — 23 methods assigned as class attributes
rather than monkey-patched at import time. The docstring at `:9` is an explicit
argument for why: the previous approach was invisible to `help()`, IDEs and mypy;
the surface differed per program depending on which modules were imported; and
three methods silently shadowed Rust methods of the same name.

### Duck typing as the extension mechanism

A step is any object with `poll(ctx)`. A transition is a plain `dict`. The
rationale at `soma-python/python/soma/agentic.py:103`: "what crosses into Rust is
data rather than a class hierarchy."

`(!)` The cost is a stringly-typed seam kept in sync across Rust, the `.pyi` stub
and the Python constructors by literal —
[D-54](/soma/internals/debt/#d-54--nine-string-match-dispatch-sites-across-the-ffi).

### Descriptor + metaclass registry

`SearchDescriptor.__set_name__` / `__get__` / `__set__`
(`soma-python/python/soma/search.py:55`) plus the `FilterMeta` metaclass
(`filter.py:9`) turn `search(...)` at class level into a search space, so
`soma.Agent(model=search(...))` and a filter's hyperparameters fold into the same
`search_space()`.

### Optional-dependency null object

torch missing → `DifferentiableFilter = None` and 8 audit names `None`
(`soma-python/python/soma/__init__.py:31`, `:59`). plotly and pandas lazily
imported inside `_go()` (`viz/_figures.py:18`) and `_pandas()`
(`viz/_frames.py:10`), so the methods always *exist* and only calling them needs
the `somatize[viz]` extra.

### Rich-repr protocol

Five `_repr_html_` implementations — `PyGraph` (`soma-python/src/graph/viz.rs:44`),
`RunView`, `RunList`, `CompileInfo`, `DifferentiableFilter`. Evaluating an object
in a notebook draws it. `Graph::to_svg` (`soma-core/src/viz/svg.rs`) exists
specifically because notebooks sanitize `<script>`, so a mermaid block would not
render.

---

## Patterns deliberately absent

Worth naming, because their absence is a decision rather than an oversight.

| Absent | Why |
|---|---|
| **Typestate** | No `PhantomData` anywhere. State that matters is checked at compile time (`compile()`) or refused at runtime, not encoded in type parameters |
| **Generic traits** | No public trait has a type parameter or associated type — that is what keeps all 29 of them object-safe |
| **`async_trait`** | Zero uses. Every trait is synchronous; concurrency is threads |
| **Inheritance simulation** | No trait hierarchies beyond `AsAny` as a supertrait. Composition and `dyn` everywhere |
| **A DI container** | Dependencies are constructor arguments and `with_*` builders |
| **Singletons** | No `lazy_static` or `OnceCell` globals in the domain crates |
| **Error-type-per-crate** | Three error enums workspace-wide, deliberately. The decision, and what it was chosen over, is at [Architecture Decisions](/soma/design/decisions/) |

---

## Where the patterns break down

Every entry here is in the [Debt Register](/soma/internals/debt/); this is the
short version, organized by which pattern failed.

| Pattern | Where it broke |
|---|---|
| Composite | `resolve_distribution` / `collapse_differentiable` do not use `children()` and skip `Loop`/`Branch` — [D-32](/soma/internals/debt/#d-32--the-compiler-never-descends-into-loop-or-branch) |
| Template method | `Transport::execute_node`'s default is wrong for every caller, and its own doc says so — [D-41](/soma/internals/debt/#d-41--transportexecute_node-runs-remotes-with-an-empty-catalog) |
| Strategy | `Runner` — a strategy interface that had one live implementation and one dead one; the dead one is deleted — [D-34](/soma/internals/debt/#d-34--remoterunner-is-never-constructed) |
| Shared primitives | `run_node` and `StreamRun::run_compute` share three primitives and duplicate everything around them, and have drifted — [D-11](/soma/internals/debt/#d-11--the-stream-path-re-implements-run_node-and-has-drifted) |
| Newtype | Not applied to any id type — [D-56](/soma/internals/debt/#d-56--nodeid-is-a-string-and-so-is-everything-else) |
| Builder | Three wide structs skipped it — [D-06](/soma/internals/debt/#d-06--wide-data-structs-with-no-builder) |
| Atomic commit | Was four implementations, two without fsync; now one — [D-12](/soma/internals/debt/#d-12--four-write_atomic-implementations-two-of-them-unsafe) |
| Decorator | `TieredCache` promotion discards the provenance it decorates — [D-46](/soma/internals/debt/#d-46--tieredcache-promotion-destroys-provenance) |
| Facade | `soma` covers 10 of 13 crates — [D-83](/soma/internals/debt/#d-83--the-facade-covers-10-of-13-crates) |
