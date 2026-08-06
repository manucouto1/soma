---
title: Codebase Map
description: How to read the Soma source — the notation, the crate graph, the ownership spine, and the ten types that carry the system.
---

Three documentation sections describe this codebase and they answer different
questions:

- [Architecture](/soma/architecture/overview/) — the **shape**. Layers, flow, responsibilities.
- [Design](/soma/design/decisions/) — the **why**. What was chosen, over what, and what would change the answer.
- **Internals** (this section) — the **what**, with `file:line`. Every public trait, struct and enum; who implements what; who owns what; what is wrong.

Use this section when you need to find something, when you need to know what
implements a trait, or when you are planning a change and want to know what it
will touch.

---

## Reading order

| If you want to… | Read |
|---|---|
| Get oriented in one page | This page — the [spine](#d0--the-ownership-spine) and the [ten types](#the-ten-types-that-carry-the-system) |
| Explore instead of read | The [Architecture Graph](/soma/internals/graph/) — click a trait to see every implementor, a type to see what owns it |
| Follow the code as it runs | [Call Paths](/soma/internals/paths/) — the five traces as one graph, with the hops they share |
| Know the vocabulary | [Foundation](/soma/internals/foundation/) — `soma-core` is the dictionary every other crate speaks |
| Understand how a graph runs | [Execution](/soma/internals/execution/) — especially the [traces](/soma/internals/execution/#execution-traces) |
| Understand how an agent works | [Agentic Stack](/soma/internals/agentic/) — start with [D4](/soma/internals/agentic/#d4--the-effect-loop) |
| Work on the Python API | [Python Bridge](/soma/internals/python/) |
| Work on remote execution | [Distribution](/soma/internals/distribution/) — start with [D5](/soma/internals/distribution/#d5--what-crosses-the-wire) |
| Recognize an idiom you keep seeing | [Design Patterns](/soma/internals/patterns/) |
| Plan a refactor | [Known Debt](/soma/internals/debt/) |
| Know how big the user-facing API is, and what nothing calls | [Surface Census](/soma/internals/surface/) |
| Find one symbol | [Symbol Index](/soma/internals/symbols/) |

---

## Notation

Rust has no classes, so a UML class diagram does not translate directly. These
pages use a fixed ASCII notation instead — greppable, diffable, and readable in a
terminal. It is used identically on every page.

```
«trait» Name        an interface (≈ UML interface)
   ▲
   ├── Type         realization      (impl Name for Type)

A ──◆ f: T          composition      owned by value or Box — dies with A
A ──◇ f: Arc<T>     aggregation      shared — may outlive A
A ──▷ B             uses / calls     no ownership
A ──? f: Option<T>  optional

[enum] E {A|B|C}    an enum, variants inline
 (!)                a documented deviation — see the Debt Register
  !                 #[non_exhaustive]
```

**There is deliberately no single diagram of all ~250 types.** A diagram of 250
nodes is decoration. Six targeted diagrams show the seams instead, and the tables
are the diagram at finer granularity:

| | Diagram | Page |
|---|---|---|
| **D0** | The ownership spine | [below](#d0--the-ownership-spine) |
| **D1** | The node seam — `Filter` / `Step` / `NodeCatalog` | [Execution](/soma/internals/execution/#d1--the-node-seam) |
| **D2** | The execution pipeline | [Execution](/soma/internals/execution/#d2--the-execution-pipeline) |
| **D3** | The cache and journal stack | [Execution](/soma/internals/execution/#d3--the-cache-and-journal-stack) |
| **D4** | The effect loop | [Agentic Stack](/soma/internals/agentic/#d4--the-effect-loop) |
| **D5** | What crosses the wire | [Distribution](/soma/internals/distribution/#d5--what-crosses-the-wire) |
| **D6** | The FFI bridge | [Python Bridge](/soma/internals/python/#d6--the-ffi-bridge) |

A `file:line` reference is written as plain inline code — `` `soma-core/src/graph/filter.rs:120` ``
— never as a link. A GitHub permalink would need a pinned commit, and two hundred
of them would rot in one commit.

---

## The domain vocabulary

Nine names, used the same way everywhere. They are directory names first and
a documentation device second: if you can guess what `optimizer/` holds, you
did not need this page.

| Domain | What lives there |
|---|---|
| `graph` | what the user builds — nodes, edges, `Filter`, `Step` |
| `data` | the values and their stores |
| `cache` | memoizing by content |
| `execution` | compiling a graph and running a plan |
| `agentic` | effects, messages, tools, models |
| `optimizer` | hyperparameter search |
| `tracking` | what a run records |
| `distributed` | the same work, on another machine |
| `viz` | drawing it |

The vocabulary applies at **two levels**, and the second is the one that
makes it useful. Four crates span several domains and carry the names as
folders. The rest each *are* one domain — there, wrapping the whole crate in
a folder of the same name would add a level that says nothing, so the crate
is the unit and its subdivisions are the domain's own.

```
                  graph  data  cache  exec  agentic  optim  track  distrib  viz
soma-core           ●     ●      ●      ·      ●       ●      ●       ●      ●
soma-runtime        ·     ·      ●      ●      ●       ●      ●       ●      ·
soma-python         ●     ●      ●      ·      ●       ●      ●       ●      ·
─────────── crates that ARE one domain ──────────────────────────────────────
soma-compiler                            ●
soma-llm                                          ●
soma-agent                                        ●
soma-memory                                       ●            ●
soma-mcp                                          ●
soma-worker                                                             ●
soma-coordinator                                                        ●
soma-store                ●
```

Reading down a column is how a capability is read across its layers. For
`optimizer`: `soma-core` says what a search space and a study *are*,
`soma-runtime` holds what walks one — samplers, pruners, the trial loop —
and `soma-python` is what a user types. Three folders, one name, one
capability.

Each domain folder's `mod.rs` opens by saying what the domain is, so that
answer lives beside the code rather than only here.

---

## The workspace at a glance

~70 000 lines of Rust across 13 crates, plus a 7 400-line pure-Python package.
Published names are prefixed `somatize-`; directory names drop the prefix.

| Crate | Lines | Traits | Page |
|---|---|---|---|
| `soma-core` | 11 463 | 12 | [Foundation](/soma/internals/foundation/#soma-core-somatize-core) |
| `soma-macros` | 607 | 0 | [Foundation](/soma/internals/foundation/#soma-macros-somatize-macros) |
| `soma-compiler` | 3 120 | 1 | [Execution](/soma/internals/execution/#soma-compiler-somatize-compiler) |
| `soma-runtime` | 17 290 | 12 | [Execution](/soma/internals/execution/#soma-runtime-somatize-runtime) |
| `soma-llm` | 3 848 | 2 | [Agentic](/soma/internals/agentic/#soma-llm-somatize-llm) |
| `soma-agent` | 620 | 0 | [Agentic](/soma/internals/agentic/#soma-agent-somatize-agent) |
| `soma-memory` | 3 746 | 2 | [Agentic](/soma/internals/agentic/#soma-memory-somatize-memory) |
| `soma-mcp` | 3 270 | 0 | [Agentic](/soma/internals/agentic/#soma-mcp-somatize-mcp) |
| `soma-worker` | 5 922 | 0 | [Distribution](/soma/internals/distribution/#soma-worker-somatize-worker) |
| `soma-coordinator` | 949 | 0 | [Distribution](/soma/internals/distribution/#soma-coordinator-somatize-coordinator) |
| `soma-store` | 1 285 | 0 | [Distribution](/soma/internals/distribution/#soma-store-somatize-store) |
| `soma-python` | 6 605 | 0 | [Python Bridge](/soma/internals/python/) |
| `soma` (facade) | 124 | 0 | [Foundation](/soma/internals/foundation/#soma-somatize--the-facade) |

**29 public traits total.** Not one of them declares an associated type or a
generic parameter, so all but `StudyIo` and `Searchable` are object-safe — which
is why every backend in the system is swappable at runtime without a generic
bound leaking into a signature.

One number to keep in mind before judging any file by its length: **60% of
`soma-runtime` is tests.** `executor.rs` is 2 472 lines of which 1 184 are inline
`#[cfg(test)]`; `executors/study.rs` is 1 915 of which 1 470 are.

### The dependency graph, with the trait seams marked

Acyclic, read top to bottom. The arrows on the right are the traits crossing each
boundary — those are the joints the system bends at.

```
soma-macros                       proc macros; no internal dependencies
    │                             ─── generates ──▷ config_hash, impl Searchable
    ▼
soma-core                         types, traits, serialization
    │   « defines: Filter, Step, CacheStore, DataStore, StateStore,
    │     EffectHandler, ActionCache, BlobStore, EventSink, Tracker,
    │     Searchable, AsAny »
    ├──▷ soma-store               ──▷ impl DataStore  (S3, Zarr)
    │
    ├──▷ soma-compiler            « defines: NodeRegistry »
    │        │
    │        ▼
    │    soma-runtime             « defines: Runner, Transport, ForwardStrategy,
    │        │                      Sampler, Pruner, TrialExecutor, PbtExecutor,
    │        │                      StrategyContext, StrategyExecutor,
    │        │                      GradientAggregator, StateAggregator, StudyIo »
    │        │                     ──▷ impl NodeRegistry for NodeCatalog
    │        │                     ──▷ impl CacheStore ×4, EventSink, Tracker
    │        │
    │        ├──▷ soma-llm        « defines: LlmProvider, Tool »
    │        │                     ──▷ impl Step ×3, impl EffectHandler ×2
    │        │
    │        ├──▷ soma-worker     ──▷ impl Transport, impl Filter
    │        │        │
    │        │        ▼
    │        │    soma-coordinator   (reuses soma-worker's wire vocabulary)
    │        │
    │        ├──▷ soma-agent      ──▷ impl Step         ┐ both also depend
    │        └──▷ soma-memory     « defines: KnowledgeBase, Embedder »
    │                 │                                 ┘ on soma-memory
    │                 ▼
    │             soma-mcp        ──▷ Box<dyn KnowledgeBase>
    │
    └──▷ soma-python              ──▷ impl Filter, Step, Tool, PbtExecutor
             │                        « the only crate implementing four
             ▼                          foreign traits by calling into Python »
       python/soma/*.py
```

`soma` (the facade) sits outside this and re-exports ten of the thirteen.

---

## D0 · The ownership spine

One screen. If you remember nothing else, remember this shape.

```
   User writes a Graph
        │
        ▼
   GraphSession                                soma-runtime/…/graph_session.rs:38
    ├──◆ Graph                                 « nodes + edges, no behaviour »
    │
    ├──◆ NodeCatalog                           « THE registry »
    │     ├──◆ HashMap<NodeId, NodeImpl>
    │     │      ├──◇ Arc<dyn Filter>          fit / forward
    │     │      └──◇ Arc<dyn Step>            poll -> Transition
    │     └──◇ Arc<dyn StateStore>             « shared across catalog clones »
    │
    ├──◇ Arc<dyn CacheStore>                   memory → local → action store
    │
    ├──◇ Arc<EventBus>
    │     ├──◆ broadcast::Sender<Event>        lossy: live subscribers
    │     └──◆ RwLock<Vec<Arc<dyn EventSink>>> lossless: JSONL to the run dir
    │
    ├──? Option<Arc<dyn DataStore>>            local / S3 / Zarr
    ├──? Option<Arc<dyn Transport>>            ┐ (!) two transport fields
    ├──◆ Vec<Arc<dyn Transport>>               ┘     D-04
    │
    └──? Option<EffectDriver>                  « present only if steps exist »
          ├──◆ Vec<Arc<dyn EffectHandler>>     llm · tools · sub-graph · sleep
          ├──◆ EffectJournal                   record once, replay forever
          │     ├──◇ Arc<dyn ActionCache>      kept forever
          │     └──◇ Arc<dyn BlobStore>        BLAKE3 CAS, evictable
          └──? Option<Arc<NodeCatalog>>        needed only for Transition::Spawn

   compile()  ──▷ ExecutionPlan  ──▷ LocalRunner ──▷ Context ──▷ run_node
                                                                    │
                                     output_key · compute_node · store_output
```

---

## The ten types that carry the system

If you learn these, most of the rest follows.

| # | Type | file:line | Why it matters |
|---|---|---|---|
| 1 | `Filter` | `soma-core/src/graph/filter.rs:120` | `fit()` learns state, `forward()` transforms. Both independently cacheable. Everything pipeline-shaped is this |
| 2 | `Step` | `soma-core/src/graph/step.rs:250` | `poll(ctx) -> Transition`. Everything agent-shaped is this. Holds no state between turns — history arrives through `StepCtx` |
| 3 | `NodeMeta` | `soma-core/src/graph/node.rs:72` | The adapter that erases the Filter/Step distinction. `From<StepMeta>` sets `cacheable: false`, so "a step is not cacheable" is *data*, not a branch |
| 4 | `NodeCatalog` | `soma-runtime/src/execution/node_catalog.rs:79` | One registry for both kinds, and the compiler's `NodeRegistry`. Two registries joined by an adapter is what made `.compile()` skip step schemas |
| 5 | `Value` | `soma-core/src/data/value.rs:15` | Six variants, all `Arc`-backed, so `Clone` is a refcount bump |
| 6 | `CacheKey` | `soma-core/src/cache/mod.rs:21` | `state = hash(config‖x‖y)`, `output = hash(config‖state‖input_hash)`. Downstream keys use input **content**, so an unchanged intermediate cuts off the rest of the graph |
| 7 | `ExecutionPlan` | `soma-compiler/src/plan.rs:19` | What the compiler produces and the executor walks. Recursive in four shapes; `children()` is the one traversal |
| 8 | `Context` | `soma-runtime/src/execution/executor.rs:124` | The executor's mutable state through the whole walk. `(!)` Also the biggest god object in the runtime |
| 9 | `Transition` | `soma-core/src/graph/step.rs:43` | `Await` / `Spawn` / `Goto` / `Suspend` / `Done`. Deliberately **not** `#[non_exhaustive]` — every consumer must decide |
| 10 | `Effect` / `EffectJournal` | `soma-core/src/agentic/effect.rs:35`, `soma-runtime/src/agentic/journal.rs:51` | An effect is data; the journal keys pure ones by content and impure ones by site. That is the whole durability story |

### The one distinction to internalize

**A filter memoizes by content. A step journals by site.**

A filter's output is a function of its config, its state and its input, so an
identical call anywhere can reuse the result. A step's effects are not: asking a
model the same question twice can give two answers, so an impure effect is keyed
by *where and when* it happened — `(run, node, turn, index)` — recorded once and
replayed on resume, never re-run.

Everything else about caching, resumption and reproducibility follows from that
one sentence.

---

## How a run actually happens

The narrative version of D2, for orientation. Every step links to the detail.

1. **You build a `Graph`** — nodes and edges, no behaviour. Nodes come in five structural kinds (`Filter`, `SubGraph`, `Loop`, `Branch`, `Step`); every *behaviour* is library code.
2. **You register implementations** in a `NodeCatalog`, which holds filters and steps side by side.
3. **`compile()`** ([Execution](/soma/internals/execution/#soma-compiler-somatize-compiler)) walks the graph, validates schemas between connected nodes, claims loop bodies and branch arms by dominance, wraps remote nodes, and returns an `ExecutionPlan` plus diagnostics.
4. **`LocalRunner::walk`** builds a `Context` from the plan and the topology — note *topology*, not plan order: input resolution follows predecessors, which is what makes a diamond work.
5. **`execute`** recurses over the plan. Each leaf reaches **`run_node`**, which is the one execution site for both kinds.
6. **`run_node`** resolves the input, fits state if needed, derives an `output_key`, checks the cache, and on a miss calls `compute_node` — the only place a filter's `forward` and a step's driver are told apart.
7. **A step's `poll`** returns a `Transition`. If it is `Await`, the **`EffectDriver`** performs the effects on threads, consults the **`EffectJournal`** first, and calls `poll` again with the results. ([D4](/soma/internals/agentic/#d4--the-effect-loop))
8. **Results are stored** with provenance (`Origin::Computed { node_id, run_id }`), events are emitted to the `EventBus`, and a `LocalTracker` writes them to a run directory as JSONL.
9. **Afterwards**, `RunReader` and `summarize` turn that directory into a `RunSummary`, an `ExperimentRecord` lands in the pool, and `.soma/HEAD` advances — but only on success, and never inferred from a timestamp.

Remote execution replaces step 5 with a serialized plan over a WebSocket
([D5](/soma/internals/distribution/#d5--what-crosses-the-wire)); the executor
itself does not change. Streaming replaces it with `StreamRun`, which composes
the *same four primitives* per chunk — which is why a single-chunk stream and a
plain forward produce identical cache keys.

---

## Conventions this codebase keeps

Worth knowing before you write anything in it, because they are enforced by
review rather than by the compiler.

- **`#[non_exhaustive]` is a decision, not a default.** Data enums get it; control-flow enums every consumer must decide over (`NodeOutcome`, `Transition`, `StreamMode`) deliberately do not, so adding a variant breaks every match. The reason is in each doc comment. → [Patterns](/soma/internals/patterns/#non_exhaustive-as-a-deliberate-policy)
- **Unknown variants refuse, they do not guess.** `other => Err(…)` naming the situation, at four sites.
- **Errors are typed at the edges, shared at the seams.** Three error enums workspace-wide. → [Decisions](/soma/design/decisions/)
- **Nothing is async.** Zero `async_trait`. Concurrency is `std::thread::scope`.
- **Every crate opts into `#![warn(missing_docs)]`.**
- **Commits** are Conventional Commits with a crate scope: `feat(core): add Schema type`.
- **`cargo clippy --workspace -- -D warnings` must pass.** Ten `#[allow]` exist in ~70 000 lines, nine of them structural.

---

## A caution about this section

These pages are hand-written and carry ~700 `file:line` anchors. Line numbers
drift on the first edit above them.

The `docs/scripts/check-anchors.mjs` guard, wired into `npm run check`, verifies
that every referenced **file** exists and that every named symbol still appears
in it, and warns (without failing) when a line number has drifted more than 30
lines. That catches deletion and renaming — the failures that make a reference
actively misleading — but it cannot tell you whether a description is still
*true*.

Links *within* the docs are checked separately, by
`docs/scripts/check-debt-refs.mjs`: every `D-nn` reference must name an entry
that exists, its slug must match that entry's heading, and where the link text
says `D-mm`, mm must agree with the target. That guard was written after the
register was renumbered in blocks of ten and five links kept the old ids, so
`[D-51]` pointed at the text of D-61. Both halves of such a link read plausible
in isolation, which is why nothing caught them for months.

When you find a claim here that is wrong, fix it. A reference nobody trusts is
worse than no reference, which is exactly what happened to
[Architecture Review](/soma/development/architecture-review/), now kept as a
historical document.
