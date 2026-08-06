---
title: Foundation — core, macros, facade
description: The vocabulary every other crate speaks — every public trait, struct and enum in soma-core, plus the derive macros and the facade.
---

`soma-core` is the workspace's dictionary. Every type below is referenced from
at least one other crate, and nothing here executes anything. If you are
rebuilding a mental model of Soma, start with the eleven **contracts** in the
first section — those are the joints the whole system bends at.

The [notation](/soma/internals/map/) legend applies throughout. `(!)` marks a
documented deviation, with the entry in the [Debt Register](/soma/internals/debt/).

---

## soma-core (`somatize-core`)

### Mandate

Types, traits and serialization. The rule is **no runtime, no network, no
optional heavy dependency** — verifiable with
`cargo tree -p somatize-core | grep tokio`, which returns nothing.

The rule is deliberately *not* "no I/O": `LocalDataStore` and its `std::fs`
usage stay, because a filesystem costs a caller nothing. What was split out is
`soma-store`, because S3 and Zarr each own a `tokio::runtime::Runtime` — see
[Distribution](/soma/internals/distribution/).

`11 463 lines across 30 files in 9 domains · 11 traits · 45 structs · 35 enums · deps: somatize-macros`

### Modules

Nine domain folders and three files at the root. The root is not a leftover
pile: `lib.rs` is the map, and `error.rs` and `util.rs` are the two things
every domain needs and none of them owns.

Each folder's `mod.rs` opens with what that domain *is*, so the answer to
"what is `optimizer/`?" is the first thing in `optimizer/mod.rs` rather than
something to reconstruct from its contents.

The nine names are not local to this crate. They are the workspace's domain
vocabulary, and the crates that span several use the same ones — so a
capability can be read across its layers: `optimizer/` here says what a search
space *is*, in `soma-runtime` what walks one, in `soma-python` what a user
types. Crates that span only one domain do not repeat the name in a folder;
for those, the crate *is* the domain. See the [map](/soma/internals/map/).

| File | Lines | Owns |
|---|---|---|
| `soma-core/src/lib.rs` | 86 | `#![warn(missing_docs)]`, 10 `pub mod`, ~60 convenience re-exports `(!)` |
| `soma-core/src/error.rs` | 158 | `SomaError` (13 variants), `Result<T>` |
| `soma-core/src/util.rs` | 102 | `timestamp_id`, `extract_json`, `truncate` |

**`graph/` — what the user builds.** The topology and the two node traits.

| File | Lines | Owns |
|---|---|---|
| `soma-core/src/graph/mod.rs` | 1 036 | `NodeKind`, `Node`, `EdgeKind`, `Edge`, `Graph` + topo sort + the mermaid/text renderers |
| `soma-core/src/graph/filter.rs` | 299 | `FilterKind`, `StreamMode`, `Distribution`, `RemoteTarget`, `FilterMeta`, `Filter` |
| `soma-core/src/graph/step.rs` | 370 | `Transition`, `StepCtx<'a>`, `StepMeta`, `Step` |
| `soma-core/src/graph/node.rs` | 224 | **The unifying layer**: `NodeOutcome`, `NodeMeta`, `From<FilterMeta>`, `From<StepMeta>` |
| `soma-core/src/graph/control.rs` | 209 | `LoopCondition`, `LoopSignal`, `read_loop_signal`, `read_arm_selector` |
| `soma-core/src/graph/any.rs` | 22 | `AsAny` + a blanket impl — the supertrait that lets `Filter` and `Step` be downcast |

**`data/` — what flows, and where it is kept.** Two stores live here and the
difference between them is why both exist: a `DataStore` moves values between
machines, a `StateStore` holds what `fit` learned and is authoritative.

| File | Lines | Owns |
|---|---|---|
| `soma-core/src/data/value.rs` | 267 | `Value` — 6 variants, all `Arc`-backed |
| `soma-core/src/data/schema.rs` | 365 | `DataType`, `Schema`, `Dimension`, compatibility predicates |
| `soma-core/src/data/virtual_value.rs` | 410 | `VirtualValue`, `ValueStatus` |
| `soma-core/src/data/store.rs` | 480 | `DataRef`, `StorageConfig`, `DataStore`, `LocalDataStore`, `StoreMeta` `(!)` |
| `soma-core/src/data/state.rs` | 141 | `StateStore`, `MemoryStateStore` |
| `soma-core/src/data/codec.rs` | 257 | The `SOMA1` binary frame for `Value` |
| `soma-core/src/data/keys.rs` | 76 | Reserved output-store key prefixes (`__state_`, `__input_`, `__input__`) |

**`cache/` — memoize by content.** The discardable store, as against `data/`'s
two authoritative ones.

| File | Lines | Owns |
|---|---|---|
| `soma-core/src/cache/mod.rs` | 393 | `CacheKey`, `CacheTier`, `Origin`, `EntryMeta`, `CacheStore` |
| `soma-core/src/cache/action.rs` | 200 | The Bazel-style two-table model: `HashAlgo`, `ContentHash`, `ActionResult`, `ActionCache`, `BlobStore` |
| `soma-core/src/cache/canon.rs` | 174 | Deterministic CBOR (RFC 8949 §4.2 + dCBOR floats) — `canonical_bytes`, `hash_canonical` |

**`agentic/` — the side that decides.** Nothing here performs anything;
`soma-runtime` drives the effects and `soma-llm` reaches the models.

| File | Lines | Owns |
|---|---|---|
| `soma-core/src/agentic/effect.rs` | 713 | The effect vocabulary: `Effect`, `LlmRequest`, `EffectHandler`, `EffectResult`, `LlmResponse`, `NodeSpec`, `JoinPolicy`, `SuspendReason` |
| `soma-core/src/agentic/message.rs` | 346 | `Role`, `ContentBlock`, `Message`, `Messages` |
| `soma-core/src/agentic/tool.rs` | 112 | `ToolSpec` — the MCP wire shape |

**`optimizer/` — hyperparameter search.** The space and the campaign over it.

| File | Lines | Owns |
|---|---|---|
| `soma-core/src/optimizer/search.rs` | 523 | `Scale`, `SearchDimension`, `SearchSpace`, `Searchable` |
| `soma-core/src/optimizer/study.rs` | 1 064 | `Direction`, `Objective`, `SearchStrategy`, `PruningStrategy`, `TrialState`, `Trial`, `Study` |

**`tracking/` — what a run records.**

| File | Lines | Owns |
|---|---|---|
| `soma-core/src/tracking/mod.rs` | 480 | `RunKind`, `RunState`, `RunManifest`, `EventEnvelope`, `EventSink`, `Tracker` |
| `soma-core/src/tracking/event.rs` | 913 | `Event` — 30 variants across six levels `(!)` |
| `soma-core/src/tracking/summary.rs` | 622 | `RunOutcome`, `NodeCost`, `FlagCount`, `RunConclusion`, `RunSummary` |
| `soma-core/src/tracking/fingerprint.rs` | 512 | `ArchitectureFingerprint`, `structural_similarity`, `pipeline_summary` |

**`distributed/` and `viz/` — one file and two.** `distributed` is a file
rather than a folder because the domain has exactly one module here; the
folder exists in `soma-runtime` and `soma-worker`, where it has contents.

| File | Lines | Owns |
|---|---|---|
| `soma-core/src/distributed.rs` | 285 | `TrainingStrategy` + 6 satellite enums + `Partition` — description only |
| `soma-core/src/viz/mod.rs` | 205 | `NodeStatus`, `NodeOverlay`, `GraphOverlay` — pure data for the renderers |
| `soma-core/src/viz/svg.rs` | 355 | `Graph::to_svg` — self-contained SVG, longest-path layering. Declares no public type `(!)` |

### Public contracts

Eleven traits. **None declares an associated type or a generic parameter** — a
uniformity that makes every one of them `dyn`-able, which is why the whole
system can swap backends at runtime without a single generic bound leaking into
a signature.

#### `Filter` — `soma-core/src/graph/filter.rs:120`

```rust
pub trait Filter: AsAny + Send + Sync {
    fn config_hash(&self) -> CacheKey;                              // required
    fn fit(&self, x: &Value, y: Option<&Value>) -> Result<Value>;   // required
    fn forward(&self, x: &Value, state: &Value) -> Result<Value>;   // required
    fn meta(&self) -> FilterMeta;                                   // required
    fn composite_fit(&self, peers, x, y) -> Option<Result<…>> { None } // provided :148
}
```

The central abstraction: `fit()` learns state, `forward()` transforms, and both
are independently cacheable. `composite_fit` exists for differentiable groups
that must train jointly — returning `None` means "I have nothing special to say",
so a normal filter never mentions it.

| Implementor | Crate | Distinguishing behaviour |
|---|---|---|
| `PyFilterBridge` | `soma-python/src/graph/bridge.rs:224` | Calls into a live Python object; identity delegated to `soma._identity` |
| `SubprocessFilter` | `soma-worker/src/python_process.rs:1025` | Delegates over a pipe to an out-of-process interpreter |

Roughly 40 further implementations exist in tests and fixtures. `(!)` The trait
mixes computation with cache identity — [D-91](/soma/internals/debt/#d-91--the-filter-trait-mixes-computation-with-cache-identity).

#### `Step` — `soma-core/src/graph/step.rs:250`

```rust
pub trait Step: AsAny + Send + Sync {
    fn config_hash(&self) -> CacheKey;
    fn meta(&self) -> StepMeta;
    fn poll(&self, ctx: &StepCtx<'_>) -> Result<Transition>;
}
```

`Filter`'s sibling, and the whole agentic layer in one method. `poll` is
**synchronous and re-entrant**: it returns a `Transition` describing what it
wants, and a driver performs the effects and calls it again with the results.
The driver loop is sketched in the doc at `soma-core/src/graph/step.rs:13`.

The consequence that matters: a step holds **no hidden state between turns**.
Everything it knows arrives through `StepCtx::history`
(`soma-core/src/graph/step.rs:128`), which is what makes journal replay exact rather
than approximate.

| Implementor | Crate |
|---|---|
| `ReactStep`, `LlmStep`, `JudgeStep` | `soma-llm/src/steps.rs:214`, `:491`, `:586` |
| `PyStepBridge` | `soma-python/src/agentic.rs:883` — duck-typed: any object with `poll(ctx)` |
| `ResearchStep` | `soma-agent/src/research.rs:260` |

#### `CacheStore` — `soma-core/src/cache/mod.rs:207`

```rust
pub trait CacheStore: Send + Sync {
    fn get / put / exists / remove / metadata                            // required
    fn put_with_origin(&self, key, value, origin) -> Result<()>          // provided :222
    fn put_computed(&self, key, value, origin, compute, deterministic)   // provided :232
    fn tier(&self) -> CacheTier { CacheTier::Memory }                    // provided :249
    fn get_located(&self, key) -> Result<Option<(Value, CacheTier)>>     // provided :259
}
```

A **template method**: the defaults discard the extra information so a minimal
backend needs only five methods, and richer ones override. Implementors:
`MemoryCache`, `LocalCache`, `TieredCache`, `FsActionStore` — all in
[soma-runtime](/soma/internals/execution/#soma-runtime-somatize-runtime).

#### `DataStore` — `soma-core/src/data/store.rs:208`

`put` / `get` / `exists` / `remove` / `config` required; `get_rows` and `meta`
provided by downloading everything and slicing locally (`:227`, `:234`).
`ZarrStore` is the only implementor that overrides them, which is exactly the
point — a chunked backend can serve a row range without the whole array.

Implementors: `LocalDataStore` (here, `:262`), `S3DataStore`, `ZarrStore`
([Distribution](/soma/internals/distribution/)).

#### `EffectHandler` — `soma-core/src/agentic/effect.rs:262`

```rust
pub trait EffectHandler: Send + Sync {
    fn handles(&self, effect: &Effect) -> bool;
    fn perform(&self, effect: &Effect) -> Result<EffectResult>;
}
```

Chain of responsibility, with the contract written into the doc at `:256`:
"Handlers are tried in order; the first that claims an effect wins."
Implementors: `LlmHandler` (`soma-llm/src/lib.rs:205`), `Toolbox`
(`soma-llm/src/tools.rs:168`), `GraphHandler` and `SleepHandler`
(`soma-runtime/src/effects/`).

#### `ActionCache` and `BlobStore` — `soma-core/src/cache/action.rs:134`, `:143`

The two halves of the persistent cache, deliberately separate because they have
different lifetimes: action records are kept forever, CAS blobs are evictable.
Both are implemented by `FsActionStore`
(`soma-runtime/src/cache/fs_store.rs:281`, `:255`) — one type, two roles.

`(!)` Neither is re-exported from `soma-core/src/lib.rs` —
[D-84](/soma/internals/debt/#d-84--soma-cores-re-export-surface-is-asymmetric).

#### `StateStore` — `soma-core/src/data/state.rs:24`

`get` / `set` / `remove` / `clear` / `keys`. One implementor
(`MemoryStateStore`, `:68`). `NodeCatalog::with_state_store` is the only
injection site, and only a test uses it — see
[D-36](/soma/internals/debt/#d-36--unreached-methods-and-a-pluggable-seam-with-no-injection-site),
which explains why it survived the deletion pass.

#### `EventSink` and `Tracker` — `soma-core/src/tracking/mod.rs:247`, `:255`

`EventSink::record(&self, event: &Event)` with a defaulted no-op `flush`.
`Tracker` is 7 required methods over a run directory. One implementor each:
`JsonlEventSink` and `LocalTracker`, both in `soma-runtime/src/tracking/`.

#### `Searchable` — `soma-core/src/optimizer/search.rs:321`

The one **non-object-safe** trait here: `search_space()` has no receiver and
`from_sample` carries `where Self: Sized`. Its only implementations are
**macro-generated** (`soma-macros/src/lib.rs:173`) — there is not one
hand-written impl in the workspace, which is the strongest possible evidence
that the derive is the right interface.

#### `AsAny` — `soma-core/src/graph/any.rs:13`

`fn as_any(&self) -> &dyn Any`, with a blanket `impl<T: Any> AsAny for T`, used
only as a supertrait of `Filter` and `Step` so that a concrete type can be
recovered from a trait object. Exactly three downcast sites exist workspace-wide:
`soma-python/src/graph/bridge.rs:355`, `soma-worker/src/worker.rs:158` and `:482`.

### Types — structs

| Name | Role | Key fields | file:line |
|---|---|---|---|
| `CacheKey` | SHA-256 newtype | `[u8; 32]` | `soma-core/src/cache/mod.rs:21` |
| `ContentHash` | BLAKE3/SHA-256 CAS address | `algo`, `digest` (`Copy`) | `soma-core/src/cache/action.rs:52` |
| `ActionResult` | One cached action record | key, outputs, bytes, `compute_ms`, `deterministic`, origin, timestamps | `soma-core/src/cache/action.rs:110` |
| `EntryMeta` | Cache entry metadata | key, size, timestamps, ttl, origin | `soma-core/src/cache/mod.rs:188` |
| `FilterMeta` | What a filter says about itself | name, kind, cacheable, differentiable, deterministic, `stream_mode`, distribution, 2× schema | `soma-core/src/graph/filter.rs:73` |
| `StepMeta` | What a step says about itself | name, `max_turns` (24), journal, 2× schema, distribution | `soma-core/src/graph/step.rs:177` |
| `NodeMeta` | **The unified metadata** | name, `effectful`, kind, cacheable, deterministic, differentiable, distribution, 2× schema | `soma-core/src/graph/node.rs:72` |
| `StepCtx<'a>` | Everything a step sees | `node_id`, `run_id`, `input`, `turn`, `results`, `history` | `soma-core/src/graph/step.rs:115` |
| `LlmRequest` | A model call, as data | model, messages, system, `max_tokens`, tools, effort, schema | `soma-core/src/agentic/effect.rs:144` |
| `LlmResponse` | …and its reply | message, `stop_reason`, usage, model | `soma-core/src/agentic/effect.rs:330` |
| `Usage` | Token accounting | 4× `u64`, `Copy`, `impl AddAssign` | `soma-core/src/agentic/effect.rs:411` |
| `NodeSpec` | A spawn target | `runs`, `input`, `label` | `soma-core/src/agentic/effect.rs:450` |
| `ToolSpec` | MCP tool description | name, description, `inputSchema` | `soma-core/src/agentic/tool.rs:16` |
| `Message` / `Messages` | Conversation | `role` + `Vec<ContentBlock>`; `Messages` is a transparent newtype | `soma-core/src/agentic/message.rs:137`, `:189` |
| `Schema` | dtype + shape | `dtype`, `shape: Option<Vec<Dimension>>`, 9 named constructors | `soma-core/src/data/schema.rs:104` |
| `Node` / `Edge` / `Graph` | The user-facing structure | `Graph` = nodes + edges + optional strategy | `soma-core/src/graph/mod.rs:74`, `:229`, `:293` |
| `Study` | A search, its trials and its provenance | 15 fields `(!)` | `soma-core/src/optimizer/study.rs:319` |
| `Trial` | One point in the space | id, params, state, metrics, timings | `soma-core/src/optimizer/study.rs:235` |
| `SearchSpace` | Dimensions + frozen values | `dimensions`, `frozen` | `soma-core/src/optimizer/search.rs:172` |
| `ArchitectureFingerprint` | Structure-only identity | digest, node tokens, edge refs, config hashes | `soma-core/src/tracking/fingerprint.rs:36` |
| `RunManifest` | What a run was | 20 fields `(!)` | `soma-core/src/tracking/mod.rs:97` |
| `EventEnvelope` | A sequenced, timestamped event | `seq`, `ts`, flattened `Event` | `soma-core/src/tracking/mod.rs:229` |
| `RunSummary` / `RunConclusion` | The deterministic run story | 17 + 9 fields | `soma-core/src/tracking/summary.rs:331`, `:157` |
| `NodeOverlay` / `GraphOverlay` | Per-node render annotations | status, duration, tier, flags | `soma-core/src/viz/mod.rs:35`, `:53` |
| `LocalDataStore` | Filesystem `DataStore` | config, `base_path` | `soma-core/src/data/store.rs:241` |
| `MemoryStateStore` | In-process `StateStore` | `Mutex<HashMap<String, Arc<Value>>>` | `soma-core/src/data/state.rs:46` |
| `Partition` | Nodes → a remote target | `node_ids`, `target` | `soma-core/src/distributed.rs:107` |

`GraphOverlay`'s doc (`soma-core/src/viz/mod.rs:6`) states the design rule behind
this whole file group: it is "pure data — computed elsewhere and passed in, so
rendering stays a dependency-free data→string transform". The same argument
appears in `soma-core/src/tracking/summary.rs:5`. That is why `soma-core` can render a
graph to SVG without pulling in a rendering library.

### Types — enums

The five in **bold** are the ones worth memorizing.

| Name | Variants | `!` | Why that choice | file:line |
|---|---|---|---|---|
| **`Value`** | `Tensor{values, shape}`, `Text`, `Json`, `Bytes`, `Object`, `Empty` — all `Arc`-backed | yes | Data enum; new payload kinds must not break consumers. `Arc` makes `Clone` O(1) | `soma-core/src/data/value.rs:15` |
| **`NodeOutcome`** | `Produced(Value)`, `HandOff{target, carry}`, `Paused{turn, reason}` | **no** | Control flow. "A wildcard arm here is a silent wrong answer" (`:37`) | `soma-core/src/graph/node.rs:44` |
| **`Transition`** | `Await(Vec<Effect>)`, `Spawn{specs, join}`, `Goto{target, carry}`, `Suspend{reason}`, `Done(Value)` | **no** | Same reason (`:38`) | `soma-core/src/graph/step.rs:43` |
| **`Effect`** | `Llm`, `Tool{name, args}`, `Graph{graph, input, mode}`, `Sleep`, `Custom{kind, payload}` | yes | Data — a handler that does not claim an effect ignores it | `soma-core/src/agentic/effect.rs:35` |
| **`NodeKind`** | `Filter{filter_name}`, `SubGraph{graph}`, `Loop{max_iterations, until}`, `Branch{arms}`, `Step{step_name}` | yes | Five structural kinds; every behaviour is library | `soma-core/src/graph/mod.rs:32` |
| `SomaError` | 13 variants incl. `Suspended`, `Pruned`, `SchemaMismatch`, `Execution`, `Other(String)` | yes | See [what is healthy](/soma/internals/debt/#what-is-already-healthy) — 3 error enums workspace-wide | `soma-core/src/error.rs:14` |
| `Event` | **30 variants**, six levels `(!)` | yes | Also the JSONL wire format — [D-05](/soma/internals/debt/#d-05--event-is-30-variants-across-six-unrelated-concerns) | `soma-core/src/tracking/event.rs:54` |
| `StreamMode` | `FixedState`, `Evolving`, `Barrier` | **no** | Control flow, deliberate (`:32`) | `soma-core/src/graph/filter.rs:37` |
| `FilterKind` | `Stateless`, `Trainable`, `Opaque` | yes | | `soma-core/src/graph/filter.rs:16` |
| `Distribution` / `RemoteTarget` | `Local`/`Remote(t)`/`Any`; `WorkerId`/`Tag` | no | `(!)` shadowed by `Node.target: Option<String>` — [D-52](/soma/internals/debt/#d-52--node-placement-has-one-typed-mechanism-and-one-stringly-one) | `soma-core/src/graph/filter.rs:53`, `:64` |
| `EffectResult` | `Llm`, `Tool{output, is_error}`, `Graph`, `Node`, `Slept`, `Custom`, `Failed{message}` | yes | | `soma-core/src/agentic/effect.rs:278` |
| `StopReason` | `EndTurn`, `MaxTokens`, `ToolUse`, `Refusal{category}` | yes | `MaxTokens` and `Refusal` are **errors** in `ReactStep`, not empty replies | `soma-core/src/agentic/effect.rs:392` |
| `JoinPolicy` | `All`, `AllSettled`, `First` | yes | | `soma-core/src/agentic/effect.rs:482` |
| `SuspendReason` | `Human{prompt, schema}`, `External{token}` | yes | | `soma-core/src/agentic/effect.rs:507` |
| `GraphEffectMode` | `Forward`, `Fit` | yes | Only `Forward` filter-only sub-graphs are pure | `soma-core/src/agentic/effect.rs:134` |
| `LoopCondition` | `BodyTerminal`, `WhenSignaled(NodeId)`, `Exhaust` | yes | Adjacently tagged — forced by the newtype variant (`:21`) | `soma-core/src/graph/control.rs:28` |
| `EdgeKind` | `Data`, `Control` | no | | `soma-core/src/graph/mod.rs:226` |
| `DataType` | `Float64`, `Float32`, `Int64`, `Bool`, `Utf8`, `Bytes`, `Json`, `Messages` | yes | | `soma-core/src/data/schema.rs:12` |
| `Dimension` | `Fixed(usize)`, `Dynamic(String)` | no | | `soma-core/src/data/schema.rs:115` |
| `Role` / `ContentBlock` | system/user/assistant; text/tool-use/tool-result | yes | | `soma-core/src/agentic/message.rs:22`, `:59` |
| `VirtualValue` | `Materialized`, `Cached{key}`, `Deferred{producer, key}`, `Stream{source}` | yes | Lazy reference — what the executor actually stores | `soma-core/src/data/virtual_value.rs:26` |
| `DataRef` | `Local`, `S3`, `Cached`, `Stream` `(!)`, `Inline`, `Zarr` | yes | | `soma-core/src/data/store.rs:98` |
| `StorageConfig` | `Local`, `S3`, `Zarr` | yes | | `soma-core/src/data/store.rs:161` |
| `CacheTier` / `Origin` | memory/local/remote `(!)`; computed/ingested/streamed `(!)` | no | | `soma-core/src/cache/mod.rs:153`, `:161` |
| `TrainingStrategy` | `Local`, `DataParallel`, `ModelParallel`, `Federated`, `PopulationBased`, `Custom` | yes | Description only — execution lives in `soma-runtime` | `soma-core/src/distributed.rs:22` |
| `GradientAggregation`, `CommunicationProtocol`, `FederatedAggregation`, `ClientSelection`, `ExploitStrategy`, `ExploreStrategy` | satellites of the above | yes | | `soma-core/src/distributed.rs:89`–`:200` |
| `SearchStrategy` | `Grid`, `Random`, `Bayesian` | no | | `soma-core/src/optimizer/study.rs:113` |
| `PruningStrategy` | `None`, `Median`, `Percentile` | no | | `soma-core/src/optimizer/study.rs:156` |
| `SearchDimension` | `Float`, `Int`, `Categorical`, `Conditional{parent, dimension}` | yes | Recursive through `Box` | `soma-core/src/optimizer/search.rs:36` |
| `TrialState` / `Direction` / `Scale` / `Scalarizer` | search vocabulary | mixed | | `soma-core/src/optimizer/study.rs:210`, `:15`, `search.rs:17`, `study.rs:48` |
| `RunKind` / `RunState` / `RunOutcome` / `NodeStatus` | tracking vocabulary | yes | `RunKind` has a `#[serde(other)]` catch-all | `soma-core/src/tracking/mod.rs:31`, `:46`, `summary.rs:27`, `viz.rs:20` |
| `HashAlgo` | `Blake3`, `Sha256` | yes | | `soma-core/src/cache/action.rs:32` |
| `LoopSignal` / `ValueStatus` | small vocabularies | mixed | | `control.rs:46`, `virtual_value.rs:65` |

**The `#[non_exhaustive]` policy is the thing to take away.** It is not applied
uniformly, and the non-uniformity is the design: data enums get it so a consumer
need not have an opinion about a new variant; control-flow enums every consumer
must decide over — `NodeOutcome`, `Transition`, `StreamMode` — deliberately do
not, so that adding a variant breaks every `match` and forces the decision. The
reason is stated in each doc comment.

### Ownership and relationships

```
Graph                                            soma-core/src/graph/mod.rs:293
 ├──◆ Vec<Node> ──◆ [enum] NodeKind
 │                    ├──◆ Box<Graph>            SubGraph — recursive
 │                    └──◆ LoopCondition ──▷ NodeId
 ├──◆ Vec<Edge> ──◆ [enum] EdgeKind
 └──? Option<TrainingStrategy> ──◆ { GradientAggregation
                                   | Vec<Partition> + CommunicationProtocol
                                   | FederatedAggregation + ClientSelection
                                   | ExploitStrategy + ExploreStrategy }

«trait» Filter ──▷ FilterMeta ─┐
«trait» Step   ──▷ StepMeta   ─┴──▷ NodeMeta      « the adapter »
                                     soma-core/src/graph/node.rs:72
   From<FilterMeta>  → effectful: false
   From<StepMeta>    → effectful: true, cacheable: false, deterministic: false

Step::poll ──▷ [enum] Transition
                 ├──◆ Vec<Effect> ──◆ LlmRequest ──◆ Messages ──◆ Vec<Message>
                 │                                              ──◆ Vec<ContentBlock>
                 │                 ──◆ Vec<ToolSpec>
                 │                 ──◆ Box<Graph>    « Effect::Graph — a pipeline as a tool »
                 ├──◆ Vec<NodeSpec> + JoinPolicy
                 ├──◆ NodeId + Value
                 ├──◆ SuspendReason
                 └──◆ Value

CacheKey ◁── CacheStore keys, ActionResult.key, DataRef::Cached,
             VirtualValue::{Cached, Deferred}
ActionResult ──◆ BTreeMap<String, ContentHash> ──◆ HashAlgo

Study ──◆ SearchSpace ──◆ Vec<SearchDimension> ──◆ Box<SearchDimension>  « Conditional »
      ├──◆ SearchStrategy, PruningStrategy, Vec<Objective> ──◆ Direction
      └──◆ Vec<Trial> ──◆ TrialState, Vec<MetricRecord>

RunSummary ──◆ RunConclusion ──◆ RunOutcome, NodeCost, Vec<FlagCount>,
                                 TrialSummary, AgentCost
           └──? Option<ArchitectureFingerprint> ──◆ Vec<EdgeRef>
```

#### Conversions

| From → To | file:line |
|---|---|
| `FilterMeta → NodeMeta` | `soma-core/src/graph/node.rs:116` |
| `StepMeta → NodeMeta` | `soma-core/src/graph/node.rs:132` |
| `NodeMeta → FilterMeta` (lossy, an inherent method, **not** `From`) | `soma-core/src/graph/node.rs:160` |
| `Vec<f64> → Value` (1-D tensor) | `soma-core/src/data/value.rs:185` |
| `serde_json::Value → Value::Json` | `soma-core/src/data/value.rs:195` |
| `Vec<Message> → Messages`, `IntoIterator for Messages` | `soma-core/src/agentic/message.rs:253`, `:259` |
| `Value → VirtualValue::Materialized` | `soma-core/src/data/virtual_value.rs:229` |
| `io::Error → SomaError::Io` (`#[from]`) | `soma-core/src/error.rs:108` |
| `AddAssign for Usage` | `soma-core/src/agentic/effect.rs:434` |

`NodeMeta → FilterMeta` is deliberately *not* a `From` impl: it drops the
`effectful` bit, and making the lossy direction inconvenient is the point.

### Entry points

| Symbol | file:line | Why you would look |
|---|---|---|
| `CacheKey::for_state` / `for_output` | `soma-core/src/cache/mod.rs:21` | The whole caching model in two functions |
| `CacheKey::absorb` | `soma-core/src/cache/mod.rs:89` | Exhaustive `match` on `Value` by design (`:123`) — the one place a new `Value` variant must be handled |
| `canonical_bytes` | `soma-core/src/cache/canon.rs` | Why two structurally equal configs hash the same |
| `Graph::topological_sort` | `soma-core/src/graph/mod.rs:456` | `(!)` sorts ascending then pops — roots come out descending |
| `Graph::validate` | `soma-core/src/graph/mod.rs` | Cycle detection |
| `Graph::contains_steps` | `soma-core/src/graph/mod.rs:524` | Decides whether an `Effect::Graph` can be pure |
| `Effect::is_pure` / `cache_key` | `soma-core/src/agentic/effect.rs:80`–`:127` | The journal's keying rule |
| `LlmResponse::reject_non_answers` | `soma-core/src/agentic/effect.rs` | Why `length` and `content_filter` are errors |
| `read_loop_signal` / `read_arm_selector` | `soma-core/src/graph/control.rs` | How data-dependent control flow reads its input |
| `RunConclusion::render_headline` | `soma-core/src/tracking/summary.rs:212` | The templated, deterministic run story |

### Patterns in use

- **Strategy via `dyn`** — every backend seam: `CacheStore`, `DataStore`, `StateStore`, `EffectHandler`. → [Patterns](/soma/internals/patterns/#strategy)
- **Adapter** — `NodeMeta` erases the Filter/Step distinction; the module doc at `soma-core/src/graph/node.rs:1` is the clearest statement of the design in the repo.
- **Chain of responsibility** — `EffectHandler::handles`.
- **Interpreter / command** — `Effect` describes work; the runtime performs it.
- **State machine / trampoline** — `Step::poll → Transition`, deliberately avoiding `async fn` in a trait.
- **Composite** — `NodeKind::SubGraph`, `SearchDimension::Conditional`.
- **Template method** — `CacheStore` and `DataStore` defaults.
- **Newtype** — `CacheKey([u8; 32])`, `Messages(Vec<Message>)`, `ContentHash`.
- **Flyweight / COW** — every `Value` payload is `Arc`-wrapped, so `Clone` is a refcount bump.
- **Memento / journal** — `Effect::cache_key` as the journal key; `StepCtx::history` explicitly replacing hidden step state.
- **Null object** — `Value::Empty`, `ExecutionPlan::Empty`, `TrainingStrategy::Local`.
- **Data-transfer object** — `GraphOverlay`, `RunSummary`, `RunConclusion`.

Notably **absent**: typestate, `PhantomData`, and generic type parameters on any
public trait. That absence is what keeps everything `dyn`-able.

### Debt

- [D-05](/soma/internals/debt/#d-05--event-is-30-variants-across-six-unrelated-concerns) `Event` — 30 variants, six concerns · [D-06](/soma/internals/debt/#d-06--wide-data-structs-with-no-builder) wide structs with no builder
- [D-23](/soma/internals/debt/#d-23--a-serialization-failure-gives-every-failing-value-the-same-cache-key) a serialization failure collides cache keys
- [D-33](/soma/internals/debt/#d-33--value-to_plain_json-contradicts-its-own-contract) `to_plain_json` contradicts its contract · [D-35](/soma/internals/debt/#d-35--enum-variants-that-exist-only-to-be-refused-or-ignored) nine never-constructed variants, now deleted
- [D-51](/soma/internals/debt/#d-51--four-style-tables-keyed-by-the-same-magic-strings) four style tables keyed by magic strings · [D-52](/soma/internals/debt/#d-52--node-placement-has-one-typed-mechanism-and-one-stringly-one) two placement mechanisms · [D-53](/soma/internals/debt/#d-53--typed-enums-shadowed-by-their-own-string-forms) enums shadowed by strings · [D-56](/soma/internals/debt/#d-56--nodeid-is-a-string-and-so-is-everything-else) `NodeId = String`
- [D-15](/soma/internals/debt/#d-15--five-formatters-for-a-duration-two-for-a-truncation) two duration formatters that disagree · [D-17](/soma/internals/debt/#d-17--four-renderers-four-independent-match-nodekind) four renderers
- [D-84](/soma/internals/debt/#d-84--soma-cores-re-export-surface-is-asymmetric) asymmetric re-exports · [D-91](/soma/internals/debt/#d-91--the-filter-trait-mixes-computation-with-cache-identity), [D-92](/soma/internals/debt/#d-92--graphpredecessors--successors-are-linear-scans), [D-94](/soma/internals/debt/#d-94--soma-core-owns-seven-domains), [D-95](/soma/internals/debt/#d-95--somaerrorpruned-coexists-with-trialoutcomepruned)

---

## soma-macros (`somatize-macros`)

### Mandate

Two derive macros, and one job: make it impossible for a field to escape a cache
key silently.

`607 lines in one file · 0 traits · 0 public types · 2 proc macros · deps: syn, quote`

### What it generates

| Macro | file:line | Generates |
|---|---|---|
| `#[derive(SomaFilter)]` | `soma-macros/src/lib.rs:30` | `config_hash()` from the canonical CBOR of every field, plus `impl Searchable` when `#[soma(search(…))]` attributes are present |
| `#[derive(SomaStep)]` | `soma-macros/src/lib.rs:533` | The same `config_hash()` for a `Step` — "what gives every step its journal key" |

Supporting internals: `StructAttrs` (`:201`), `FieldAttrs` (`:210`),
`SearchAttrs` (`:215`), parsers at `:231` / `:301` / `:384`, codegen at `:403`
(`generate_search_dimension`) and `:473` (`generate_from_sample`).

`#[soma(cache_version = "…")]` lets an implementor bump the key deliberately when
the *behaviour* changes without the fields changing — the escape hatch that makes
field-derived identity safe.

### Why this matters more than it looks

Filter identity is the foundation of the whole cache, and the two languages solve
it differently:

- **Rust** — canonical CBOR of the field list, plus `cache_version`. Adding a field changes the key automatically.
- **Python** — qualname + canonical config + a source-hash ladder (`_cache_version` → `inspect.getsource` → cloudpickle with a warning), in `soma-python/python/soma/_identity.py:124`. An unhashable config raises `CacheConfigError`, never a silent key.

`(!)` The generated code panics on a non-CBOR-serializable field, from inside
`config_hash()`, which the executor calls on every node —
[D-29](/soma/internals/debt/#d-29--macro-generated-code-panics-inside-the-cache-key-path).

Note the dependency direction: `soma-macros` has a **dev**-dependency back on
`soma-core`, path-only and deliberately unversioned. The comment at
`soma-macros/Cargo.toml:16` explains the publish cycle that would otherwise
result.

---

## soma (`somatize`) — the facade

`124 lines in one file.` Nine crate re-exports (`core`, `compiler`, `runtime`,
`memory`, `worker`, `agent`, `llm`, `coordinator`, `macros`), a feature-gated
`store`, and a `prelude` with 22 re-exports at `soma/src/lib.rs:93`.

`(!)` It covers 10 of 13 crates — `somatize-mcp` and `somatize-python` are
workspace members it does not depend on — and it hand-rolls `any(s3, zarr)` with
two complementary `#[cfg]` attributes at `soma/src/lib.rs:77`. See
[D-83](/soma/internals/debt/#d-83--the-facade-covers-10-of-13-crates). The
comments at `soma/src/lib.rs:64` and `:82` record two previous instances of
exactly this gap being found and fixed, which suggests the shape of the fix
matters more than the fix.
