---
title: Python Bridge — PyO3 layer and the soma package
description: The complete pyclass and pyfunction surface, the pure-Python package map, every FFI conversion, and where Rust and Python encode the same concept twice.
---

Python is Soma's primary interface, which makes this the crate a user actually
touches — and the one with the highest concept density per line, because every
type here exists in two languages at once.

The layering is worth stating before anything else, because it is not obvious
from the file names:

```
     user code
         │
   python/soma/*.py        7 377 lines   ← the API's ergonomics: mixins, dataclasses,
         │                                  duck-typed steps, viz, CLI
   soma/_soma (extension)                ← built by maturin from:
         │
   soma-python/src/*.rs    6 605 lines   ← the bridge: pyclass wrappers + trait impls
         │
   the Rust workspace
```

The [notation](/soma/internals/map/) legend applies. `(!)` marks a documented
deviation, with the entry in the [Debt Register](/soma/internals/debt/).

---

## D6 · The FFI bridge

Four Rust types implement a Rust trait by calling *into* Python. They are the
entire mechanism by which a Python object becomes a first-class node.

```
   Python side                        Rust side                   Trait satisfied
   ───────────                        ─────────                   ───────────────
   class MyFilter(Filter)             PyFilterBridge         ──▷  «trait» Filter
     forward(x, state)                bridge.rs:7                 soma-core/…filter.rs:120
     fit(x, y)                        ├─ py_obj: Py<PyAny>
     _cache_version                   ├─ pickled_bytes  (cloudpickle → workers)
                                      └─ config_hash_val ◁── soma._identity.filter_identity

   any object with poll(ctx)          PyStepBridge           ──▷  «trait» Step
     poll(ctx) -> dict                agentic.rs:872              soma-core/…step.rs:250
     « no base class, on purpose »

   @soma.tool def f(...)              PyTool ──▷ PyToolAdapter ─▷ «trait» Tool
                                      agentic.rs:46 / :194        soma-llm/…tools.rs:53

   Graph.train / evaluate fns         PyPbtExecutor          ──▷  «trait» PbtExecutor
                                      pbt.rs:40                   soma-runtime/…pbt.rs:55

   soma.Agent(...)  ─┐
   soma.Judge(...)  ─┴─▷ to_step_spec ──▷ ReactStep | JudgeStep   « already Rust steps »
                         agentic.rs:978

   dict transitions            dict effects
   {"transition": "done"} ──▷ parse_transition ──▷ [enum] Transition
   {"effect": "llm"}      ──▷ parse_effect      ──▷ [enum] Effect
                              agentic.rs:587 / :755    (!) stringly-typed, D-54
```

The `PyStepBridge` case is the one to internalize: **a step is any object with
`poll(ctx)`**. There is no base class and no registration. The rationale is at
`soma-python/python/soma/agentic.py:103` — "what crosses into Rust is data rather
than a class hierarchy" — and it is why `Fanout`
(`soma-python/python/soma/agentic.py:740`) is a plain class with two attributes.

---

## soma-python (Rust side)

### Mandate

Expose the workspace to Python and let Python objects re-enter it as filters,
steps and tools. It owns no domain logic — everything here either wraps a Rust
type for Python or wraps a Python object for Rust.

`6 605 lines across 15 files in 6 domains · 0 traits defined · 10 #[pyclass] · 33 #[pyfunction] · 4 bridge impls`

### Modules

The same domain names as `soma-core` and `soma-runtime`, so the three layers
of one capability line up: `optimizer/` here is what a user types, in
`soma-runtime` what walks a space, in `soma-core` what a space *is*. Three
domains are a single module and stay a file rather than a folder.

`graph/` is also where [D-01](/soma/internals/debt/#d-01--pygraph-is-the-workspaces-god-object)
gets fixed: the folder exists, and `PyGraph`'s 2 362 lines have somewhere to
be broken into.

| File | Lines | Owns |
|---|---|---|
| `soma-python/src/lib.rs` | 228 | `prelude`, exceptions, `#[pymodule] _soma` |
| `soma-python/src/agentic.rs` | 1 224 | `PyTool`, `PyAgent`, `PyJudge`, `PyStepCtx`, `PyStepBridge`, transition/effect parsers |
| `soma-python/src/distributed.rs` | 296 | `PyWorker` |
| `soma-python/src/cache.rs` | 182 | 5 cache functions |

**`graph/` — the primary API and the filter bridge.**

| File | Lines | Owns |
|---|---|---|
| `soma-python/src/graph/mod.rs` | 2 362 | `PyGraph` — the whole primary API `(!)` |
| `soma-python/src/graph/bridge.rs` | 437 | `PyFilterBridge` |

**`optimizer/` — search, as Python types.**

| File | Lines | Owns |
|---|---|---|
| `soma-python/src/optimizer/study.rs` | 706 | `PyStudy`, `PyTrial`, search-dimension parsing |
| `soma-python/src/optimizer/pbt.rs` | 209 | `PyPbt`, `PyPbtExecutor` |

**`tracking/` — writing a run, and reading one back.**

| File | Lines | Owns |
|---|---|---|
| `soma-python/src/tracking/readers.rs` | 449 | 24 JSON readers over run dirs and the experiment pool |
| `soma-python/src/tracking/run.rs` | 209 | `PyRun` |

**`data/` — the boundary itself.** `convert.rs` is the narrowest and most
consequential surface in the crate: `py_to_value` decides that a numeric list
is a tensor, and that decision lands in every cache key derived from it.

| File | Lines | Owns |
|---|---|---|
| `soma-python/src/data/convert.rs` | 228 | `py_to_value` / `value_to_py` / `json_to_py` / `py_any_to_json` / `as_json` |
| `soma-python/src/data/store.rs` | 73 | `build_data_store` — shared by `Graph` and `Worker` |

### The `#[pyclass]` surface

Registered in `#[pymodule] fn _soma` at `soma-python/src/lib.rs:178`.

| Rust type | Python name | file:line | Notes |
|---|---|---|---|
| `PyGraph` | `Graph` | `soma-python/src/graph/mod.rs:29` | `subclass` — required by `soma._graph.Graph` |
| `PyAgent` | `Agent` | `soma-python/src/agentic.rs:282` | model / system / max_turns / max_tokens / effort settable; `search_space()` |
| `PyJudge` | `Judge` | `soma-python/src/agentic.rs:421` | model / rubric / threshold; `search_space()` |
| `PyTool` | `Tool` | `soma-python/src/agentic.rs:46` | manual `impl Clone` via `clone_ref` (`:54`) |
| `PyStepCtx` | `StepCtx` | `soma-python/src/agentic.rs:505` | all fields `#[pyo3(get)]` — what a Python `poll` receives |
| `PyStudy` | `Study` | `soma-python/src/optimizer/study.rs:172` | `subclass` — `soma._study.Study` adds the plots |
| `PyTrial` | `Trial` | `soma-python/src/optimizer/study.rs:111` | `__getitem__` / `__contains__` / `report` / `should_prune` |
| `PyRun` | `Run` | `soma-python/src/tracking/run.rs:12` | `log`, `log_epoch`, `step_completed`, `heartbeat`, `finish` |
| `PyWorker` | `Worker` | `soma-python/src/distributed.rs:53` | `(!)` `#[allow(too_many_arguments)]` on the whole impl block |
| `PyPbt` | `Pbt` | `soma-python/src/optimizer/pbt.rs:34` | `run(train, evaluate)` |

Three of these are **subclassable on purpose**, and that is the assembly
mechanism: `PyGraph` and `PyStudy` are subclassed in Python to attach the pure-
Python methods (below).

Non-`#[pyclass]` types that still cross the boundary: `PyFilterBridge`,
`PyStepBridge`, `PyToolAdapter`, `PyPbtExecutor` (see [D6](#d6--the-ffi-bridge)),
plus three private enums — `StepSpec` (`soma-python/src/agentic.rs:936`),
`Behaviour` (`soma-python/src/graph/mod.rs:13`) and `StoreConfig`
(`soma-python/src/distributed.rs:72`, deferred configuration held until `serve`).

### `PyGraph` — the full method surface

**~47 public methods and 22 private helpers on one type.** Grouped so it is
navigable:

| Group | Methods (all `soma-python/src/graph/mod.rs`) |
|---|---|
| Construction & topology | `__new__` `:820`, `node` `:899`, `edge` `:1290`, `branch` `:1026`, `loop_` `:1093`, `handoff` `:1308`, `optional` `:1177`, `optional_edges` `:1205`, `set_edge` `:1214` |
| Agentic registration | `register_graph` `:959`, `register_step` `:987`, `use_provider` `:1156`, `add_tool` `:1256`, `add_mcp_server` `:1265`, `steps` `:1245` |
| Execution | `fit` `:1322` `(!)`, `forward` `:1584`, `resume` `:1628`, `compile` `:1663` |
| Rendering | `to_mermaid` `:1725`, `to_svg` `:1736`, `_repr_html_` `:1745`, `to_text` `:1762`, `graph_json` `:1818` |
| Events & tracking | `on_event` `:1780`, `emit_event` `:1805`, `begin_run` `:1841` |
| Distribution | `add_worker` `:1901`, `set_data_store` `:1918`, `set_strategy` `:1954` `(!)`, `strategy` `:2079` `(!)`, `shutdown_worker` `:2099`, `shutdown_workers` `:2114`, `set_coordinator` `:2129`, `workers` `:2136` |
| Introspection | `filter_source` `:2278`, `filter_requirements` `:2289`, `filter_sources_dict` `:2296`, `filter` `:2313`, `filter_ids` `:2323`, `filters` `:2349`, `set_node_state` `:2366`, `edges` `:2384`, `get_node_state` `:2396`, `mark_fitted` `:2411`, `py_state` `:2422`, `__len__` `:2430`, `__repr__` `:2434`, `__str__` `:2443` |

**19 fields** (`soma-python/src/graph/mod.rs:30`–`:84`), five of them parallel maps
keyed by node id:

```
graph: Graph                    library: NodeCatalog        cache: Arc<dyn CacheStore>
event_bus: Arc<EventBus>        fitted: bool                data_store: Option<Arc<dyn DataStore>>
workers: Vec<(addr, token, tags)>                           coordinator: Option<(url, token)>
tools: HashMap<String, PyTool>  default_provider: Option<String>
mcp_toolboxes: Vec<Toolbox>     py_state: Option<Py<PyDict>>
optional_edges: Vec<(String, String)>
cut_edges: HashMap<(String, String), (usize, Edge)>

(!) five parallel node-keyed maps, written together, never removed from:
    pickled_filters · filter_sources · filter_trainable · live_filters · live_steps
```

See [D-01](/soma/internals/debt/#d-01--pygraph-is-the-workspaces-god-object).

### The `#[pyfunction]` surface

Thirty-three functions, and one convention worth knowing: **everything in
`readers.rs` returns a JSON `String`** which the Python wrapper `json.loads`.
That is a deliberate FFI simplification, argued at `soma-python/src/tracking/readers.rs:7`
— one conversion path instead of twenty-five hand-written `IntoPy` impls. `(!)`
It also means every `RunView` property pays a serialize→parse round trip —
[D-63](/soma/internals/debt/#d-63--runreader-re-parses-eventsjsonl-once-per-accessor).

| Module | Functions |
|---|---|
| `soma-python/src/agentic.rs` | `tool` `:1147`, `providers` `:1179`, `models` `:1200` |
| `soma-python/src/cache.rs` | `cache_stats` `:24`, `cache_gc` `:64`, `cache_pin` `:89`, `cache_verify` `:109`, `cache_purge_v1` `:146` |
| `soma-python/src/tracking/readers.rs` | 24 functions: 4 run/HEAD (`run_summary_json` `:38`, `checkout_run` `:51`, `read_head_run` `:59`, `clear_head_run` `:66`), 5 knowledge-base (`kb_find_similar_json` `:85`, `kb_record_conclusion` `:144`, `kb_lineage_json` `:176`, `kb_diff_json` `:196`, `kb_reindex` `:220`), 11 run readers (`list_runs_json` `:271` … `run_overlay_json` `:373`), 4 renderers (`run_to_mermaid` `:383`, `graph_json_to_mermaid` `:403`, `graph_json_to_svg` `:420`, `run_to_svg` `:436`) |

### Conversions across the boundary

| Direction | Mechanism | file:line |
|---|---|---|
| Python object → `Value` | `py_to_value` | `soma-python/src/data/convert.rs` |
| `Value` → Python | `value_to_py` | `soma-python/src/data/convert.rs` |
| Python → `serde_json::Value` | `py_any_to_json`, via `json.dumps` | `soma-python/src/data/convert.rs:5` |
| `serde_json::Value` → Python | `json_to_py`, via `json.loads` | `soma-python/src/data/convert.rs:20` |
| Python → JSON, lossless only | `as_json` | `soma-python/src/data/convert.rs:56` |
| `SomaError` → `PyErr` | `soma_err_to_py` | `soma-python/src/lib.rs:129` |
| `PyErr` → `SomaError` | `py_err_to_soma` `(!)` **lossy** | `soma-python/src/lib.rs:158` |
| Python dict → `Transition` | `parse_transition` | `soma-python/src/agentic.rs:582` |
| Python dict → `Effect` | `parse_effect` | `soma-python/src/agentic.rs:755` |
| `_input_schema` / `_output_schema` → `Schema` | `parse_schema_attr` | `soma-python/src/agentic.rs:719` |
| Python dict → `SearchDimension` | `parse_py_search_dim` | `soma-python/src/optimizer/study.rs:16` |
| Python args → `Arc<dyn DataStore>` | `build_data_store` | `soma-python/src/data/store.rs:20` |

Two of these are more interesting than they look.

`json_to_py` (`soma-python/src/data/convert.rs:20`) routes through `json.loads`
rather than a hand-written match, and the doc records why: the hand-written
version returned arrays and objects **as strings**. Round-tripping through the
`json` module is slower and correct.

`as_json` (`soma-python/src/data/convert.rs:56`) walks the object in Rust and
*rejects* tuples, integer keys, `NaN`/`±inf`, and integers outside `i64`/`u64` —
explicitly replacing a `dumps → loads → ==` round-trip check. It is the strictest
conversion in the file, and it is the one used where a wrong answer would become
a wrong cache key.

The error direction is asymmetric on purpose in one direction and by accident in
the other. Rust → Python is **structured**: four exceptions
(`SomaSuspended`, `SomaPruned`, `SomaSchemaMismatch`, `SomaNodeNotFound`) all
deriving `RuntimeError` specifically so existing `except RuntimeError` keeps
working (`soma-python/src/lib.rs:92`), with `Suspended` carrying `run_id`,
`node_id`, `turn`, `kind` and `reason` as attributes. `(!)` Python → Rust
collapses **every** `PyErr` to `SomaError::Other(e.to_string())`, so a
`KeyboardInterrupt` inside a filter is indistinguishable from a `ValueError` by
the time the runner sees it.

---

## The pure-Python package

`7 377 lines across 28 modules in `soma-python/python/soma/`.`

### The assembly point

`soma-python/python/soma/_graph.py:35` is where the API becomes what a user
sees:

```python
class Graph(_RustGraph):
    materialize = _orchestrator.materialize      # 14 methods from _orchestrator
    train       = _orchestrator.train
    state       = _checkpoint.state              # 4 from _checkpoint
    load_state  = _checkpoint.load_state
    search_space = _study.graph_search_space     # 3 from _study
    study        = _study.graph_study
    track_run    = _tracking.track_run
    gradient_audit = _audit.gradient_audit
    compile      = _compile.compile_with_repr    # SHADOWS the Rust compile
```

**23 methods assigned in the class body**, and the docstring at
`soma-python/python/soma/_graph.py:9` explains why it is written this way: these
used to be *monkey-patched* onto the Rust class at import time from six modules.
Nothing could see them — not `help()`, not an IDE, not mypy — the surface
differed depending on which modules a program had imported, and three of them
silently shadowed Rust methods of the same name.

Assignment in a class body fixes all of that at the cost of one import-order
constraint. It is the single best structural decision in the Python layer.

### Modules

| File | Lines | Role |
|---|---|---|
| `_audit.py` | 1 338 | Gradient/activation audit: 7 dataclasses + `Audit` (30 methods) `(!)` |
| `agentic.py` | 820 | 5 filters/steps, 11 transition constructors, 8 pattern factories |
| `_orchestrator.py` | 650 | The torch training loop, bolted onto `Graph` |
| `viz/_figures.py` | 582 | 9 plotly figures |
| `library.py` | 421 | `Eval`, `Accumulator`, `Retriever`, `Compact` |
| `_runs.py` | 417 | `RunView` (30 methods), `RunList` |
| `viz/_health.py` | 412 | 5 audit figures |
| `viz/_report.py` | 410 | The self-contained HTML report builder |
| `_cache_cli.py` | 364 | The `somatize` CLI |
| `_checkpoint.py` | 324 | save / load / state / restore_optimizer |
| `_composite.py` | 280 | `DifferentiableFilter` (torch, optional) |
| `_study.py` | 253 | `search_space`, `apply_params`, `study`, `Study(_Study)` |
| `filter.py` | 186 | `FilterMeta` metaclass + the `Filter` base |
| `_lineage.py` | 147 | Thin JSON wrappers over `_soma.kb_*` |
| `_identity.py` | 132 | Canonical config JSON + code fingerprint + `CacheConfigError` |
| `chain.py` | 128 | `Chain`, `Fork` — the operator DSL |
| `__init__.py` | 125 | The public surface |
| `viz/_theme.py` | 111 | The plotly template |
| `_graph.py` | 89 | The assembly point above |
| `_compile.py` | 89 | `CompileInfo(dict)` with `_repr_html_` |
| `viz/_frames.py` | 87 | pandas projections |
| `search.py` | 86 | `SearchDescriptor` (descriptor protocol) + `search()` |
| `cli.py` | 74 | Worker CLI shim |
| `builder.py` | 69 | `somatize(topology)` fluent builder |
| `_tracking.py` | 67 | `track_run` context manager |
| `viz/__init__.py` | 63 | 16 re-exports |
| `_experiments.py` | 30 | `experiments(root)` |
| `_soma.pyi` | 738 | The hand-written stub for the extension |

### Typed versus duck-typed — a deliberate split

The package ships `py.typed`, so what it says about itself is public API. But it
is typed in some places and duck-typed in others, and the line is drawn on
purpose.

**Typed** — anything a user *reads*:

- `_audit.py` dataclasses: `Thresholds` `:67` (frozen), `AuditScope` `:92` (frozen), `StepRecord` `:128`, `ChannelConfig` `:203` (frozen), `FilterReport` `:230`, `AuditReport` `:243`
- `_soma.pyi` — 738 lines with a `Protocol` (`_SearchDim` `:31`), four `TypedDict`s (`:599`, `:611`, `:618`, `:623`) and `@overload`s for `Graph.node` (`:212`, `:216`) and `tool` (`:381`, `:391`)

**Duck-typed** — anything a user *writes*:

- A **step** is any object with `poll(ctx)` — no base class, no registration
- A **filter** is any `Filter` subclass with `forward(x, state)` — metaclass-registered at `filter.py:6`
- A **transition or effect** is a plain `dict`, built by 11 constructor functions at `agentic.py:123`–`:193` (`Done`, `Await`, `Spawn`, `Goto`, `Suspend`, `Sleep`, `Custom`, `Run`, `RunGraph`, `Llm`, `ToolCall`)
- A **search descriptor** is anything with `to_dict()` and `field_name`, sniffed by `hasattr` at `soma-python/src/agentic.rs:216`
- `AuditScope` accepts `True`, an int, a list of fnmatch patterns, or the dataclass — coerced by `_coerce_scope` (`_audit.py:1042`)

A stub can lie, so `soma-python/tests/test_stubs.py` checks the hand-written
`.pyi` against the module that was actually **built**: same classes, methods,
attributes, parameter names and defaults, and no constructor for the three
classes that have no `#[new]`. What no test can check is whether a type is
*right*.

Two PyO3 facts that shape the stub and are easy to trip over: a `#[new]`'s
signature lands on the *type* (`cls.__text_signature__`), not on `__new__`; and a
method bound dynamically in a class body is `Any` to a checker, which is why the
`soma.viz` methods on `Study` and `RunView` are written out one by one instead of
attached in a loop.

### `soma.agentic` — patterns as functions

Every pattern is a function that returns a plain `soma.Graph`. There is no
pattern class hierarchy.

| Factory | Line | Shape |
|---|---|---|
| `react` | `agentic.py:487` | The ReAct loop |
| `route` | `:514` | Selector → arms |
| `refine` | `:537` | Generate → critique → revise |
| `debate` | `:574` | Two agents, N rounds |
| `board` | `:612` | Du et al. multi-agent debate: `brief → members → chair` |
| `self_consistency` | `:669` | One agent sampled N times |
| `parallel_vote` | `:714` | N agents, one vote |
| `orchestrate` | `:785` | `planner → fanout → synthesize`, pool sized from the plan |

Filters and steps: `Revise` `:69`, `Brief` `:202`, `MajorityVote` `:245`,
`Validate` `:354`, `Fanout` `:740` (a step, not a filter).

`board` is worth reading as the reference implementation: the chair also reads
the brief (or round 2 forgets the question), `MajorityVote` is a filter rather
than a model call, and `done` is unanimity — so a converged panel stops early.

`(!)` Three places parse prose as control flow — `PANEL_MARKER` (`:196`),
`MajorityVote.extract` (`:288`), `Fanout.tasks` (`:759`) —
[D-57](/soma/internals/debt/#d-57--prose-parsing-as-control-flow).

### `soma.library`

`Eval` `:81` (accuracy / exact-match / token-F1 / top-k — scoring nothing is an
**error**, not a 0.0), `Accumulator` `:227` (stateful, `_deterministic=False`, the
documented exception), `Retriever` `:284` (over the experiment pool), `Compact`
`:361` (sliding window — enabling it invalidates replay of earlier runs).

The docstring at `library.py:13` states the boundary: "They live in Python
because that is where the primary interface is… A Rust user does not get them."

### Optional-dependency degradation

A pattern in its own right, applied consistently: torch missing →
`DifferentiableFilter = None` and 8 audit names set to `None`
(`__init__.py:31`, `:59`); plotly and pandas lazily imported inside `_go()`
(`viz/_figures.py:18`) and `_pandas()` (`viz/_frames.py:10`) so the methods always
*exist* and only calling them needs the `somatize[viz]` extra. rich and tqdm the
same, with plain fallbacks.

---

## Where Rust and Python encode the same concept

This table is the reason the Python layer is worth auditing separately. Some of
these duplications are correct layering; some are real debt. The difference is in
the last column.

| Concept | Rust | Python | Verdict |
|---|---|---|---|
| Filter identity | `PyFilterBridge::new` (`soma-python/src/graph/bridge.rs:27`) | delegates to `soma._identity.filter_identity` (`_identity.py:124`) | ✅ **Correctly not duplicated** — Rust calls into Python |
| Data-store config | `build_data_store` (`soma-python/src/data/store.rs:20`) | — | ✅ Shared by `Graph.set_data_store` and `Worker.set_data_store`; the file docstring says the sharing is the point |
| Agentic patterns | `ReactStep` (`soma-llm/src/steps.rs:32`) is the loop | `agentic.react()` builds a Graph around `Agent`, which *is* a `ReactStep` | ✅ Layering, not duplication |
| Knowledge-base retrieval | `readers.rs:85` | `_lineage.py:62` — a thin `json.loads` | ✅ on the Python side; ❌ `readers.rs` duplicates `soma-mcp` — [D-16](/soma/internals/debt/#d-16--two-knowledge-base-front-ends-already-divergent) |
| Search dimension | `parse_py_search_dim` (`soma-python/src/optimizer/study.rs:16`), `searchable` (`soma-python/src/agentic.rs:212`) | `SearchDescriptor` + `search()` (`search.py:4`) | ❌ **Three** encodings, counting `_searchable` inside the MCP driver string (`soma-mcp/src/exec.rs:96`) |
| Step/effect vocabulary | `Transition`, `Effect` enums | 11 dict constructors (`agentic.py:123`) | ❌ Kept in sync **by string literals only** — [D-54](/soma/internals/debt/#d-54--nine-string-match-dispatch-sites-across-the-ffi) |
| Graph rendering | `PyGraph::to_mermaid` `:1836`, `run_to_mermaid` (`readers.rs:382`), `graph_json_to_mermaid` (`readers.rs:435`) | `_runs.py:151`/`:233`/`:238` plus `_inner_overlay` `:178` | ❌ Three entry points into one renderer, with overlay assembly on both sides |
| Report rendering | `soma-mcp/src/render.rs` — Markdown for models | `viz/_report.py` — HTML for humans | ⚠️ Different audiences, but three duration formatters between them — [D-15](/soma/internals/debt/#d-15--five-formatters-for-a-duration-two-for-a-truncation) |
| Training strategy | `TrainingStrategy` enum | `set_strategy(kind: str, …)` / `strategy() -> str` | ❌ Lossy round trip — [D-55](/soma/internals/debt/#d-55--set_strategy--strategy-is-a-lossy-round-trip) |
| Study | `PyStudy` (`subclass`) | `class Study(_Study)` adding 8 plot methods | ✅ Mirrored for plotting only |

---

## Patterns in use

- **Bridge / adapter ×4** — `PyFilterBridge`, `PyStepBridge`, `PyToolAdapter`, `PyPbtExecutor`. → [Patterns](/soma/internals/patterns/#adapter--bridge)
- **Mixin assembly in a class body** — `_graph.py:35`, replacing runtime monkey-patching. The docstring is an explicit anti-monkey-patch argument.
- **Descriptor protocol** — `SearchDescriptor.__set_name__` / `__get__` / `__set__` (`search.py:55`).
- **Metaclass registry** — `FilterMeta` collects `SearchDescriptor`s into `_soma_search_space` (`filter.py:9`).
- **Operator DSL** — `Filter.__rshift__` / `__or__` (`filter.py:172`, `:181`), `Chain` / `Fork` (`chain.py:36`, `:85`), `builder.somatize` (`builder.py:11`).
- **Data-as-JSON-string across the FFI** — the 25 `*_json` functions.
- **Facade + lazy view** — `RunView` (`_runs.py:30`) with cached properties and `refresh()`.
- **Rich-repr protocol** — five `_repr_html_` implementations: `PyGraph` (`graph.rs:1856`), `RunView` (`_runs.py:260`), `RunList` (`:386`), `CompileInfo` (`_compile.py:26`), `DifferentiableFilter` (`_composite.py:179`).
- **Context manager** — `track_run` (`_tracking.py:36`), `Graph.context` (`_orchestrator.py:461`), `audit_modules` (`_audit.py:1016`), `gradient_audit` (`_audit.py:1239`).
- **Deferred configuration** — `StoreConfig` (`worker.rs:72`) held until `serve` can build the store on its own thread.
- **Null-object degradation** on missing optional dependencies.
- **Exception hierarchy under one base** — all four Rust-defined exceptions derive `RuntimeError` (`lib.rs:92`).

## Debt

**High** — [D-01](/soma/internals/debt/#d-01--pygraph-is-the-workspaces-god-object) `PyGraph` god object, including the 262-line `fit` with a five-times-duplicated tail

**Medium** — [D-09](/soma/internals/debt/#d-09--audit-is-a-30-method-class-in-a-1-338-line-module) `Audit` ·
[D-27](/soma/internals/debt/#d-27--unwrap-inside-a-detached-thread-makes-bind-failures-unreportable) `unwrap` in a detached thread ·
[D-54](/soma/internals/debt/#d-54--nine-string-match-dispatch-sites-across-the-ffi) nine string-match dispatch sites ·
[D-57](/soma/internals/debt/#d-57--prose-parsing-as-control-flow) prose as control flow ·
[D-16](/soma/internals/debt/#d-16--two-knowledge-base-front-ends-already-divergent) duplicated KB front-ends

**Low** — [D-15](/soma/internals/debt/#d-15--five-formatters-for-a-duration-two-for-a-truncation) three Python duration formatters ·
[D-37](/soma/internals/debt/#d-37--dead-helper-in-the-python-bindings) `split_value_into_batches` dead ·
[D-47](/soma/internals/debt/#d-47--a-cross-crate-contract-carried-by-an-environment-variable) `SOMA_LOCAL_PACKAGE` ·
[D-55](/soma/internals/debt/#d-55--set_strategy--strategy-is-a-lossy-round-trip) lossy strategy round trip

Plus two Python-specific observations not severe enough for their own entries:
`eprintln!` is used as the logging strategy in `soma-python/src/distributed.rs` (7
occurrences) and `run.rs` (4), where the rest of the workspace uses `tracing` and
these go to stderr uncontrollably from Python; and 9 of the workspace's 10
`#[allow(...)]` live in this crate, all `clippy::too_many_arguments` on PyO3
keyword constructors — a structural consequence of the binding style rather than
a smell.

## Tests

699 Python tests (14 deselected by default: slow + live). No hypothesis — the
property tests are Rust-side in `soma-core/tests/proptests.rs`.

```bash
cd soma-python && maturin develop && pytest tests/     # the fast set
cd soma-python && pytest tests/ -m slow                # SIGKILL crash-sim, statistical TPE
cd soma-python && SOMA_LIVE=1 pytest tests/ -m live    # real endpoints
cd soma-python && mypy                                 # the package ships py.typed
```
