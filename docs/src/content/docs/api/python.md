---
title: Python API
description: Python API reference for the Soma package.
---

## Installation

```bash
pip install somatize            # core
pip install 'somatize[viz]'     # + plotly, pandas, rich, tqdm
```

The `viz` extra powers figures, DataFrames, HTML reports, the colored
CLI tables and `progress=` bars. The core install never requires it;
calls that need it raise a message telling you to add it.

New to Soma? Start with the [Quickstart](/soma/getting-started/quickstart/)
or the [tutorial notebooks](/soma/getting-started/notebooks/).

## Core Classes

### Filter

Base class for computational nodes. Subclass to define custom transformations.

```python
from soma import Filter, search

class MyScaler(Filter):
    scale: float = search(0.1, 10.0, scale="log")
    method: str = search(choices=["standard", "robust"])

    def fit(self, x, y=None):
        """Learn state from training data. Returns state dict."""
        return {"mean": sum(x) / len(x)}

    def forward(self, x, state):
        """Transform data using learned state. Returns transformed data."""
        return [(v - state["mean"]) * self.scale for v in x]
```

#### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `fit` | `(x, y=None) -> dict` | Learn internal state from training data |
| `forward` | `(x, state) -> list` | Transform data using learned state |
| `kwargs` | `() -> dict` | Constructor kwargs (used by `Graph.save`/`Graph.load`) |
| `class_path` | `() -> str` | Class method. Fully-qualified import path (`"module.Class"`) |
| `to` | `(other) -> Chain` | Chain this filter to another (fluent builder) |
| `>>` | `filter >> other` | Chain operator (same as `.to()`) |
| `\|` | `filter \| other` | Fork operator (parallel branches) |

#### Class attributes

Behavior is declared with class attributes — all optional:

| Attribute | Default | Effect |
|---|---|---|
| `_kind` | `"trainable"` | `"stateless"` skips the fit phase |
| `_cacheable` | `True` | `False` never caches this filter's outputs |
| `_differentiable` | `False` | `True` lets the compiler fuse it into a gradient-flowing block |
| `_deterministic` | `True` | `False` marks a stochastic forward (not cached unless the run pins a `seed`) |
| `_stream_mode` | `"fixed"` | `"evolving"` (checkpointed state) or `"barrier"` (needs the whole stream) |
| `_cache_version` | unset | Pins the code identity for cache keys — **set it for filters defined in notebooks/REPLs**, where the source is unavailable |
| `_audit_scope` | unset | Pre-selects submodules for `gradient_audit(inside=True)` (differentiable filters) |
| `class_version` | `1` | Bump when constructor kwargs or saved-state layout change; `Graph.load` warns on mismatch |

Attributes prefixed with `_` never enter the cache key. For an
unhashable public attribute, either prefix it or define
`__soma_config__() -> dict`.

#### Search Descriptors

Use `search()` to define hyperparameter search spaces:

```python
scale: float = search(0.1, 10.0, scale="log")      # Float range
epochs: int = search(10, 100)                        # Integer range
method: str = search(choices=["a", "b", "c"])        # Categorical
```

### Graph

The primary API for Soma. A computational DAG of filter nodes.

#### Construction

```python
from soma import Graph, Filter

class Scaler(Filter):
    def forward(self, x, state):
        return [v * 2 for v in x]

class Model(Filter):
    def fit(self, x, y=None):
        return {"w": 1.0}
    def forward(self, x, state):
        return [v * state["w"] for v in x]

# Method 1: Fluent builder with Graph.somatize()
g = Graph.somatize(Scaler() >> Model())

# Method 2: Manual construction
g = Graph()
g.node(Scaler())
g.node(Model())
g.connect("scaler", "model")
```

By default every `Graph()` shares a **persistent cache** at
`$SOMA_CACHE_DIR` (or `~/.soma/cache`): fit states and forward outputs
survive crashes and are reused across processes and projects. Options:

```python
g = Graph()                                   # persistent tiered cache (default)
g = Graph(cache="memory")                     # process-local, nothing persists
g = Graph(cache="local", cache_path="/data")  # explicit store directory
g = Graph(cache_max_bytes=2 * 2**30)          # in-memory LRU tier budget
```

#### Fluent Operators

```python
# >> chains filters linearly
g = Graph.somatize(Scaler() >> PCA() >> Model())

# | creates parallel branches
g = Graph.somatize(
    Scaler() >> (HeadA() | HeadB()) >> Ensemble()
)

# Nested branches with long chains
g = Graph.somatize(
    (LoadA() >> NormA() | LoadB() >> NormB())
    >> Aggregate()
    >> Backbone()
    >> (ClassA() | ClassB())
)

# .to() / .collect() method syntax
g = Graph.somatize(
    Scaler().to([
        PCA() >> ClassA(),
        UMAP() >> ClassB(),
    ]).collect(Ensemble())
)
```

#### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `somatize` | `(topology) -> Graph` | Class method. Materialize a Chain/Fork into a graph |
| `node` | `(filter, target=None) -> str` or `(node_id, filter)` | Add a filter node, returns its id (snake_case class name, deduped with `_2`). `target="local"` pins it off remote workers |
| `edge` / `connect` | `(source, target)` | Connect two nodes with a data edge |
| `fit` | `(x, y=None, batch_size=None, mode="inference", seed=None)` | Fit all trainable filters in topological order; `seed` is hashed into every cache key |
| `forward` | `(x, stream=False, chunk_size=1024, seed=None)` | Forward data through the fitted graph (`stream=True` chunks it). Returns a list for pure-inference graphs; `(out, aux_by_node)` while any differentiable filter is in `train()` mode |
| `compile` | `(mode="inference") -> CompileInfo` | Compile and return diagnostics (a dict that renders as tiles + callouts + plan diagram in notebooks) |
| `to_mermaid` | `(overlay=None) -> str` | Mermaid diagram; `overlay=` annotates nodes (see [Visualization](/soma/design/visualization/)) |
| `to_svg` | `(overlay=None) -> str` | Self-contained SVG — no JavaScript, renders inline anywhere |
| `to_text` | `() -> str` | ASCII tree (what `print(g)` shows) |
| `_repr_html_` | `() -> str` | Notebook display: evaluating `g` draws the architecture diagram |
| `on_event` | `(callback)` | Register event callback (background thread) |
| `materialize` | `(sample_input)` | Build every `DifferentiableFilter._module` once, threading shapes |
| `train` / `eval` | `()` | Flip `training` on every live filter and its `_module` |
| `to` | `(device, *, dtype=None) -> Graph` | Move every materialised filter `_module` to `device`/`dtype`; target persists so lazy-built modules inherit it |
| `parameters` | `() -> Iterator[Parameter]` | Topological iterator over all materialised filter parameters |
| `make_optimizer` | `(cls=Adam, **kw)` | Build + register an optimiser over `g.parameters()` |
| `set_optimizer` | `(opt)` | Register an externally-built optimiser |
| `context` | `() -> ctx` | Autograd context (no-op locally; `dist.autograd.context()` under RPC) |
| `backward` | `(ctx, loss)` | Local `loss.backward()`; RPC `dist.autograd.backward(ctx, [loss])` |
| `step` | `(ctx=None)` | Local `opt.step()`; RPC `DistributedOptimizer.step(ctx)` |
| `zero_grad` | `(set_to_none=True)` | Wrapper over registered optimiser; silent no-op before `make_optimizer` |
| `freeze` | `()` | Snapshot every live `_module.state_dict()` into runtime state, switch to eval |
| `state` | `() -> dict[node_id, state]` | Snapshot per-node runtime state |
| `load_state` | `(sd, strict=True)` | Apply a state dict; `strict=False` warns on missing/unknown keys |
| `save` | `(path, include_optimizer=False)` | Persist full graph (manifest + safetensors + JSON) to a zip bundle |
| `load` | `(path, strict=True)` | Class method. Rebuild topology + restore state from a checkpoint |
| `restore_optimizer` | `() -> bool` | Apply a pending optimiser snapshot bundled by `save(include_optimizer=True)` |
| `edges` | `() -> list[(src, tgt)]` | Data edges in insertion order (used by `save`) |
| `get_node_state` / `set_node_state` | `(node_id [, state])` | Low-level state accessor used by `state` / `load_state` |
| `gradient_audit` | `(thresholds=None, channels=None, inside=None) -> ctx[Audit]` | Install forward/backward hooks for a training pass. `channels=` adds per-channel diagnostics, `inside=` audits submodules *within* each node — see [the guide](/soma/guides/gradient-audit/) |
| `add_worker` | `(address, token?, tags?)` | Add a remote worker |
| `set_coordinator` | `(url, token?)` | Set coordinator for auto-discovery |
| `workers` | `() -> list[dict]` | List known workers |
| `track_run` | `(name, *, root=".soma", kind="train", tags=(), params=None, parent=None, hypothesis=None) -> ctx[Run]` | Context manager: create a run directory, snapshot the graph into it, finalize on exit (even on exception). `params` are the hyperparameters that live outside the graph; `parent` overrides the run this one descends from; `hypothesis` records what you expected *before* seeing the result |
| `begin_run` | `(name, root=".soma", kind="train", tags=None) -> Run` | Lower-level: start a run without the context manager (you call `run.finish()`) |
| `emit_event` | `(dict)` | Emit a custom event onto the bus (must match an `Event` variant) |
| `search_space` | `() -> list[dict]` | Aggregate every searchable dimension — filters' `search()` descriptors, agents'/judges' `search()`-valued constructor args, and each `optional()` edge as a boolean dimension — prefixed with the node id (`"encoder.lr"`, `"edge:a->b"`) |
| `apply_params` | `(params)` | Write a sampled configuration back onto the live filters/agents, and keep or cut optional edges |
| `study` | `(name, **kwargs) -> Study` | A `Study` over this graph's search space |
| `optional` | `(source, target)` | Make an existing edge part of the search space: a study may keep it or cut it |
| `optional_edges` | `() -> list[(src, tgt)]` | The edges a study is allowed to cut |
| `set_edge` | `(source, target, enabled)` | Keep or cut one optional edge, restoring it whole and in place |
| `branch` | `(node_id, condition, arms, target=None) -> str` | Add a routing node: runs `condition` and executes only the arm its output names |
| `loop` | `(node_id, body, until=None, max_iterations=None) -> str` | Repeat a body until it signals completion; `until=False` runs the full count |
| `handoff` | `(source, target)` | Declare that `source` may hand control to `target` — what a step's `Goto` needs |
| `register_step` | `(step_id, obj) -> str` | Register a spawn target *without* adding a node (see [the agentic layer](#the-agentic-layer)) |
| `register_graph` | `(sub)` | Make `sub`'s node implementations runnable by this graph's steps (`soma.agentic.RunGraph`) |
| `resume` | `(run_id, node_id, turn, reason, answer)` | Answer what a suspended run was waiting for; every argument comes off `SomaSuspended` |
| `use_provider` | `(provider)` | Set the provider that serves model names given without a prefix |
| `add_tool` | `(tool)` | Register a tool without attaching it to a particular agent |
| `add_mcp_server` | `(command, args=None) -> list[str]` | Start an MCP server and make its tools callable; returns the tool names found |
| `steps` | `() -> list[(node_id, obj)]` | The live object behind each step node, sorted by id |
| `filters` | `() -> list[(node_id, filter)]` | Live filter instances in topological order |
| `filter` / `filter_ids` | `(node_id)` / `()` | One filter instance / all node ids |
| `graph_json` | `() -> str` | Serialized topology (what `graph.json` holds in a run directory) |
| `optimizer` | `() -> Optimizer \| None` | The registered optimiser, if any |
| `py_state` | property `-> dict` | Python-side scratch space (`active_run`, `active_audit`, optimiser) |

#### Compile Modes

```python
info = g.compile("inference")       # Full caching
info = g.compile("differentiable")  # Cache states, re-execute forwards
info = g.compile("no_cache")        # Force re-execution
# Returns CompileInfo (a dict): {total_nodes, cached_nodes, parallel_branches,
#   diagnostics: [{node, level, message}], plan_text, plan_mermaid, plan_svg}
# cached_nodes is always 0: cache hits are resolved at RUNTIME per node
# (key = hash(config + state + input)), not baked into the plan.
```

#### Events

```python
def on_event(event):
    print(event["event_type"], event.get("node_id", ""))

g.on_event(on_event)
g.fit(data)
# Events: NodeStarted, NodeCompleted, NodeCacheHit, NodeCacheMiss, NodeFailed, ...
```

#### Workers

```python
# Mode B: Direct workers
g.add_worker("ws://gpu-0:8080", token="sk-xxx", tags=["gpu"])
g.add_worker("ws://cpu-0:8080", tags=["cpu"])

# Mode C: Coordinator auto-discovery
g.set_coordinator("http://coord:9090", token="sk-xxx")

# List all workers
for w in g.workers():
    print(w["address"], w["tags"])
```

## The agentic layer

An agentic flow is an ordinary `Graph` whose nodes are *effectful steps*
instead of (or alongside) filters. There is no second engine: the same
compiler, cache, events, search spaces and tracking apply. This section
is the reference; the rationale is in
[Agentic Graphs](/soma/design/agentic/), and there are two task-shaped
guides: the [agentic quickstart](/soma/guides/agentic-quickstart/) and
[writing a step](/soma/guides/writing-a-step/).

### Agent

A ReAct agent: ask a model, run the tools it asks for, repeat. Used as a
graph node like any filter.

```python
soma.Agent(
    model,                 # "provider/name", or bare with g.use_provider()
    system=None,           # system prompt
    tools=None,            # list[soma.Tool]
    max_turns=None,        # turn budget for the reason-act loop
    max_tokens=None,
    effort=None,           # reasoning depth, in the provider's terms
    text_only=True,        # reply must be prose (tool calls still allowed)
    schema=None,           # JSON Schema the reply must satisfy
    max_repairs=1,         # violations that buy a correction round
)
```

`model`, `system`, `max_turns`, `max_tokens` and `effort` all accept a
`search()` descriptor: an agent's constructor arguments are its
hyperparameters, so the space is declared where the value goes and lands
in the same `g.search_space()` a filter contributes to.

```python
g.node("writer", soma.Agent(
    model=soma.search(choices=["ollama/qwen2.5", "kimi/kimi-k2"]),
    system=soma.search(choices=["be terse", "be detailed"]),
))
```

`schema=` asks the endpoint to enforce the shape when it supports
constrained decoding (`response_format`) and asks in the system prompt
when it does not; either way the reply is checked structurally, and one
violation buys one correction with the violation quoted back
(`max_repairs=` raises the ceiling). Truncated (`finish_reason: length`)
and refused replies are **errors**, never returned as answers.

Attributes: `model`, `system`, `max_turns`, `max_tokens`, `effort`,
`schema`, `tools` (a fresh list each time), plus
`search_space() -> list[dict]`.

### Judge

Grade something with a model against a rubric. Returns a mapping with a
`done` key, which is what `Graph.loop(until=...)` and `refine()` read.

```python
soma.Judge(model, rubric, threshold=None)
```

All three arguments accept `search()` — how strictly to grade is a real
thing to tune, and so is the wording of the rubric.

### Tool and `@soma.tool`

A tool backed by a Python callable. Without an explicit `schema`, one is
derived from the function's signature; a lambda with neither name nor
description raises `ValueError` at construction.

```python
soma.Tool(func, description=None, name=None, schema=None)

@soma.tool                       # or @soma.tool(description=...)
def lookup(city: str) -> str:
    """Current weather for a city."""
    ...

g.node("agent", soma.Agent(model="ollama/qwen2.5", tools=[lookup]))
g.add_tool(lookup)               # or: graph-wide, not bound to one agent
g.add_mcp_server("uvx", ["some-mcp-server"])   # MCP tools, by command
```

### Writing a step

A step is any object with `poll(ctx)`, duck-typed exactly like a
filter's `forward`. Each `poll` advances one turn and returns one of five
transitions — plain dicts built by helpers from `soma.agentic`:

```python
from soma.agentic import Done, Await, Spawn, Goto, Suspend, Llm, ToolCall, Run

class Fanout:
    _cache_version = "1"          # steps need a code identity, like filters

    def poll(self, ctx):
        if ctx.turn == 0:
            return Spawn([Run("worker", task) for task in ctx.input])
        return Done([r["output"] for r in ctx.results])
```

| Transition | Signature | Meaning |
|---|---|---|
| `Done` | `(value=None)` | Finished, with this output |
| `Await` | `(*effects)` | Perform these effects concurrently, then poll again with the results in `ctx.results`, in order |
| `Spawn` | `(specs, join="all")` | Run these nodes now (`Run(runs, input, label)` each), then poll again with their outputs. `join` is `"all"` \| `"all_settled"` \| `"first"` |
| `Goto` | `(target, carry=None)` | Hand control to another node (needs `g.handoff(source, target)` declared) |
| `Suspend` | `(reason="waiting")` | Stop and persist the run; resuming replays to here |

Effects to `Await`: `Llm(model, prompt, system=None)`,
`ToolCall(name, args=None)`, `RunGraph(graph, input=None, mode="forward")`,
`Sleep(seconds)`, `Custom(kind, payload=None)`. Results arrive as dicts
with a `"kind"` key (`"llm"` carries `text`; `"tool"`, `"graph"`,
`"node"` and `"custom"` carry `output`; `"failed"` carries `message`;
all carry `is_error`).

`poll` must be **deterministic given the same context**: the journal
records each effect's result once and replays it on resume, so a
deterministic step retakes the identical path. Put anything that is not
deterministic in an effect. Accordingly a step keeps no state of its
own — it rebuilds what it has accumulated from `ctx.history`.

#### StepCtx

What `poll` is handed. Read-only.

| Attribute | Type | Meaning |
|---|---|---|
| `node_id` / `run_id` | `str` | Where and in which run this is happening |
| `input` | `Any` | Input resolved from predecessors |
| `turn` | `int` | Which turn, counting from 0 |
| `results` | `list[dict]` | What last turn's effects returned, in request order |
| `history` | `list[list[dict]]` | Every turn's results, oldest first |
| `result()` | `dict \| None` | The single result of a one-effect turn; `None` on turn 0 |

#### `register_step` and `handoff`

`g.register_step(step_id, obj)` registers a spawn target **without**
adding a node — a node with no incoming edges is a root and would also
run once on the graph's own input. `g.handoff(a, b)` declares the
control edge a `Goto` needs; handing control somewhere the graph never
said it could is an error, not a silent jump.

#### A pipeline as a tool: `RunGraph` + `register_graph`

```python
sub = soma.Graph()
sub.node("featurize", Featurize())

class Planner:
    _cache_version = "1"
    def __init__(self, sub):
        self._sub = sub           # underscored — see below
    def poll(self, ctx):
        if ctx.turn == 0:
            return Await(RunGraph(self._sub, input=ctx.input))
        return Done(ctx.result()["output"])

g.node("planner", Planner(sub))
g.register_graph(sub)             # make sub's implementations runnable here
```

The effect carries the sub-graph's structure; `g.register_graph(sub)`
merges its implementations into the outer graph (same id under a
different implementation is a `ValueError`). `mode="fit"` fits the
sub-graph instead of running forward. A pipeline that fails comes back as
`{"kind": "failed", "message": ...}` — a finding for the step to read,
not a crash. Sub-graphs may themselves contain steps
(agent → pipeline → agent), capped at nesting depth 8.

One trap: store a live `Graph` (or any unhashable object) on a step
under an **underscored** attribute. Public attributes enter the step's
config identity, and a live graph cannot be hashed into it — the journal
keys the effect by the graph's own content, so nothing is lost.

#### Suspend and resume

A `Suspend` transition raises `soma.SomaSuspended` from
`fit`/`forward` — a pause, not a failure. The exception carries
`run_id`, `node_id`, `turn`, `kind` and `reason`, which is exactly what
`Graph.resume(...)` takes; after `resume`, re-running under the same
`run_id` replays the journal to the suspension point and continues with
the answer.

```python
try:
    g.forward(x, run_id="review-1")
except soma.SomaSuspended as s:
    answer = input(s.reason["prompt"])
    g.resume(s.run_id, s.node_id, s.turn, s.reason, answer)
    result = g.forward(x, run_id="review-1")   # replays, then continues
```

#### Schemas on Python nodes

Python filters and steps may declare `_input_schema` / `_output_schema`
class attributes — the shorthand strings `"text"`, `"json"`,
`"messages"`, `"bytes"`, or a `{dtype, shape}` mapping. Declared schemas
make the compiler's edge check fire from Python: two connected nodes
that disagree raise `soma.SomaSchemaMismatch` at `compile()`, not
mid-run.

```python
class Summarize:
    _cache_version = "1"
    _input_schema = "text"
    _output_schema = "json"
    def poll(self, ctx): ...
```

### `soma.agentic`: the patterns

Every pattern is a function returning an ordinary `Graph` — searchable,
cacheable, trackable. All take `provider=` and `cache=` keywords.

| Function | Signature | Shape |
|---|---|---|
| `react` | `(model, tools=(), *, system=None, max_turns=None)` | One agent that thinks, calls tools, and answers |
| `route` | `(classifier, arms)` | Send each request to exactly one handler; the chosen arm receives the original request, not the label |
| `refine` | `(worker, judge, *, max_rounds=3, revise=None)` | Draft, grade, redraft until the judge reports `done` |
| `debate` | `(agents, *, rounds=2, judge=None)` | Agents answer in turn, each seeing what came before |
| `board` | `(members, chair=None, *, rounds=2, brief=None)` | Du et al. multi-agent debate: panel answers, chair moderates, panel answers again; unanimity stops early |
| `parallel_vote` | `(agents, aggregator)` | Several agents at once, then reconcile |
| `self_consistency` | `(agent, *, n=5, aggregator=None)` | One agent sampled `n` times; majority wins |
| `orchestrate` | `(planner, worker, synthesizer, *, max_workers=16)` | Planner → dynamic fan-out (`Spawn`) → synthesizer; the pool is sized from the plan |

Plus the filters the patterns are made of: `Validate(schema, strict=False)`
(a JSON-Schema verdict a `branch` routes on — note it lives in
`soma.agentic`, not `soma.library`), `Revise`, `Brief`,
`MajorityVote(brief_node="brief", mode="number"|"text")`, `Fanout`.

`soma.library` holds the evaluation side: `Eval`, `Accumulator`,
`Retriever`, `Compact` (see the [design page](/soma/design/agentic/#the-library-the-patterns-are-made-of)).

### DifferentiableFilter

Filter base class for trainable `nn.Module` wrappers. Available when
`torch` is installed; `None` otherwise. A materialized instance
displays as an architecture diagram in notebooks (submodules with
their parameter counts), and may declare `_audit_scope` to pre-select
which of its submodules [`gradient_audit(inside=True)`](/soma/guides/gradient-audit/)
should watch.

```python
from soma import Graph, DifferentiableFilter
import torch, torch.nn as nn

class Dense(DifferentiableFilter):
    def __init__(self, out_dim, lr=1e-3):
        super().__init__(out_dim=out_dim, lr=lr)
    def build_module(self, input_shape):
        return nn.Linear(input_shape[-1], self.out_dim)
    def output_shape(self, input_shape):
        return (self.out_dim,)

g = Graph.somatize(Dense(8) >> Dense(2))
g.materialize(sample_x)
g.train()
g.make_optimizer(torch.optim.Adam, lr=1e-3)
for x, y in batches:
    with g.context() as ctx:
        g.zero_grad()
        out, aux = g.forward(x)
        loss = nn.functional.mse_loss(out, y)
        g.backward(ctx, loss)
    g.step(ctx)
g.freeze(); g.eval()
preds = g.forward(x_test)
```

#### Subclass hooks

| Hook | Signature | Purpose |
|---|---|---|
| `build_module` | `(input_shape) -> nn.Module` | Construct the trainable module (called once) |
| `output_shape` | `(input_shape) -> tuple` | Forward shape so cascade-materialise can size successors |
| `forward` | `(x, state=None) -> (out, aux_dict)` | Provided by base; override to surface aux signals (gates etc.) |
| `compute_loss` | `(output, y, aux=None) -> tensor` | Default MSE; override for BCE/CE/custom |
| `make_optimizer` | `(modules) -> Optimizer` | Default `Adam(lr=self.lr)`; override for per-filter LRs |

`forward(x, state=None)` is **polymorphic on `self.training`**: when
training, the `state` argument is ignored and the filter runs the live
`_module` with autograd; when not training, it loads
`state["weights_b64"]` if present, runs `no_grad`, and returns lists.
Always returns `(out, aux_dict)`.

See the [gradients design doc](/soma/design/gradients/#native-training-loop-python)
for the full training-loop pattern and RPC-ready notes.

### Study

Hyperparameter optimization study.

```python
from soma import Study

study = Study(
    name="my_study",
    search_space=[
        {"type": "float", "name": "lr", "low": 0.001, "high": 0.1, "scale": "log"},
        {"type": "categorical", "name": "kernel", "choices": ["rbf", "linear"]},
    ],
    strategy="bayesian",    # "grid", "random", or "bayesian"
    n_trials=50,
    objectives=[("f1", "maximize")],
    seed=42,
)

def train(trial):
    """One trial. `trial` behaves like a params mapping and adds
    report()/should_prune() for pruning-aware loops."""
    g = Graph.somatize(Scaler() >> Model(lr=trial["lr"]))
    g.fit(train_data)
    for epoch in range(20):
        f1 = evaluate(g)
        if trial.report("f1", f1, step=epoch):
            return None                    # pruned by the median rule
    return {"f1": f1}

study.run(train, on_event=lambda e: print(e["event_type"]))
print(study.best_trial)     # {"id": "...", "params": {...}, "metrics": {...}}
print(study.trials)         # every trial as a dict
print(study.run_dir)        # .soma/runs/study_.../  (tracking=True default)
```

Additional constructor keywords: `objective=` (a Python callable over
the final metrics dict, recorded as metric `"score"`), `direction=`,
`pruning=("median", warmup)` or `("percentile", pct, warmup)`,
`tracking=`, `root=".soma"`, `tags=[...]`, `frozen={...}` (fixed params
injected into every trial), and `seeds=[...]` — **experiment seeds**:
every sampled config runs once per seed, `trial["seed"]` carries it
(wire it into your framework: `torch.manual_seed(trial["seed"])`), the
manifest records them, and each (config, seed) pair is an independent,
resumable trial with its own cache line. Legacy `fn(params) -> dict`
executors keep working (the trial handle supports `params.get(...)` /
`params["x"]`), and a bare-float return becomes the `"score"` metric.

Studies persist to their run directory after every trial and can be
followed or continued from anywhere:

```python
study = soma.Study.load(".soma/runs/study_20260726T101502_a3f1")
print(study.progress, len(study.trials))
study.run(train, resume=True)   # continues at trial N, no repeats
```

Graph-level search spaces come from `search()` descriptors on filters:

```python
space = g.search_space()        # dims named "<node_id>.<param>"
g.apply_params(trial.params)    # write a sampled config onto filters
study = g.study("tune", strategy="grid", n_trials=4,
                objectives=[("f1", "maximize")])
```

#### Study members

| Member | Signature | Description |
|---|---|---|
| `run` | `(executor, on_event=None, resume=False, progress=False)` | Run the study. `progress=True` draws a tqdm bar fed by live `StudyProgress` events (needs the `viz` extra) |
| `load` | `(run_dir, objective=None) -> Study` | Static. Reload a study from its run directory |
| `save` | `(path=None)` | Write `study.json` (also written automatically after every trial) |
| `best_trial` | property `-> dict \| None` | Best trial as a dict (see below) |
| `trials` | property `-> list[dict]` | Every trial |
| `n_trials` | property `-> int` | Trials recorded so far |
| `progress` | property `-> float` | Completed / planned |
| `objectives` | property `-> list[(metric, direction)]` | Declared objectives (a composite objective reports as `("score", direction)`) |
| `name` | property `-> str` | Study name |
| `run_dir` | property `-> str \| None` | Run directory, or `None` when `tracking=False` |

Each trial dict carries `id`, `params`, `state`
(`completed`/`pruned`/`failed`/`running`/`pending`), `metrics` (last
value per name), `series` (**every** reported `{name, value, step,
timestamp}` — the learning curve), `started_at`, `finished_at` and
`duration_ms`.

#### Trial handle

Inside your executor, `trial` exposes `params`, `id`, `report(name,
value, step) -> bool` (True ⇒ the pruner says stop), `should_prune()`,
plus dict-style access: `trial["lr"]`, `trial.get("lr", default)`,
`"lr" in trial`, `trial.keys()`.

#### Figures

With the `viz` extra, `Study` gains (Optuna-aligned names):
`plot_optimization_history`, `plot_intermediate_values`,
`plot_parallel_coordinate`, `plot_param_importances`, `plot_timeline`,
`plot_pareto_front`, `trials_dataframe()`, and
`to_html(path=None, inline=False)`. See
[Visualization](/soma/design/visualization/).

### Cache management

```python
from soma import _soma
_soma.cache_stats()                    # dict: records, blobs, bytes, compute banked
_soma.cache_gc(max_bytes=20 * 2**30)   # evict low-value blobs (records retained)
```

Or from the shell (every subcommand takes `--dir` to point at a cache
other than `$SOMA_CACHE_DIR`):

```console
$ soma cache stats
$ soma cache gc --max-size 20G [--min-age 3600]
$ soma cache pin best-run <action-key-hex>
$ soma cache verify
$ soma cache purge-v1
```

`CacheConfigError` (importable from `soma`) is raised when a filter
attribute cannot enter the cache key — prefix it with `_` or define
`__soma_config__()`. Set `_cache_version = "..."` on a filter class to
pin its code identity explicitly; `_deterministic = False` marks a
stochastic forward (never cached unless the run pins a `seed`).

### Tracked runs

```python
with g.track_run("baseline", tags=["mos"]) as run:
    with g.gradient_audit(channels=True) as audit:
        ...  # native training loop
        run.log("val_f1", 0.85, step=epoch)
print(soma.experiments())       # journal of completed runs/studies
```

`track_run` writes `.soma/runs/<run_id>/` — manifest, heartbeat status,
graph topology, lossless `events.jsonl`/`metrics.jsonl`, and (with the
audit) `diagnostics/` including per-channel safetensors snapshots. See
the [Experiment Tracking](/soma/design/tracking/) design page for the full
layout.

#### Run

The handle `track_run` (or `begin_run`) yields:

| Method | Signature | Description |
|---|---|---|
| `log` | `(name, value, step=None, node=None)` | Record a metric; also updates the run's summary |
| `log_epoch` | `(epoch, total=None)` | Epoch marker + heartbeat |
| `log_epoch_completed` | `(epoch, metrics=None)` | Epoch-end metrics + heartbeat |
| `step_completed` | `(step, epoch=None)` | Optimizer-step marker (liveness) |
| `heartbeat` | `()` | Refresh liveness so readers don't call the run crashed |
| `finish` | `(status="completed")` | Finalize; `track_run` calls it for you (`"failed"` on exception). On success it also appends the run to `<root>/experiments.jsonl` and advances `.soma/HEAD` |
| `id` / `dir` | properties | Run id and absolute run directory |

### Reading runs back

Readers never write, so they work on live, finished and crashed runs
alike. `soma.runs()` returns a list that renders as a table in
notebooks; each entry is a `RunView`.

```python
import soma

soma.runs()                       # newest first; stale heartbeat ⇒ "crashed"
view = soma.runs()[0]
view = soma.RunView(".soma/runs/train_20260728T093011_9c2e")   # or by path
```

| Member | Returns | Description |
|---|---|---|
| `id` / `name` / `kind` / `state` / `dir` | `str` | Identity; `state` is `running`/`completed`/`failed`/`crashed` |
| `info` | `dict` | The listing entry (adds `created_at`, `duration_ms`, `tags`) |
| `refresh()` | `RunView` | Re-read `status.json` for a live run |
| `manifest()` | `dict` | Environment, git, seeds, graph summary |
| `events()` | `list[dict]` | Every envelope `{seq, ts, event_type, …}`; torn/unknown lines skipped |
| `metric_series(name=None)` | `list[dict]` | `{ts, name, value, step, trial_id, node_id}` |
| `node_timings()` | `list[dict]` | Per-node spans: wall times, duration, outcome, cache tier |
| `cache_activity()` | `dict` | `{hits, misses, by_node}` |
| `health_flags()` | `list[dict]` | `HealthFlag` events with wall time |
| `trial_timeline()` | `list[dict]` | Trial lifetimes (study runs) |
| `overlay()` | `dict` | Per-node annotations for the renderers |
| `to_mermaid(overlay=True, node=None)` | `str` | Annotated diagram; `node=` renders that node's inner architecture |
| `to_svg(overlay=True, node=None)` | `str` | Self-contained SVG (no JavaScript) |

With the `viz` extra it also gains `plot_metrics`, `plot_gantt`,
`plot_health`, `plot_audit`, `plot_module_flow`, `plot_channels`,
`plot_channel_evolution`, `metrics_dataframe()` and
`to_html(path=None, inline=False)`.

```python
soma.experiments()              # the flat journal of completed runs/studies
soma.experiments_dataframe()    # ...as a DataFrame (viz extra)
```

### Lineage and the experiment pool

Runs descend from one another. The parent is resolved as `parent=` →
`$SOMA_PARENT_RUN` → `.soma/HEAD` → none, and HEAD advances after every
*successful* run. See [Experiment Pool](/soma/design/experiment-pool/).

| Function | Signature | Description |
|---|---|---|
| `soma.head` | `(*, root=".soma") -> str \| None` | Which run the next one will descend from |
| `soma.checkout` | `(run_id, *, root=".soma")` | Point HEAD at an existing run so the next one branches from it |
| `soma.detach` | `(*, root=".soma")` | Clear HEAD; the next run starts its own research line |
| `soma.reindex` | `(*, root=".soma") -> int` | Rebuild `experiments.jsonl` from `<root>/runs/` |
| `soma.find_similar` | `(query="", *, like_run=None, limit=5, research_line=None, tags=None, half_life_days=None, root=".soma") -> list[dict]` | Rank past experiments: `0.40·text + 0.25·architecture + 0.15·recency + 0.20·importance`. Each hit has `score`, `why`, `components`, `record` |
| `soma.record_conclusion` | `(run_id, notes, *, hypothesis=None, tags=None, root=".soma") -> str` | Retain what you learned, as an append-only amendment. Returns its id |
| `soma.lineage` | `(run_id, *, root=".soma") -> dict \| None` | `focus` + `ancestors` + `descendants` (pre-order, with depth) |
| `soma.diff` | `(a, b, *, root=".soma") -> dict` | The move between any two experiments — including siblings, which have no recorded edge |

```python
soma.find_similar("dropout collapse", limit=3)
# [{'score': 0.81, 'why': 'score 0.81 (text 0.94, structure 0.00, …)', ...}]

soma.record_conclusion(run_id, "depth past 3 vanishes; needs residuals")
```

Failures rank: `importance` is floored for any run that failed, crashed
or regressed **and** carries a conclusion. Not repeating a dead end
saves as much as repeating a win — which is also why
`record_conclusion` is worth calling on the runs that did not work.

### Command line

```console
$ soma runs [--root .soma] [--json] [--plain]
$ soma graph <run_id|path> [--format mermaid|dot] [--no-overlay] [--root .soma]
$ soma report <run_id|path> [-o FILE] [--inline] [--open] [--root .soma]
$ soma cache stats|gc|pin|verify|purge-v1 [--dir PATH]
$ soma kb reindex|head|detach [--root .soma]
$ soma kb checkout <run_id> [--root .soma]
$ somatize-worker --port 8080 --tags gpu --token sk-xxx [--cpus N] [--memory 8G]
                  [--gpus N] [--max-concurrent N] [--id ID] [--coordinator URL]
```

`soma runs` prints a colored table when `rich` is installed (`--plain`
forces plain text, handy in pipes). `soma report` writes one
self-contained HTML file per run; `--inline` embeds its assets so it
opens with no network access.

### Gradient audit types

```python
from soma import AuditScope, ChannelConfig, Thresholds
```

**`Thresholds`** — flag bounds (`grad_lo=1e-7`, `grad_hi=1e3`,
`activation_saturation=50.0`, `saturation_frac=0.5`, `dead_eps=1e-7`,
`dead_frac=0.95`). Flags: `NAN`, `INF`, `VANISHING`, `EXPLODING`,
`DEAD`, `SATURATED`.

**`ChannelConfig`** — per-channel diagnostics (`channel_dim=1`,
`snapshot_every=50`, `corr_threshold=0.95`, `dead_channel_frac=0.95`,
`dormancy_tau=0.1`, `ignored_grad_eps=1e-9`, `groups=None`). `groups`
is keyed by *filter id*, then by group name:
`groups={"encoder": {"audio": range(0, 64), "text": range(64, 128)}}`.
Flags: `DEAD_CHANNELS(n)`, `IGNORED_CHANNELS(n)`, `LEAKAGE`.

**`AuditScope`** — which submodules to audit inside a node
(`depth=None`, `patterns=None`, `sample_every=1`, `max_modules=32`).
Both `depth` and `patterns` unset ⇒ automatic selection.

**`Audit`** (yielded by `gradient_audit`) — `report() -> AuditReport`,
`records() -> dict[id, list[StepRecord]]`, `timeseries(filter_id)`,
`assert_healthy()` (raises `GradientHealthError`).
**`AuditReport`** — `filters: list[FilterReport]`, `n_steps`,
`is_healthy()`, `by_id()`, `pretty()`, `dataframe()`. Each
`FilterReport` carries `filter_id`, `n_steps`, `metrics`, `flags`.

`soma.audit_modules([(name, module), ...])` is the standalone form for
code that does not drive training through a `Graph`.

## Type checking

The package ships `py.typed`, so a type checker will use what it says
about itself. No configuration is needed — install it and mypy or pyright
picks up the annotations.

```python
from soma import Graph, Study, Trial

g = Graph(cache="memory")
g.edge("a")                  # error: missing positional argument "target"
g.node("x", f, targt="gpu")  # error: did you mean "target"?

def objective(t: Trial) -> str:      # error: expected metrics, a number, or None
    return "not a metric"
Study("s").run(objective)
```

The extension module is a compiled `.so`, so its surface lives in a
hand-written stub, `soma/_soma.pyi`; everything above it is annotated in
its own source. A hand-written stub can drift from the binary silently —
it keeps type-checking, it just stops describing anything — so the test
suite compares the stub against the module that was actually built: the
same classes, methods, getters, parameter names and defaults, and no
constructor for the three classes that have none.

What that cannot check is whether a type is *correct*. If you hit one that
is wrong, it is a bug worth reporting.

Two notes for contributors touching the bindings:

- PyO3 puts a `#[new]`'s signature on the **type** (`cls.__text_signature__`),
  not on `__new__`, which reports an unhelpful `(*args, **kwargs)`.
- A method bound dynamically in a class body is `Any` to a checker. Write
  them out; that is why the `soma.viz` methods on `Study` and `RunView` are
  seventeen separate definitions rather than a loop.

## Rust API

The full Rust API documentation is auto-generated from source code:

**[View Rust API Docs](/soma/api/somatize_core/)**

Key crates:
- [`somatize_core`](/soma/api/somatize_core/) — Types, traits, enums
- [`somatize_compiler`](/soma/api/somatize_compiler/) — Graph compilation
- [`somatize_runtime`](/soma/api/somatize_runtime/) — Execution engine
- [`somatize_worker`](/soma/api/somatize_worker/) — Worker daemon + coordinator
- [`somatize_memory`](/soma/api/somatize_memory/) — Knowledge base
- [`somatize_agent`](/soma/api/somatize_agent/) — Research agent
- [`somatize_coordinator`](/soma/api/somatize_coordinator/) — Worker registry & routing
- [`somatize_mcp`](/soma/api/somatize_mcp/) — MCP server
