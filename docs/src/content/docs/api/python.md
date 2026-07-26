---
title: Python API
description: Python API reference for the Soma package.
---

## Installation

```bash
pip install somatize
```

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

Class attribute `class_version: int = 1` — bump in subclasses when
constructor kwargs or saved-state layout change. `Graph.load` reads it
from the manifest and warns on mismatch.

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
| `node` | `(filter) -> str` | Add a filter node, returns node ID |
| `edge` / `connect` | `(source, target)` | Connect two nodes with a data edge |
| `fit` | `(x, y=None)` | Fit all trainable filters in topological order |
| `forward` | `(x) -> list` | Forward data through fitted graph |
| `run` | `() -> dict` | Compile and execute, return all outputs |
| `compile` | `(mode="inference") -> dict` | Compile and return diagnostics |
| `to_mermaid` | `() -> str` | Render graph as Mermaid diagram |
| `to_graphviz` | `() -> str` | Render graph as Graphviz DOT |
| `to_text` | `() -> str` | Render graph as ASCII tree |
| `on_event` | `(callback)` | Register event callback (background thread) |
| `set_strategy` | `(strategy)` | Set training strategy (from soma-core) |
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
| `gradient_audit` | `(thresholds=None) -> ctx[Audit]` | Install per-filter forward/backward hooks for the duration of a training pass |
| `add_worker` | `(address, token?, tags?)` | Add a remote worker |
| `set_coordinator` | `(url, token?)` | Set coordinator for auto-discovery |
| `workers` | `() -> list[dict]` | List known workers |

#### Compile Modes

```python
info = g.compile("inference")       # Full caching
info = g.compile("differentiable")  # Cache states, re-execute forwards
info = g.compile("no_cache")        # Force re-execution
# Returns: {total_nodes, cached_nodes, parallel_branches, diagnostics, plan_text, plan_mermaid}
```

#### Events

```python
def on_event(event):
    print(event["event_type"], event.get("node_id", ""))

g.on_event(on_event)
g.fit(data)
# Events: NodeStarted, NodeCompleted, NodeCacheHit, NodeFailed, ...
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

### DifferentiableFilter

Filter base class for trainable `nn.Module` wrappers. Available when
`torch` is installed; `None` otherwise.

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

See the [gradients design doc](/design/gradients/#native-training-loop-python)
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

def executor(params):
    """Execute one trial. Returns dict of metric_name -> value."""
    g = Graph.somatize(Scaler() >> Model(lr=params["lr"]))
    g.fit(train_data)
    outputs = g.forward(test_data)
    return {"f1": compute_f1(outputs)}

study.run(executor)
print(study.best_trial)     # {"id": "...", "params": {...}, "metrics": {...}}
print(study.n_trials)       # 50
print(study.progress)       # 1.0
```

### Lab

Connect to a remote Soma worker.

```python
from soma import Lab

lab = Lab.connect("http://localhost:8080")
lab.health()        # "ok"
lab.info()          # Worker capabilities dict
```

## Rust API

The full Rust API documentation is auto-generated from source code:

**[View Rust API Docs](/api/soma_core/)**

Key crates:
- [`soma_core`](/api/soma_core/) — Types, traits, enums
- [`soma_compiler`](/api/soma_compiler/) — Graph compilation
- [`soma_runtime`](/api/soma_runtime/) — Execution engine
- [`soma_worker`](/api/soma_worker/) — Worker daemon + coordinator
- [`soma_memory`](/api/soma_memory/) — Knowledge base
- [`soma_agent`](/api/soma_agent/) — Research agent
