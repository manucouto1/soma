---
title: Gradient Propagation
description: How Soma pipelines support end-to-end gradient flow through differentiable filters.
---

## Two Types of Graphs

Soma distinguishes between two kinds of graphs that serve different purposes:

| | Computational Graph (Soma `Graph`) | Orchestration Graph (Platform) |
|---|---|---|
| **Nodes** | Filters (data transformations) | LLM, Agent, Condition, Tool, IO |
| **Purpose** | Transform data | Orchestrate reasoning |
| **Gradients** | Yes, if filters are differentiable | No (not meaningful) |
| **Example** | `[Scaler] → [Linear] → [Softmax]` | `[Agent: Hypothesize] → [Graph] → [Agent: Analyze]` |

A Soma `Graph` **is** a computational graph, and supports gradient propagation when its filters are differentiable.

## How Gradients Flow

Each filter declares whether its `forward()` function is differentiable via `FilterMeta::differentiable`:

```rust
pub struct FilterMeta {
    pub kind: FilterKind,
    pub differentiable: bool,  // key flag
    pub cacheable: bool,
    pub stream_mode: StreamMode,
}
```

When all filters in a graph are differentiable, `g.forward()` maintains the autograd graph and gradients flow end-to-end:

```
Graph: [Scaler] → [Linear] → [ReLU] → [Linear2]
        diff ✓     diff ✓    diff ✓     diff ✓

x → forward(Scaler) → forward(Linear) → forward(ReLU) → forward(Linear2) → output
                                                                               │
loss = criterion(output, target)                                               │
loss.backward()  ← gradients flow through all forwards ◄──────────────────────┘
```

### The methods that drive it

Gradient flow is driven from Python, where the tensors and the optimizer live.
`soma/_orchestrator.py` installs these on `Graph`:

| Method | What it does |
|---|---|
| `g.materialize(sample)` | Threads shapes through the graph and builds each filter's `nn.Module`. Must run before training or auditing. |
| `g.train()` / `g.eval()` | Flip every differentiable filter's mode |
| `g.parameters()` | Every parameter, in topological order, deduplicated |
| `g.make_optimizer(cls, **kw)` | Build and attach an optimizer (default Adam) |
| `g.context()` | Context manager scoping one training step |
| `g.forward(x)` | Returns the output, in both modes. Auxiliaries land in `g.py_state["last_aux"]` |
| `g.backward(ctx, loss)` | `loss.backward()`, plus the audit hook and step event |
| `g.step(ctx)` | Optimizer step |
| `g.freeze()` | Fold module weights into node state and switch to eval |

Nothing detaches between filters while training, so the autograd graph
accumulates across the whole topology:

```python
g.materialize(x)
g.train()
g.make_optimizer(torch.optim.Adam, lr=1e-2)

for _ in range(epochs):
    with g.context() as ctx:
        g.zero_grad()
        out = g.forward(x)
        aux = g.py_state["last_aux"]
        g.backward(ctx, nn.functional.mse_loss(out, y))
    g.step(ctx)
```

In eval — after `freeze()`, or with `training=False` — each filter loads its
saved state and runs under `no_grad`, so no autograd graph is built at all.

### Gradient Interruption

When a non-differentiable filter appears in the pipeline, it breaks the gradient chain:

```
Graph: [Scaler] → [DecisionTree] → [Linear]
        diff ✓     diff ✗           diff ✓

Gradients from Linear CANNOT reach Scaler.
Linear can still receive gradients from its own output.
```

## Compiler Diagnostics

The compiler analyzes gradient flow and produces warnings:

```rust
impl Compiler {
    fn check_gradient_flow(&self, graph: &Graph) -> Vec<Diagnostic> {
        let mut diagnostics = vec![];
        let topo_order = self.topological_sort(graph);
        let mut gradient_flows = true;

        for node in &topo_order {
            let meta = node.filter.meta();

            if gradient_flows && !meta.differentiable {
                diagnostics.push(Diagnostic::Warning {
                    node: node.id.clone(),
                    message: format!(
                        "Gradient flow interrupted at `{}` (FilterKind::{:?}). \
                         Downstream filters will not receive gradients from \
                         upstream filters.",
                        node.label, meta.kind
                    ),
                });
                gradient_flows = false;
            }

            if !gradient_flows && meta.differentiable {
                // Gradient flow restarts from this point
                gradient_flows = true;
            }
        }

        diagnostics
    }
}
```

### Example Diagnostic Output

```
warning[gradient]: Gradient flow interrupted at `DecisionTree` (FilterKind::Opaque)
  --> pipeline node 2
  |
  | [Scaler] → [DecisionTree] → [Linear]
  |              ^^^^^^^^^^^^^^
  |              This filter is not differentiable.
  |              Gradients from [Linear] will not reach [Scaler].
  |
  = help: If you need end-to-end gradients, replace with a differentiable
          alternative or wrap the non-differentiable section in a single
          opaque Process.
```

## Interaction with Caching

There is an important interaction between caching and gradients:

### In eval mode (inference)

Caching works normally. Each filter's output is cached and reused. No gradients needed.

### During `forward()` (differentiable)

Cached outputs **cannot** be used for gradient flow because the cached
tensor has no computational graph attached. `CompileMode::Differentiable`
therefore disables output caching at runtime — every forward re-executes
so the live computational graph exists end to end (fit states still
cache).

The compiler takes a `mode` parameter:

```rust
impl Compiler {
    pub fn compile(
        &self,
        graph: &Graph,
        cache: &dyn CacheStore,
        mode: CompileMode,
    ) -> ExecutionPlan {
        match mode {
            CompileMode::Inference => {
                // Resolve cache aggressively
                self.resolve_cache(plan, cache)
            }
            CompileMode::Differentiable => {
                // Only cache states (from fit), not forward outputs
                self.resolve_state_cache_only(plan, cache)
            }
        }
    }
}
```

### State Caching in Differentiable Mode

Even in differentiable mode, **filter states** (from `fit()`) can still be cached. The state is constant during forward, so it doesn't participate in the gradient computation:

```
fit() results  → cacheable always (state is constant)
forward() results → cacheable only in inference mode
```

## Use Cases

### End-to-end differentiable graph

```python
g = Graph.somatize(
    MyScaler(with_mean=True)
    >> LinearLayer(hidden=128)
    >> ReLU()
    >> LinearLayer(hidden=10)
)

g.fit(x_train, y_train)

# Differentiable forward
output = g.forward(x_test)
loss = cross_entropy(output, y_test)
loss.backward()  # gradients flow through the whole graph

# Access gradients
x_test.grad  # gradient with respect to input
```

### Mixed graph (partial gradients)

```python
g = Graph.somatize(
    SQLQuery("SELECT * FROM features")   # Opaque
    >> MyScaler()                        # Differentiable
    >> NeuralNet(layers=3)               # Differentiable
)

# Compiler warns: gradient flow interrupted at SQLQuery
# But Scaler → NeuralNet gradient flow works fine
output = g.forward(x)
loss.backward()  # gradients flow from NeuralNet through Scaler
```

### Training mode vs eval mode

There is one `forward()`; the mode decides what it does.

| Mode | Gradients | Caching | Use case |
|---|---|---|---|
| `g.eval()` (default) | No | Full output caching | Inference, evaluation, production |
| `g.train()` | Yes | State-only caching | Training, fine-tuning, gradient analysis |

## Native Training Loop (Python)

The conceptual model above describes the runtime; the Python orchestrator
exposes it as a small, RPC-ready surface on `Graph`. Filters that subclass
`DifferentiableFilter` keep a persistent `nn.Module` on the instance,
so a user-driven training loop can run batches through the graph with
gradients flowing **natively** between filters — no per-batch
serialization of weights or activations.

### Anatomy of `DifferentiableFilter`

A subclass implements two hooks plus an optional `forward` override:

```python
from soma import DifferentiableFilter
import torch.nn as nn

class Dense(DifferentiableFilter):
    def __init__(self, out_dim, lr=1e-3):
        super().__init__(out_dim=out_dim, lr=lr)

    def build_module(self, input_shape):       # built once
        return nn.Linear(input_shape[-1], self.out_dim)

    def output_shape(self, input_shape):       # for cascade materialise
        return (self.out_dim,)
```

`forward(x, state=None)` is provided by the base. It is **polymorphic on
`self.training`**:

- `training=True` → returns `(out_tensor, aux_dict)` with autograd live.
  The `state` argument is ignored — parameters live on `self._module`.
- `training=False` → optionally loads `state["weights_b64"]`, runs
  `no_grad`, returns `(out_list, aux_dict)`. This is the path the Rust
  runtime uses for cached/distributable inference after `freeze()`.

A **filter's** `forward` returns `(out, aux)`; `aux` is an empty dict
unless the override surfaces auxiliary signals (gates, routing weights,
auxiliary losses). The **graph's** `forward` does not pass that tuple on —
it collects the auxiliaries by node into `g.py_state["last_aux"]` and
returns the output alone, so the call has one shape whatever mode the
graph is in.

### Graph orchestration API

| Method | Description |
|---|---|
| `g.materialize(sample_input)` | Walk topology, build every `_module` once, threading shapes through `output_shape`. |
| `g.train()` / `g.eval()` | Toggle `training` on every live filter (and its `_module`). |
| `g.parameters()` | Iterate `nn.Parameter`s of every materialised filter, in topological order, deduplicated. |
| `g.forward(x)` | Dispatches on the `_differentiable` declaration: a graph holding differentiable filters is walked in Python with autograd live, otherwise it goes to the Rust path. Returns the output either way; auxiliaries are collected into `py_state["last_aux"]`. The Python walk refuses `stream`, `chunk_size`, `seed` and `run_id`, which it cannot honour. |
| `g.make_optimizer(cls=Adam, **kw)` | Build and register an optimiser over `g.parameters()`. |
| `g.set_optimizer(opt)` | Register an externally-built optimiser. |
| `g.context()` | Autograd context manager. Local: no-op. RPC: `dist.autograd.context()`. |
| `g.backward(ctx, loss)` | Local: `loss.backward()`. RPC: `dist.autograd.backward(ctx, [loss])`. |
| `g.step(ctx)` | Local: `opt.step()`. RPC: `DistributedOptimizer.step(ctx)`. |
| `g.zero_grad()` | Wrapper around the registered optimiser; silent no-op before `make_optimizer`. |
| `g.freeze()` | Snapshot every live `_module.state_dict()` into the runtime's filter-state library and switch to eval. After `freeze()`, the Rust forward path is ready. |

### Canonical training loop

```python
from soma import Graph, DifferentiableFilter
import torch, torch.nn as nn

g = Graph.somatize(Dense(8) >> Dense(2))
g.materialize(sample_x)
g.train()
g.make_optimizer(torch.optim.Adam, lr=1e-3)

for epoch in range(epochs):
    for x, y in batches:
        with g.context() as ctx:
            g.zero_grad()
            out = g.forward(x)
            aux = g.py_state["last_aux"]
            loss = nn.functional.mse_loss(out, y)
            g.backward(ctx, loss)
        g.step(ctx)

g.freeze()                 # weights → state library, switch to eval
g.eval()
preds = g.forward(x_test)  # Rust inference path
```

### Auxiliary signals

When a filter override returns `(out, {"gate": gate_tensor})`, the
graph's training-mode `forward` collects per-node aux into
`aux_by_node = {node_id: aux_dict}`. The user combines them with the
main loss explicitly:

```python
out = g.forward(x)
aux = g.py_state["last_aux"]
main = nn.functional.cross_entropy(out, y)
gate_l1 = aux["classifier"]["gate"].abs().mean()
total = main + 0.1 * gate_l1
g.backward(ctx, total)
```

`aux` tensors keep autograd live, so gradients from the auxiliary term
reach the same parameters as the main loss.

### Gradient health audit

Pair the training loop with `graph.gradient_audit()` to record per-filter
activation and gradient statistics and flag vanishing / exploding /
NaN / dead / saturated nodes with no manual hooks:

```python
with g.gradient_audit() as audit:
    for x, y in batches:
        with g.context() as ctx:
            g.zero_grad()
            out = g.forward(x)
            aux = g.py_state["last_aux"]
            loss = my_loss(out, y, aux)
            g.backward(ctx, loss)
        g.step(ctx)

print(audit.report().pretty())
audit.assert_healthy()
```

See the [Gradient Health Audit guide](../../guides/gradient-audit/) for
metrics, flags, and threshold tuning.

### RPC-ready by design

The `context() / backward(ctx, loss) / step(ctx)` triplet is intentionally
shaped after `torch.distributed.autograd`. Locally each call reduces to
the obvious single-process action; once filters live on remote workers
backed by `torch.distributed.rpc`, the same call sites swap in the
distributed equivalents (`dist.autograd.context()`,
`dist.autograd.backward(ctx, [loss])`,
`DistributedOptimizer.step(ctx)`) without touching user code.
