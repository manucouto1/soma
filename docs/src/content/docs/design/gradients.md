---
title: Gradient Propagation
description: How Soma pipelines support end-to-end gradient flow through differentiable filters.
---

## Two Types of Graphs

Soma distinguishes between two kinds of graphs that serve different purposes:

| | Computational Graph (Soma Pipeline) | Orchestration Graph (Platform) |
|---|---|---|
| **Nodes** | Filters (data transformations) | LLM, Agent, Condition, Tool, IO |
| **Purpose** | Transform data | Orchestrate reasoning |
| **Gradients** | Yes, if filters are differentiable | No (not meaningful) |
| **Example** | `[Scaler] → [Linear] → [Softmax]` | `[Agent: Hypothesize] → [Pipeline] → [Agent: Analyze]` |

Soma pipelines **are** computational graphs and **should** support gradient propagation when the user implements differentiable filters.

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

When all filters in a pipeline are differentiable, `pipeline.forward()` maintains the computational graph and gradients flow end-to-end:

```
Pipeline: [Scaler] → [Linear] → [ReLU] → [Linear2]
           diff ✓     diff ✓    diff ✓     diff ✓

x → forward(Scaler) → forward(Linear) → forward(ReLU) → forward(Linear2) → output
                                                                               │
loss = criterion(output, target)                                               │
loss.backward()  ← gradients flow through all forwards ◄──────────────────────┘
```

### Pipeline Methods

```rust
impl Pipeline {
    /// Differentiable forward: maintains computational graph
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut current = x.clone();
        for (filter, state) in &self.fitted_filters {
            current = filter.forward(&current, state)?;
            // No detach: gradient graph accumulates
        }
        Ok(current)
    }

    /// Inference: detaches after each filter (no gradients)
    pub fn predict(&self, x: &Tensor) -> Result<Tensor> {
        let mut current = x.clone();
        for (filter, state) in &self.fitted_filters {
            current = filter.forward(&current, state)?.detach();
        }
        Ok(current)
    }

    /// Training: fit each filter sequentially, detach between them
    pub fn fit(&mut self, x: &Tensor, y: Option<&Tensor>) -> Result<()> {
        let mut current = x.clone();
        for (filter, state_slot) in &mut self.filters {
            let state = self.resolve_or_fit(filter, &current, y)?;
            // Detach between fits: each filter trains independently
            current = filter.forward(&current, &state)?.detach();
            *state_slot = Some(state);
        }
        Ok(())
    }
}
```

### Gradient Interruption

When a non-differentiable filter appears in the pipeline, it breaks the gradient chain:

```
Pipeline: [Scaler] → [DecisionTree] → [Linear]
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

### During `predict()` (inference)

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

### End-to-End Differentiable Pipeline

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
loss.backward()  # gradients flow through entire pipeline

# Access gradients
x_test.grad  # gradient with respect to input
```

### Mixed Pipeline (partial gradients)

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

### When to Use `forward()` vs `predict()`

| Method | Gradients | Caching | Use Case |
|---|---|---|---|
| `predict()` | No | Full caching | Inference, evaluation, production |
| `forward()` | Yes | State-only caching | Training, fine-tuning, gradient analysis |

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

The contract is **always `(out, aux)`**; `aux` is an empty dict unless
the filter override surfaces auxiliary signals (gates, routing weights,
auxiliary losses).

### Graph orchestration API

| Method | Description |
|---|---|
| `g.materialize(sample_input)` | Walk topology, build every `_module` once, threading shapes through `output_shape`. |
| `g.train()` / `g.eval()` | Toggle `training` on every live filter (and its `_module`). |
| `g.parameters()` | Iterate `nn.Parameter`s of every materialised filter, in topological order, deduplicated. |
| `g.forward(x)` | Polymorphic. If any filter is in training, walks live filters with autograd live and returns `(out, aux_by_node)`. Otherwise delegates to the Rust inference path. |
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
            out, aux = g.forward(x)
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
out, aux = g.forward(x)
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
            out, aux = g.forward(x)
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
