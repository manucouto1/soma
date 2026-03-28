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

Cached outputs **cannot** be used for gradient flow because the cached tensor has no computational graph attached. The compiler handles this:

```rust
ExecutionPlan::Cached { id, key }
// ↑ Used in predict() mode

ExecutionPlan::Execute { id, filter }
// ↑ Used in forward() mode, even if cache exists,
//   because we need the live computational graph
```

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
pipeline = Pipeline([
    MyScaler(with_mean=True),
    LinearLayer(hidden=128),
    ReLU(),
    LinearLayer(hidden=10),
])

pipeline.fit(x_train, y_train)

# Differentiable forward
output = pipeline.forward(x_test)
loss = cross_entropy(output, y_test)
loss.backward()  # gradients flow through entire pipeline

# Access gradients
x_test.grad  # gradient with respect to input
```

### Mixed Pipeline (partial gradients)

```python
pipeline = Pipeline([
    SQLQuery("SELECT * FROM features"),  # Opaque
    MyScaler(),                           # Differentiable
    NeuralNet(layers=3),                  # Differentiable
])

# Compiler warns: gradient flow interrupted at SQLQuery
# But Scaler → NeuralNet gradient flow works fine
output = pipeline.forward(x)
loss.backward()  # gradients flow from NeuralNet through Scaler
```

### When to Use `forward()` vs `predict()`

| Method | Gradients | Caching | Use Case |
|---|---|---|---|
| `predict()` | No | Full caching | Inference, evaluation, production |
| `forward()` | Yes | State-only caching | Training, fine-tuning, gradient analysis |
