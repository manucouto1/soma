---
title: Filter Model
description: The two-phase filter lifecycle -- fit and forward.
---

## The Filter Trait

A **Filter** is the fundamental unit of computation in Soma. Every filter has two phases:

1. **`fit(x, y)`** -- Learn internal state from training data
2. **`forward(x, state)`** -- Transform data using the learned state

This separation is key because each phase is **independently cacheable** and has different semantics.

```rust
/// The fundamental computation unit in Soma
#[async_trait]
pub trait Filter: Send + Sync {
    /// The learned state type (weights, statistics, etc.)
    type State: Serialize + Deserialize + Clone;

    /// Learn state from training data
    fn fit(&self, x: &Tensor, y: Option<&Tensor>) -> Result<Self::State>;

    /// Transform data using learned state (can be differentiable)
    fn forward(&self, x: &Tensor, state: &Self::State) -> Result<Tensor>;

    /// Metadata for the compiler
    fn meta(&self) -> FilterMeta;
}
```

## Filter Metadata

Each filter declares its characteristics via `FilterMeta`. The compiler uses this for optimization, validation, and planning:

```rust
pub struct FilterMeta {
    pub kind: FilterKind,
    pub cacheable: bool,
    pub differentiable: bool,
    pub stream_mode: StreamMode,
}

pub enum FilterKind {
    /// No state needed. forward() ignores state.
    /// Example: activation function, fixed projection
    Stateless,

    /// Learns state in fit(), uses it in forward()
    /// Example: scaler, PCA, classifier
    Trainable,

    /// Not differentiable. Breaks gradient flow.
    /// Example: decision tree, SQL query, file I/O
    Opaque,
}
```

## Lifecycle

```
┌─────────────────────────────────────────────────────┐
│                   FILTER LIFECYCLE                    │
│                                                      │
│   ┌─────────┐                                        │
│   │  CREATE  │  MyScaler { scale: 2.0 }              │
│   └────┬────┘                                        │
│        │                                             │
│        ▼                                             │
│   ┌─────────┐         ┌─────────────┐               │
│   │   FIT   │────────►│   STATE     │               │
│   │ (x, y)  │         │ { mean, std }│               │
│   └─────────┘         └──────┬──────┘               │
│        │                     │                       │
│        │    cacheable:       │   cacheable:           │
│        │    hash(config      │   hash(config          │
│        │     + data_xy)      │    + state + data_x)   │
│        │                     │                       │
│        ▼                     ▼                       │
│   ┌──────────┐        ┌──────────┐                   │
│   │ PREDICT  │        │ FORWARD  │                   │
│   │ (detach) │        │ (grads)  │                   │
│   └──────────┘        └──────────┘                   │
│                                                      │
│   predict = forward + detach (no gradient tracking)  │
│   forward = raw differentiable computation           │
└─────────────────────────────────────────────────────┘
```

### `predict` vs `forward`

- **`forward(x, state)`**: The raw transformation. If the filter is differentiable, this maintains the computational graph for backpropagation.
- **`predict(x)`**: Convenience method. Calls `forward()` and detaches the result. Used in inference when gradients are not needed.

```rust
impl Pipeline {
    /// Inference: no gradient tracking
    pub fn predict(&self, x: &Tensor) -> Result<Tensor> {
        let mut current = x.clone();
        for (filter, state) in &self.fitted_filters {
            current = filter.forward(&current, state)?.detach();
        }
        Ok(current)
    }

    /// Training/differentiable: maintains computational graph
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut current = x.clone();
        for (filter, state) in &self.fitted_filters {
            current = filter.forward(&current, state)?;
        }
        Ok(current)
    }
}
```

## Examples

### Stateless Filter (no training needed)

```rust
#[derive(Filter)]
#[soma(kind = "Stateless", cacheable = true, differentiable = true)]
struct ReLU;

impl Filter for ReLU {
    type State = ();  // no state

    fn fit(&self, _x: &Tensor, _y: Option<&Tensor>) -> Result<()> {
        Ok(())  // nothing to learn
    }

    fn forward(&self, x: &Tensor, _state: &()) -> Result<Tensor> {
        Ok(x.maximum(&Tensor::zeros_like(x)))
    }
}
```

### Trainable Filter

```rust
#[derive(Filter)]
#[soma(kind = "Trainable", cacheable = true, differentiable = true)]
struct StandardScaler {
    #[soma(search(choices = [true, false]))]
    with_mean: bool,

    #[soma(search(choices = [true, false]))]
    with_std: bool,
}

#[derive(Serialize, Deserialize, Clone)]
struct ScalerState {
    mean: Tensor,
    std: Tensor,
}

impl Filter for StandardScaler {
    type State = ScalerState;

    fn fit(&self, x: &Tensor, _y: Option<&Tensor>) -> Result<ScalerState> {
        let mean = if self.with_mean { x.mean(0)? } else { Tensor::zeros(x.dim(1))? };
        let std = if self.with_std { x.std(0)? } else { Tensor::ones(x.dim(1))? };
        Ok(ScalerState { mean, std })
    }

    fn forward(&self, x: &Tensor, state: &ScalerState) -> Result<Tensor> {
        // Differentiable: gradients flow through subtraction and division
        Ok((&(x - &state.mean)?) / &state.std)?)
    }
}
```

### Trainable Model with Intermediate Metrics

```rust
#[derive(Filter)]
#[soma(kind = "Trainable", cacheable = true, differentiable = true)]
struct LinearClassifier {
    #[soma(search(low = 1e-5, high = 1e-1, scale = "log"))]
    lr: f64,

    #[soma(search(low = 10, high = 200))]
    epochs: usize,
}

#[derive(Serialize, Deserialize, Clone)]
struct LinearState {
    weights: Tensor,
    bias: Tensor,
}

impl Filter for LinearClassifier {
    type State = LinearState;

    fn fit(&self, x: &Tensor, y: Option<&Tensor>) -> Result<LinearState> {
        let y = y.ok_or(SomaError::RequiresLabels)?;
        let mut w = Tensor::randn(&[x.dim(1), y.dim(1)])?;
        let mut b = Tensor::zeros(&[y.dim(1)])?;

        for epoch in 0..self.epochs {
            let pred = (x.matmul(&w)? + &b)?;
            let loss = cross_entropy(&pred, y)?;
            loss.backward()?;

            // Update weights (inside fit, gradients are internal)
            w = (&w - &(self.lr * w.grad())?)?;
            b = (&b - &(self.lr * b.grad())?)?;

            // Report metric for Study pruning
            ctx.report_metric("loss", loss.item(), epoch)?;
        }

        Ok(LinearState {
            weights: w.detach(),
            bias: b.detach(),
        })
    }

    fn forward(&self, x: &Tensor, state: &LinearState) -> Result<Tensor> {
        // Differentiable: if someone backprops from here,
        // gradients flow through matmul to x
        Ok((x.matmul(&state.weights)? + &state.bias)?)
    }
}
```

### Opaque Filter (non-differentiable)

```rust
#[derive(Filter)]
#[soma(kind = "Opaque", cacheable = true)]
struct DecisionTree {
    #[soma(search(low = 2, high = 50))]
    max_depth: usize,
}

// State = the trained tree structure
#[derive(Serialize, Deserialize, Clone)]
struct TreeState { /* internal tree nodes */ }

impl Filter for DecisionTree {
    type State = TreeState;

    fn fit(&self, x: &Tensor, y: Option<&Tensor>) -> Result<TreeState> {
        // Train decision tree (not differentiable)
        Ok(build_tree(x, y, self.max_depth))
    }

    fn forward(&self, x: &Tensor, state: &TreeState) -> Result<Tensor> {
        // Lookup in tree (not differentiable -- breaks gradient flow)
        Ok(predict_tree(state, x))
    }
}
```

## The Derive Macro

`#[derive(Filter)]` generates:

1. **`Searchable` impl**: Collects `#[soma(search)]` annotations into a `SearchSpace`
2. **`Serialize`/`Deserialize`**: For remote execution
3. **`config_hash()`**: SHA hash of all public fields (for cache key computation)
4. **`from_sample()`**: Construct instance from sampled hyperparameters
5. **`current_params()`**: Extract current parameters as key-value pairs

### Cache Key Hash Rules

- Only public fields that are constructor parameters contribute to the hash
- Fields marked `#[soma(skip_hash)]` are excluded
- The `State` type is hashed separately (it depends on training data)
- The hash is deterministic: same config always produces the same key

```rust
#[derive(Filter)]
struct MyFilter {
    scale: f64,          // ✓ included in config_hash
    method: String,      // ✓ included in config_hash

    #[soma(skip_hash)]
    verbose: bool,       // ✗ excluded from config_hash
}
```

## Python API

```python
from soma import Filter, Tensor, search

class MyScaler(Filter):
    with_mean: bool = search(choices=[True, False])
    with_std: bool = True  # not searchable, fixed

    def fit(self, x: Tensor, y: Tensor = None):
        mean = x.mean(0) if self.with_mean else Tensor.zeros(x.shape[1])
        std = x.std(0) if self.with_std else Tensor.ones(x.shape[1])
        return {"mean": mean, "std": std}

    def forward(self, x: Tensor, state):
        return (x - state["mean"]) / state["std"]
```

The Python `Filter` base class uses metaclasses to:

- Register `search()` descriptors as `SearchDimension` entries
- Generate `config_hash()` from `__init__` parameters
- Serialize the filter for remote execution via pickle + config JSON
