---
title: Hyperparameter Optimization
description: Search spaces co-located with filters, with Bayesian, Hyperband, and multi-objective strategies.
---

## Core Principle: Search Spaces Live in the Filter

Unlike external optimization frameworks where you define search spaces separately, Soma's search spaces are **co-located with the parameters they describe**:

```rust
#[derive(Filter)]
struct MyClassifier {
    #[soma(search(low = 0.001, high = 100.0, scale = "log"))]
    C: f64,

    #[soma(search(choices = ["linear", "rbf", "poly"]))]
    kernel: String,

    // Not searchable -- fixed value
    verbose: bool,
}
```

This co-location provides:

- **Compile-time type validation**: The macro verifies that search annotations match field types
- **No string references**: No risk of typos in parameter names
- **Single source of truth**: Change the parameter, change the search space, in one place
- **Automatic aggregation**: The graph collects all search spaces from its filters

## Search Dimensions

```rust
#[derive(Serialize, Deserialize, Clone)]
pub enum SearchDimension {
    /// Continuous range
    Float {
        name: String,
        low: f64,
        high: f64,
        scale: Scale,
        default: Option<f64>,
    },

    /// Integer range
    Int {
        name: String,
        low: i64,
        high: i64,
        scale: Scale,
    },

    /// Set of discrete choices
    Categorical {
        name: String,
        choices: Vec<Value>,
    },

    /// Active only when parent has specific values
    Conditional {
        name: String,
        parent: String,
        parent_values: Vec<Value>,
        dimension: Box<SearchDimension>,
    },
}

#[derive(Serialize, Deserialize, Clone)]
pub enum Scale {
    Linear,
    Log,
    ReverseLog,
}
```

### Type Inference from Rust Types

The derive macro infers the dimension type from the field's Rust type:

| Rust Type | Inferred Dimension | Required Annotation |
|---|---|---|
| `f32`, `f64` | `Float` | `low`, `high` (required), `scale` (optional) |
| `i8`..`i64`, `u8`..`u64`, `usize` | `Int` | `low`, `high` (required) |
| `bool` | `Categorical { [true, false] }` | None (automatic) |
| `String` + `choices` | `Categorical` | `choices` (required) |
| `enum` (with `#[derive(SomaEnum)]`) | `Categorical` | None (variants become choices) |

### Compile-Time Validation

The macro rejects invalid combinations at compile time:

```rust
// ✗ Float range on String field
#[soma(search(low = 0.1, high = 10.0))]
method: String,
// compile_error!("Float range on String field. Use `choices` instead.")

// ✗ low > high
#[soma(search(low = 10.0, high = 0.1))]
lr: f64,
// compile_error!("`low` must be less than `high`")

// ✗ Log scale on integer (ambiguous)
#[soma(search(low = 1, high = 10, scale = "log"))]
n_layers: usize,
// compile_error!("Log scale not supported for integer fields. Use f64 or remove scale.")
```

### Enums as Categories

```rust
#[derive(Serialize, Deserialize, Clone, SomaEnum)]
pub enum Kernel {
    Linear,
    Rbf,
    Polynomial,
}

#[derive(Filter)]
struct MySVM {
    #[soma(search)]  // no args needed: enum variants become choices
    kernel: Kernel,

    #[soma(search(low = 0.001, high = 100.0, scale = "log"))]
    C: f64,
}
```

## The Searchable Trait

Generated automatically by `#[derive(Filter)]`:

```rust
pub trait Searchable {
    /// The search space defined by field annotations
    fn search_space() -> SearchSpace;

    /// Construct an instance from sampled parameters
    fn from_sample(params: &HashMap<String, Value>) -> Result<Self>
    where Self: Sized;

    /// Extract current parameters as key-value pairs
    fn current_params(&self) -> HashMap<String, Value>;
}
```

## Graph Search Space Aggregation

The graph automatically collects all search spaces:

```rust
impl Graph {
    pub fn search_space(&self) -> SearchSpace {
        let mut combined = SearchSpace::new();
        for node in &self.nodes {
            let space = node.filter.search_space();
            // Prefix with filter label to avoid collisions:
            // "MyScaler.scale" vs "MyClassifier.scale"
            combined.merge_with_prefix(&node.label, space);
        }
        combined
    }
}
```

```
Graph.somatize(MyScaler(scale=2.0) >> MySVM(kernel=Rbf, C=1.0))

search_space():
  MyScaler.scale:  Float[0.1, 10.0] log
  MySVM.kernel:    Categorical[Linear, Rbf, Polynomial]
  MySVM.C:         Float[0.001, 100.0] log
```

## Search Strategies

```rust
#[derive(Serialize, Deserialize)]
pub enum SearchStrategy {
    /// Exhaustive grid search
    Grid { points_per_dim: usize },

    /// Random sampling
    Random { n_trials: usize, seed: Option<u64> },

    /// Bayesian optimization (Tree-Parzen Estimator)
    Bayesian {
        n_trials: usize,
        n_startup: usize,       // random trials before modeling
        seed: Option<u64>,
    },

    /// Successive halving with early stopping
    Hyperband {
        max_resource: usize,     // max epochs/iterations
        reduction_factor: usize, // halving factor (typically 3)
    },

    /// Multi-objective optimization (NSGA-II or similar)
    MultiObjective {
        n_trials: usize,
        objectives: Vec<Objective>,
    },

    /// Agent-guided exploration (future: agent decides what to try)
    AgentGuided { agent_id: String },
}
```

## Pruning Strategies

Pruning stops unpromising trials early, saving computation:

```rust
#[derive(Serialize, Deserialize)]
pub enum PruningStrategy {
    /// No pruning
    None,

    /// Prune if metric is below median of completed trials at same step
    Median { n_warmup_steps: usize },

    /// Prune if metric is below given percentile
    Percentile {
        percentile: f64,
        n_warmup_steps: usize,
    },

    /// Bracket-based pruning (used with Hyperband)
    Hyperband,
}
```

Pruning integrates with the filter via `ctx.report_metric()`:

```rust
// Inside a filter's fit() method:
for epoch in 0..self.epochs {
    let loss = train_epoch(&model, &data);
    let val_f1 = evaluate(&model, &val_data);

    // Report to the study. If pruned, this returns Err(Pruned).
    ctx.report_metric("f1", val_f1, epoch)?;
}
```

## The Study

A Study orchestrates the full optimization:

```rust
#[derive(Serialize, Deserialize)]
pub struct Study {
    pub id: StudyId,
    pub name: String,
    pub graph: Graph,
    pub search_space: SearchSpace,     // aggregated from graph
    pub strategy: SearchStrategy,
    pub pruning: PruningStrategy,
    pub objectives: Vec<Objective>,
    pub trials: Vec<Trial>,
    pub best_trials: Vec<TrialId>,
}

#[derive(Serialize, Deserialize)]
pub struct Objective {
    pub metric: String,
    pub direction: Direction,
}

#[derive(Serialize, Deserialize)]
pub enum Direction { Minimize, Maximize }
```

### Trial Lifecycle

```rust
#[derive(Serialize, Deserialize, Clone)]
pub struct Trial {
    pub id: TrialId,
    pub params: HashMap<String, Value>,
    pub state: TrialState,
    pub metrics: Vec<MetricRecord>,
    pub duration: Option<Duration>,
}

#[derive(Serialize, Deserialize, Clone)]
pub enum TrialState {
    Pending,
    Running,
    Completed,
    Pruned { step: usize, reason: String },
    Failed { error: String },
}
```

### Freezing Parameters

Users can fix some parameters and only search over the rest:

```python
study = Study(
    graph=g,
    freeze={"MySVM.kernel": "rbf"},  # fix kernel, search only C
    strategy=Bayesian(n_trials=50),
    objectives=[("f1", "maximize")],
)
```

## Python API

```python
from soma import Graph, Study, Bayesian, Median

g = Graph.somatize(
    MyScaler(scale=2.0) >> MySVM(kernel="rbf", C=1.0)
)

# View the auto-collected search space
print(g.search_space())
# MyScaler.scale:  Float[0.1, 10.0] log
# MySVM.kernel:    Categorical[Linear, Rbf, Polynomial]
# MySVM.C:         Float[0.001, 100.0] log

# Run optimization
study = Study(
    name="svm_optimization",
    graph=g,
    strategy=Bayesian(n_trials=100),
    pruning=Median(n_warmup_steps=5),
    objectives=[("f1", "maximize")],
)

study.run(train_data, eval_data)

# Results
print(study.best_trial)
print(study.best_trial.params)
# {'MyScaler.scale': 1.23, 'MySVM.kernel': 'rbf', 'MySVM.C': 12.5}

# Visualization (events-driven)
study.plot()  # parallel coordinates, learning curves, importance
```

## Events Produced

During a study, the following events flow:

```
StudyStarted { study summary }
  TrialStarted { trial_001, params: {scale: 0.5, C: 10.0} }
    RunStarted { run_001 }
      NodeStarted { scaler }
      NodeCompleted { scaler, 0.02s }
      NodeStarted { svm }
        TrialMetric { f1: 0.72, step: 1 }
        TrialMetric { f1: 0.78, step: 2 }
        TrialMetric { f1: 0.81, step: 3 }
      NodeCompleted { svm, 1.2s }
    RunCompleted { run_001, 1.22s }
  TrialCompleted { trial_001, f1: 0.81 }
  BestUpdated { trial_001 }
  StudyProgress { completed: 1, total: 100, best: 0.81 }

  TrialStarted { trial_002, params: {scale: 5.0, C: 0.1} }
    ...
    TrialMetric { f1: 0.45, step: 1 }
    TrialMetric { f1: 0.46, step: 2 }
  TrialPruned { trial_002, step: 2, reason: "below median" }
  StudyProgress { completed: 2, total: 100, best: 0.81 }

  ...

StudyCompleted { best_trials: [trial_042] }
```
