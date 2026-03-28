---
title: Python API
description: Python API reference for the Soma package.
---

## Installation

```bash
pip install soma
```

## Core Classes

### Filter

Base class for pipeline nodes. Subclass to define custom transformations.

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

#### Search Descriptors

Use `search()` to define hyperparameter search spaces:

```python
scale: float = search(0.1, 10.0, scale="log")      # Float range
epochs: int = search(10, 100)                        # Integer range
method: str = search(choices=["a", "b", "c"])        # Categorical
enabled: bool = search()                             # Auto: [True, False]
```

### Pipeline

Compose filters into a sequential pipeline with automatic caching.

```python
from soma import Pipeline

pipeline = Pipeline([MyScaler(scale=2.0), MyClassifier(C=1.0)])
pipeline.fit(x_train, y_train)
result = pipeline.predict(x_test)
```

#### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(filters: list[Filter])` | Create pipeline from filter instances |
| `fit` | `(x, y=None)` | Train all filters sequentially |
| `predict` | `(x) -> list` | Forward data through fitted pipeline |
| `is_fitted` | `() -> bool` | Whether pipeline has been fitted |
| `filter_names` | `() -> list[str]` | Names of filters in order |
| `search_space` | `() -> list` | Aggregated search space |

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
    # Build and run pipeline with params...
    return {"f1": 0.85}

study.run(executor)
print(study.best_trial)     # {"id": "...", "params": {...}, "metrics": {...}}
print(study.n_trials)       # 50
print(study.progress)       # 1.0
```

#### Constructor

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Study name |
| `search_space` | `list[dict]` | List of dimension dicts |
| `strategy` | `str` | `"grid"`, `"random"`, or `"bayesian"` |
| `n_trials` | `int` | Number of trials |
| `objectives` | `list[tuple]` | `[(metric_name, "maximize"\|"minimize")]` |
| `seed` | `int\|None` | Random seed for reproducibility |

#### Search Space Dimensions

```python
# Float
{"type": "float", "name": "lr", "low": 0.001, "high": 0.1, "scale": "log"}

# Integer
{"type": "int", "name": "epochs", "low": 10, "high": 100}

# Categorical
{"type": "categorical", "name": "kernel", "choices": ["rbf", "linear"]}
```

### Lab

Connect to a remote Soma worker.

```python
from soma import Lab

lab = Lab.connect("http://localhost:3000")
lab.health()        # "ok"
lab.info()          # Worker capabilities dict
lab.workers()       # List of available workers
```

## Rust API

The full Rust API documentation is auto-generated from source code:

**[View Rust API Docs](/api/soma_core/)**

Key crates:
- [`soma_core`](/api/soma_core/) — Types, traits, enums
- [`soma_compiler`](/api/soma_compiler/) — Graph compilation
- [`soma_runtime`](/api/soma_runtime/) — Execution engine
- [`soma_memory`](/api/soma_memory/) — Knowledge base
- [`soma_agent`](/api/soma_agent/) — Research agent
- [`soma_worker`](/api/soma_worker/) — Worker daemon
