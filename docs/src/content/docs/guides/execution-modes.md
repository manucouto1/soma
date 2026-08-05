---
title: Execution Modes & Data Transport
description: How to use strategies, runners, executors, and data transport in Soma.
---

## Overview

Soma separates **WHERE** code runs (Runner) from **WHAT** runs (Executor) and **HOW** data moves (Transport).

```
┌─────────────────────────────────────────────────────────┐
│                         User API                         │
│  g = Graph()                                            │
│  g.node("encoder", MyEncoder())                         │
│  g.node("classifier", MyClassifier())                   │
│  g.edge("encoder", "classifier")                        │
│  g.fit(data)          ← one call, everything handled    │
│  g.forward(new_data)                                    │
└────────────────────────┬────────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              │    Runner (WHERE)   │
              ├─────────────────────┤
              │ LocalRunner         │ ← same machine
              │ RemoteRunner        │ ← worker via WS
              └──────────┬──────────┘
                         │
              ┌──────────┴──────────┐
              │   Executor (WHAT)   │
              ├─────────────────────┤
              │ SimpleExecutor      │ ← one-shot fit+forward
              │ StudyExecutor       │ ← hyperparameter search
              │ PbtExecutor         │ ← population-based training
              │ StreamExecutor      │ ← chunked data
              └──────────┬──────────┘
                         │
              ┌──────────┴──────────┐
              │  Transport (DATA)   │
              ├─────────────────────┤
              │ WS inline  (<10MB)  │ ← small payloads
              │ HTTP bulk  (≥10MB)  │ ← large payloads
              │ DataStore (opt-in)  │ ← persistent S3/local
              │ WS Binary chunks    │ ← streaming
              └─────────────────────┘
```

---

## Data Transport

### Automatic routing (transparent to user)

```python
from soma import Graph, Filter

g = Graph()
g.node("model", MyModel())
g.add_worker("ws://gpu-server:8080", token="sk-xxx")

# Small data → WebSocket inline (automatic)
g.fit([1.0, 2.0, 3.0])

# Large data (≥10MB) → HTTP POST /upload (automatic)
big_data = [float(i) for i in range(2_000_000)]
g.fit(big_data)
```

### DataStore (opt-in, persistent)

```python
# Local storage
g.set_data_store("local", path="/data/soma")

# S3 storage
g.set_data_store("s3",
    bucket="my-lab",
    prefix="experiments/",
    endpoint="s3.amazonaws.com",
    access_key="AK...",    # or env AWS_ACCESS_KEY_ID
    secret_key="SK...",    # or env AWS_SECRET_ACCESS_KEY
)

# Now all data goes through the store
g.fit(data)  # → uploaded to store, worker reads by reference
```

### Streaming (chunked)

```python
# Forward in chunks via WebSocket Binary
result = g.forward(large_data, stream=True, chunk_size=1024)
```

Each chunk is processed independently by the worker's StreamExecutor.
Supports three modes per filter:

| StreamMode | Behavior |
|------------|----------|
| `FixedState` | Each chunk independent, cacheable |
| `Evolving` | State mutates per chunk, periodic checkpoints |
| `Barrier` | Accumulates all chunks, processes as batch on flush |

---

## Training Strategies

Set a strategy on the graph to control distributed training.

### Local (default)

```python
g = Graph()
g.node("model", MyModel())
g.fit(data)  # runs locally, no workers needed
```

:::danger[These four strategies are a design, not a running feature]
Read this before the code below, because the code below does not do what
its comments say.

`Graph::set_strategy` exists in Rust (`soma-core/src/graph.rs`) and records
the strategy on the graph, which serializes with it. That is all that
happens. Verified on 2026-08-05:

- **Python has no `set_strategy`.** Not "not yet wired" — the method does
  not exist, so every `g.set_strategy(...)` on this page raises
  `AttributeError`.
- **The scheduler does not read it.** `soma-compiler/src/scheduler.rs`
  contains no reference to the strategy; it groups differentiable nodes
  and distributes parallel branches, and would do so identically whatever
  strategy is set.
- **The training loops have no caller.** `soma-runtime/src/strategy.rs`
  implements `StrategyExecutor` for `TrainingStrategy` — the sharding, the
  gradient aggregation, the federated round loop are all written — and
  nothing in the workspace calls `fit` on it. `ModelParallel` and
  `PopulationBased` return "not yet implemented"; `Local`, `DataParallel`
  and `Federated` are implemented and equally unreachable.

So `g.fit(data)` does not shard, does not AllReduce, and does not run
federated rounds. What *does* work today is placement: `add_worker`,
`set_coordinator`, `target=` on a node, and remote execution of the
compiled plan — the sections above this one.

The snippets below are kept because they describe the intended API and
the semantics each strategy is meant to have. Treat them as a
specification.
:::

### Data Parallel

Replicates the model on N workers, each trains on a shard of the data.
Gradients are synchronized after each step.

```python
from soma import Graph

g = Graph()
g.node("model", MyModel())
g.set_strategy("data_parallel", num_replicas=4, aggregation="all_reduce")

g.add_worker("ws://gpu-0:8080", tags=["gpu"])
g.add_worker("ws://gpu-1:8080", tags=["gpu"])
g.add_worker("ws://gpu-2:8080", tags=["gpu"])
g.add_worker("ws://gpu-3:8080", tags=["gpu"])

g.fit(data)  # shards data across 4 workers, AllReduce gradients
```

### Federated

Data stays on workers. Only model updates are shared.

```python
g.set_strategy("federated",
    num_clients=10,
    rounds=50,
    aggregation="fed_avg",
)

# Each client trains on local data
# Coordinator aggregates states after each round
g.fit(data)
```

### Model Parallel

Split the model across workers. Different nodes run on different machines.

```python
g = Graph()
g.node("encoder", Encoder(), target="gpu-0")
g.node("classifier", Classifier(), target="gpu-1")
g.edge("encoder", "classifier")

g.add_worker("ws://gpu-0:8080", tags=["gpu-0"])
g.add_worker("ws://gpu-1:8080", tags=["gpu-1"])

g.fit(data)  # encoder on gpu-0, classifier on gpu-1
```

### Population-Based Training

Evolutionary hyperparameter optimization. Each generation trains a population,
evaluates, then evolves the best.

```python
g.set_strategy("pbt",
    population_size=20,
    generations=50,
    exploit="truncation",
    explore="perturbation",
)
```

---

## Workers

### Starting a worker

```bash
# Basic
somatize-worker --port 8080

# With GPU routing
CUDA_VISIBLE_DEVICES=0 somatize-worker --port 8080 --tags gpu-0

# With authentication
somatize-worker --port 8080 --token sk-my-secret

# With resource limits
somatize-worker --port 8080 --cpus 4 --memory 8G --gpus 1

# Multiple workers per machine (one per GPU)
CUDA_VISIBLE_DEVICES=0 somatize-worker --port 8080 --tags gpu-0 &
CUDA_VISIBLE_DEVICES=1 somatize-worker --port 8081 --tags gpu-1 &
```

### Worker architecture

The worker is a **LocalRunner** that listens on a port. Python filters execute
in a **child subprocess** — the GIL is completely isolated from Rust/Tokio.

```
somatize-worker process (Rust + Tokio)
  ├── HTTP server (health, upload, download)
  ├── WebSocket handler (receive plans, send results)
  └── Python child process (per plan)
       ├── cloudpickle.loads() → filters loaded
       ├── model on GPU
       └── fit/forward via stdin/stdout JSON Lines
```

### Connecting from Python

```python
from soma import Graph

g = Graph()
g.add_worker("ws://gpu-server:8080", token="sk-xxx", tags=["gpu"])

# Or via SSH tunnel
# ssh -L 8080:localhost:8080 gpu-server
g.add_worker("ws://localhost:8080", token="sk-xxx")

# Shutdown a worker
g.shutdown_worker("ws://gpu-server:8080")
g.shutdown_workers()  # all workers
```

---

## Executors

### SimpleExecutor (default)

One-shot: compile → fit → forward. Used internally by `g.fit()` and `g.forward()`.

### StudyRunner (hyperparameter optimization)

```python
from soma import Graph, Study, search

class MyModel(Filter):
    _kind = "trainable"
    lr: float = search(1e-4, 1e-1, scale="log")
    hidden: int = search(32, 256)

    def fit(self, x, y=None):
        # train with self.lr, self.hidden
        ...

study = Study(
    graph=g,
    objective="minimize",
    metric="loss",
    strategy="bayesian",
    n_trials=100,
)
study.run()
```

### StreamExecutor (chunked processing)

For datasets too large to fit in memory. Each filter declares its StreamMode:

```python
class MyEncoder(Filter):
    _kind = "stateless"
    _stream_mode = "fixed_state"  # each chunk independent

    def forward(self, x, state):
        return encode(x)

class MyAggregator(Filter):
    _kind = "stateless"
    _stream_mode = "barrier"  # accumulate all chunks, process as batch

    def forward(self, x, state):
        return aggregate(x)
```

---

## Composite Execution (Autograd)

When consecutive filters are differentiable, the compiler groups them into
a `Composite` block. All filters execute in a single Python process with
PyTorch tensors passed directly — autograd stays connected.

```python
class Encoder(Filter):
    _kind = "trainable"
    # _differentiable = True (default for trainable)

    def __init__(self):
        self.linear = torch.nn.Linear(768, 256)
        self.optimizer = torch.optim.Adam(self.linear.parameters())

    def forward(self, x, state):
        return self.linear(x)

class Classifier(Filter):
    _kind = "trainable"

    def __init__(self):
        self.linear = torch.nn.Linear(256, 10)
        self.optimizer = torch.optim.Adam(self.linear.parameters())
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x, state):
        return self.linear(x)

g = Graph()
g.node("encoder", Encoder())
g.node("classifier", Classifier())
g.edge("encoder", "classifier")

# Both are differentiable → Composite block
# backward() flows through classifier → encoder
g.fit(data, labels)
```

---

## Filter Introspection (for Nous agents)

```python
# Get source code of a filter (for agent editing)
source = g.filter_source("encoder")

# Get all sources
sources = g.filter_sources_dict()
# {"encoder": "class Encoder(Filter):...", "classifier": "class Classifier(Filter):..."}
```
