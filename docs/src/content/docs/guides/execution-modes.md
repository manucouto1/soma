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

:::caution[Three of these four run. Read this before the code below]
Verified on 2026-08-05, and the answer differs per strategy:

- **`federated` runs.** `g.set_strategy("federated", num_clients=…,
  rounds=…)` exists in Python, `fit` hands execution to the strategy when
  more than one worker is registered, and FedAvg averages the clients'
  states element-wise — including the dicts a Python filter's `fit`
  returns. Verified over two real workers: two shards whose means are 1.5
  and 5.5 produce 3.5, which no single client can.
- **`data_parallel` trains.** Each replica fits its own shard, the
  gradients are read off the workers, averaged, applied, and the stepped
  weights are read back. Verified over two real workers against a
  reference computed by hand: same initialisation, each shard's gradient
  taken separately, the two averaged, one SGD step — the weights match
  exactly, and differ from what either shard alone produces.

  Four things had to be true at once, and none of them were:
  a remote fit of a `DifferentiableFilter` now runs a **backward pass**
  (its `fit` learns no state — the parameters live in `_module` — so the
  worker used to report a trained model whose parameters had never seen a
  gradient); gradients cross the wire as **JSON**, not a torch pickle,
  because the aggregator is in Rust and cannot average an opaque blob;
  inputs and targets are **sharded together** (sharding only `x` sent every
  replica the whole `y`, shapes that broadcast rather than fail, so each
  replica trained on pairs that were never pairs); and the final state is
  **read back over the wire** rather than recalled from the fit, which
  returned the weights from before the averaged gradient was applied.
- **`model_parallel` runs.** Partitions tile the graph and each one is a
  *stage*: it runs on the worker it was pinned to, and its output is the
  next stage's input. The model is split; the data is not.

  ```python
  g.set_strategy("model_parallel", partitions=[
      {"nodes": ["encoder"],    "tag": "gpu0"},
      {"nodes": ["classifier"], "tag": "gpu1"},
  ])
  ```

  The partitions must tile the plan, and three things are refused rather
  than run: a node claimed by two partitions (it would run twice, on two
  machines, and the second activation would overwrite the first), a node
  claimed by nobody (model parallelism has no default target), and a
  stage interleaved with another (the activation would have to cross
  back). Use `"worker"` instead of `"tag"` to pin one by address.
- **`population_based` refuses, and it is not a gap.** PBT gives every
  member *different* hyperparameters, and applying those means rebuilding
  the graph's filters — which a worker cannot be asked to do, because it
  is sent a plan, not a way to construct one. That is the same shape as
  `Study`, so PBT is an executor driven from Python:

  ```python
  pbt = soma.Pbt(
      search_space=[{"type": "float", "name": "lr",
                     "low": 1e-4, "high": 1e-1, "scale": "log"}],
      population_size=8, generations=5,
  )
  best = pbt.run(train, evaluate)   # best[0] is the fittest member
  ```

  `train(member)` receives `{"id", "params", "state", "fitness"}` and
  returns the new state; `evaluate(member)` returns a number where
  **higher is better**. A single member that cannot be evaluated is
  scored at negative infinity and exploited away; a generation where
  *none* could be evaluated is an error, because a population of
  negative infinities looks ranked and has no signal in it.
- **`fed_prox` and `fed_yogi`** refuse too: the first needs the previous
  global model to measure drift against, the second the optimizer moments
  it carries between rounds, and this aggregator is given neither. A plain
  mean under their name would be FedAvg lying.

Until this was wired, `set_strategy` did not exist in Python at all, and
in Rust nothing read the attribute back — setting a strategy recorded it
and changed nothing.
:::

:::tip[Developing against a worker]
A worker builds an isolated venv per pipeline and installs `somatize`
into it from PyPI. A working tree is normally *ahead* of the last
release, so there is no such version to install and the worker would
quietly run an older Soma than the one that pickled the filters — which
is exactly how a `DifferentiableFilter` came to train on a worker and
diverge while the same graph ran fine locally.

A worker started from Python (`soma.Worker(...)`) now points its
environments at that interpreter's own `soma` package. For the standalone
`somatize-worker` binary, set it yourself:

```bash
SOMA_LOCAL_PACKAGE=/path/to/soma/soma-python/python somatize-worker --port 8080
```
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

Split the model across workers. Each partition is a stage: it runs where
it was pinned, and hands its activation to the next one.

```python
g = Graph()
g.node("encoder", Encoder(), target="gpu-0")
g.node("classifier", Classifier(), target="gpu-1")
g.connect("encoder", "classifier")

g.add_worker("ws://gpu-0:8080", tags=["gpu-0"])
g.add_worker("ws://gpu-1:8080", tags=["gpu-1"])

g.set_strategy("model_parallel", partitions=[
    {"nodes": ["encoder"],    "tag": "gpu-0"},
    {"nodes": ["classifier"], "tag": "gpu-1"},
])

g.fit(data)  # encoder on gpu-0, its output feeds classifier on gpu-1
```

The `target=` on a node routes a single remote node; `partitions=` is what
makes the graph a pipeline of stages. A partition that does not tile the
plan is refused — see the note above for the three cases.

### Population-Based Training

Evolutionary hyperparameter optimization: each generation trains a
population, evaluates it, and lets the underperformers copy and mutate
the leaders.

It is **not** a distribution strategy — see the note above — so it is
driven from Python like a `Study`:

```python
import soma

pbt = soma.Pbt(
    search_space=[
        {"type": "float", "name": "lr", "low": 1e-4, "high": 1e-1, "scale": "log"},
    ],
    population_size=20,
    generations=50,
    exploit="truncation",     # or "binary"
    explore="perturbation",   # or "resample"
)

def train(member):
    g = build_graph(lr=member["params"]["lr"])
    g.fit(train_x, train_y)
    return g.state()

def evaluate(member):
    return accuracy(member)   # higher is better

population = pbt.run(train, evaluate)
print(population[0]["params"], population[0]["fitness"])   # the fittest
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
