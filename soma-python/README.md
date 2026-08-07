# Soma

**Soma** (σῶμα — *body*) is a computational graph runtime for research pipelines, agent orchestration, and data virtualization. Written in Rust with Python bindings.

Part of the **Nous-Soma-Chronos** ecosystem:
- **[Nous](https://github.com/manucouto1/nous)**: Understands, reasons — research IDE, agent graphs, automation
- **Soma** (this project): Executes, materializes — graphs, optimization, distributed workers
- **[ChronosVector](https://github.com/manucouto1/chronos-vector)**: Remembers — temporal vector database

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Filter** | Data transformation with `fit()` (learn state) and `forward()` (transform). Independently cacheable. |
| **Graph** | Computational DAG of filters. Build with `.node()`/`.connect()` or `>>` / `\|` operators. |
| **Graph.somatize()** | *"You think it. Soma somatizes it."* — Materialize a chain/fork topology into an executable graph. |
| **TrainingStrategy** | Graph-level attribute: Local, DataParallel, ModelParallel, Federated, PopulationBased. |
| **Study** | Hyperparameter optimization: Grid, Random, or Bayesian (TPE) search with median/percentile pruning. |
| **PBT** | Population-Based Training: evolutionary train→evaluate→exploit/explore cycles. |
| **Agentic layer** | `soma.Agent`, `soma.Judge` and Python steps (any object with `poll(ctx)`) are nodes like any filter; a pipeline is a tool an agent can run (`soma.agentic.RunGraph`) and an agent is a node in a pipeline. Patterns (`react`, `refine`, `board`, …) live in `soma.agentic` and return ordinary graphs — searchable, cached, tracked. |
| **ExecutionPlan** | Compiled from graph. Variants: Sequence, Parallel, Execute, Step, Loop, Branch, Remote, Composite, Stream, Empty. |
| **DataStore** | Abstraction for data movement: Local, S3, Zarr (chunked tensors), Cached, Stream. |
| **Worker** | Remote execution daemon. Auto-detects hardware, Slurm-style resource limits, token auth. |
| **Coordinator** | Lightweight gateway: worker registration, routing, health monitoring. |

## Workspace (10 crates)

```
soma-macros     → proc macro (#[derive(SomaFilter)])
soma-core       → types + traits: Filter, Value, Graph, TrainingStrategy, Schema, Event
                  DataStore (Local/S3/Zarr), VirtualValue, StreamCache
soma-compiler   → Graph → ExecutionPlan (caching, parallelism, distribution)
                  Scheduler, plan visualization (Mermaid/Graphviz)
soma-runtime    → GraphSession, executor, NodeCatalog (filters AND steps), caches,
                  samplers, pruners, EffectDriver + journal, StudyRunner, PbtRunner,
                  stream executor
soma-memory     → KnowledgeBase trait + MemoryKB + ChronosKB
soma-worker     → Worker, Coordinator, Protocol, EnvManager, token auth
                  Auto-detect capabilities, resource limits, CLI binary
soma-agent      → Research agent loop (observe → hypothesize → experiment → conclude)
soma-mcp        → MCP server (13 tools for code, execution, knowledge)
soma-python     → PyO3 bindings: Graph, Filter, Study, Chain/Fork operators
```

## Quick Start

```bash
# Run all tests (355 Rust + 29 Python)
cargo test --workspace
cd soma-python && maturin develop && pytest tests/ -v

# With S3/Zarr DataStore
cargo test -p soma-core --features s3
cargo test -p soma-core --features zarr

# With ChronosVector
cargo test -p soma-memory --features chronos

# MCP server
cargo run -p soma-mcp -- /path/to/project
```

## Python Usage

```python
from soma import Filter, Graph, Study, search

class Scaler(Filter):
    _differentiable = True

    def fit(self, x, y=None):
        return {"mean": sum(x) / len(x)}

    def forward(self, x, state):
        return [v - state["mean"] for v in x]

class Model(Filter):
    lr: float = search(0.001, 1.0, scale="log")

    def fit(self, x, y=None):
        return {"weights": [0.5] * len(x)}

    def forward(self, x, state):
        return [v * w for v, w in zip(x, state["weights"])]

# Build with >> (chain) and | (fork)
g = Graph.somatize(Scaler() >> Model())
g.fit(train_data)
result = g.forward(test_data)

# Visualize
print(g.to_mermaid())
print(g.to_text())

# Complex topologies
g = Graph.somatize(
    (LoadA() >> NormA() | LoadB() >> NormB())
    >> Aggregate()
    >> Backbone()
    >> (HeadA() | HeadB())
)

# Events
g.on_event(lambda e: print(e["event_type"], e.get("node_id", "")))

# Distributed training
g.set_strategy(DataParallel(num_replicas=4))
g.set_coordinator("http://coord:9090", token="sk-xxx")
```

## Workers

```bash
# Start a worker with auto-detected capabilities
soma-worker --port 8080 --tags gpu,training --token sk-xxx

# With resource limits (Slurm-style)
soma-worker --cpus 4 --memory 8G --gpus 1 --max-concurrent 2

# With coordinator auto-registration
soma-worker --coordinator http://coord:9090 --token sk-xxx --tags gpu
```

Workers auto-detect CPU cores, RAM, GPUs (nvidia-smi), and Python environments.
Each worker creates isolated venv/conda environments per job with incremental dependency updates.

## Feature Flags

- `soma-core/s3` — S3-compatible DataStore (AWS, Backblaze B2, MinIO)
- `soma-core/zarr` — Zarr v3 chunked tensor storage with compression
- `soma-memory/chronos` — ChronosVector-backed KnowledgeBase

## License

[Elastic License 2.0](LICENSE)
