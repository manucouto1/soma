# Soma

**Soma** (σῶμα — *body*) is a computational graph runtime for research pipelines, agent orchestration, and data virtualization. Written in Rust with Python bindings.

Part of the **Nous-Soma-Chronos** ecosystem:
- **[Nous](https://github.com/manucouto1/nous)**: Understands, reasons — research IDE, agent graphs, automation
- **Soma** (this project): Executes, materializes — pipelines, optimization, distributed workers
- **[ChronosVector](https://github.com/manucouto1/chronos-vector)**: Remembers — temporal vector database

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Filter** | Data transformation with `fit()` (learn state) and `forward()` (transform). Independently cacheable. |
| **Pipeline** | Sequence of filters. Automatic SHA-256 content-addressable caching with cascade invalidation. |
| **Study** | Hyperparameter optimization: Grid, Random, or Bayesian (TPE) search with median/percentile pruning. |
| **ExecutionPlan** | Compiled from graph. Variants: Sequence, Parallel, Execute, Cached, Remote, Loop, Branch. |
| **DataStore** | Abstraction for data movement: Local, S3, Cached, Stream. Workers exchange DataRefs. |
| **Scheduler** | Distributes plan across workers: sequential→same worker, parallel→distribute, differentiable→group. |
| **Worker** | Remote execution daemon. Isolated Python environments per pipeline with incremental dependency updates. |

## Workspace (10 crates)

```
soma-macros     → proc macro (#[derive(SomaFilter)])
soma-core       → Filter, Value, Graph, CacheKey, DataStore, StreamCache, Schema
soma-compiler   → Graph → ExecutionPlan, Scheduler
soma-runtime    → Executor, Pipeline, StreamExecutor, Samplers, Pruners, StudyRunner
soma-memory     → KnowledgeBase trait + MemoryKB + ChronosKB
soma-worker     → Protocol, Worker, EnvManager, WebSocket server, Dockerfile
soma-agent      → Research agent loop (observe → hypothesize → experiment → conclude)
soma-mcp        → MCP server (13 tools for code, execution, knowledge)
soma-python     → PyO3 bindings: Pipeline, Filter, Study, Lab
```

## Quick Start

```bash
# Run all tests (297 Rust + 19 Python)
cargo test --workspace
cd soma-python && maturin develop && pytest tests/ -v

# With S3 DataStore
cargo test -p soma-core --features s3

# With ChronosVector
cargo test -p soma-memory --features chronos

# MCP server
cargo run -p soma-mcp -- /path/to/project
```

## Python Usage

```python
from soma import Filter, Pipeline, Study, search

class MyScaler(Filter):
    _differentiable = True
    clip = search.float(1.0, 10.0)

    def fit(self, x, y=None):
        return {"mean": x.mean(), "std": x.std()}

    def forward(self, x, state):
        return (x - state["mean"]) / state["std"]

pipe = Pipeline([("scale", MyScaler()), ("classify", KNN(k=5))])
pipe.fit(X_train, y_train)
predictions = pipe.predict(X_test)

study = Study(pipeline=pipe, strategy="bayesian", max_trials=50)
study.optimize(X_train, y_train, metric="accuracy")
```

## Worker (Distributed Execution)

```bash
# Docker (with common ML packages pre-installed)
docker run -d \
  -e NOUS_API_KEY=nous_xxx \
  -e NOUS_URL=wss://server/nous/ws/worker \
  --gpus all \
  ghcr.io/manucouto1/soma-worker:gpu

# Or native
pip install soma
soma-worker --api-key nous_xxx --coordinator wss://server/nous/ws/worker
```

Workers create isolated venv/conda environments per pipeline, with incremental dependency updates (only install/upgrade/remove what changed).

## Feature Flags

- `soma-core/s3` — S3-compatible DataStore (AWS, MinIO)
- `soma-memory/chronos` — ChronosVector-backed KnowledgeBase

## License

[Elastic License 2.0](LICENSE)
