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
| **Persistent cache** | Crash-safe cache-by-default: content-addressed action store, `soma cache stats\|gc\|pin\|verify`. |
| **Run tracking** | Every `track_run`/`Study` writes a run directory (`.soma/runs/<id>/`): manifest, lossless event log, metrics, diagnostics. Crash-safe resume; readers never block training. |
| **Visualization** | Annotated architecture diagrams, Optuna-style HPO charts, gradient-health views, and one-file HTML reports — all read from run directories. |
| **TrainingStrategy** | Graph-level attribute: Local, DataParallel, ModelParallel, Federated, PopulationBased. |
| **Study** | Hyperparameter optimization: Grid, Random, or Bayesian (TPE) search with median/percentile pruning. |
| **PBT** | Population-Based Training: evolutionary train→evaluate→exploit/explore cycles. |
| **ExecutionPlan** | Compiled from graph. Variants: Sequence, Parallel, Execute, Cached, Remote, Loop, Branch. |
| **DataStore** | Abstraction for data movement: Local, S3, Zarr (chunked tensors), Cached, Stream. |
| **Worker** | Remote execution daemon. Auto-detects hardware, Slurm-style resource limits, token auth. |
| **Coordinator** | Lightweight gateway: worker registration, routing, health monitoring. |

## Workspace (11 crates)

```
soma-macros      → proc macro (#[derive(SomaFilter)])
soma-core        → types + traits: Filter, Value, Graph, Event, Schema, Study,
                   DataStore (Local/S3/Zarr), VirtualValue, tracking schema, GraphOverlay
soma-compiler    → Graph → ExecutionPlan (caching, parallelism, distribution)
                   Scheduler, plan visualization
soma-runtime     → GraphSession, executor, FilterLibrary, caches, samplers, pruners
                   StudyRunner, PbtRunner, LocalTracker + RunReader (run directories)
soma-memory      → KnowledgeBase trait + MemoryKB + ChronosKB
soma-worker      → Worker, Protocol, EnvManager, token auth, CLI binary
soma-coordinator → worker registry, routing, heartbeat monitoring
soma-agent       → Research agent loop (observe → hypothesize → experiment → conclude)
soma-mcp         → MCP server (13 tools for code, execution, knowledge)
soma-python      → PyO3 bindings: Graph, Study, Run, RunView, soma.viz, Chain/Fork operators
somatize (soma/) → facade crate re-exporting the workspace
```

## Quick Start

```bash
# Run all tests (875+: 577 Rust + 298 Python)
cargo test --workspace
cd soma-python && maturin develop && pytest tests/ -v

# With S3/Zarr DataStore
cargo test -p somatize-core --features s3
cargo test -p somatize-core --features zarr

# With ChronosVector
cargo test -p somatize-memory --features chronos

# MCP server
cargo run -p somatize-mcp -- /path/to/project
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

# Complex topologies
g = Graph.somatize(
    (LoadA() >> NormA() | LoadB() >> NormB())
    >> Aggregate()
    >> Backbone()
    >> (HeadA() | HeadB())
)

# Distributed training
g.set_strategy(DataParallel(num_replicas=4))
g.set_coordinator("http://coord:9090", token="sk-xxx")
```

## See your experiments

Everything a run produces lands in a run directory; everything below
just reads it back.

```python
# Track a run — architecture snapshot + lossless event log + metrics
with g.track_run("mos-baseline", tags=["mos"]) as run:
    g.fit(train_data)
    run.log("val_f1", evaluate(g), step=0)

# List and inspect runs (crashed runs are detected via stale heartbeat)
for run in soma.runs():
    print(run.id, run.state, run.name)

view = soma.runs()[0]
view.node_timings()         # per-node wall times, durations, cache hits
print(view.to_mermaid())    # architecture annotated with timing/cache/health
```

```text
graph LR
    scaler["scaler<br/>26ms"]
    model["model<br/>27ms · ⚠ DEAD_CHANNELS(2)"]
    scaler --> model
    class scaler soma_completed
    class model soma_flagged
```

With the `viz` extra (`pip install 'somatize[viz]'`) you get interactive
Plotly figures and pandas projections:

```python
study.plot_optimization_history()   # objective per trial + best-so-far
study.plot_parallel_coordinate()    # params → objective
study.plot_param_importances()      # Spearman rank correlation
study.plot_timeline()               # trial gantt
study.trials_dataframe()            # pandas table

view.plot_metrics()                 # logged metric curves
view.plot_gantt()                   # where the run's wall time went
view.plot_health()                  # gradient-audit health flags
view.plot_channels("encoder")      # channel-correlation heatmap
```

And from the shell:

```bash
soma runs                            # list tracked runs
soma graph <run_id>                  # annotated mermaid/dot diagram
soma report <run_id> -o report.html  # self-contained HTML report
soma report <run_id> --inline        # fully offline (embeds plotly.js)
```

`soma report` packages the annotated DAG, efficiency tiles, metric
curves, the full HPO section with trial table, and the health section
into one shareable file. A future live GUI reads the same run-directory
files and the same embedded JSON shapes — see
`docs design/visualization.md`.

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

## Feature Flags & Extras

- `somatize-core/s3` — S3-compatible DataStore (AWS, Backblaze B2, MinIO)
- `somatize-core/zarr` — Zarr v3 chunked tensor storage with compression
- `somatize-memory/chronos` — ChronosVector-backed KnowledgeBase
- `somatize[viz]` (pip) — Plotly figures, DataFrames, HTML reports

## License

[Elastic License 2.0](LICENSE)
