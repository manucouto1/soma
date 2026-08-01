---
title: Workers & Remote Execution
description: Distributed execution with serialized graphs and configurable infrastructure.
---

## Overview

Soma supports remote execution by serializing compiled graphs and sending them to **workers** -- daemon processes running on lab machines. This enables:

- Offloading heavy computation to GPU clusters
- Parallel execution across multiple machines
- Shared caching via remote storage (S3)
- Lab-wide resource management

## Architecture

```
┌─────────────────┐         ┌──────────────┐
│  User / Agent   │         │  Worker 1    │
│                 │         │  (GPU, 32GB) │
│  graph.run()    │────────►│              │
│  lab.run(study) │         │  soma-worker │
│                 │◄────────│  daemon      │
│  ← events       │         └──────────────┘
│  ← results      │
│                 │         ┌──────────────┐
│                 │         │  Worker 2    │
│                 │────────►│  (CPU, 128GB)│
│                 │         │              │
│                 │◄────────│  soma-worker │
│                 │         │  daemon      │
└─────────────────┘         └──────────────┘
         │
         │ shared cache
         ▼
    ┌──────────┐
    │  S3 / R2 │
    └──────────┘
```

## Worker Daemon

Each worker runs a `soma-worker` daemon that:

1. **Registers** with the coordinator (capabilities, available resources)
2. **Heartbeats** periodically (load metrics, availability)
3. **Receives** serialized plans from the coordinator
4. **Executes** plans using a local `soma-runtime` instance
5. **Streams** events back to the coordinator in real-time
6. **Returns** results (or stores them in shared cache)

```rust
pub struct Worker {
    pub id: WorkerId,
    pub capabilities: Capabilities,
    event_bus: Arc<EventBus>,
    cache: Arc<dyn CacheStore>,
    filters: NodeCatalog,
    /// Optional persistent DataStore (S3, Zarr, …), configured by the user.
    data_store: Option<Arc<dyn DataStore>>,
    /// Temporary local store for HTTP bulk uploads — auto-created, auto-cleaned.
    temp_store: Arc<LocalDataStore>,
    /// Creates venvs carrying the filters' dependencies.
    env_manager: EnvManager,
}

pub struct Capabilities {
    pub gpus: Vec<GpuInfo>,
    pub ram_bytes: u64,
    pub cpu_cores: usize,
    pub python_envs: Vec<String>,  // available conda/venv envs
    pub tags: Vec<String>,         // user-defined: "training", "inference"
}
```

## Serialization Strategy

The hybrid approach: workers have `soma-runtime` + lab's base image pre-installed. Soma only sends what changes: the plan and user-defined filters.

```rust
#[derive(Serialize, Deserialize)]
pub struct SerializedPlan {
    pub plan: ExecutionPlan,
    pub filters: Vec<SerializedFilter>,
    pub data_refs: Vec<DataRef>,        // references to data in shared storage
    pub cache_config: CacheConfig,
}

#[derive(Serialize, Deserialize)]
pub enum SerializedFilter {
    /// Built-in filter (already in worker's soma-runtime)
    Builtin(String),

    /// User's Python filter (sent as module + class + config)
    PythonModule {
        module_path: String,
        class_name: String,
        config: Value,
    },

    /// User's Rust filter (compiled artifact reference)
    RustPlugin {
        artifact_hash: String,
        symbol: String,
        config: Value,
    },
}
```

## Worker Protocol

Communication between coordinator and workers:

```rust
pub enum WorkerMessage {
    // Worker → Coordinator
    Register {
        id: WorkerId,
        capabilities: Capabilities,
    },
    Heartbeat {
        id: WorkerId,
        load: LoadMetrics,
    },
    Event(Event),           // streamed during execution
    PlanResult {
        plan_id: PlanId,
        result: Result<Value, String>,
    },

    // Coordinator → Worker
    AssignPlan {
        plan_id: PlanId,
        plan: SerializedPlan,
    },
    CancelPlan {
        plan_id: PlanId,
    },
}

pub struct LoadMetrics {
    pub cpu_usage: f32,
    pub memory_usage: f32,
    pub gpu_usage: Vec<f32>,
    pub active_plans: usize,
    pub queue_depth: usize,
}
```

## Distribution in the Compiler

The compiler's distribution planner assigns nodes to targets based on filter metadata:

```rust
pub enum Distribution {
    /// Execute locally (default)
    Local,
    /// Execute on a specific worker or worker tag
    Remote(RemoteTarget),
    /// Execute anywhere (let scheduler decide)
    Any,
}

pub enum RemoteTarget {
    WorkerId(WorkerId),
    Tag(String),           // e.g., "gpu", "high-memory"
    Any,
}
```

Filters declare their preferred distribution:

```rust
#[derive(SomaFilter)]
#[soma(distribution = "Remote(Tag(\"gpu\"))")]
struct GpuTrainer {
    // This filter should run on a GPU worker
}
```

The compiler wraps these nodes in `ExecutionPlan::Remote`:

```
Graph: [Preprocess] → [GpuTrainer] → [Evaluate]
       local           gpu            local

Plan:
Sequence([
    Execute(Preprocess),
    Remote {
        target: Tag("gpu"),
        plan: Execute(GpuTrainer),
    },
    Execute(Evaluate),
])
```

## Lab Configuration

:::caution[Partly implemented]
`soma.connect(...)` and `lab.workers()` work. `lab.run(study, data=...)` does
not exist — today you drive a study locally with `study.run(...)` and route
individual nodes to workers with `target=`, as shown above.
:::

Labs configure their workers and shared resources:

```python
lab = soma.connect("https://my-lab.soma.dev")

# List available workers
lab.workers()
# [Worker { id: "gpu-01", gpus: 4xA100, ram: 128GB, tags: ["gpu", "training"] },
#  Worker { id: "cpu-01", cpus: 64, ram: 256GB, tags: ["cpu", "preprocessing"] }]

# Run a study on the lab
lab.run(study, data=train_data)
# → Coordinator schedules trials across workers
# → GPU trials go to gpu-01, CPU trials go to cpu-01
# → Events streamed back in real-time
# → Results stored in shared cache (S3)
```

## Shared Caching

When workers share a remote cache (S3), computation is deduplicated across the entire lab:

```
Worker 1 runs: [Scaler] → [PCA] → [SVM]
  → Caches Scaler output to S3

Worker 2 runs: [Scaler] → [PCA] → [RandomForest]
  → Scaler output: S3 cache HIT (same config + same data)
  → Skips Scaler entirely
  → Only executes PCA (if not cached) and RandomForest
```

This is where the content-addressable caching becomes powerful at scale: no coordination needed between workers. If the same computation has been done anywhere in the lab, it's available everywhere.
