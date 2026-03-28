---
title: Architecture Overview
description: High-level architecture of the Soma runtime.
---

## Layered Architecture

Soma is organized in three conceptual layers, each with clear responsibilities:

```
┌─────────────────────────────────────────────────────┐
│  Layer 3: PLATFORM                                   │
│  Agents, knowledge base, visual graph editor         │
│  Pipelines become nodes in orchestration graphs      │
├─────────────────────────────────────────────────────┤
│  Layer 2: COMPILER + PLANNER                        │
│  Graph → ExecutionPlan                              │
│  Cache resolution, gradient flow analysis            │
│  Cost estimation, distribution planning              │
├─────────────────────────────────────────────────────┤
│  Layer 1: RUNTIME                                   │
│  Plan execution, event system, parallelism           │
│  Tiered cache (memory/disk/remote), stream support   │
│  Optimization engine (samplers, pruners)             │
└─────────────────────────────────────────────────────┘
```

### Layer 1: Runtime

The execution engine. Receives a compiled `ExecutionPlan` and executes it:

- **Tree-walk executor**: Recursively walks the plan tree (Sequence, Parallel, Loop, Branch)
- **Event bus**: Broadcasts structured events via async channels
- **Tiered cache**: Memory / RocksDB / S3 with automatic promotion and eviction
- **Optimization engine**: Runs Studies with configurable samplers and pruners
- **Stream support**: Processes chunks with configurable filter semantics

### Layer 2: Compiler

The intelligence layer. Converts a user-defined `Graph` into an optimized `ExecutionPlan`:

- **Topological analysis**: Detects parallelizable branches, barriers, and dependencies
- **Cache resolution**: Computes cache keys and replaces cached nodes before execution
- **Gradient flow verification**: Warns when non-differentiable filters break the gradient chain
- **Schema validation**: Ensures type compatibility between connected filters
- **Cost estimation**: Queries cache metadata to estimate execution time

### Layer 3: Platform (Future)

The orchestration layer. Enables visual composition of pipelines and agents:

- **Pipeline publishing**: A compiled pipeline becomes a node in the platform graph
- **Agent integration**: Autonomous agents build, execute, and analyze pipelines
- **Knowledge base**: ChronosVector-powered temporal experiment tracking
- **Workers**: Remote execution with configurable infrastructure

## Data Flow

```
User defines Graph (code or visual)
        │
        ▼
┌─── Compiler ───┐
│  Validate       │
│  Resolve cache  │
│  Check grads    │
│  Build plan     │
└────────┬────────┘
         │
    ExecutionPlan
         │
         ▼
┌─── Runtime ────┐
│  Execute plan   │
│  Emit events    │◄──── Event subscribers (UI, logging, agents)
│  Cache results  │
│  Return value   │
└────────┬────────┘
         │
    VirtualValue (lazy, materializable)
         │
         ▼
┌─── CacheStore ─┐
│  Memory   <1ms  │
│  RocksDB  ~1ms  │
│  S3       ~50ms │
└─────────────────┘
```

## Execution Modes

Soma supports three execution modes, all using the same pipeline definition:

### Local Execution

```python
pipeline = Pipeline([MyScaler(), MyClassifier()])
pipeline.fit(x_train, y_train)
result = pipeline.predict(x_test)
```

The runtime compiles and executes the pipeline in the current process. Cache is local (memory + disk).

### Remote Execution

```python
lab = soma.connect("https://my-lab.soma.dev")
result = lab.run(pipeline, data=x_train)
```

The pipeline is serialized (filters + config + plan), sent to a worker, executed remotely, and results returned. Cache can be shared (S3) across workers.

### Platform Execution (Future)

```python
lab.publish(pipeline, name="my_experiment")
```

The pipeline becomes a node in the platform's visual graph editor, where it can be connected with agents and other pipelines in an orchestration graph.

## Type System

Soma uses Rust's type system to enforce correctness at compile time:

```rust
// Values flowing between filters
enum Value {
    Tensor(Tensor),                               // numeric data
    Json(serde_json::Value),                       // structured data
    DataFrame(LazyFrame),                          // tabular data
    Bytes(Vec<u8>),                                // raw bytes
    Stream(Pin<Box<dyn Stream<Item = Value>>>),    // chunked stream
    Virtual { key: CacheKey, schema: Schema },     // lazy reference
}

// Events emitted during execution
enum Event {
    // Pipeline level (per-run)
    RunStarted { .. }, NodeStarted { .. }, NodeCompleted { .. }, ..
    // Trial level (per-hyperparameter evaluation)
    TrialStarted { .. }, TrialMetric { .. }, TrialPruned { .. }, ..
    // Study level (per-optimization)
    StudyProgress { .. }, BestUpdated { .. }, ParetoUpdated { .. }, ..
}

// Execution plans produced by the compiler
enum ExecutionPlan {
    Sequence(Vec<ExecutionPlan>),
    Parallel(Vec<ExecutionPlan>),
    Execute { id: NodeId, process: Arc<dyn Filter> },
    Cached { id: NodeId, key: CacheKey },
    Loop { .. },
    Branch { .. },
    Remote { target: RemoteTarget, plan: Box<ExecutionPlan> },
}
```
