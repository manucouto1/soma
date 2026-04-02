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
│  Graphs become nodes in orchestration graphs          │
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

The orchestration layer. Enables visual composition of graphs and agents:

- **Graph publishing**: A compiled graph becomes a node in the platform orchestration layer
- **Agent integration**: Autonomous agents build, execute, and analyze graphs
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

Soma supports three execution modes, all using the same graph definition:

### Local Execution

```python
g = Graph.somatize(Scaler() >> Model())
g.fit(train_data)
result = g.forward(test_data)
```

The runtime compiles and executes the graph in the current process. Cache is local (memory + disk).

### Remote Execution

```python
g = Graph.somatize(Scaler() >> Model())
g.add_worker("ws://gpu-0:8080", token="sk-xxx")
g.set_strategy(DataParallel(num_replicas=2))
g.fit(train_data)
```

The graph is compiled locally, the plan is sent to workers, executed remotely, and results returned. Cache can be shared (S3) across workers.

### Coordinator Execution

```python
g = Graph.somatize(Scaler() >> Model())
g.set_coordinator("http://coord:9090", token="sk-xxx")
g.fit(train_data)  # coordinator routes to available workers
```

Workers self-register with the coordinator. The client submits plans and the coordinator routes to the best available worker based on tags, capacity, and strategy.

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
    // Run level (per-execution)
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
