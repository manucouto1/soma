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
│  Validation, gradient flow analysis, distribution     │
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
- **Tiered cache**: in-memory LRU over a filesystem action store (BLAKE3 CAS), with promotion and value-density eviction
- **Optimization engine**: Runs Studies with configurable samplers and pruners
- **Stream support**: Processes chunks with configurable filter semantics

### Layer 2: Compiler

The intelligence layer. Converts a user-defined `Graph` into an optimized `ExecutionPlan`:

- **Topological analysis**: Detects parallelizable branches, barriers, and dependencies
- **Validation**: Cycle detection and schema compatibility between connected filters
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
│  Plan the DAG   │
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
│   files   ~1ms  │
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
g.fit(train_data)
```

:::note[Placement and strategy are two different things]
**Placement** is `add_worker` / `set_coordinator` and `target=` on a
node: the scheduler assigns the compiled plan across the registered
workers. That is what the snippet above uses, and it needs no strategy.

**A `TrainingStrategy`** changes what a fit *means*. Set one and
`GraphSession::fit` hands execution to it instead of the local runner:

```python
g.set_strategy("data_parallel", num_replicas=2)
g.fit(x, y)   # each replica fits a shard; the gradients are averaged
```

Three of the four run today — `federated`, `data_parallel` and
`model_parallel`. `population_based` refuses on purpose, because a
member's hyperparameters cannot be applied over the wire; PBT lives in
`soma.Pbt` instead. [Execution Modes](/soma/guides/execution-modes/) has
the verified account of each, including what data-parallel had to get
right to be real training rather than a loop that reports success.

Until 0.5.0 this note said the opposite, and was correct at the time:
`set_strategy` did not exist in Python, and nothing in Rust read the
attribute back.
:::

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

:::caution[Illustrative, not current]
The sketches below convey the *shape* of the type system and have drifted from
the code — `Value` has no `DataFrame` or `Stream` variant, and
`ExecutionPlan::Execute` carries only a `node_id`. For the actual definitions,
with `file:line` for every public trait, struct and enum, see
[Internals → Foundation](/soma/internals/foundation/) and the
[Symbol Index](/soma/internals/symbols/).
:::

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
    Loop { .. },
    Branch { .. },
    Remote { target: RemoteTarget, plan: Box<ExecutionPlan> },
}
```
