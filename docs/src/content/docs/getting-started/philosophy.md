---
title: Philosophy
description: Core principles that guide Soma's design.
---

## Design Principles

### Everything is a Transformation Graph

Every process in Soma, from a simple data normalization to a multi-stage ML experiment, is represented as a directed graph of transformations. This single abstraction covers:

- Sequential pipelines
- Parallel branches
- Conditional logic
- Iterative loops
- Nested sub-graphs

There is no separate concept for "pipeline", "workflow", or "DAG". A pipeline is just a graph. A workflow is just a graph. Composition is recursive.

### Lazy by Default, Eager by Request

Values in Soma are virtual references until someone needs the actual data. This means:

- No computation happens until `materialize()` is called
- The compiler can inspect the entire graph before executing anything
- Intermediate results that nobody reads are never computed
- Cache lookups happen before any real work

```
g.forward(x)           # lazy: returns VirtualValue
g.forward(x).collect() # eager: materializes the result
```

### Co-location of Concerns

Configuration lives where it belongs:

- **Search spaces** are defined in the filter struct, not in an external config
- **Cache behavior** is declared per filter, not globally
- **Stream semantics** are part of the filter contract, not a runtime flag
- **Gradient support** is declared in filter metadata, verified by the compiler

This reduces the distance between "what a filter does" and "how it's configured", preventing the drift that happens when configuration lives in separate files.

### The Compiler is the Optimizer

Soma's compiler does more than convert graphs to execution plans. It:

1. **Validates** type compatibility between connected filters
2. **Resolves caching** by computing keys and checking the store
3. **Detects gradient flow** and warns about interruptions
4. **Plans parallelism** by identifying independent branches
5. **Schedules distribution** by matching filters to available workers
6. **Estimates cost** by querying cache metadata without loading data

The plan that comes out of the compiler is already optimized. The runtime just executes it.

### Rust Core, Python Interface

Soma is written in Rust for:

- **Performance**: Zero-cost abstractions, no GC pauses
- **Safety**: Ownership model prevents data races in parallel execution
- **Serialization**: Serde enables efficient plan serialization for remote execution
- **Ecosystem**: Tokio for async, Polars for DataFrames, Candle/Burn for tensors

But the primary user interface is Python (via PyO3), because that's where researchers work. The Python API must be **as simple or simpler than LabChain**:

```python
from soma import Graph, Filter

class MyFilter(Filter):
    scale: float = search(0.1, 10.0, scale="log")

    def fit(self, x, y=None):
        return {"mean": x.mean(0)}

    def forward(self, x, state):
        return (x - state["mean"]) * self.scale

g = Graph.somatize(MyFilter(scale=2.0))
g.fit(train_data)
result = g.forward(test_data)
```

### Extensibility Through Derivation

Users extend Soma by implementing the `Filter` trait (Rust) or inheriting from `Filter` (Python). No plugin system, no configuration files, no registration boilerplate. A filter that implements the trait is automatically:

- A valid node in any graph
- Cacheable (if declared)
- Serializable for remote execution
- Searchable (if search spaces are annotated)
- Streamable (based on declared semantics)

### Events as First-Class Citizens

Every execution produces a stream of structured events at three levels:

1. **Graph level**: Node started, completed, cache hit, failed
2. **Trial level**: Metrics reported, trial pruned, trial completed
3. **Study level**: Best updated, Pareto front changed, study completed

These events enable real-time visualization, monitoring, logging, and agent decision-making without coupling the runtime to any specific UI or tracking system.

### The Agent is a User, Not a Component

Soma doesn't embed agent logic into the runtime. Instead, agents interact with Soma the same way a human user does:

- Define a graph (or ask the agent to generate one)
- Submit it for execution (local or remote)
- Receive events and results
- Query the knowledge base
- Decide next steps

The difference is that agents can do this programmatically, in a loop, and at scale. But the API is the same. This keeps the runtime clean and the agent layer independent.
