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

### The Whole Graph Before the First Node

`g.forward(x)` is eager: it returns the value. What is deferred is not the
call but the *work inside it*, and the deferral is the compiler's, not the
caller's:

- The graph is compiled before anything runs, so schema mismatches and cycles
  are errors before the first `forward`, not during it
- Every node's cache key is derived and looked up **before** its work is
  attempted, so a warm node costs a hash and a read
- A downstream key is derived from the *content* that arrived, so a node whose
  output did not change stops the invalidation there — an ancestor recomputing
  does not force its descendants to

Values do travel through the executor as `VirtualValue`s, but that is how the
run's store holds them; it is not an API you call `.collect()` on.

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
2. **Plans distribution** by assigning nodes to workers
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

### One Substrate, Two Profiles

Soma used to say the agent was a user of the runtime and not a part of it. That was a way of saying the runtime should stay clean — a good instinct with the wrong conclusion, because it left agentic work outside everything the runtime is good at. An agent that talks to Soma from the outside gets no cache, no schema checking on its handoffs, no search space over its prompts, and no lineage on what it tried.

So the line moved. It now runs between *structure* and *behaviour*, not between *computation* and *agency*:

- **The substrate** — the runtime — understands a small set of structural things: sequence, parallel, branch, loop, subgraph, and one effectful node. Six shapes, and it has to understand them because it has to schedule them.
- **The profiles** — everything above — are libraries. An LLM call, a tool, a judge, a router, a retriever, a debate: none of these are node types the runtime knows. They are filters and steps registered like any other, and the named patterns are functions that build graphs.

Concretely, adding "debate between three agents, judged, up to four rounds" is a function in `soma.agentic`. It costs nothing in the core and nothing to maintain. The frameworks that made each pattern an enum variant all ended up with dead variants in their catalog, still documented, still shipped, no longer working.

The dividend is that everything Soma already does applies unchanged. An agentic graph is content-addressed and cached. Its edges are schema-checked, which turns the largest documented category of multi-agent failure — incompatible handoffs — into a compile error. Its prompts and models are search-space dimensions, its topology too. Its runs land in the experiment pool with lineage, so "which of these two flows was better" is a question with a recorded answer.

### Two Kinds of Node, Because There Are Two Kinds of Work

A `Filter` is deterministic: given the same configuration, state and input, it produces the same output, so it can be memoized by content and shared across runs forever.

A `Step` is effectful: it calls a model, runs a tool, sleeps, waits for a person. Memoizing it by content would be wrong — the second call is supposed to be able to differ. So its effects are journaled instead, the way durable-execution systems do it: recorded once, replayed on resume, never re-run. That is what makes a nondeterministic run reproducible, which is what makes it comparable to its parent in the pool.

Both are nodes in the same graph, on the same edges, under the same compiler.

### Suspension is a Feature, Not a Failure

A step that needs a human, an approval, or a result that will not exist for an hour does not block a thread. It suspends, and its position is a journal entry like everything else. Resuming is replaying to that point and continuing — in a new process if need be.
