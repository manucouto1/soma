---
title: Problem & Solution
description: The problems Soma solves and how it solves them.
---

## The Fragmentation Problem

Current systems separate multiple responsibilities across incompatible tools:

| Responsibility | Typical Tools | Limitation |
|---|---|---|
| ETL & pipelines | Apache Airflow, Luigi | Execution graphs, but no caching or ML awareness |
| Distributed processing | Apache Spark | Powerful but no inter-run caching, no agents, no streaming unification |
| Stream processing | Kafka Streams, Flink | Separate paradigm from batch, different APIs |
| ML experiment tracking | W&B, MLflow | Logging layer only, doesn't execute pipelines |
| Hyperparameter optimization | Optuna, Ray Tune | External to the pipeline definition |
| Agent frameworks | LangChain, CrewAI | Focus on LLM orchestration, not data processing |
| Vector databases | Pinecone, Weaviate | No temporal dimension, no trajectory analysis |

This creates friction when building systems that require:

- Rapid iteration on data experiments
- Experimental reproducibility with automatic caching
- Hybrid execution (batch + streaming) with the same code
- Direct integration with autonomous agents
- Temporal analysis of research trajectories

## What Soma Unifies

Soma provides a single execution model where:

> Every process is a data transformation flow represented as an executable computational graph.

```
┌──────────────────────────────────────────────────────────┐
│                         SOMA                              │
│                                                           │
│  Pipelines    ──┐                                         │
│  Caching      ──┤                                         │
│  Streaming    ──┼──► Unified computational graph runtime  │
│  Optimization ──┤                                         │
│  Distribution ──┤                                         │
│  Agents       ──┘                                         │
└──────────────────────────────────────────────────────────┘
```

### Solution 1: Computation as Graphs

Every pipeline is a directed graph where:

- **Nodes** are filters (trainable transformations)
- **Edges** define data flow and dependencies
- The **compiler** converts graphs into optimized execution plans
- The **runtime** executes plans with parallelism, events, and caching

### Solution 2: Data Virtualization

In Soma, data is not a static entity but a **potential result of a transformation that can be materialized on demand**:

```
VirtualValue::Cached    → stored in K/V, load on access
VirtualValue::Deferred  → not computed yet, has a "recipe"
VirtualValue::Stream    → materializes chunk by chunk
```

This enables lazy evaluation, deferred execution, and working with virtual datasets without immediate materialization -- like Denodo's data virtualization, but for computation rather than SQL queries.

### Solution 3: Cache-Aware Compilation

The compiler resolves caching **before execution**:

1. Analyze the graph topology
2. Compute cache keys for each node: `hash(filter_config + input_data_hash)`
3. Replace cached nodes with `ExecutionPlan::Cached`
4. The resulting plan already knows what to execute and what to reuse

This is more powerful than post-hoc caching because it can **optimize the entire plan** before running anything.

### Solution 4: Filter-Level Search Spaces

Hyperparameter search spaces are defined where the parameters live -- in the filter itself:

```rust
#[derive(Filter)]
struct MyClassifier {
    #[soma(search(low = 0.001, high = 100.0, scale = "log"))]
    C: f64,

    #[soma(search(choices = ["linear", "rbf", "poly"]))]
    kernel: String,
}
```

The pipeline aggregates all search spaces automatically. The Study orchestrates optimization without the user manually mapping parameters. Type validation happens at compile time.

### Solution 5: Unified Batch + Stream

Soma eliminates the traditional distinction between offline pipelines and real-time processing. A single filter definition works on both:

- Complete datasets (batch)
- Continuous data streams (chunked)

The filter declares its stream semantics (`FixedState`, `Evolving`, `Barrier`) and the runtime adapts execution accordingly.

### Solution 6: Pipelines as Platform Nodes

A compiled pipeline can be published to the platform, where it becomes a node in a larger orchestration graph alongside agents. This enables visual composition of research workflows where agents analyze results, refine hypotheses, and launch new experiments.
