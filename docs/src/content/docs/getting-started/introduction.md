---
title: Introduction
description: What is Soma and why it exists.
---

## What is Soma?

**Soma** is a computational graph runtime written in Rust with Python bindings. It provides a unified execution model for research pipelines, data processing, and agent orchestration.

The name comes from the Greek word for "body". If an autonomous agent is the brain that decides *what to do*, Soma is the body that defines *how it's done* -- materializing decisions into reproducible, efficient, and composable computational flows.

## Origins

Soma converges ideas from three prior projects:

### LabChain

A Python framework for defining ML experiments as pipelines of reusable filters with content-addressable caching. Key contributions to Soma:

- **Filter model**: `fit()` / `predict()` lifecycle for trainable transformations
- **Content-addressable caching**: SHA-based identity for filters and data, enabling automatic deduplication of computation across experiments
- **Reproducibility**: Full experiment serialization to JSON
- **Storage backends**: Local filesystem and S3 with distributed locking

### Chatty the Lab

A Rust/SvelteKit platform for LLM agent orchestration with a visual graph editor. Key contributions to Soma:

- **Graph compilation**: Converting DAGs into structured `ExecutionPlan` trees (Sequence, Parallel, Loop, Branch)
- **Event system**: Real-time streaming of execution progress (NodeStarted, NodeToken, NodeCompleted)
- **Parallel execution**: Tokio JoinSet with context store snapshots for fork-join patterns
- **Agent model**: OpenFang-based autonomous agents with skills, hands, and memory

### ChronosVector

A temporal vector database that indexes embeddings by both semantic proximity and time. Key contributions to Soma:

- **Temporal knowledge base**: Track how experiments evolve over time
- **Trajectory analysis**: Velocity, acceleration, and change point detection on embedding sequences
- **Tiered storage**: Hot (RocksDB), Warm (Parquet), Cold (S3) with automatic promotion/demotion
- **Semantic cache**: Find "sufficiently similar" cached results, not just exact hash matches

## Key Capabilities

| Capability | Description |
|---|---|
| **Computational Graphs** | Pipelines as executable, optionally differentiable graphs |
| **Two-Phase Filters** | `fit()` learns state, `forward()` transforms data -- both cacheable independently |
| **Content-Addressable Caching** | Automatic deduplication with cascade invalidation, resolved at compile time |
| **Data Virtualization** | Every value is a lazy reference materialized on demand |
| **Batch + Stream Unification** | Same filters work on complete datasets or chunked streams |
| **Gradient Propagation** | Differentiable filters enable end-to-end backpropagation through the pipeline |
| **Hyperparameter Optimization** | Search spaces defined at the filter level with Bayesian, Hyperband, and multi-objective strategies |
| **Remote Execution** | Serialize and send pipelines to workers for distributed computation |
| **Agent Integration** | Agents build, execute, and analyze pipelines autonomously |
| **Temporal Knowledge Base** | ChronosVector-powered experiment tracking with trajectory analysis |

## Who is Soma for?

- **Research labs** that want to automate experimentation, track results, and let agents explore hypotheses
- **ML engineers** who need reproducible pipelines with intelligent caching
- **Data scientists** who want a unified batch/stream processing framework
- **Agent developers** who need a robust execution layer for autonomous systems
