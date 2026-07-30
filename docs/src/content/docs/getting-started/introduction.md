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

- **Filter model**: `fit()` / `forward()` lifecycle for trainable transformations
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

A temporal vector database that indexes embeddings by both semantic proximity
and time. Soma can use it as an experiment backend behind the `chronos` feature
flag, but **does not by default** — the shipped knowledge base is an append-only
JSONL journal with BM25 + structural ranking, and Soma has no embedding model.
See [Knowledge Base](/soma/platform/knowledge-base/) for what is and is not
implemented.

## Key Capabilities

| Capability | Description |
|---|---|
| **Computational Graphs** | Executable, optionally differentiable graphs |
| **Two-Phase Filters** | `fit()` learns state, `forward()` transforms data -- both cacheable independently |
| **Content-Addressable Caching** | Automatic deduplication with cascade invalidation, resolved per node at runtime |
| **Data Virtualization** | Every value is a lazy reference materialized on demand |
| **Batch + Stream Unification** | Same filters work on complete datasets or chunked streams |
| **Gradient Propagation** | Differentiable filters enable end-to-end backpropagation through the pipeline |
| **Hyperparameter Optimization** | Search spaces defined at the filter level; grid, random and Bayesian (TPE) search, multi-objective, median/percentile pruning |
| **Remote Execution** | Serialize and send graphs to workers for distributed computation |
| **Agent Integration** | Agents build, execute, and analyze graphs autonomously |
| **Experiment Pool** | Every run records its conclusion, architecture fingerprint and the change from its parent; ranked retrieval over the lot |

## Who is Soma for?

- **Research labs** that want to automate experimentation, track results, and let agents explore hypotheses
- **ML engineers** who need reproducible pipelines with intelligent caching
- **Data scientists** who want a unified batch/stream processing framework
- **Agent developers** who need a robust execution layer for autonomous systems
