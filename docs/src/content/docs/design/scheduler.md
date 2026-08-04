---
title: Scheduler
description: Distributes ExecutionPlan across workers based on topology and capabilities
---

# Scheduler

The Scheduler analyzes an ExecutionPlan's topology and distributes nodes across available workers.

## Rules

1. **Sequential phases** → single worker (avoids data transfer)
2. **Parallel branches** → round-robin across workers by capability
3. **Differentiable connected nodes** → same worker (gradient flow must be preserved)
4. **Cache hits** → resolved at runtime on the assigned worker (the plan itself carries no cached nodes)
5. **Loop/Branch bodies** → same worker as controller

## DistributionPlan

The scheduler produces a `DistributionPlan` containing:

| Field | Description |
|-------|-------------|
| `assignments` | Node → worker mapping with reason |
| `phases` | Sequential/parallel groupings |
| `data_transfers` | S3 transfers needed between workers |
| `warnings` | Capacity issues, single-worker fallback |

## Example

```
Graph: [Load] → [Normalize] → [Train SVM]
                                → [Train KNN]

Scheduler with 2 workers:
  Worker A (GPU): Load → Normalize → Train SVM  (sequential, data locality)
  Worker B (CPU): Train KNN                       (parallel branch)

  Data Transfer: Normalize → Train KNN (via S3)
```

## Usage

```rust
use somatize_compiler::{schedule, WorkerInfo};

let workers = vec![
    WorkerInfo { id: "gpu-1", gpu: true, cpu_cores: 16, ... },
    WorkerInfo { id: "cpu-1", gpu: false, cpu_cores: 64, ... },
];

let plan = schedule(&execution_plan, &workers, &differentiable_nodes);
// plan.assignments, plan.data_transfers, plan.warnings
```
