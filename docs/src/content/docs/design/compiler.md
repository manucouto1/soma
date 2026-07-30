---
title: Compiler & Execution Plans
description: How Soma compiles graphs into optimized, cache-aware execution plans.
---

## Overview

The Soma compiler is the intelligence layer between the user's graph definition and the runtime execution. It performs:

1. **Validation**: Cycle detection, schema compatibility
2. **Cache resolution**: Replace cached nodes before execution
3. **Gradient analysis**: Detect and warn about gradient flow interruptions
4. **Parallelism detection**: Identify independent branches for concurrent execution
5. **Cost estimation**: Estimate execution time from cache metadata
6. **Distribution planning**: Assign nodes to local or remote targets

## The ExecutionPlan

The compiler's output is a recursive `ExecutionPlan` tree:

```rust
#[derive(Serialize, Deserialize, Clone)]
pub enum ExecutionPlan {
    /// Execute steps sequentially
    Sequence(Vec<ExecutionPlan>),

    /// Execute branches concurrently (fork-join)
    Parallel(Vec<ExecutionPlan>),

    /// Execute a single filter
    Execute {
        id: NodeId,
        filter: Arc<dyn Filter>,
    },

    /// Iterate over a collection or repeat until condition
    Loop {
        id: NodeId,
        body: Box<ExecutionPlan>,
        config: LoopConfig,
    },

    /// Conditional branching
    Branch {
        id: NodeId,
        arms: Vec<(Predicate, ExecutionPlan)>,
    },

    /// Execute on a remote worker
    Remote {
        id: NodeId,
        target: RemoteTarget,
        plan: Box<ExecutionPlan>,
    },
}
```

## Compilation Process

```
    Graph (user-defined)
        │
        ▼
┌─── validate() ──────┐
│  - Cycle detection    │
│  - Schema compat      │
│  - Required fields    │
└──────────┬───────────┘
           │
           ▼
┌─── topological_sort() ─┐
│  - Kahn's algorithm      │
│  - Deterministic order   │
└──────────┬──────────────┘
           │
           ▼
┌─── detect_patterns() ──┐
│  - Parallel branches     │
│  - Fork-join points      │
│  - Loop bodies           │
│  - Conditional arms      │
└──────────┬──────────────┘
           │
           ▼
┌─── check_gradients() ──┐
│  - Analyze diff. flow    │
│  - Emit warnings         │
└──────────┬──────────────┘
           │
           ▼
┌─── plan_distribution() ┐
│  - Match node targets    │
│  - Wrap in Remote {}     │
└──────────┬──────────────┘
           │
           ▼
    ExecutionPlan (optimized)
```

## Pattern Detection

The compiler analyzes graph topology to detect structural patterns:

### Parallel Branches

When a node has multiple successors that converge later:

```
        [A]
       / | \
     [B] [C] [D]
       \ | /
        [E]

Compiled plan:
Sequence([
    Execute(A),
    Parallel([
        Execute(B),
        Execute(C),
        Execute(D),
    ]),
    Execute(E),
])
```

### Multiple Roots

Independent subgraphs that share no dependencies:

```
[A] → [B]     [C] → [D]
  (independent)

Compiled plan:
Parallel([
    Sequence([Execute(A), Execute(B)]),
    Sequence([Execute(C), Execute(D)]),
])
```

### Conditional Branches

A node whose outgoing edges carry predicates:

```
        [Evaluate]
        /        \
  (score > 0.8)  (else)
      /              \
  [Deploy]        [Retrain]

Compiled plan:
Sequence([
    Execute(Evaluate),
    Branch {
        condition: Evaluate,
        arms: [
            (score > 0.8, Execute(Deploy)),
            (else,        Execute(Retrain)),
        ],
    },
])
```

### Loop Bodies

A node marked as a loop with identified body nodes:

```
[ForEach dataset] → [Train] → [Evaluate] → [Collect]

Compiled plan:
Loop {
    id: ForEach,
    body: Sequence([Execute(Train), Execute(Evaluate)]),
    config: LoopConfig { collection: "datasets", flatten: true },
}
```

## Cache resolution happens at runtime, not here

The compiler never sees the dataset, so it cannot decide cache hits: any
compile-time key would be independent of the input data, and the same
graph run on two datasets would collide. Instead the **executor**
computes `hash(config + state + input)` per node with the materialized
input in hand and skips execution on a hit (see the
[caching design](/soma/design/caching/)). The compiled plan contains only
`Execute` nodes.

## Compile Modes

The compiler accepts a mode that affects runtime caching behavior:

```rust
pub enum CompileMode {
    /// Full caching. Used for inference/predict.
    Inference,

    /// Cache states only, re-execute forwards.
    /// Used when gradients are needed.
    Differentiable,

    /// No caching at all. Force re-execution.
    /// Used for debugging or benchmarking.
    NoCache,
}
```

## Cost Estimation

The compiler can estimate execution cost without running anything:

```rust
impl Compiler {
    pub fn estimate_cost(&self, plan: &ExecutionPlan, cache: &dyn CacheStore) -> Cost {
        match plan {
            ExecutionPlan::Execute { filter, .. } => {
                // Cost = estimated compute time (from filter metadata)
                filter.meta().estimated_cost()
            }
            ExecutionPlan::Sequence(steps) => {
                steps.iter().map(|s| self.estimate_cost(s, cache)).sum()
            }
            ExecutionPlan::Parallel(branches) => {
                // Cost = max of all branches (they run concurrently)
                branches.iter().map(|b| self.estimate_cost(b, cache)).max()
            }
            _ => Cost::unknown(),
        }
    }
}
```

## Validation

### Cycle Detection

The compiler uses topological sorting (Kahn's algorithm) to detect cycles. If the graph cannot be fully sorted, it contains a cycle and compilation fails.

### Schema Compatibility

The compiler checks that connected filters have compatible input/output schemas:

```rust
// Node A outputs shape [batch, 128] with dtype f32
// Node B expects shape [batch, 128] with dtype f32
// → Compatible ✓

// Node A outputs shape [batch, 128]
// Node B expects shape [batch, 256]
// → Incompatible ✗ : compile error with diagnostic
```

## Serialization

The entire `ExecutionPlan` is serializable (via serde), which enables:

- **Remote execution**: Send the plan to a worker
- **Plan inspection**: Pretty-print the plan for debugging
- **Plan caching**: Cache the compiled plan itself for repeated executions
- **Plan comparison**: Diff two plans to see what changed

```rust
let plan = compiler.compile(&graph, &cache, CompileMode::Inference)?;

// Serialize for remote worker
let bytes = bincode::serialize(&plan)?;
worker.send(bytes).await?;

// Pretty-print for debugging
println!("{}", plan.display());
// Sequence:
//   Cached(scaler, key=abc123)
//   Execute(classifier)
//   Execute(evaluator)
```
