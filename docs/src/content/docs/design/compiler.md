---
title: Compiler & Execution Plans
description: How Soma compiles graphs into optimized, cache-aware execution plans.
---

## Overview

The Soma compiler is the intelligence layer between the user's graph definition and the runtime execution. It performs:

1. **Validation**: Cycle detection, schema compatibility
2. **Validation**: Reject cycles and incompatible schemas before anything runs
3. **Gradient analysis**: Detect and warn about gradient flow interruptions
4. **Parallelism detection**: Identify independent branches for concurrent execution
5. **Cost estimation**: Estimate execution time from cache metadata
6. **Distribution planning**: Assign nodes to local or remote targets

## The ExecutionPlan

The compiler's output is a recursive `ExecutionPlan` tree
(`soma-compiler/src/plan.rs`):

```rust
#[derive(Serialize, Deserialize, Clone)]
pub enum ExecutionPlan {
    /// Execute sub-plans sequentially, one after another
    Sequence(Vec<ExecutionPlan>),

    /// Execute branches concurrently (fork-join)
    Parallel(Vec<ExecutionPlan>),

    /// Execute a single filter node
    Execute { node_id: NodeId },

    /// Run an effectful step to completion: poll, perform its
    /// effects, repeat. `handoffs` lists where it may pass control.
    Step {
        node_id: NodeId,
        handoffs: Vec<(NodeId, ExecutionPlan)>,
    },

    /// Iterate: run `body` until `until` says stop, or the cap is hit
    Loop {
        node_id: NodeId,
        body: Box<ExecutionPlan>,
        max_iterations: Option<usize>,
        until: LoopCondition,          // resolved — never BodyTerminal here
        carry_from: Option<NodeId>,    // what each pass hands the next
    },

    /// Conditional branching: evaluate condition, pick an arm
    Branch {
        node_id: NodeId,
        arms: Vec<(String, ExecutionPlan)>,
    },

    /// Execute a sub-plan on a remote worker
    Remote {
        node_id: NodeId,
        target: RemoteTarget,
        plan: Box<ExecutionPlan>,
    },

    /// Execute multiple differentiable nodes as one block, passing
    /// tensors directly so autograd survives the node boundaries
    Composite { node_ids: Vec<NodeId> },

    /// Streaming execution: chunks through a filter chain, each
    /// filter's StreamMode defining its per-chunk contract
    Stream { node_ids: Vec<NodeId>, chunk_size: usize },

    /// Nothing to execute (e.g. empty graph)
    Empty,
}
```

Three variants deserve a word beyond their comment.

**`Step` is not `Execute` with a flag.** The runtime has to drive a turn
loop — poll the step, perform the effects it asks for, journal what was
performed, poll again — rather than call a function once. Its `handoffs`
are the branch the *step* decides instead of a condition value, so they
compile the same way a branch's arms do: each target is claimed by the
step and appears exactly once, inside it. A `Goto` naming a target not
listed there is an error, not a jump into the dark.

**`Loop` separates what it carries from what stops it.** `until` says
when to stop; `carry_from` names the node whose output each pass hands to
the next one. They are different questions: a fixed-round debate has no
stop signal at all, but every round still has to start from what the last
one said — otherwise the loop just repeats its first iteration. A
`LoopCondition::BodyTerminal` declared on the graph is resolved to
`WhenSignaled(node)` at compile time, so the executor reads the signal
from exactly one node; a body with several terminals is a compile error,
not a race.

**Ownership is decided by dominance.** A loop owns its body-entry nodes
and everything they dominate; a branch owns each arm's entry and its
dominated subgraph. Without that exclusion the body would be emitted
twice — once inside the loop, once after it.

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

A loop node claims its body by dominance and compiles with a resolved
stop condition and a carry:

```
[revise] → [worker] → [judge]     loop("refine", body=revise, until=judge)

Compiled plan:
Loop {
    node_id: refine,
    body: Sequence([Execute(revise), Execute(worker), Execute(judge)]),
    max_iterations: Some(4),
    until: WhenSignaled(judge),     // was BodyTerminal on the graph
    carry_from: Some(judge),        // the verdict the next pass reads
}
```

## Cache resolution happens at runtime, not here

The compiler never sees the dataset, so it cannot decide cache hits: any
compile-time key would be independent of the input data, and the same
graph run on two datasets would collide. Instead the **executor**
computes `hash(config + state + input)` per node with the materialized
input in hand and skips execution on a hit (see the
[caching design](/soma/design/caching/)). There is no `Cached` plan
variant: a hit is a runtime outcome, not a compile-time decision.

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
//   Execute(scaler)
//   Execute(classifier)
//   Execute(evaluator)
```
