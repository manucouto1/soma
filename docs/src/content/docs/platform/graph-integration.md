---
title: Graph Integration
description: How graphs become nodes in the platform's orchestration graphs.
---

:::caution[Superseded, and mostly unnecessary]
This page sketched a `publish` mechanism for exposing a Soma graph to a
separate orchestration layer. That layer no longer exists as a separate
thing: orchestration nodes are [steps in the same graph](/soma/design/agentic/),
and running a pipeline from one is `Effect::Graph`, which is implemented.

What remains unimplemented is the *hosted* half — `lab.publish`,
`lab.get_graph`, `PublishedGraph`. The `Lab` client that once held
`connect`/`health`/`info`/`workers` was deleted: nothing but its own test
called it, and `Graph.add_worker` already talks to a worker. Read the rest
of this page as history.
:::

## The bridge, as it actually turned out

The premise below was that Soma graphs and "platform graphs" are different
kinds of object needing a bridge between them. They are not. A graph with an
LLM node and a graph with a classifier node are the same kind of graph with
different nodes in it — see [Agentic Graphs](/soma/design/agentic/) for why
that turned out to be the right split.

So there is no publishing step. A step that wants to run a pipeline emits
one:

```rust
Effect::Graph {
    graph: Box::new(pipeline),
    input: params,
    mode: GraphEffectMode::Fit,
}
```

`GraphHandler` runs it through a `GraphSession`, with the pipeline's own
cache, schema checks and events intact, and the result comes back to the
step. Because it is an effect, it is journaled: a loop that crashes after
its fourth experiment replays the first three instead of paying for them
again. Nothing needs to be registered anywhere first.

The table below is kept because the contrast in it is still worth reading —
just note that both columns describe nodes in one graph now, not two systems.

| | Filter node | Step node |
|---|---|---|
| **Purpose** | Compute | Reach outside the graph |
| **Determinism** | Same input ⇒ same output | Not promised |
| **Gradients** | Yes (when differentiable) | No |
| **Reuse** | Content-addressed cache | Journal: record once, replay |
| **Execution** | Compiled to ExecutionPlan | Compiled to ExecutionPlan |

## Publishing a Graph *(historical)*

```python
# User defines and tests a graph locally
g = Graph.somatize(
    MyPreprocessor(scale=2.0) >> MyClassifier(model="svm", C=1.0)
)
g.fit(train_data, y_train)

# Publish to the platform
lab.publish(g, name="svm_classifier")
```

Once published, `svm_classifier` appears as a node type in the platform's visual graph editor, alongside LLM nodes, agent nodes, and other platform nodes.

## How It Works

A published graph is wrapped as a platform node:

```rust
#[derive(Serialize, Deserialize)]
pub struct PublishedGraph {
    pub id: GraphId,
    pub name: String,
    pub graph: Graph,                       // the computation graph
    pub search_space: Option<SearchSpace>,  // if optimization is available
    pub input_schema: Schema,               // what it expects
    pub output_schema: Schema,              // what it produces
    pub fitted_states: Option<Vec<CacheKey>>, // pre-trained states
}
```

The platform treats it as any other node:

```
Platform Graph:
  [Agent: Generate Hypothesis]
      │
      ▼
  [Published Graph: svm_classifier]  ← Soma graph as a node
      │
      ▼
  [Agent: Analyze Results]
      │
      ▼
  [Agent: Write Report]
```

## Orchestration Patterns

### Agent-Driven Experimentation

```
┌────────────────────────────────────────────────┐
│  Platform Orchestration Graph                   │
│                                                 │
│  [Agent: Hypothesis] ──► [Graph: Train+Eval]    │
│         ▲                       │               │
│         │                       ▼               │
│         └──── [Agent: Analyze Results]          │
│                       │                         │
│                       ▼                         │
│              [Agent: Document]                  │
│                       │                         │
│                       ▼                         │
│         [Condition: Continue?]                  │
│              /            \                     │
│          (yes)           (no)                   │
│            │               │                    │
│            ▼               ▼                    │
│   [loop back]    [Agent: Final Report]          │
└────────────────────────────────────────────────┘
```

### Multi-Graph Comparison

```
                [Agent: Design Experiment]
                    /          \
                   /            \
  [Graph: SVM Approach]  [Graph: Neural Approach]
                   \            /
                    \          /
              [Agent: Compare Results]
                        │
                  [Agent: Report]
```

### Graph as Sub-Component

A published graph can also be used as a filter within another graph (recursive composition):

```python
# A published graph IS a filter
svm_graph = lab.get_graph("svm_classifier")

# Use it inside a larger graph
meta_graph = Graph.somatize(
    DataLoader(source="s3://datasets/ucr")
    >> svm_graph  # ← nested graph
    >> MetricAggregator(metrics=["f1", "accuracy"])
)
```

## Event Flow

When a platform graph executes a published graph, events from both layers are emitted:

```
Platform events:
  PlatformNodeStarted { node: "svm_classifier" }

    Graph events (nested):
      RunStarted { run_id }
        NodeStarted { node: "preprocessor" }
        NodeCacheHit { node: "preprocessor", tier: "Memory" }
        NodeStarted { node: "classifier" }
        NodeCompleted { node: "classifier", duration: 1.2s }
      RunCompleted { run_id, duration: 1.22s }

  PlatformNodeCompleted { node: "svm_classifier", duration: 1.22s }
```

The platform UI can display both levels: the high-level orchestration flow and the detailed graph execution within each node.

## Future: Visual Graph Editor

Beyond publishing existing graphs, the platform will support visual graph construction:

- Drag-and-drop filters from a library
- Configure search spaces visually
- Connect filters with typed edges
- Validate schemas in real-time
- Launch optimization studies from the UI
- View results in integrated dashboards

This bridges the gap between code-defined graphs and visual experimentation, letting researchers work in whichever mode suits them.
