---
title: Graph Integration
description: How graphs become nodes in the platform's orchestration graphs.
---

:::caution[Not implemented yet]
This page describes the intended design. None of the API below exists today:
`Lab` has only `connect`, `health`, `info` and `workers` — there is no
`lab.publish`, `lab.get_graph` or `PublishedGraph`. Read it as a design
sketch, not as a reference.
:::

## The Bridge

Soma graphs and the platform's orchestration graphs serve different purposes:

| | Soma Graph | Platform Graph |
|---|---|---|
| **Node type** | Filters (data transformations) | LLM, Agent, Graph, Tool, IO |
| **Purpose** | Compute | Orchestrate |
| **Gradients** | Yes (when differentiable) | No |
| **Caching** | Content-addressable | Not applicable |
| **Execution** | Compiled to ExecutionPlan | Event-driven execution |

The bridge between them: **a compiled graph can be published as a node** in the platform graph.

## Publishing a Graph

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
