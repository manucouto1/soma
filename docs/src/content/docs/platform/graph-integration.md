---
title: Graph Integration
description: How pipelines become nodes in the platform's orchestration graphs.
---

## The Bridge

Soma pipelines and the platform's orchestration graphs serve different purposes:

| | Soma Pipeline | Platform Graph |
|---|---|---|
| **Node type** | Filters (data transformations) | LLM, Agent, Pipeline, Tool, IO |
| **Purpose** | Compute | Orchestrate |
| **Gradients** | Yes (when differentiable) | No |
| **Caching** | Content-addressable | Not applicable |
| **Execution** | Compiled to ExecutionPlan | Event-driven execution |

The bridge between them: **a compiled pipeline can be published as a node** in the platform graph.

## Publishing a Pipeline

```python
# User defines and tests a pipeline locally
pipeline = Pipeline([
    MyPreprocessor(scale=2.0),
    MyClassifier(model="svm", C=1.0),
])
pipeline.fit(train_data, y_train)

# Publish to the platform
lab.publish(pipeline, name="svm_classifier")
```

Once published, `svm_classifier` appears as a node type in the platform's visual graph editor, alongside LLM nodes, agent nodes, and other platform nodes.

## How It Works

A published pipeline is wrapped as a platform node:

```rust
#[derive(Serialize, Deserialize)]
pub struct PublishedPipeline {
    pub id: PipelineId,
    pub name: String,
    pub graph: Graph,                       // the pipeline as a graph
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
  [Published Pipeline: svm_classifier]  ← Soma pipeline as a node
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
│  [Agent: Hypothesis] ──► [Pipeline: Train+Eval] │
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

### Multi-Pipeline Comparison

```
                [Agent: Design Experiment]
                    /          \
                   /            \
  [Pipeline: SVM Approach]  [Pipeline: Neural Approach]
                   \            /
                    \          /
              [Agent: Compare Results]
                        │
                  [Agent: Report]
```

### Pipeline as Sub-Component

A published pipeline can also be used as a filter within another pipeline (recursive composition):

```python
# A published pipeline IS a filter
svm_pipeline = lab.get_pipeline("svm_classifier")

# Use it inside a larger pipeline
meta_pipeline = Pipeline([
    DataLoader(source="s3://datasets/ucr"),
    svm_pipeline,  # ← nested pipeline
    MetricAggregator(metrics=["f1", "accuracy"]),
])
```

## Event Flow

When a platform graph executes a published pipeline, events from both layers are emitted:

```
Platform events:
  PlatformNodeStarted { node: "svm_classifier" }

    Pipeline events (nested):
      RunStarted { run_id }
        NodeStarted { node: "preprocessor" }
        NodeCacheHit { node: "preprocessor", tier: "Memory" }
        NodeStarted { node: "classifier" }
        NodeCompleted { node: "classifier", duration: 1.2s }
      RunCompleted { run_id, duration: 1.22s }

  PlatformNodeCompleted { node: "svm_classifier", duration: 1.22s }
```

The platform UI can display both levels: the high-level orchestration flow and the detailed pipeline execution within each node.

## Future: Visual Pipeline Editor

Beyond publishing existing pipelines, the platform will support visual pipeline construction:

- Drag-and-drop filters from a library
- Configure search spaces visually
- Connect filters with typed edges
- Validate schemas in real-time
- Launch optimization studies from the UI
- View results in integrated dashboards

This bridges the gap between code-defined pipelines and visual experimentation, letting researchers work in whichever mode suits them.
