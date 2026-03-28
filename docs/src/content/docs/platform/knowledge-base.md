---
title: Knowledge Base
description: Temporal experiment tracking and navigable research history powered by ChronosVector.
---

## Purpose

The Knowledge Base is a navigable, queryable record of all experiments executed in a Soma lab. It answers questions like:

- "What experiments have we run on this dataset?"
- "Which hyperparameter configurations worked best?"
- "How has our accuracy evolved over the last month?"
- "Where did a breakthrough happen?"
- "Which research lines are worth continuing?"

It is powered by **ChronosVector**, which provides both semantic similarity search and temporal trajectory analysis.

## Experiment Records

Every completed study or pipeline run is indexed as an `ExperimentRecord`:

```rust
#[derive(Serialize, Deserialize)]
pub struct ExperimentRecord {
    pub id: ExperimentId,
    pub name: String,
    pub hypothesis: Option<String>,
    pub pipeline: PipelineSummary,          // what was executed
    pub params: HashMap<String, Value>,     // hyperparameters used
    pub metrics: HashMap<String, f64>,      // final metrics
    pub embedding: Vec<f32>,               // semantic embedding
    pub timestamp: DateTime<Utc>,
    pub duration: Duration,
    pub parent: Option<ExperimentId>,       // which experiment this derives from
    pub research_line: Option<String>,      // grouping tag
    pub tags: Vec<String>,
    pub notes: Option<String>,             // agent or user notes
}
```

### Automatic Indexing

When a Study completes, the runtime automatically:

1. Generates an embedding from the experiment description + params + results
2. Creates an `ExperimentRecord`
3. Stores it in ChronosVector with the current timestamp
4. Links it to the parent experiment (if any)

## Querying the Knowledge Base

### Semantic Search

Find experiments similar to a natural language query:

```python
kb = lab.knowledge_base()

results = kb.search("normalization impact on short time series")
# Returns experiments semantically similar to the query
# Ranked by combined semantic + temporal proximity
```

### Trajectory Analysis

Track how a metric evolves across experiments in a research line:

```python
trajectory = kb.trajectory(
    research_line="rocket_normalization",
    metric="f1_weighted",
)

# trajectory.values    → [0.72, 0.78, 0.81, 0.85, 0.84, 0.86]
# trajectory.velocity  → rate of improvement (slowing down)
# trajectory.change_points → [experiment_003: breakthrough]
```

### Change Point Detection

Find where significant shifts occurred in experimental results:

```python
change_points = kb.change_points(
    research_line="rocket_normalization",
    metric="accuracy",
)

# [ChangePoint {
#     experiment: "exp_003",
#     timestamp: "2026-03-15T14:22:00Z",
#     metric_before: 0.78,
#     metric_after: 0.85,
#     reason: "Switched from min-max to z-norm"
# }]
```

### Promising Lines

Identify which research directions are worth continuing:

```python
promising = kb.promising_lines()

# [ResearchLine {
#     name: "rocket_znorm",
#     trend: "improving",
#     velocity: 0.02,           # metric improvement per experiment
#     acceleration: -0.005,     # slowing down but still positive
#     best_metric: 0.86,
#     n_experiments: 8,
#     recommendation: "Continue with focus on hyperparameter tuning"
# },
# ResearchLine {
#     name: "inception_minmax",
#     trend: "plateaued",
#     velocity: 0.001,
#     best_metric: 0.79,
#     recommendation: "Consider abandoning or pivoting"
# }]
```

### Comparison

Compare two research lines or experiments:

```python
comparison = kb.compare(
    line_a="rocket_znorm",
    line_b="inception_znorm",
)

# comparison.metric_comparison → side-by-side metrics
# comparison.divergence_point  → when they started differing
# comparison.resource_usage    → which used more compute
```

## Tiered Storage

ChronosVector's tiered storage maps to experiment lifecycle:

| Tier | Content | Latency | Retention |
|---|---|---|---|
| **Hot** (RocksDB) | Current session experiments | <1ms | Until session ends |
| **Warm** (Parquet) | Recent experiments (this month) | <10ms | Configurable (default: 6 months) |
| **Cold** (S3/Object Store) | Historical archive | <100ms | Indefinite |

Automatic promotion/demotion:
- New experiments → Hot
- After session → demote to Warm
- After retention period → demote to Cold
- When queried, Cold results promote back to Warm

## Navigable Interface

The knowledge base exposes a navigable structure for UI rendering:

```python
# Browse by research line
lines = kb.list_lines()

# Browse experiments in a line (chronological)
experiments = kb.list_experiments(line="rocket_znorm")

# Get full details of an experiment
exp = kb.get_experiment("exp_042")
exp.pipeline       # what was executed
exp.params         # hyperparameters
exp.metrics        # results
exp.parent         # derived from which experiment
exp.children       # experiments derived from this one

# Tree view: experiment genealogy
tree = kb.experiment_tree("exp_001")
# exp_001
# ├── exp_003 (changed normalization)
# │   ├── exp_005 (tuned learning rate)
# │   └── exp_006 (added augmentation)
# └── exp_004 (different classifier)
```

## Report Generation

The knowledge base can generate reports from experiment history:

```python
report = kb.generate_report(
    research_line="rocket_normalization",
    format="markdown",
)

# Generated report includes:
# - Research objective
# - Methodology (pipelines used)
# - Results table (all experiments, sorted by metric)
# - Trajectory plot description
# - Change points and breakthroughs
# - Conclusions and recommendations
# - References to specific experiments
```

## Integration with Agents

Agents use the knowledge base as their primary memory:

```
Agent: "I want to try z-norm on the Coffee dataset"
  → kb.search("z-norm Coffee dataset")
  → "You already ran this in exp_012. F1 was 0.91.
     Changing the classifier might be more productive."
  → Agent adjusts strategy based on existing knowledge
```

This prevents redundant experiments and enables the agent to build on prior work rather than starting from scratch.
