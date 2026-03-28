---
title: Agents & Memory
description: Autonomous research agents that build, execute, and analyze pipelines.
---

## Overview

Soma's agent layer enables autonomous research workflows. An agent can:

- Generate hypotheses and translate them into pipelines
- Execute pipelines (locally or remotely)
- Analyze results using temporal knowledge
- Decide which research lines to pursue
- Document findings automatically

The agent interacts with Soma through the **same API** as a human user. It is not embedded in the runtime -- it sits above it.

## Agent Structure

Inspired by the OpenFang model from Chatty the Lab:

```rust
pub struct Agent {
    /// Identity and personality (system prompt)
    pub soul: String,

    /// Capabilities: domain knowledge, reasoning templates
    pub skills: Vec<Skill>,

    /// Tools: shell, web, file system, Soma API
    pub hands: Vec<Hand>,

    /// Temporal memory (ChronosVector)
    pub memory: SomaMemory,

    /// LLM driver for reasoning
    pub driver: Box<dyn LlmDriver>,
}
```

### Soul

The agent's identity. Defines its role, expertise, and behavior:

```markdown
You are a research agent specializing in time series classification.
Your goal is to systematically explore normalization techniques
and their impact on classification accuracy across the UCR archive.

When results are inconclusive, prefer exploring a new technique
over repeating experiments with minor parameter changes.
```

### Skills

Prompt-based capabilities that guide the agent's reasoning:

```rust
pub struct Skill {
    pub name: String,
    pub description: String,
    pub prompt: String,          // instructions injected into context
    pub tools_provided: Vec<String>, // tools this skill unlocks
}
```

Examples:
- **Hypothesis generation**: Templates for formulating testable hypotheses
- **Experiment design**: Best practices for pipeline construction
- **Result analysis**: Statistical analysis and interpretation patterns
- **Report writing**: Documentation templates and formatting

### Hands

Actual tools the agent can execute:

```rust
pub struct Hand {
    pub name: String,
    pub tools: Vec<ToolDefinition>,
}
```

Examples:
- **soma_pipeline**: Create, compile, and run pipelines
- **soma_study**: Launch hyperparameter optimization studies
- **soma_knowledge**: Query the knowledge base
- **file_io**: Read and write files (datasets, reports)
- **web_search**: Search for papers and related work

## Agent Loop

```
┌─────────────────────────────────────────────────────────┐
│                    AGENT RESEARCH LOOP                    │
│                                                          │
│  1. Receive objective (from user or self-generated)      │
│     "Investigate impact of normalization on TS classif." │
│                                                          │
│  2. Consult knowledge base                               │
│     "What have I tried before? What worked?"             │
│     → ChronosVector: trajectory, change points           │
│                                                          │
│  3. Generate hypothesis                                  │
│     "Z-norm may outperform min-max on short series"      │
│                                                          │
│  4. Build pipeline                                       │
│     Pipeline([ZNorm(), TSClassifier(model="rocket")])    │
│                                                          │
│  5. Execute (local or remote)                            │
│     study = Study(pipeline, Bayesian(n=50))              │
│     lab.run(study, data)                                 │
│                                                          │
│  6. Analyze results                                      │
│     Compare with previous experiments                    │
│     Detect change points, trends                         │
│                                                          │
│  7. Decide next action                                   │
│     a) New hypothesis → go to 3                          │
│     b) Refine current → modify pipeline, go to 5         │
│     c) Conclude → generate report                        │
│                                                          │
│  8. Document                                             │
│     Index experiment in knowledge base                   │
│     Generate report with findings                        │
└─────────────────────────────────────────────────────────┘
```

## Memory: ChronosVector Integration

The agent's memory is powered by ChronosVector, providing temporal-aware recall:

```rust
pub struct SomaMemory {
    vector_store: ChronosVector,
}

impl SomaMemory {
    /// Episodic recall: "What did I try recently?"
    pub async fn recall(
        &self,
        context: &[f32],         // embedding of current situation
        temporal_weight: f64,     // how much to weight recency
    ) -> Vec<Episode> { .. }

    /// Trajectory: "How has F1 evolved across my experiments?"
    pub async fn trajectory(
        &self,
        experiment_line: &str,
    ) -> Trajectory { .. }

    /// Change points: "When did results change significantly?"
    pub async fn change_points(
        &self,
        metric: &str,
    ) -> Vec<ChangePoint> { .. }

    /// Drift: "Is my approach converging or diverging?"
    pub async fn drift(
        &self,
        line: &str,
        window: TimeRange,
    ) -> DriftMetrics { .. }

    /// Store a new experiment
    pub async fn record_experiment(
        &self,
        record: ExperimentRecord,
    ) -> Result<()> { .. }
}
```

### ChronosVector Capabilities Used

| ChronosVector Feature | Agent Use Case |
|---|---|
| Snapshot kNN | "Find experiments similar to this hypothesis" |
| Evolutionary Path | "How has this research line evolved?" |
| Vector Velocity | "Is improvement accelerating or slowing?" |
| Change Point Detection | "When did a breakthrough happen?" |
| Drift Quantification | "Is this line converging?" |
| Temporal Analogy | "What worked 2 weeks ago may apply now" |
| Tiered Storage | Hot: current session, Warm: recent, Cold: archive |

## Python API

```python
from soma.agent import Researcher

# Create a research agent
agent = Researcher(
    lab=lab,
    plan="""
    Investigate the impact of different normalization techniques
    on time series classification accuracy.

    Datasets: UCR archive subset (GunPoint, ECG200, Coffee)
    Techniques: z-normalization, min-max, robust scaling, none
    Classifiers: ROCKET, 1-NN DTW, InceptionTime
    Metrics: accuracy, F1-weighted, training time
    """,
)

# Let the agent run autonomously
report = agent.investigate(max_iterations=20)

# The agent:
# 1. Downloads/loads datasets
# 2. Creates pipelines for each combination
# 3. Runs studies with hyperparameter optimization
# 4. Records results in knowledge base
# 5. Analyzes trajectories to find promising lines
# 6. Generates a final report with findings

# Interactive: ask the agent about its findings
agent.ask("Which normalization worked best for short series?")
agent.ask("Show me the trajectory for the ROCKET experiments")
```

## Events from Agent Execution

The agent loop produces its own events layered on top of Study/Trial/Run events:

```
AgentStarted { objective }
  HypothesisGenerated { hypothesis, confidence }
  PipelineBuilt { pipeline_summary }
  StudyStarted { ... }
    TrialStarted { ... }
      RunStarted { ... }
        NodeStarted { ... }
        NodeCompleted { ... }
      RunCompleted { ... }
    TrialCompleted { ... }
  StudyCompleted { ... }
  AnalysisCompleted { findings }
  DecisionMade { action: "new_hypothesis" | "refine" | "conclude" }
  ExperimentRecorded { id, summary }
AgentCompleted { report }
```
