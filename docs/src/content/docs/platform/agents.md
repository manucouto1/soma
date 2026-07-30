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

    /// The experiment pool: what has been tried, and what came of it
    pub memory: Box<dyn KnowledgeBase>,

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
│     Graph.somatize(ZNorm() >> TSClassifier("rocket"))    │
│                                                          │
│  5. Execute (local or remote)                            │
│     study = Study("exp", strategy="bayesian", ...)       │
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

## Memory: the experiment pool

The agent's memory is the [experiment pool](/design/experiment-pool/) —
the same `.soma/experiments.jsonl` a human's runs write to. There is no
separate agent memory store: an agent remembers what was run because
running it recorded it.

The interface is the `KnowledgeBase` trait
([Knowledge Base](/platform/knowledge-base/)):

```rust
use chrono::Utc;
use somatize_memory::{FileKnowledgeBase, KnowledgeBase, RetrievalQuery};

let mut kb = FileKnowledgeBase::open(".soma/experiments.jsonl")?;
kb.refresh()?;   // pick up runs that finished elsewhere

// "What have I tried that bears on this?" — ranked by text relevance,
// architectural resemblance, recency and importance.
let hits = kb.retrieve(&RetrievalQuery::new("z-norm on short series", Utc::now()))?;

// "What came of that starting point?" — the tree, with the change
// applied to the parent labelling every edge.
let lineage = kb.lineage(&hits[0].record.id)?;

// "How has this line moved?"
kb.trajectory("rocket-znorm", "val_f1")?;
kb.change_points("rocket-znorm", "val_f1", 0.05)?;
kb.promising_lines("val_f1")?;
```

Over MCP the same capabilities are `kb_find_similar`, `kb_lineage`,
`kb_diff`, `kb_record_conclusion`, `kb_branch_from`, `kb_summarize_run`
and `kb_stats`.

Dead ends are retrievable on purpose: `importance` puts a floor under
any run that failed, crashed or regressed and carries a conclusion.
Not repeating a failed idea saves an agent as much time as repeating a
successful one.

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
