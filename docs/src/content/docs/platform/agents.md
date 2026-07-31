---
title: Agents & Memory
description: The research loop as a Step, and the experiment pool it remembers with.
---

## Overview

An autonomous researcher in Soma is a node, not a runtime. `ResearchStep`
(crate `somatize-agent`) is a [`Step`](/soma/design/agentic/) that proposes
an experiment, runs it, reads the metrics, and decides whether to keep
going. Everything around it already existed and is not reimplemented: the
loop is the effect driver's, the durability is the journal's, the record is
`somatize-memory`'s, and running a pipeline is `GraphHandler`'s.

What is left is the reasoning, and the reasoning is the model's.

## The loop

One turn:

1. **Ask** the model what to try next, given the objective and what the pool
   already knows.
2. **Read an `Action`** out of the reply — a constrained schema, not prose.
3. `RunExperiment` → **`Effect::Graph`**, which runs the pipeline with the
   proposed parameters and hands back its metrics.
4. `Conclude` → **done**, with every experiment behind it.

```rust
use somatize_agent::ResearchStep;

let agent = ResearchStep::new("ollama/qwen2.5", "beat 0.8 held-out F1", pipeline)
    .with_history(kb.all()?)        // start from what is already known
    .with_max_iterations(10);

let mut steps = StepLibrary::new();
steps.register("researcher", Box::new(agent));
```

The driver needs two handlers: one that serves `Effect::Llm` (from
`somatize-llm`) and a `GraphHandler` holding the filters the pipeline is
built from.

```rust
let driver = EffectDriver::new(journal)
    .with_handler(Arc::new(LlmHandler::new(router)))
    .with_handler(Arc::new(GraphHandler::new(library)));
```

### The two actions

```json
{"action": "run_experiment", "name": "exp_0007",
 "research_line": "regularization",
 "hypothesis": "C=4 lifts held-out F1 above 0.8",
 "params": {"classifier.C": 4.0}}
```
```json
{"action": "conclude", "reason": "the line plateaued at 0.79"}
```

Two, and deliberately no more. An agent that can run an experiment and an
agent that can stop covers the whole loop; every other verb people reach for
("analyze", "compare", "summarize") is the model thinking, and thinking does
not need a protocol.

### What the contract refuses

- **Prose instead of an action ends the run.** Guessing would mean launching
  an experiment nobody asked for, on a budget somebody is paying.
- **An experiment without a falsifiable hypothesis does not deserialize.** A
  result nobody can interpret later is a result nobody will read, and the
  pool exists to be read later.
- **The iteration budget stops a model that will not stop itself.**

### A failed experiment is a finding

`Effect::Graph` returns `EffectResult::Failed` rather than erroring, and the
step records it with the failure in its notes. A configuration that will not
run is information — often the valuable kind — and it is what stops the
agent proposing the same broken thing next turn. Ending the run instead
would discard everything learned before it.

### The step keeps no state

The history is rebuilt from `ctx.history` each turn, by pairing each model
reply with the result that followed it. That is not a style preference: a
replay reconstructs exactly the history the original run had, because it
replays exactly the same results. Kept in a field, the two can drift.

### Metrics

A pipeline reports its metrics as a node's output, the same way a `Study`
objective reads them. Every number in the result becomes a metric named by
where it sits — `{"classifier": {"f1": 0.9}}` gives `classifier.f1` — so two
nodes both reporting `loss` stay two series when experiments are compared
months later. Arrays are data, not metrics.

## Memory: the experiment pool

The agent's memory is the [experiment pool](/soma/design/experiment-pool/) —
the same `.soma/experiments.jsonl` a human's runs write to. There is no
separate agent memory store: an agent remembers what was run because
running it recorded it.

The interface is the `KnowledgeBase` trait
([Knowledge Base](/soma/platform/knowledge-base/)):

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

:::caution[Rust only, for now]
`ResearchStep` has no Python binding yet. From Python, the agentic surface
is `soma.Agent`, `soma.Judge` and `soma.agentic` — see
[Agentic Graphs](/soma/design/agentic/) — and the memory half above, which
works today from either language.
:::

## Events

The research loop emits the ordinary step events, so its trace nests inside
whatever else is running:

```
AgentTurnStarted { node_id, turn }
  EffectRequested  { label: "llm:ollama/qwen2.5" }
  EffectCompleted  { .. }
  EffectRequested  { label: "graph" }
    RunStarted     { .. }        # the pipeline it launched
      NodeStarted  { .. }
      NodeCompleted{ .. }
    RunCompleted   { .. }
  EffectCompleted  { .. }
AgentStepCompleted { node_id, turns }
```

There is no separate agent event level. An agent's experiment is a run like
any other, which is exactly why it lands in the pool like any other.
