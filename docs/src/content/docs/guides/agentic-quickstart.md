---
title: Agentic Quickstart
description: Build a refine loop, hand an agent a tool, and search over the agent — in about twenty lines.
---

An agentic flow in Soma is a graph. Its nodes reach outside the process —
they call a model, run a tool — but they sit on the same edges, go through
the same compiler, write to the same cache, and land in the same
experiment pool. So the prompt an agent uses is a hyperparameter, the
model it calls is a hyperparameter, and whether two nodes should be
connected at all is a hyperparameter — one search space, one `Study`.

This guide walks that arc. It mirrors
[notebook 13](https://github.com/manucouto1/soma/blob/main/notebooks/13_agentic_flows.ipynb),
which ships executed and embeds a **mock provider** so it runs with no API
key and no local model — read it there if you want the zero-setup version.
Here we assume a local [Ollama](https://ollama.com) (`ollama pull
qwen2.5`); any OpenAI-compatible endpoint in the provider catalog works
the same way.

## One agent

`soma.Agent` is a node like any other: `g.node(id, thing)` takes it
exactly the way it takes a filter.

```python
import soma

g = soma.Graph()
g.node("writer", soma.Agent(model="ollama/qwen2.5", system="be helpful"))
print(g.forward("explain compilers"))
```

There is no fit phase — a graph containing only steps runs `forward`
directly.

## Add a tool

A tool is a Python function with a docstring. The docstring is not
decoration: it is what the model reads to decide whether to call the
tool, so a function without one is refused rather than registered inert.

```python
@soma.tool
def lookup(term: str) -> str:
    """Look a term up in the glossary. Call this for unfamiliar jargon."""
    return f"{term}: a program that translates programs."

g = soma.Graph()
g.node("scholar", soma.Agent(model="ollama/qwen2.5",
                             system="Answer using the glossary.",
                             tools=[lookup]))
g.forward("what is a compiler?")
```

The model decides when to call it; the runtime performs and journals the
call like any other effect, and the answer lands back in the
conversation.

## A refine loop

`soma.Judge` grades against a rubric and reports `done` — exactly the
signal a loop reads to stop. `soma.agentic.refine` wires the two
together: a worker drafts, the judge grades, the worker sees the critique
and tries again, until the judge passes it or the round budget runs out.

```python
from soma.agentic import refine

def flow(system="be helpful", threshold=0.8):
    return refine(
        worker=soma.Agent(model="ollama/qwen2.5", system=system),
        judge=soma.Judge(model="ollama/qwen2.5",
                         rubric="Is it accurate and useful?",
                         threshold=threshold),
        max_rounds=3,
    )

verdict = flow().forward("explain compilers")
print(verdict["score"], verdict["reason"])
```

The loop *carries*: after each pass, the judge's verdict becomes what the
worker reads next. Without that, the loop would redraft the same thing
three times — the difference between "refine" meaning something and not.

## Search over the agent

An agent's constructor arguments are its hyperparameters, so the space is
declared where the value goes — the same `search()` a filter uses:

```python
g = refine(
    worker=soma.Agent(
        model="ollama/qwen2.5",
        system=soma.search(choices=["be terse", "be helpful", "be detailed"]),
    ),
    judge=soma.Judge(
        model="ollama/qwen2.5",
        rubric="Is it accurate and useful?",
        threshold=soma.search(0.5, 0.9),
    ),
    max_rounds=3,
)
print(g.search_space())    # writer.system, judge.threshold
```

## Run a study

The graph builds the study; the trial writes sampled params onto the live
agents and runs the flow:

```python
def trial(t):
    g.apply_params(t.params)
    verdict = g.forward("explain compilers")
    return {"score": verdict["score"]}

study = g.study("prompt-and-strictness", strategy="bayesian", n_trials=12,
                objectives=[("score", "maximize")])
study.run(trial, progress=True)
print(study.best_trial)
```

Every trial is a tracked run with lineage in the experiment pool, and the
report shows tokens per trial next to the metric — a study over an
agentic graph spends real money, which is why median pruning is not
optional here.

Topology can be a dimension too: `g.optional(a, b)` makes an existing
edge something a study may keep or cut.

## What's next

- [Writing a step](/soma/guides/writing-a-step/) — the `poll` contract,
  dynamic fan-out, suspend/resume, and running a pipeline from a step.
- [Agentic Graphs](/soma/design/agentic/) — why the design is shaped
  this way.
- Notebook 14 replicates Du et al.'s multi-agent debate with `board()`.
