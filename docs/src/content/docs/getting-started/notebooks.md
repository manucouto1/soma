---
title: Tutorials
description: Fifteen runnable notebooks, from your first filter to a research campaign that uses its own recorded history — and a panel of agents that debate.
---

The notebooks in [`notebooks/`](https://github.com/manucouto1/soma/tree/main/notebooks)
are the hands-on path through Soma. They ship **executed, with their
outputs saved** — tables, diagrams and figures included — so you can
read them on GitHub before running anything.

```bash
git clone https://github.com/manucouto1/soma && cd soma
pip install 'somatize[viz]'
jupyter lab notebooks/
```

Figures are stored twice: interactive (Plotly) for Jupyter and VS Code,
plus a PNG fallback so they also render on GitHub.

## The path

| # | Notebook | What you learn |
|---|---|---|
| 01 | [Filters and Graphs](https://github.com/manucouto1/soma/blob/main/notebooks/01_filters_and_pipelines.ipynb) | The `fit`/`forward` contract, stateful and parameterized filters, composing with `.node()`/`.connect()` and the `>>` / `\|` DSL, and reading a compiled plan |
| 02 | [Caching and state](https://github.com/manucouto1/soma/blob/main/notebooks/02_caching_and_state.ipynb) | Why `fit()` runs once per (config, data), what invalidates a cache line, seeds as cache lines, code as part of the identity, `soma cache`, and opting out |
| 03 | [Search and optimization](https://github.com/manucouto1/soma/blob/main/notebooks/03_search_and_optimization.ipynb) | `Study` with random and Bayesian (TPE) search, experiment seeds, and trials that drive a whole graph |
| 04 | [Streaming](https://github.com/manucouto1/soma/blob/main/notebooks/04_streaming.ipynb) | Chunked execution, per-chunk caching, and barrier filters that need the whole stream |
| 05 | [Advanced patterns](https://github.com/manucouto1/soma/blob/main/notebooks/05_advanced_patterns.ipynb) | Trunks shared across graphs, `.somack` checkpoints, a study over a full graph with seeds, and distribution |
| 06 | [Tracking and diagnostics](https://github.com/manucouto1/soma/blob/main/notebooks/06_tracking_and_diagnostics.ipynb) | `track_run` directories, a training run with a gradient audit, a study with a composite objective and pruning, and resuming from disk |
| 07 | [Visualization and reports](https://github.com/manucouto1/soma/blob/main/notebooks/07_visualization_and_reports.ipynb) | `soma.runs()`/`RunView`, annotated architecture diagrams, the Optuna-style study figures, DataFrames, and the self-contained HTML report |
| 08 | [Auditing inside nodes](https://github.com/manucouto1/soma/blob/main/notebooks/08_auditing_inside_nodes.ipynb) | `gradient_audit(inside=...)`, hierarchical ids, the per-layer gradient-flow staircase, the annotated inner architecture, and flag rollup |
| 09 | [Complex architectures and health](https://github.com/manucouto1/soma/blob/main/notebooks/09_complex_architectures_and_health.ipynb) | A multimodal fork/fan-in pipeline, and a branched model with four injected pathologies — dead channels, CKA leakage, a gradient-starved branch, a vanishing trunk — all caught at default thresholds |

### One campaign, in three parts

Notebooks 10–12 are a single story about a single model, and are best
read in order. A 1-D sensor stream with three regimes; two feature views
that are each blind to one of them; an encoder with a copy-paste wiring
bug and a trunk that will not pass a gradient.

| # | Notebook | What you learn |
|---|---|---|
| 10 | [Building an architecture you can read](https://github.com/manucouto1/soma/blob/main/notebooks/10_building_an_architecture.ipynb) | Fork/fan-in filters, what `compile()` can prove before you run, why preprocessing and the trainable region belong in separate graphs, and the annotated diagram of what actually happened |
| 11 | [Auditing before tuning](https://github.com/manucouto1/soma/blob/main/notebooks/11_auditing_before_tuning.ipynb) | Four pathologies caught in six training steps — leakage, dead channels, a gradient-starved branch, a vanishing trunk — then a grid sweep over the version that deserves one |
| 12 | [The research campaign](https://github.com/manucouto1/soma/blob/main/notebooks/12_research_campaign.ipynb) | Four variants and one lineage: `checkout` to branch, the move recorded on every edge, `diff` between siblings, and `find_similar` returning the conclusion you wrote three cells earlier |

The result notebook 12 arrives at is worth spoiling, because it is the
argument for keeping a lineage at all: two plausible fixes each buy
nothing on their own (+0.008 and −0.016), and together they are worth
+0.305. One-variable-at-a-time would have reported that neither helps.

### Agentic flows

| # | Notebook | What you learn |
|---|---|---|
| 13 | [Tuning an agentic flow](https://github.com/manucouto1/soma/blob/main/notebooks/13_agentic_flows.ipynb) | `soma.Agent` and `soma.Judge` as graph nodes, a `refine()` loop, tools, and a `Study` where the prompt and the topology are hyperparameters. Ships with an embedded mock provider, so it runs with **no API key and no local model** |
| 14 | [A panel of agents](https://github.com/manucouto1/soma/blob/main/notebooks/14_a_panel_of_agents.ipynb) | `board()` — the multi-agent debate of Du et al. (ICML 2024) — replicated on GSM8K, sweeping panel size and rounds with a grid. **Needs a live model** (Ollama): no mock can measure an accuracy gap |
| 15 | [Pipelines and agents, each calling the other](https://github.com/manucouto1/soma/blob/main/notebooks/15_pipelines_and_agents.ipynb) | The seam itself, both directions: an `Agent` inside a compute pipeline scored by `Eval`, schema contracts refused at compile, a step running a pipeline with `RunGraph`, the journal replaying a run with zero model calls, `orchestrate`, and `Suspend`/`resume`. Embedded mock — **no key needed** |

Notebooks 06–12 need PyTorch (`pip install torch`); 01–05 and 13–15 run
on the core install alone. Notebooks 10–12 share `notebooks/campaign.py`,
which holds the data generator and the model — a module rather than a
cell, because `Graph.load` resolves filters by import path.

## Suggested routes

- **"I want to build a pipeline"** → 01 → 02 → 05
- **"I want to tune hyperparameters"** → 01 → 03 → 07
- **"My model trains badly and I don't know why"** → 06 → 08 → 09
- **"I want to show results to someone"** → 07 (report + figures)
- **"I want to run a research campaign"** → 10 → 11 → 12
- **"I want to build and tune an agentic flow"** → 01 → 13 → 15 → 14

Notebooks 01–05 and 10–15 are in English; 06–09 are in Spanish, and
translating them is outstanding.

Re-execute them with `python notebooks/execute.py` (all of them) or
`python notebooks/execute.py 10 11 12` (a subset). Each runs in a fresh
temporary directory with an empty cache — the cache-miss demonstrations
are only honest against one — and a notebook whose output contains a
warning is refused rather than written back.

Each notebook ends with a *What's next* pointer. For the reference
material behind them, see the [Python API](/soma/api/python/); for the
design rationale, the Design section.
