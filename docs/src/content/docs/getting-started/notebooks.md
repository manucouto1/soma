---
title: Tutorials
description: Nine runnable notebooks, from your first filter to auditing gradient health inside a multimodal model.
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

Notebooks 06–09 need PyTorch (`pip install torch`); 01–05 run on the
core install alone.

## Suggested routes

- **"I want to build a pipeline"** → 01 → 02 → 05
- **"I want to tune hyperparameters"** → 01 → 03 → 07
- **"My model trains badly and I don't know why"** → 06 → 08 → 09
- **"I want to show results to someone"** → 07 (report + figures)

Each notebook ends with a *What's next* pointer. For the reference
material behind them, see the [Python API](/soma/api/python/); for the
design rationale, the Design section.
