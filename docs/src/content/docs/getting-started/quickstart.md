---
title: Quickstart
description: Install Soma and go from a first filter to a tracked, visualized, cached pipeline in five minutes.
---

Everything below is runnable top to bottom.

## Install

```bash
pip install somatize            # core: Graph, Study, tracking, cache, CLI
pip install 'somatize[viz]'     # + plotly, pandas, rich, tqdm (figures & reports)
```

The `viz` extra is optional: the core install never needs it, and every
plotting call raises a clear error telling you to add it.

## 1. A filter

A filter is a class with two methods: `fit` learns state, `forward`
transforms. Both are cached independently.

```python
from soma import Filter, Graph

class Normalizer(Filter):
    _cache_version = "quickstart-v1"     # see the note below

    def fit(self, x, y=None):
        mean = sum(x) / len(x)
        std = (sum((v - mean) ** 2 for v in x) / len(x)) ** 0.5
        return {"mean": mean, "std": std or 1.0}

    def forward(self, x, state):
        return [(v - state["mean"]) / state["std"] for v in x]

class Scale(Filter):
    _cache_version = "quickstart-v1"

    def __init__(self, factor=2.0, **kwargs):
        super().__init__(factor=factor, **kwargs)   # kwargs → cache key

    def forward(self, x, state):
        return [v * self.factor for v in x]
```

:::note[Why `_cache_version`]
A filter's cache identity is derived from its source code. In a REPL or
notebook the source is not retrievable, so Soma falls back to
cloudpickle and warns. Declaring `_cache_version` pins the identity
explicitly — bump the string when the filter's behavior changes. See
[Caching](/soma/design/caching/).
:::

## 2. A graph

Two equivalent ways to compose. Explicit:

```python
g = Graph()
g.node("normalizer", Normalizer())
g.node("scale", Scale(factor=3.0))
g.connect("normalizer", "scale")
```

Or the fluent DSL — `>>` chains, `|` forks:

```python
g = Graph.somatize(Normalizer() >> Scale(factor=3.0))
```

Fit, then transform new data with the learned state:

```python
g.fit([10.0, 20.0, 30.0, 40.0])
g.forward([15.0, 25.0])
```

In a notebook, evaluating `g` draws the architecture as a diagram; in a
terminal, `print(g)` shows the text tree. `g.compile()` returns the
execution plan plus diagnostics (also self-displaying):

```python
g.compile()          # tiles, per-level diagnostics, the plan drawn
print(g.to_mermaid())  # or to_graphviz() / to_svg() / to_text()
```

## 3. Caching is already on

Run the same fit twice and the second one is free — results are stored
content-addressed under `$SOMA_CACHE_DIR` (default `~/.soma/cache`),
survive crashes, and are shared across graphs and processes:

```python
g2 = Graph.somatize(Normalizer() >> Scale(factor=3.0))
g2.fit([10.0, 20.0, 30.0, 40.0])   # cache hit — no recomputation
```

```bash
soma cache stats      # records, CAS size, compute banked
soma cache gc --max-size 20G
```

## 4. Track a run

`track_run` writes a run directory (`.soma/runs/<run_id>/`) with the
graph snapshot, a lossless event log, metrics and diagnostics. It is
crash-safe and nothing reads it while training.

```python
with g.track_run("first-run", tags=["quickstart"]) as run:
    g.fit([10.0, 20.0, 30.0, 40.0])
    for step in range(5):
        run.log("loss", 1.0 / (step + 1), step=step)
```

## 5. Look at what happened

Readers never write, so they work on live, finished and crashed runs
alike:

```python
import soma

soma.runs()                    # table of every run (newest first)
view = soma.runs()[0]
view.node_timings()            # per-node wall time, durations, cache hits
view.metric_series("loss")     # the logged curve
print(view.to_mermaid())       # architecture annotated with timing/cache/health
```

With the `viz` extra, the same data as interactive figures:

```python
view.plot_metrics()            # metric curves
view.plot_gantt()              # where the wall time went, node by node
```

And from the shell:

```bash
soma runs                              # list runs
soma graph <run_id>                    # annotated diagram (mermaid/dot)
soma report <run_id> -o report.html    # one self-contained HTML report
```

## 6. Optimize

Declare the search space where the parameter lives, then let the study
sample it. `trial.report` feeds live curves and pruning:

```python
from soma import Study, search

class Model(Filter):
    _cache_version = "quickstart-v1"
    lr: float = search(1e-4, 1e-1, scale="log")

    def fit(self, x, y=None):
        return {"w": 1.0 - self.lr}

    def forward(self, x, state):
        return [v * state["w"] for v in x]

g = Graph.somatize(Normalizer() >> Model())

study = g.study("quickstart-hpo", strategy="bayesian", n_trials=8,
                objectives=[("score", "maximize")])

def train(trial):
    g.apply_params(trial.params)
    for step in range(5):
        score = 1.0 - abs(trial["model.lr"] - 0.01)
        if trial.report("score", score, step):
            return None            # pruned
    return {"score": score}

study.run(train, progress=True)    # progress= draws a tqdm bar
study.best_trial["params"]
```

Then the W&B-style views:

```python
study.plot_optimization_history()
study.plot_parallel_coordinate()
study.plot_param_importances()
study.trials_dataframe()
study.to_html("study.html")
```

## Where to go next

- **[Tutorials](/soma/getting-started/notebooks/)** — nine runnable
  notebooks, from filters to auditing a multimodal model's health.
- **[Python API](/soma/api/python/)** — the complete reference.
- **[Visualization](/soma/design/visualization/)** — diagrams, figures,
  reports, and the data contract a front-end reads.
- **[Gradient Health Audit](/soma/guides/gradient-audit/)** — find
  vanishing gradients, dead channels and leakage inside your models.
