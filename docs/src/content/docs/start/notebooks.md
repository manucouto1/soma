---
title: The notebooks
description: Thirteen runnable notebooks, shipped executed with their outputs — declaring a graph, watching a run, and eleven more.
---

The notebooks in [`examples/`](https://github.com/manucouto1/soma/tree/main/examples)
are the hands-on path. They ship **executed, with their outputs saved** —
tables, diagrams and figures included — so opening one shows what it does
without running anything, and every one of them is readable here: each title
below links to the notebook rendered as a page.

Each one runs on its own and needs nothing downloaded. The data is made up in
the cell that uses it, on purpose: an example that fetches a dataset is an
example that stops working.

## The path

| | what it is about |
|---|---|
| [01 — Declaring a graph](/soma/tutorials/01-declaring-a-graph/) | the DSL, `.on()` / `.at()` / `.cached()` / `.frozen()` / `.mapped()`, and the figure a graph draws of itself before it has ever run |
| [02 — Watching a run](/soma/tutorials/02-watching-a-run/) | `watching=`, a `Recorder`, reading a run back, `progress` / `spent` / `Live`, the cache seen, and a node on a real worker |
| [03 — Training](/soma/tutorials/03-training/) | a `Trainer`, `Opaque`, the loss drawn live, gradient accumulation, freezing, and exporting what a run learnt |
| [04 — A study](/soma/tutorials/04-a-study/) | `Space` / `Sampler` / `Pruner`, the distributed loop, a table of results, hyper-parameter influence and parallel coordinates |
| [05 — The health of a network](/soma/tutorials/05-the-health-of-a-network/) | `auditing=`, each pathology built and caught, and the invariant: a diagnosis taken from the record, argued with by moving a bound, and taken again |
| [06 — A problem, end to end](/soma/tutorials/06-a-problem-end-to-end/) | the whole loop on one problem: propose an architecture, find what is wrong, fix it, check the fix — five times, and the last one is not a bug in the network at all |
| [07 — A real architecture](/soma/tutorials/07-a-real-architecture/) | convolutions with residuals, a transformer stack, a recurrent cell and a bottleneck, with the architecture drawn inside each; three problems in **problem → symptoms → solution → healthy** cycles, and two that showed nothing |
| [08 — Before a step is taken](/soma/tutorials/08-before-a-step-is-taken/) | `probe`, one recorded forward that never trained, and the rule: what separates is a runaway, what ranks is a proxy |
| [09 — A fleet](/soma/tutorials/09-a-fleet/) | `fleet` and `machines`, working against waited on, what only a machine can say about itself, and the idle one that writes on a clock |
| [10 — Where the data comes from](/soma/tutorials/10-a-dataset/) | a source is a node, a graph handed a **coordinate** instead of a batch, what that saves the cache, the version the store already knew, and a frame crossing a wire |
| [11 — What an edit did](/soma/tutorials/11-what-an-edit-did/) | `foreseen.names` / `unneeded` / `changes` / `snapshot`, an afternoon of edits answered without running any of it, the three ways a name moves, and the half a notebook cannot answer about its own cells |
| [12 — Where a value came from](/soma/tutorials/12-where-a-value-came-from/) | why a key does not run backwards, the five things written beside a kept value and who is standing where each is knowable, the four that land with nobody asking, and what a caller may not say |
| [13 — The reasoning of an investigation](/soma/tutorials/13-the-reasoning-of-an-investigation/) | the five kinds written from the terminal, `depends` and why it is not a dispute, what folds and why, going back to a move by name, and a standing that comes back on its own |

## Running them

```bash
git clone https://github.com/manucouto1/soma && cd soma
pip install 'somatize[viz]'   # plotly, for every figure here
pip install ipywidgets        # optional: `Live` redraws in place with it
jupyter lab examples/
```

Notebooks 03 to 07 need `torch`. Notebook 02 starts a real worker process,
which needs nothing but the same interpreter. Notebook 13 needs the
`somatize-tree` command, which the wheel does not carry — it is a binary, and a
wheel has no use for an argument parser:

```bash
cargo install --path soma-tree
```

Every figure is stored twice, and that is on purpose: the Plotly JSON, so
JupyterLab and nbviewer draw it live and you can hover a node or zoom a curve;
and a PNG beside it, so a static viewer — GitHub, a diff, the pages here —
shows the same figure instead of an empty cell. It is the renderer that
decides:

```bash
PLOTLY_RENDERER="plotly_mimetype+png" jupyter lab
```

## What is not here, and why

**A graph marked with what an edit did.** Notebook 11 answers with findings per
node and stops there. `overlaid` already puts findings on a figure, but its
channel is health: the outline turns red, and red means ill. A node whose
recipe changed is not ill, so saying **where** an edit landed needs a channel
of its own rather than borrowing one that means something else.

**A run spread over real machines.** Notebook 02 starts one worker on this
machine and notebook 09 reads a fleet back out of a record. Containers on
separate hosts, a GPU among them, and a study handed out of a shared folder
live in `soma-python/tests/cluster/`, which needs docker and is opt-in — an
example that needs a cluster is an example nobody can open.

For the notes on re-executing them after an API change — including why a debug
build writes timings ten times worse into the record — see
[`examples/README.md`](https://github.com/manucouto1/soma/blob/main/examples/README.md).
