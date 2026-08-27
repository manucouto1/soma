---
title: A study
description: Level 3 has no type — N training runs are a `for` — and the pieces that loop asks for are indices in, indices out, never a tensor.
---

The graph is a network, one `forward`. The `Trainer` is a training run, an
afternoon. This is the level above, and **it has no type**: N training runs are
a `for` loop.

That is a decision and not an omission. A graph earns its keep when there are
dependencies to declare, and N runs have none. Making level 3 a graph would
have been the same mistake one level up — see
[the model](/soma/model/overview/).

What lives in `somatize.study` are the pieces that `for` asks for, and they all
have one shape:

> **Indices and keys in, indices out. Never a tensor.**

Which is what lets all of it be Rust while the loop stays in Python. The one
step that cannot move is *train*, because training is torch — and a trait that
calls back out for it is not an abstraction, it is the loop leaking. The
original measured that: its `TrialExecutor` has one implementor, and it is a
closure wrapper.

## Three families of the same shape

Each is an enum of structs, because in every case the set is closed and known.

### Where to look

| | looks at |
|---|---|
| `Grid` | **the space's shape**, and the one that runs out |
| `Random` | **nothing**. Over a space where few knobs matter, it beats a grid |
| `Halton`, `Sobol` | nothing either, but uniform for **every prefix** |
| `Tpe` | **what already happened** |

`ask` is a function of the **index**, not of what was asked before. So a
machine that claimed trial 7 out of a shared folder derives the same point
without replaying six. `Tpe` is the exception, and it says so.

Uniform for every prefix is what stops two machines proposing neighbours.
`Random` is uniform only *in expectation*, which merely makes it unlikely.
`Halton` is arithmetic and has no ceiling; `Sobol` uses Joe & Kuo's table and
tops out at 32 knobs.

### When to give up

| | judged against |
|---|---|
| `Percentile` | **the others** at the same step. The median pruner is `p = 50` |
| `Threshold` | **a constant** already known to be hopeless |
| `Patience` | **itself**: it has stopped improving |

**A pruner stops nothing.** It answers a `Verdict` and the loop stops calling
the trainer — which is why none of this added a line to level 2. Nothing is
asked of a `Trainer`.

### Where to cut the samples

| | |
|---|---|
| `KFold` | `k` parts, each held out in turn |
| `Stratified` | a k-fold **inside each class** |
| `Grouped` | a k-fold **over the groups**, so a group never splits |
| `StratifiedGrouped` | both, as far as both can be had at once |
| `TimeSeries` | growing prefixes, so nothing trains on its own future |

Five schemes and not sklearn's fifteen, because **stratifying and grouping are
not different algorithms** and the rest are parameters: `LeaveOneOut` is
`KFold { k: n }`, and purged cross-validation is `TimeSeries { gap }`.

It is not called `Split`, because `somatize.torch.Split` is already split
learning — two traits with a same-named method in scope make that name
unusable, and two types with one name make a reader guess.

## The loop

```python
from somatize.study import Sampler, Space, finished, report, take

space = Space().real("lr", 1e-5, 1e-1, log=True).choice("opt", ["adam", "sgd"])
sampler = Sampler.tpe(goal="min")

for trial in range(100):
    point = sampler.ask(space, trial, finished(store, space, study="spam"))
    if not take(store, point, study="spam", trial=trial, me=me, goal="min"):
        continue                                   # somebody else has that one
    ...                                            # train, and collect reports
    report(store, point, reports, study="spam", trial=trial, me=me,
           state="done", goal="min")
```

That is the whole thing, and the same script runs on every machine.

## Handing out work costs no message, because nothing is handed out

A trial is a **number**. `ask` is a function of that number. `take` settles who
gets it with a conditional write. So **the state *is* the queue**, and a claim
is exactly-once by construction — no server, no port, no protocol, and nothing
to keep alive.

Slurm distributes. The shared folder is the only thing the machines have in
common.

## The record, and why it is shaped that way

A trial lives at `<study>/trial/<n>/<attempt>`.

| where | what |
|---|---|
| the **record**, which a scan already carries | `state`, `point`, `score`, `who`, `goal` |
| the **blob**, which costs a fetch | the whole curve, and why it stopped |

That split is the cost model. **A sampler's whole history is one scan and zero
fetches**, because the configuration is kept as text beside the score. Only a
pruner's curves cost a fetch each.

`goal` is written **per trial**, and that is denormalised on purpose: a score is
good or bad and the number does not say which, so whoever reads the study
without this script has no other way to find out. Per trial rather than per
study because it records what was meant **at the time**.

And it is **one record rewritten as it goes**, not five events. A bus's
`TrialStarted` / `TrialPruned` / `TrialCompleted` are the *diff* of this record
— derivable from it, while the record is not derivable from them without
replaying all three.

## A guided sampler that knows what the others are holding

```python
sampler.ask(space, trial, finished(...) + in_flight(...))
```

`ask` takes a score that **may be absent**, and absent means *running*. Those
points are kept away from **without voting on how big the good pile is**.

That distinction was measured. Handing a guided sampler a made-up bad score for
a running trial gets it backwards: it does not just fail to help, it moves the
threshold the wrong way. See
[handing trials out of a folder](/soma/searching/handing-out-trials/).

## Reading it back

```python
from somatize.study import coordinates, importance, influence, table, trials

trials(store, space, study="spam")       # every trial, whatever state
table(store, space, study="spam")
influence(store, space, study="spam")
importance(store, space, study="spam")   # Spearman's ρ
coordinates(store, space, study="spam")  # parallel coordinates
```

`importance` is **Spearman's ρ**, which the original names as fANOVA-deferred
and never wrote. `coordinates` is hand-drawn out of splines because plotly's
`Parcoords` only draws straight segments; it trades brushing for a trial
reading as one curve.

In all three, **pruned and finished are never ranked together** — a pruned
trial's score was measured after fewer epochs, and it is not comparable.

## And a cheap objective

```python
from somatize.torch import proxies
```

Five zero-cost proxies, scored before training. They are an **objective a study
loops over**, never a `Flag`, because *what separates is a runaway and what
ranks is a proxy* — and a ranking only means something next to another
candidate's, which is exactly this level. The only question worth asking of one
is in `soma-health/tests/proxies.py`: *does it beat counting parameters?*
