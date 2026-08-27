---
title: The health of a network
description: Sixteen flags computed from the stored record and never from a second training run — plus the one that was measured and left switched off.
---

This is the third row of the split. The declaration is **drawn**, the record
says what **happened**, and this is the **diagnosis** — which is an opinion
about the record and not a fact in it.

The line between them is not a preference. It is a test:

> **A diagnosis has to be reproducible from the stored record, without training
> again.**

Which is why `soma-health` is a crate with **no dependencies at all, not even
the core's**. Numbers in, flags out, no measuring and no clock. Change a bound
and ask again; the record has not moved, and an argument about a threshold
costs a scan instead of an afternoon of GPU.

## Measuring, and diagnosing, are different jobs

```python
from somatize.torch import Trainer

t = Trainer(g, ..., watching=recorder, auditing=True)
t.fit(data, epochs=10)
```

`auditing=True` hooks the nodes and emits `health` facts through the same
`watching=`. **Thresholds never go near it.** Baked into the measurement they
would make an argument about a bound cost a re-run, which is exactly what the
invariant exists to prevent.

Then, from the record, at any time and as many times as you like:

```python
from somatize.health import Thresholds, about, alerts, diagnose, flags, history, overlaid, profile, seen, where

diagnose(store, run="tuesday")                                    # the findings
diagnose(store, run="tuesday", thresholds=Thresholds(dead_frac=0.9))   # argue with a bound
```

## The taxonomy

Two of these are inherited from the original, and they are **knowledge, not
design** — they were measured once and the measurement holds:

| flag | what it says |
|---|---|
| `Dead` | most of what this node outputs is zero, on at least one step |
| `Saturated` | most of it is pinned at the far end of its non-linearity, where the derivative is nothing |

Both are read off the **maximum** over a window and never the mean. **A layer
that dies one step in four is dead**, and the mean is exactly what hides it.
And dormant is not dead.

The classic pair, and `Vanishing` is a **profile over depth** rather than a
property of a network — the early layers go quiet while the last one learns:

| flag | |
|---|---|
| `Vanishing` | the parameter gradients are so small this node is not being trained |
| `Exploding` | so large the next step will not be a step |
| `Nan` / `Inf` | a number stopped being a number, or stopped being finite |

Three came in from the literature. `Stalled` and `Overstepping` read the
**update-to-weight ratio**, which the original measured and never said anything
about, and which lands a healthy layer near `1e-3` — the cheapest signal there
is:

| flag | |
|---|---|
| `Stalled` | moving, but by so little relative to its own weights that it will not arrive |
| `Overstepping` | moving so much that each step throws away where it was |
| `LosingPlasticity` | **a conjunction** — weights growing *or* units going quiet on their own is a network that is training |

Three are about width rather than depth, and the distinction is the point:

| flag | |
|---|---|
| `DeadChannels(n)` | how many channels output near zero. Separate from `Dead`: a layer can be alive with a quarter of its width doing nothing, which is a width problem |
| `IgnoredChannels(n)` | channels that are **alive and never asked for** — they compute something and no gradient comes back. Gradient starvation, and a dormant channel is not the same thing: it is computing nothing to be ignored |
| `Leakage` | two groups of channels the architecture means to keep apart are carrying the same information, by linear CKA |

## The one that was measured and left off

`Narrowing` is in the vocabulary and **off by default, because it was measured
and the measurement did not support it.** The published monitor's certificate is
a deviation from a healthy baseline, and one run has none.

The metric is recorded and drawn. The alarm was not invented. See
`soma-health/tests/narrowing.py`.

## A node is often a whole architecture

*This node is unhealthy* is not an answer when the node is twenty layers:

```python
from somatize.torch import Audit

Trainer(g, ..., auditing=Audit(inside=True))
```

Findings are then keyed `node.path.to.submodule`, and
`overlaid(..., inside=...)` puts each one **on the layer it is about** — which
is what makes *what is measured has a box* true rather than hopeful. See
[the graph, drawn](/soma/looking/the-graph-drawn/).

## Where is a question the graph answers

```python
where(store, run="tuesday")       # which nodes, and on which machines
overlaid(g, store, run="tuesday") # the ill ones marked on the figure
alerts(store, run="tuesday")      # the loud one: cards a cell shows on its own
profile(store, run="tuesday")     # the shape over depth
```

Health gets **a channel of its own** on the figure: the fill goes on saying
where a node runs and the **outline** turns red. On a graph spread over three
machines, *where does this run* is the answer somebody came for, and taking that
channel away to say something else would cost more than it buys.

Findings are coloured by **family** — numeric, signal, activation, step,
capacity, data — with a legend of the ones actually on the figure, because six
alarms that all look the same are one alarm.

## And a question that is not about the network at all

```python
from somatize.data import contribution

contribution(g, batches, objective=accuracy, over=["text", "symptoms"])
```

It shuffles one input and scores again; the drop is what that input was worth.

`health` asks whether a network is **learning**. This asks whether it is
learning **what you meant**, which no amount of looking at a gradient will ever
say. `IgnoredInput` is the finding — *a network with a perfectly healthy
gradient can be ignoring an input all afternoon without a single other flag
firing* — and `SoleReliance` is the other end of the same worry.

**Shuffled and not zeroed.** A zero is a value, and what is being asked about is
the correspondence with the answer.

It exists because of a real project: symptom channels for a mental-health
condition, months spent on the architecture, and the signal was in the
self-disclosure. `IgnoredInput` would have said so in an afternoon.

## Before the first step

Everything above needs the network to have been **learning**. To ask whether it
*can*, there is `probe`: one recorded forward that never trained, written under
the same keys and read back through all of the same functions — `diagnose`,
`seen`, `profile`, `flags`, `where`, `overlaid` and `alerts`, with no new code
at all.

It measures three things and **none of them is a gradient norm**: at
initialisation there is no loss, so a parameter gradient would be taken against
a target somebody made up. Of the three, exactly one earned an alarm:

| flag | |
|---|---|
| `MissingNormalisation` | the signal has grown over a stretch nobody is normalising |

It is a conjunction whose structural half lives in the **measurement**: the gain
is counted from the last normalisation upstream, because resetting the reference
is what a norm layer is for. And it is **one-sided**, which the measurement
decided and the design did not — a stack whose signal arrives five
ten-thousandths of the size it went in trains as well as a healthy one, because
Adam is scale-invariant per parameter. See
`soma-health/tests/normalisation.py`.

The other two raise nothing, and the rule they left behind is the one worth
carrying:

> **What separates is a runaway. What ranks is a proxy.**

They **rank** and do not **separate** — and the network with the tighter
spectrum was the one that failed. A ranking belongs at level 3, where a number
only means something next to another candidate's, which is where the five
zero-cost proxies went: `somatize.torch.proxies` is a cheap objective a study
scores with, never a `Flag`.

See [before a step is taken](/soma/tutorials/08-before-a-step-is-taken/).
