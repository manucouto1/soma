---
title: The graph, drawn
description: The plan is a tree, so placing it needs no layout engine — and what it draws is the declaration, before anything has run.
---

```python
g.figure()   # a plotly Figure, to show or to compose
g            # in a notebook: the same figure, straight in the cell
```

A graph can be drawn **having never run**. Printing a declaration is not
observing it, and this is the first of the three things observability was split
into — the other two being
[the record of what happened](/soma/looking/the-record/) and
[the diagnosis](/soma/looking/health/).

## What is drawn is the plan

Not the list of edges. The **plan** — `compile` then `distribute` — because
that is where the decisions show: a `Wave` is what runs at once, a `Remote` is
what crosses to another machine, and a bare list of edges says neither.

| in the plan | on the figure |
|---|---|
| `Execute` | a box, filled by the device it runs on |
| `Sequence` | its children stacked, top to bottom |
| `Wave` | its children side by side, inside a frame |
| `Remote` | a frame labelled with the host |
| `Empty` | an empty figure that says so |

**The plan is a tree**, so placing it needs no layout engine and no
crossing heuristic. That is the payoff of [`compile` decomposing rather than
flattening](/soma/running/the-plan/).

## The boxes say *when*; the arrows say *what feeds what*

A graph that is not series-parallel falls back to a flat `Sequence`, and there
the nesting stops saying who feeds whom — the truth lives entirely in each
step's `from`. So the two channels are separated on purpose, and the **N**
(`a→c`, `a→d`, `b→d`) is the case that keeps the figure honest. It is in the
tests.

An edge that would cross a box it does not belong to is **routed around it**,
one lane each. An arrow drawn over a node reads as an arrow into it.

## One fact per channel

The fill says **where a node runs** and nothing else. Cached, frozen and mapped
are badges in the label, because three facts cannot share one fill.

Health gets a channel of its own — the **outline** turns red — precisely so the
fill can go on saying where. On a graph spread over three machines, *where does
this run* is the answer somebody came for.

The colours live in one table, `somatize._theme`, and are looked up with `[]`
and never with `.get(…, default)` — so a typo fails rather than quietly coming
out as the alarm colour. In the original the same colours lived in four tables
keyed by the same strings.

That table is also what this site is painted from. A library whose graph is
light and whose curves are dark is two libraries.

## Opening a node up

A node is often a whole architecture, and *this node is unhealthy* is not an
answer when it is twenty layers:

```python
from somatize.torch import architecture

g.figure(inside=architecture(g, x))
```

`architecture` traces what each node is made of — with `fx` where it can,
because `fx` sees the operations that are **not** modules and a residual
connection is exactly one; with a real forward where it cannot, **saying so**,
because a residual that is missing looks exactly like a residual that is not
there.

The unit is the **node**, since a node holding two modules composes them in its
own `forward`.

`g.figure(inside=...)` draws a node's box as a **frame** — the shape a `Wave`
and a `Remote` already are — and lays the inside out by what feeds what, so a
skip runs down a gutter and enters from the side.

## The rules that make an architecture readable

**A kind, not a class name, decides the silhouette.** A convolution is a
parallelogram, a recurrent cell has a tab, an attention block has its corners
cut and says what is in it, a normalisation is a capsule, a non-linearity is
pointed, and anything that changes the width is tapered the way it goes. The
kind is guessed **by role**, so a class the table has never heard of whose name
ends in `Norm` is a normalisation — a guess, and a good one, because the
alternative is calling half of everybody's models `other`.

**A composite everybody recognises is one box**, and `depth=` opens it.

**Blocks that are the same block collapse to `×N`** — and when the block is
more than one layer, that `×N` goes on a **frame around them** rather than on
each of them. Four encoder layers opened up are eight boxes each saying `×4`,
which is the count said eight times and the block said none.

**Identical lanes running at once get plates behind them**, never separate
boxes. The heads of an attention block and the groups of a convolution are one
projection in torch; four of them wired side by side would be a graph nobody
built.

**The shape is written on the layer**, because that is the only thing that
makes a bottleneck a picture. And every number **says what it is** —
`4 batch · 16 steps · 24 dim`, not `4×16×24`. The batch is checked rather than
assumed, since the caller knows how many rows went in, and a layer that did not
change the shape keeps the names of the one that did — so a `BatchNorm1d` in a
convolutional trunk says `ch` and `len`.

## Marking it

```python
from somatize.health import overlaid

overlaid(g, store, run="tuesday", inside=architecture(g, x))
```

The audit's scope and the drawing's are then the **same scope**, which is what
makes *what is measured has a box* true rather than hopeful — those two stop
being the same set once `fx` has had its say, and a finding on a layer with no
box lands nowhere.

Findings are coloured **by family** — numeric, signal, activation, step,
capacity, data — with a legend of the ones actually on the figure. Six alarms
that all look the same are one alarm.

## And a study draws too

```python
from somatize.study import coordinates, importance, influence, table
```

`importance` is **Spearman's ρ**, which the original names as fANOVA-deferred
and never wrote. `coordinates` is hand-drawn out of splines, because plotly's
`Parcoords` only draws straight segments — it trades brushing for a trial
reading as one curve.

In all three, pruned and finished trials are never ranked together.
