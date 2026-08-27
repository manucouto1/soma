---
title: Training
description: The graph never finds out. A trainer from outside, an update that is a group of steps, and a trainer that travels to the half that is not here.
---

There is no `g.fit(...)`. Training is done **from outside**, by something the
graph has never heard of:

```python
import torch
from somatize import Graph
from somatize.torch import Trainer, parameters

g = Graph.somatize(Encoder().on("cuda:0") >> Head().on("cuda:0"))

t = Trainer(
    g,
    objective=cross_entropy,
    optimizer=torch.optim.Adam(parameters(g), lr=1e-3),
)
result = t.fit(data, epochs=10)
result.loss
```

That is the whole point of the separation: **the same graph can be trained
three ways without touching it.** A node that also knew how to train itself
would be a node you could not run without deciding that first.

## The core does not learn what a loss is

None of this is in `soma-core`, and none of it ever will be. The core does not
know what a loss is, nor a gradient, nor an optimizer — writing that neutrally
would mean a `Backend` trait with a single implementor, which is the shape this
project deletes on sight.

So it lives in `somatize.torch`, in Python, and the core does not change a
line. The line between the two languages is drawn by **shape and not by
language**: Rust keeps what is pure, deterministic and hashable; the loop stays
where torch is.

The optimizer is **the caller's**, which is what keeps a name registry
(`optimizer="adam"`) out of the API.

## `step` is the primitive and `fit` is sugar

```python
t.step(batch)                  # forward, loss, backward, update when the group closes
t.fit(data, epochs=10)         # one step per batch, for as many epochs as you say
```

Whatever does not fit in an epoch loop is written with `step`. `fit` walks
`data` once per epoch, so with more than one epoch it has to be re-iterable — a
generator is exhausted on the first.

## An update is a group of steps

```python
Trainer(g, ..., every=4, micro=2)
```

`every` is how many steps go into one update. `micro` is how many pieces one
step is cut into. They **multiply** rather than compete: `every=4, micro=2` is
eight forwards per update.

The split between them is the level boundary doing its job. A micro-batch is
level 2 — the batch belongs to the caller, so `torch.chunk` reaches it — while
`.mapped()` is the engine's, because caching item by item has to **name** each
item and a name comes from its content, not its place. See
[the model](/soma/model/overview/).

Whatever trains itself elsewhere makes the same group out of the same steps,
and it needs nobody to tell it which step it is on.

## The half that is not here

Training a graph that has a slice on another machine is **not** training that
slice. What crosses a wire is the value, not the graph that made it, so a
`backward()` in this process reaches nothing over there.

```python
Trainer(
    g,
    objective=cross_entropy,
    optimizer=Adam(parameters(g, without=trains), lr=1e-3),  # the half that is here
    trains={"body": Split(SGD, lr=0.1)},                     # the half that is not
    broker=Broker.embedded({"gpu": Worker.at("node3:7000")}),
)
```

`trains` puts a trainer **beside** the node rather than inside it, so **the
node is never asked to know it is being trained**. Those weights belong to that
trainer, so they come out of this optimizer — `parameters(g, without=trains)` —
and holding both is *refused* rather than quietly updating them twice.

It is said **here**, in the trainer, and not on the graph, because it is a fact
of this training run and not of the graph. Where a graph gets cut is the pair
`(host, trained)`: `.at()` already said the first half and the graph owns it;
the second is told.

`Split` is one tenant of a hole called `Learning`. A technique writes
`learn(signal, ctx)` and nothing else — handed `dL/d(what the node produced)`,
it gives back `dL/d(what it was given)`, or `None`. Split learning, greedy
layer-wise, forward-forward and synthetic gradients are the same hole answered
four ways.

And none of it needed a new variant in `Plan` or anything new on the wire: a
trainer lets go of the activation exactly as a cable does, and a backward pass
is a `forward` of the transposed stage.

## An activation crossing

A torch tensor mid-autograd-graph cannot be converted without being destroyed —
round-trip it through numbers and it comes back without its `grad_fn`. So it
travels as an `Opaque` with a `Codec` in front of it, and the same node is
handed the same shape wherever it runs. See
[a node is one thing](/soma/model/a-node/).

## Freezing, and why the trainer calls it

```python
from somatize.torch import freeze

freeze(g, "encoder")   # declares and obeys
freeze(g)              # only obeys what was already declared
```

`.frozen()` in the expression is *information the core reasons over* — it
cannot make it true, because the core cannot reach a `requires_grad_`. This is
what makes it true, and `Trainer` calls the second form itself, so a `.frozen()`
written in the expression is true before the first step rather than after
somebody remembers.

That matters because of the rule in
[what is remembered](/soma/running/what-is-remembered/): a node's output can be
kept only if nothing upstream of it can change. What is restored from a store is
a **leaf**, so the backward pass stops there and everything above it quietly
stops training.

## What a run leaves behind

```python
t.export()     # {node_id: {key: tensor}} — a snapshot, detached and copied
```

Node by node, by the same two ducks everything here asks by. A snapshot and not
a view, so the next step does not move it under you.

That is what makes federation a `for` loop rather than a framework:

```python
from somatize.torch import fedavg, gather

averaged = fedavg(exports, sizes=[len(d) for d in shards])
```

`fedavg` is a **function**. A federated round is a `for`. Level 3 has no type
and that is on purpose — see [the model](/soma/model/overview/).

Across machines it is a folder they all mounted and nothing else: a `Store`
opened by hand, `claim` to hand work out, and `gather` for the round, where
**whoever finds it complete claims the averaging**. No coordinator to keep
alive, no port, no protocol, no `Plan::Remote`. Slurm distributes.

## Watching it

```python
Trainer(g, ..., watching=recorder, auditing=True)
```

`watching` is told what happens **and is handed on to every `forward`**, so one
stream carries both the engine's vocabulary and this level's `loss` and
`updated`. They meet in the record and not in Rust — see
[the record](/soma/looking/the-record/).

`auditing=True` hooks the nodes and emits `health` facts through that same
door. Thresholds never go near it: baked into the measurement, they would make
an argument about a bound cost an afternoon of GPU. See
[the health of a network](/soma/looking/health/).
