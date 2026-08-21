"""Training the half that is not here, from beside the node and not inside it.

What crosses a cable is the value and not the graph that made it, so a node on
another machine gets no gradient from a `backward()` run in this process. The
answer is not to bring it back: it is to **put a trainer over there**. It keeps
the activation where its autograd graph is, it is handed `dL/d(what the node
produced)`, it carries on with the chain rule under an optimizer of its own, and
it gives back `dL/d(what the node was given)` so whoever is above can do the
same.

That is split learning, and the same shape covers the rest: ignore the signal and
it is local greedy, run no backward at all and it is forward-forward, predict the
signal and it is synthetic gradients. **One hole, four techniques** — `learn` is
the only thing a technique writes.

## Whose job this is

The node's job is to compute. Nothing here asks it to know that it is being
trained, where it runs, or what a gradient is: it is the same node, unchanged,
whether it trains here, on another machine, or by an objective of its own. Which
of them is trained on its own is a fact of the **training run**, said by whoever
trains::

    Trainer(g, objective=…, optimizer=…, trains={"body": Split(SGD, lr=0.1)})

## Where it lives, and why that is the catalog

The optimizer has to point at the tensors that execute, and those are the ones on
the machine the node runs on: the client's copy of the weights is other objects
in another process. So the trainer **travels**, and the one thing a worker keeps
between calls is its catalog — so that is where it goes, in the two positions a
graph can put it in:

```text
    …  →  body:in  →  body:computes  →  body   →  …
          the trainer   the node        the trainer
          leafs the     computes        keeps the activation,
          input                         gives out what it let go of
```

Both positions are the **same object**, and `pickle` keeps that: two entries of
one catalog, one trainer, one optimizer, pointing at the weights that are there.
Nothing in the worker, the protocol or the core learns that training exists —
what arrived is objects with a `forward`, which is all a catalog ever holds.

## The envelope

What tells a backward message from an input is a reserved key, on the precedent
of `__soma_opaque__`, with a cheap check before anything is built. One key and
not two, because unlike a packed opaque there is no kind to carry. In a learning
pass every value on every edge is an envelope, so a **map** of envelopes is not a
fan-in of inputs but a fan-in of gradients, and the chain rule says they add. An
envelope carrying nothing is how "no gradient for you this step" is said, which
is what a technique that gives none back answers.

## What this slice does not do

The wire is not touched: activations and gradients cross as lists of floats,
built back into a tensor on arrival. And a trained node takes **one** input — one
with two producers would owe a different gradient to each, which the transpose
alone does not route.
"""

from __future__ import annotations

from functools import partial

import torch

from soma_next import Done, Node, Opaque

SIGNAL = "__soma_gradient__"
"""The reserved key that says a value is not a value: it is the gradient of one."""


class OutOfStep(Exception):
    """A gradient arrived for an activation that is not there any more.

    Through a `Graph` it comes back as the text of a `ValueError`: a node's
    failure crosses as a message and not as a type.
    """


def envelope(gradient):
    """A gradient in its envelope, which is how one crosses an edge. What a
    transposed stage is fed with, by a `Trainer` or by hand."""
    return {SIGNAL: _data(gradient)}


def gradient(value, device=None):
    """What an envelope carries, as a tensor — or the sum of what a map of them
    carries. `None` for something that is not a backward message at all, and
    `None` too for an envelope carrying nothing."""
    inside = _envelopes_in(value)
    return None if inside is None else _added(inside, device)


def leaf(value, device=None):
    """A value as something that can be differentiated back to.

    Detached on purpose, and it is the whole premise: what trains a node lets go
    of the chain that produced its input, which is why it cuts the graph exactly
    as a cable does.
    """
    if isinstance(value, dict):
        raise ValueError(
            f"a trained node takes one input, and this one was handed a map of "
            f"{len(value)}: it would owe a different gradient to each producer, "
            f"and giving one back per edge is not something the transposed graph "
            f"routes yet"
        )
    return _tensor(value, device).detach().requires_grad_(True)


class Learning(Node):
    """What trains one node, on the machine that node runs on.

    A technique writes `learn(signal, ctx)` and nothing else: it is handed
    `dL/d(what the node produced)` and gives back `dL/d(what it was given)`, or
    `None` if it hands nothing back. `Split` is the one that comes with this.

    It reaches the node it trains through `held` — the activation of this step —
    and `given` — the leaf its input became. Both are dropped after each `learn`.
    """

    given = held = built = None
    """Nothing of what one step leaves is set in `__init__`: `pickle` does not
    call it, and being rebuilt on another machine is this object's normal life."""

    def __init__(self, optimizer, **how):
        self.making = partial(optimizer, **how)
        self.node = None

    def of(self, node):
        """The node it trains. Said when the graph is put together, because that
        is the only moment somebody has both."""
        self.node = node
        return self

    def beside(self):
        """Its two positions in a graph: the one that leafs the input, and
        itself, which keeps what the node produced."""
        return Enters(self), self

    def forward(self, value, ctx):
        """As a node, in the position after the one it trains: an ordinary value
        is the activation to keep, and an envelope is the gradient to learn
        from."""
        inside = _envelopes_in(value)
        if inside is None:
            self.held = value
            return Done(_data(value))
        return Done(envelope(self.learn(_added(inside, ctx.device), ctx)))

    def entering(self, value, ctx):
        """The input as a leaf, remembered, which is what makes `dL/d(input)` a
        thing that exists. Called from the other position."""
        self.given = leaf(value, ctx.device)
        return self.given

    def learn(self, signal, ctx):
        """What to do with `dL/d(what the node produced)`, and what to give back.

        The hole. Whatever a technique is, it is this method — and `signal` being
        `None` is nobody owing it anything this step.
        """
        raise NotImplementedError

    def done(self):
        """Lets go of this step's activation, and gives back what the node was
        given so a gradient can be read off it. Every `learn` ends here."""
        given, self.held, self.given = self.given, None, None
        return given

    def waiting(self):
        """The activation this step left, or `OutOfStep` if there is none.

        Not a `None` walking into an optimizer: a gradient for an activation that
        is not there means the two halves are a step apart, and that is worth
        stopping for.
        """
        if self.held is None:
            raise OutOfStep(
                f"`{type(self).__name__}` was handed a gradient and has no "
                f"activation to apply it to: either the node never ran forward "
                f"this step, or it already learnt from this one and let it go"
            )
        return self.held

    def training(self):
        """Which parameters it updates: the node's, and whatever else a technique
        brought with it — a decoder, a guesser — which is why it is a method and
        not a line."""
        return list(self.node.parameters())

    @property
    def optimizer(self):
        """Its optimizer, built the first time it is asked for — over the
        parameters of wherever it ended up, which is the whole reason it is built
        here and not by whoever declared it."""
        if self.built is None:
            self.built = self.making(self.training())
        return self.built

    def __repr__(self):
        return f"{type(self).__name__}({getattr(self.node, '__class__', '?').__name__})"


class Split(Learning):
    """Split learning: carry on with the chain rule from the gradient of the
    seam, step, and hand back the gradient of the input."""

    def learn(self, signal, ctx):
        if signal is None:
            self.done()
            return None
        held = self.waiting()
        self.optimizer.zero_grad()
        held.backward(signal)
        self.optimizer.step()
        return self.done().grad


class Enters(Node):
    """The trainer's other position: before the node, where the input becomes a
    leaf. It holds no state of its own — what it makes it hands to the trainer."""

    def __init__(self, learning):
        self.learning = learning

    def forward(self, value, ctx):
        return Done(Opaque(self.learning.entering(value, ctx)))


def _envelopes_in(value):
    """The gradients this value carries, or `None` if it is an ordinary input."""
    if not isinstance(value, dict) or not value:
        return None
    if _is_an_envelope(value):
        return [value[SIGNAL]]
    if all(_is_an_envelope(each) for each in value.values()):
        return [each[SIGNAL] for each in value.values()]
    return None


def _is_an_envelope(value):
    """The cheap check, before anything is built: one key, and it is the one."""
    return isinstance(value, dict) and len(value) == 1 and SIGNAL in value


def _added(gradients, device):
    """The gradients of every consumer, summed — which is what the chain rule
    says a value that was read twice is owed. `None` if not one of them carried
    anything: an envelope is also how "nothing for you" is said."""
    carried = [each for each in gradients if each is not None]
    return sum(_tensor(each, device) for each in carried) if carried else None


def _tensor(value, device=None, dtype=torch.float32):
    """A value as a tensor, on the device whoever asks was placed on."""
    if torch.is_tensor(value):
        return value
    return torch.tensor(value, dtype=dtype, device=device or None)


def _data(value):
    """A tensor as plain data, which is what crosses an edge here: this slice
    does not touch the wire, so activations and gradients cross as floats."""
    return value.detach().tolist() if torch.is_tensor(value) else value
