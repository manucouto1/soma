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
"""The reserved key that makes a map a gradient instead of a map."""

CLOSING = "__soma_closes__"
"""And the one beside it whose **presence** says this gradient ends a group of
accumulated ones. It carries nothing. Absent, every step is its own group."""

_OURS = {SIGNAL, CLOSING}
"""What may be in an envelope. Anything else in there and it is a user's map."""


class OutOfStep(Exception):
    """A gradient arrived for an activation that is not there any more.

    Through a `Graph` it comes back as the text of a `ValueError`: a node's
    failure crosses as a message and not as a type.
    """


def envelope(gradient, closing=False):
    """A gradient in its envelope, which is how one crosses an edge. What a
    transposed stage is fed with, by a `Trainer` or by hand.

    `closing` says that this is the gradient that ends a group of accumulated
    ones, so whoever is accumulating applies what it has. It rides on the
    envelope and not on a message of its own because it is the same fact seen
    from the other end: *how many steps make an update* is the training run's,
    and the training run is here.
    """
    sent = {SIGNAL: _data(gradient)}
    if closing:
        # The key being there **is** the fact, so it carries nothing — which is
        # also the only shape that crosses: there is no `bool` on an edge, and a
        # closed set of variants is what stops one being invented as `1.0`.
        sent[CLOSING] = None
    return sent


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

    every = None
    seen = 0
    told = False
    """How many steps make an update, how many have gone by, and whether this one
    was told to be the last. `None` is *whatever the trainer says*, and the same
    class-attribute rule as above applies: an object rebuilt over there reads
    them before anything sets them."""

    def __init__(self, optimizer, *, every=None, **how):
        self.making = partial(optimizer, **how)
        self.node = None
        self.every = every

    def accumulating(self, every):
        """How many steps go into one update, unless this one already said.

        The trainer's number is the default and a technique that named its own
        **wins** — the same rule `trains` follows for who trains whom. Saying it
        twice is then a thing somebody meant, not a thing nobody noticed.
        """
        if self.every is None:
            self.every = every
        return self

    def opens(self):
        """Whether this step starts a group, which is where gradients are cleared
        rather than added to."""
        return self.seen == 0

    def closes(self):
        """Whether this step ends one, which is where the optimizer moves."""
        return self.told or self.seen + 1 >= (self.every or 1)

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
        # Counting is the framework's and what to do about it is the technique's:
        # `learn` asks `opens` and `closes` and never has to remember to tick
        # anything, which is one thing less for whoever fills this hole.
        self.told = _closes(value)
        back = self.learn(_added(inside, ctx.device), ctx)
        self.seen = 0 if self.closes() else self.seen + 1
        self.told = False
        return Done(envelope(back))

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
    seam, step, and hand back the gradient of the input.

    With a group of more than one step it is the same three movements taken
    apart: the gradients are cleared where the group **opens**, added to in
    between, and the optimizer moves where it **closes**. With a group of one —
    which is the default — the three fall back together into the line they were.
    """

    def learn(self, signal, ctx):
        if signal is None:
            # Nothing owed this step. If it is also the one that closes the
            # group, what was added before it still has to be applied — and if
            # nothing was, there is not even an optimizer worth building.
            if self.closes() and not self.opens():
                self.optimizer.step()
            self.done()
            return None
        held = self.waiting()
        if self.opens():
            self.optimizer.zero_grad()
        held.backward(signal)
        if self.closes():
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
    """The cheap check, before anything is built: the key is there, and nothing
    that is not one of ours is."""
    return isinstance(value, dict) and SIGNAL in value and not set(value) - _OURS


def _closes(value):
    """Whether what arrived says the group of accumulated gradients ends here.

    Asked of the whole thing and not of one envelope: a fan-in is one step, so
    either all of them close it or none does, and a single one saying so is the
    step saying so.
    """
    if _is_an_envelope(value):
        return CLOSING in value
    if isinstance(value, dict):
        return any(_closes(each) for each in value.values())
    return False


def _added(gradients, device):
    """The gradients of every consumer, summed — which is what the chain rule
    says a value that was read twice is owed. `None` if not one of them carried
    anything: an envelope is also how "nothing for you" is said."""
    carried = [each for each in gradients if each is not None]
    return sum(_tensor(each, device) for each in carried) if carried else None


def _tensor(value, device=None, dtype=torch.float32):
    """A value as a tensor, on the device whoever asks was placed on.

    It takes a wrapped one as well as a bare one: an envelope built here and read
    here never crossed anything, and one that crossed came back out of its
    wrapper — and neither of those is the caller's business.
    """
    if isinstance(value, Opaque):
        value = value.value
    if torch.is_tensor(value):
        return value
    return torch.tensor(value, dtype=dtype, device=device or None)


def _data(value):
    """A tensor wrapped so that it crosses an edge whole.

    It used to be `tolist()`, and the docstring here used to say that this slice
    did not touch the wire — so a gradient crossed as floats. Now a codec writes
    a tensor down and it crosses as bytes, in this direction as in the other one:
    a backward pass is a forward pass, and it should not be paying a different
    price.
    """
    return Opaque(value.detach()) if torch.is_tensor(value) else value
