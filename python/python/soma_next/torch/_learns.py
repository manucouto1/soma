"""A node that trains itself, which is the other reason a graph gets cut.

What crosses a cable is the value and not the graph that made it, so a node on
another machine gets no gradient from a `backward()` run here. The answer is not
to bring it back: it is for **that side to train itself**. It keeps its
activation alive where its autograd graph is, it is handed `dL/d(what it
produced)`, it carries on with the chain rule from there under an optimizer of
its own, and it gives back `dL/d(what it was given)` so whoever is above can do
the same.

That is split learning, and the same shape covers the rest: ignore the signal
and it is local greedy, run no backward at all and it is forward-forward,
predict the signal and it is synthetic gradients. **One hole, four techniques**,
which is why there is no class per technique — `learn(signal, ctx)` is a duck
beside `parameters()` and `state_dict()`, and it is the same duck `_stage` asks
for to know where a graph gets cut.

## One contract, filled in

The user writes `compute(x, ctx)` and the mixin writes `forward`, which
dispatches. It is a real concession against *a node is one single thing*, and it
is worth saying out loud: **the mixin does not add a second contract, it fills
in the dispatch of the only one there is**. A backward pass is a `forward` of
the transposed graph, and nothing in the core, the plan or the protocol learns
that gradients exist.

What tells the two apart is an **envelope**, on the precedent of
`__soma_opaque__`: a reserved key, and a cheap check before anything else. One
key and not two, because unlike a packed opaque there is no kind to carry — what
sits under the key is the gradient itself. In a learning pass every value on
every edge is an envelope, so a **map** of envelopes is not a fan-in of inputs
but a fan-in of gradients, and it is summed. A user whose own map has that exact
key gets it read as a gradient: known and accepted, the same bill the codec pays.

## What holds on to what

`held` is the activation, kept alive between the two messages, and it is
**dropped after each `learn`**. A gradient arriving for an activation that is
not there is `OutOfStep` with a name on it, and not a `None` propagating into an
optimizer: it is the failure CU12 wrote down against itself.

The optimizer is built **on first use and not in `__init__`**, and that is not a
style: `pickle` does not call `__init__` when it rebuilds an object, and this
node's normal life is to be rebuilt on another machine. What travels is a
**factory** — `partial(Adam, lr=1e-3)`, picklable by both strategies — and the
optimizer is built over there, over the parameters that are over there. Its
state survives between calls because the worker's catalog does.

Everything a rebuilt object needs is a **class attribute**, for the same reason:
whatever `__init__` would have set is not there when `pickle` hands the object
back.

## What this slice does not do

The wire is not touched: an activation and a gradient cross as lists of floats,
built back into a tensor on arrival. And a learner takes **one** input — a
learner with two producers would have to send a different gradient back to each,
which the transpose alone does not route.
"""

from __future__ import annotations

from abc import abstractmethod

import torch

from soma_next import Done, Node

SIGNAL = "__soma_gradient__"
"""The reserved key that says a value is not a value: it is the gradient of one."""


class OutOfStep(Exception):
    """A gradient arrived for an activation that is not there any more."""


def envelope(gradient):
    """A gradient in its envelope, which is how one crosses an edge. What the
    transposed graph is fed with, by a `Trainer` or by hand.

    Not called `signal` so that it stays callable from inside a `learn(self,
    signal, ctx)` somebody wrote.
    """
    return {SIGNAL: _data(gradient)}


def gradient(value, device=None):
    """What an envelope carries, as a tensor — or the sum of what a map of them
    carries. `None` if this is not a backward message at all, which is the same
    question `forward` asks to know which of the two it was handed."""
    inside = _envelopes_in(value)
    return None if inside is None else _added(inside, device)


class Learns(Node):
    """A node that runs its own backward pass and its own optimizer.

    Write `compute(x, ctx)` in plain torch and nothing else::

        class Body(Learns):
            def __init__(self):
                self.lin = torch.nn.Linear(8, 6)

            def compute(self, x, ctx):
                return self.lin(x).relu()

            def parameters(self):
                return list(self.lin.parameters())

    `learn` is the one to override for a technique that is not split learning:
    ignoring the signal is local greedy, not running a backward is
    forward-forward, predicting it is DNI.
    """

    held = None
    """The activation of the last `compute`, alive so its backward can be run."""

    given = None
    """The leaf it was handed, whose `.grad` is what goes back up."""

    learning = None
    """How to build the optimizer, said by `learns_with`."""

    optimizer = None
    """Built the first time it learns, over the parameters of wherever it is."""

    @abstractmethod
    def compute(self, x, ctx):
        """What this node computes, over a leaf that can be differentiated.

        A value and not a transition: a node that learns and also asks the world
        for something is not written yet.
        """

    def forward(self, value, ctx):
        """The single contract, dispatched: a gradient in an envelope is a
        backward message, anything else is an input."""
        signal = gradient(value, ctx.device)
        if signal is None:
            self.given = leaf(value, ctx.device)
            self.held = self.compute(self.given, ctx)
            return Done(_data(self.held))
        return Done(envelope(self.learn(signal, ctx)))

    def learn(self, signal, ctx):
        """Carries on with the chain rule from `dL/d(what I produced)`, steps its
        own optimizer and gives back `dL/d(what I was given)`."""
        if self.held is None:
            raise OutOfStep(
                f"`{type(self).__name__}` was handed a gradient and has no "
                f"activation to apply it to: either it never ran forward this "
                f"step, or it already learnt from this one and let it go"
            )
        optimizer = self._optimizer()
        optimizer.zero_grad()
        self.held.backward(signal)
        optimizer.step()
        given, self.held, self.given = self.given, None, None
        return given.grad

    def learns_with(self, factory):
        """Says what it will build its optimizer with — `partial(Adam, lr=1e-3)`
        — the first time it learns. A recipe and not an optimizer, because the
        parameters it has to point at are the ones on the machine it ends up on."""
        self.learning = factory
        return self

    def _optimizer(self):
        """Its optimizer, built the first time it is asked for."""
        if self.optimizer is not None:
            return self.optimizer
        if self.learning is None:
            raise ValueError(
                f"`{type(self).__name__}` trains itself and nobody said what "
                f"with: say it with `.learns_with(partial(torch.optim.SGD, "
                f"lr=0.1))`, or let a `Trainer` say it for every node that learns"
            )
        parameters = getattr(self, "parameters", None)
        if parameters is None:
            raise ValueError(
                f"`{type(self).__name__}` trains itself and does not say what its "
                f"parameters are: whoever learns answers `parameters()`, the same "
                f"duck the graph asks with"
            )
        self.optimizer = self.learning(parameters())
        return self.optimizer


def _envelopes_in(value):
    """The gradients this value carries, or `None` if it is an ordinary input.

    A map whose every value is an envelope is the fan-in of a node that fed
    several: it is one gradient per consumer, and they add up.
    """
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
    says a value that was read twice is owed."""
    return sum(_tensor(each, device) for each in gradients)


def leaf(value, device=None):
    """A value as something that can be differentiated back to.

    Detached on purpose, and it is the whole premise: a node that learns lets go
    of the chain that produced its input, which is why it cuts the graph exactly
    as a cable does.
    """
    if isinstance(value, dict):
        raise ValueError(
            f"a node that learns takes one input, and this one was handed a map "
            f"of {len(value)}: it would owe a different gradient to each producer, "
            f"and giving one back per edge is not something the transposed graph "
            f"routes yet"
        )
    return _tensor(value, device).detach().requires_grad_(True)


def _tensor(value, device=None, dtype=torch.float32):
    """A value as a tensor, on the device whoever asks was placed on."""
    if torch.is_tensor(value):
        return value
    return torch.tensor(value, dtype=dtype, device=device or None)


def _data(value):
    """A tensor as plain data, which is what crosses an edge here: this slice
    does not touch the wire, so activations and gradients cross as floats."""
    return value.detach().tolist() if torch.is_tensor(value) else value
