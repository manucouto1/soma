"""Training **one** graph. Neither does the graph know this exists, nor does
this know there are other training runs.

Three levels, and none knows the one above:

| level | what it is | scale |
|---|---|---|
| the graph | a network | one `forward` |
| `Trainer` | one training run | an afternoon |
| a study | N training runs | an experiment |

The third **has no type**, and that is on purpose: N independent training runs
are a Python list, and modelling a list as a graph is paying a DAG's price
without using it. A graph earns its keep when there are dependencies to declare.

And training is not a node for the same reason: a node's contract —
`forward(input, ctx)` — describes **one step**, with its turn budget and no
partial recovery. A training run lasts an afternoon, mutates its state and fails
in ways one wants to recover from. The original Soma put `fit` in the node
contract, and the bill shows in its own tests: four crates implement an empty
`fit` just to be able to exist.
"""

from __future__ import annotations

import torch

from soma_next import Opaque
from soma_next.torch._freeze import freeze
from soma_next.torch._params import parameters


class Result:
    """What a training run leaves behind: the loss, step by step."""

    def __init__(self, history):
        self.history = history

    @property
    def loss(self):
        """The last loss, or `None` if not a single step was taken."""
        return self.history[-1] if self.history else None

    def __repr__(self):
        if not self.history:
            return "Result(no steps)"
        return (
            f"Result({len(self.history)} steps, "
            f"{self.history[0]:.4f} → {self.history[-1]:.4f})"
        )


class Trainer:
    """Trains a graph, without the graph finding out — no `g.fit(...)`, so the
    same graph can be trained three ways without touching it.

    The optimizer is built by the caller, which is what keeps a name registry
    (`optimizer="adam"`) out::

        t = Trainer(g, objective=cross_entropy,
                    optimizer=torch.optim.Adam(parameters(g), lr=1e-3))
    """

    def __init__(self, graph, *, objective, optimizer):
        params = parameters(graph)
        if not params:
            raise ValueError(
                "this graph has no parameters: no node answers `.parameters()`, "
                "so training it would change nothing and the loss would come out "
                "flat"
            )
        _check_they_talk(params, optimizer)

        self.graph = graph
        self.objective = objective
        self.optimizer = optimizer
        # Whatever the expression declared settled has to **be** settled before
        # the first step, not after somebody notices the loss going flat where
        # it should not. Declaring is the graph's, obeying is torch's.
        freeze(graph)

    def step(self, batch):
        """One step: forward, loss, backward, update. Returns the loss.

        **The primitive**, and `fit` is sugar on top: whatever does not fit in an
        epoch loop is written as a `while` over this.
        """
        input_, target = batch
        self.optimizer.zero_grad()
        output = self.graph.forward(_crossable(input_))
        loss = self.objective(output, _where_the_output_is(target, output))
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def fit(self, data, epochs=1):
        """Takes one step per batch, for as many epochs as you say.

        `data` is walked once per epoch, so with more than one it has to be
        re-iterable: a generator is exhausted on the first.
        """
        history = []
        for _ in range(epochs):
            for batch in data:
                history.append(self.step(batch))
        return Result(history)

    def __repr__(self):
        return f"Trainer({len(parameters(self.graph))} parameters)"


def _crossable(input_):
    """A tensor is wrapped to cross an edge; everything else passes as it is.

    The one place `Opaque` is not asked for by hand, because here a tensor is
    the case and not a surprise.
    """
    return Opaque(input_) if isinstance(input_, torch.Tensor) else input_


def _where_the_output_is(target, output):
    """The target goes to meet the output wherever it ended up.

    The input crosses the graph and each node moves it to its device; the target
    goes straight to the loss, so nobody ever moves it. The loss is the only one
    that sees both. Moving the target and not the output: bringing the output
    back to the cpu would drag the backward pass with it at every step.
    """
    if torch.is_tensor(target) and torch.is_tensor(output):
        return target.to(output.device)
    return target


def _check_they_talk(params, optimizer):
    """That the optimizer and the graph are talking about the same weights.

    Only sharing **none at all** is rejected; covering a part is legitimate —
    freezing the encoder and training the head is exactly that.
    """
    from_graph = {id(p) for p in params}
    from_optimizer = {
        id(p) for group in optimizer.param_groups for p in group["params"]
    }
    if not (from_graph & from_optimizer):
        raise ValueError(
            f"the optimizer updates no parameter of this graph: it has "
            f"{len(from_optimizer)} and the graph {len(from_graph)}, and not one "
            f"matches. Was it built over another graph? "
            f"You do it with `Adam(parameters(g), ...)`"
        )
