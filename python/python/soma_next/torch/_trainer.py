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
from soma_next._stage import learns
from soma_next.torch._freeze import freeze
from soma_next.torch._params import parameters


class NoGradient(Exception):
    """Something the optimizer holds never got a gradient.

    Its own type and not a `ValueError`, because it is worth catching: with a
    cut on purpose — split learning, where the far side runs its own backward —
    this is the thing you expect to see, and you say so by taking those
    parameters out of the optimizer.
    """


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

    `store` is a directory, and it is what makes the case all of this was for
    work: a settled prefix declared `.cached()` runs **once per batch** and is
    read from there on every epoch after the first.

    `workers` says what each host resolves to, exactly as in
    `Graph.forward`. Training a graph with a slice on another machine is
    **not** training that slice: what crosses a wire is the value and not the
    graph that made it, so its parameters get no gradient here. A node that
    trains itself says so with `Learns`, and then it is left out of
    `parameters()` on its own; anything else that ends up without a gradient
    stops the first step with `NoGradient`.

    `optimizer` may be left out **only** when every node with weights trains
    itself: there is nothing here for it to update, and each of them says what it
    trains with through `.learns_with(...)`.
    """

    def __init__(
        self, graph, *, objective, optimizer=None, store=None, workers=None
    ):
        params = parameters(graph)
        learning = _who_learns(graph)
        _check_somebody_moves_them(params, learning, optimizer)
        _check_nobody_is_settled_and_learning(graph, learning)

        self.graph = graph
        self.objective = objective
        self.optimizer = optimizer
        self.store = store
        self.workers = workers
        self._checked = False
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
        output = self.graph.forward(
            _crossable(input_), store=self.store, workers=self.workers
        )
        loss = self.objective(output, _where_the_output_is(target, output))
        loss.backward()
        self._check_the_gradient_arrived()
        self.optimizer.step()
        return loss.item()

    def _check_the_gradient_arrived(self):
        """That nothing about to be updated was left out of the backward pass.

        Once, on the first step: what is structural is the same on every one
        afterwards, and this costs a walk of the parameters.

        It is the counterpart, at this level, of the prefix rule the cache is
        checked against — and it catches more than the cache does, because it
        asks the question after the fact: **whatever** cut the chain, the symptom
        is the same. A node that ran on another host, an output restored from a
        store, a branch that never reached the loss. All of them show up as a
        parameter the optimizer is about to move with nothing telling it where.

        Without this, the run does not fail: it trains **half the network**, the
        loss goes down because the other half is learning, and nothing says so.
        """
        if self._checked:
            return
        self._checked = True
        orphans = [
            parameter
            for group in self.optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad and parameter.grad is None
        ]
        if not orphans:
            return
        whose = _whose(self.graph, orphans)
        raise NoGradient(
            f"the optimizer is about to update {len(orphans)} parameter(s) that "
            f"received no gradient, of {whose}. Nothing joins them to the loss: "
            f"the usual reasons are a node that ran on **another host** — what "
            f"crosses a wire is the value, not the graph that made it — an output "
            f"read back from a store, which is a leaf, or a branch the loss never "
            f"reads. Training would go on and the loss would come down, because "
            f"the rest of the net is learning. If it is deliberate, leave them "
            f"out of the optimizer"
        )

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
        kept = f", keeping in {self.store}" if self.store else ""
        return f"Trainer({len(parameters(self.graph))} parameters{kept})"


def _whose(graph, parameters):
    """Which nodes those parameters belong to, named the way the graph names
    them. One that belongs to no node at all came from somewhere else, and
    saying so is more use than a number."""
    mine = {id(parameter) for parameter in parameters}
    theirs = []
    for node_id in graph.nodes():
        implementation = graph.implementation(node_id)
        collect = getattr(implementation, "parameters", None)
        if collect and any(id(p) in mine for p in collect()):
            theirs.append(f"`{node_id}`")
    return ", ".join(theirs) if theirs else "no node of this graph"


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


def _who_learns(graph):
    """Which nodes train themselves, which are exactly the ones `parameters()`
    leaves out."""
    return [
        node_id
        for node_id in graph.nodes()
        if learns(graph.implementation(node_id))
    ]


def _check_somebody_moves_them(params, learning, optimizer):
    """That every weight in this graph has somebody who will update it.

    Said before the first step, because the symptom otherwise is the one this
    class exists to prevent: a loss that comes down while half the net stands
    still.
    """
    if not params and not learning:
        raise ValueError(
            "this graph has no parameters: no node answers `.parameters()`, "
            "so training it would change nothing and the loss would come out "
            "flat"
        )
    if params and optimizer is None:
        raise ValueError(
            f"this graph has {len(params)} parameter(s) and no optimizer to move "
            f"them: leaving `optimizer` out is only for a graph where **every** "
            f"node with weights trains itself, and these do not"
        )
    if optimizer is None:
        return
    if not params:
        raise ValueError(
            "every node with weights in this graph trains itself, so an optimizer "
            "here has nothing to update: what each of them trains with is said "
            "with `.learns_with(...)`, over the parameters of wherever it runs"
        )
    _check_they_talk(params, optimizer)


def _check_nobody_is_settled_and_learning(graph, learning):
    """That nobody was declared settled **and** writes its own weights.

    `.frozen()` says this node's state does not change while the graph runs, and
    learning changes it every step. Both at once is a contradiction, and it is
    the kind that a cache would turn into the wrong tensor coming back.
    """
    both = [f"`{node_id}`" for node_id in learning if node_id in graph.frozen()]
    if both:
        raise ValueError(
            f"a node cannot be settled and train itself at the same time, and "
            f"these are both: {', '.join(both)}. `.frozen()` says its state does "
            f"not change while the graph runs, and learning is changing it every "
            f"step"
        )


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
