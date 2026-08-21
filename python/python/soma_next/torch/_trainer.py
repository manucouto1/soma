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
from soma_next._stage import around, stages, takes_a_gradient
from soma_next.torch._freeze import freeze
from soma_next.torch._learning import envelope, gradient, leaf
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

    `workers` says what each host resolves to, exactly as in `Graph.forward`.
    Training a graph with a slice on another machine is **not** training that
    slice: what crosses a wire is the value and not the graph that made it, so
    its parameters get no gradient here, and the first step stops with
    `NoGradient`.

    `trains` is how that half gets trained anyway, and it is said **here**
    because it is a fact of this training run and not of the graph::

        Trainer(g, objective=cross_entropy,
                optimizer=Adam(parameters(g), lr=1e-3),   # the half that is here
                trains={"body": Split(SGD, lr=0.1)},      # the half that is not
                workers={"gpu": Worker.at("node3:7000")})

    What that puts on the far side is a trainer of its own, beside the node and
    not inside it: the node is not asked to know it is being trained, and the
    same node runs untouched with or without any of this. Their weights are that
    trainer's, so they come **out** of this optimizer —
    `parameters(g, without=trains)` — and holding both is refused rather than
    quietly updating them twice.

    `optimizer` may be left out **only** when everything with weights is trained
    that way: there is nothing here for it to update.

    `every` is how many steps go into one update, for a batch that does not fit
    but whose gradient does::

        Trainer(g, objective=cross_entropy, optimizer=..., every=4)

    Four `step`s, one update, and the loss of each divided by four so that the
    four together pull exactly as one step over the four batches would. Said
    **here** for the same reason `trains` is — how many steps make an update is
    a fact of this training run — and told to whatever trains itself elsewhere,
    so both sides make the same group out of the same steps. A technique that
    named its own (`Split(SGD, lr=0.1, every=8)`) keeps it: two numbers is then
    something somebody meant.

    A group the run ends in the middle of is closed by `update`, which `fit`
    calls at the end of every epoch and whoever writes their own loop calls when
    theirs ends.
    """

    def __init__(
        self,
        graph,
        *,
        objective,
        optimizer=None,
        trains=None,
        every=1,
        store=None,
        workers=None,
    ):
        trains = dict(trains or {})
        _check_the_group(every)
        _check_who_is_trained(graph, trains)
        theirs = {
            id(parameter)
            for node_id in trains
            for parameter in graph.implementation(node_id).parameters()
        }
        mine = [p for p in parameters(graph) if id(p) not in theirs]
        _check_nobody_moves_them_twice(theirs, trains, optimizer)
        _check_somebody_moves_them(mine, trains, optimizer)
        _check_nobody_is_settled_and_trained(graph, trains)

        self.graph = graph
        self.objective = objective
        self.optimizer = optimizer
        self.store = store
        self.workers = workers
        self.trains = trains
        self.theirs = theirs
        self.every = every
        self.seen = 0
        self._checked = False
        # Everything structural, decided once: where this graph is cut does not
        # change from one step to the next, and neither do the transposes its
        # backward pass runs.
        #
        # Driven stage by stage when something is trained where it runs, and
        # **only** then. It is not "when the graph is cut": a lone trained node
        # is one stage and still needs its backward run over the transpose, and a
        # slice on another host with nobody training it needs none of this — for
        # that one the step below is the one it always was, line for line, which
        # is what keeps the blast radius of all this to whoever asked for it.
        self.by_stages = bool(trains)
        if self.by_stages:
            self.running, beside = around(
                graph,
                {
                    node_id: learning.accumulating(every)
                    .of(graph.implementation(node_id))
                    .beside()
                    for node_id, learning in trains.items()
                },
            )
            self.stages = stages(self.running, learns=beside)
            self.backs = [stage.transposed() for stage in self.stages]
            _check_what_is_kept_is_at_the_front(self.stages)
        else:
            self.running, self.stages, self.backs = graph, [], []
        # Whatever the expression declared settled has to **be** settled before
        # the first step, not after somebody notices the loss going flat where
        # it should not. Declaring is the graph's, obeying is torch's.
        freeze(graph)

    def step(self, batch):
        """One step: forward, loss, backward, and update when the group closes.
        Returns the loss, **whole** — divided for the backward pass and not for
        whoever is reading, or a history would change shape with `every`.

        **The primitive**, and `fit` is sugar on top: whatever does not fit in an
        epoch loop is written as a `while` over this.

        With something in the graph trained where it runs it is the same four
        movements taken over the stages — forward in order, the loss, backward in
        reverse — and with nobody it is, line for line, the single pass it always
        was. With `every=1`, which is the default, so is the update.
        """
        input_, target = batch
        if self.by_stages:
            return self._over_the_stages(input_, target)
        if self._opens():
            self.optimizer.zero_grad()
        output = self.graph.forward(
            _crossable(input_), store=self.store, workers=self.workers
        )
        loss = self.objective(output, _where_the_output_is(target, output))
        self._shared(loss).backward()
        self._check_the_gradient_arrived()
        if self._closes():
            self.optimizer.step()
        self._counted()
        return loss.item()

    def _over_the_stages(self, input_, target):
        """One step of a graph that is cut, which is the same step with the
        stages in between: each one is handed what the ones before it produced,
        and the gradients go back the way the values came."""
        if self.optimizer is not None and self._opens():
            self.optimizer.zero_grad()
        produced, seams = {}, {}
        for stage in self.stages:
            stage.fill(
                {
                    producer: self._handed(stage, producer, produced[producer], seams)
                    for producer in stage.holds
                }
            )
            produced.update(
                stage.read(
                    stage.graph.forward(
                        self._the_input(input_, stage) if stage.level == 0 else None,
                        store=self.store if stage.level == 0 else None,
                        workers=self.workers,
                    )
                )
            )
        output = self._as_the_output(produced, seams)
        loss = self.objective(output, _where_the_output_is(target, output))
        self._shared(loss).backward()
        self._hand_the_gradients_back(produced, seams, closing=self._closes())
        # After handing them back and not before: with a cut in the middle, a
        # node on this side gets its gradient from the stage behind it, and
        # asking any earlier would be calling every one of them an orphan.
        self._check_the_gradient_arrived()
        if self.optimizer is not None and self._closes():
            self.optimizer.step()
        self._counted()
        return loss.item()

    def _opens(self):
        """Whether this step starts a group, which is where gradients are cleared
        rather than added to."""
        return self.seen == 0

    def _closes(self):
        """Whether this step ends one, which is where the optimizer moves."""
        return self.seen + 1 >= self.every

    def _counted(self):
        """This step, gone by. The far side counts the same steps from the same
        start, which is what keeps the two groups the same group."""
        self.seen = 0 if self._closes() else self.seen + 1

    def _shared(self, loss):
        """The loss each step of a group is answerable for.

        The usual idiom written down: `N` steps accumulated are meant to be the
        one step of a batch `N` times as long, and an objective that takes the
        mean of its batch has to be divided for that to be true. It assumes the
        steps are the same size, which is the assumption everybody makes and
        nobody says: whoever accumulates uneven ones divides them themselves.

        Untouched with a group of one, so that the graph a `backward` walks is
        the same graph it always was.
        """
        return loss if self.every == 1 else loss / self.every

    def update(self):
        """Applies what has been accumulated so far and starts a new group.

        For the group that a run ends in the middle of: `fit` calls it at the end
        of every epoch, and whoever writes their own loop over `step` calls it
        when their loop ends. Does nothing, and says so, if no step is waiting.

        Across a cut it costs one pass over the transposed stages — not a step,
        no forward, no gradient: what travels is the fact that the group is over,
        by the same road a gradient goes and in an envelope carrying nothing.
        """
        if self.seen == 0:
            return False
        if self.by_stages:
            self._close_the_group()
        if self.optimizer is not None:
            self.optimizer.step()
        self.seen = 0
        return True

    def _close_the_group(self):
        """Tells whoever trains itself elsewhere that the group is over.

        Every hold of a transposed stage feeds a trainer directly — only what
        takes a gradient is transposed — so an envelope carrying nothing reaches
        all of them and nothing else. A stage with nobody training in it has no
        transpose to run.
        """
        for back in reversed(self.backs):
            if not back.nodes:
                continue
            back.fill({node_id: envelope(None, closing=True) for node_id in back.holds})
            back.graph.forward(None, workers=self.workers)

    def _the_input(self, input_, stage):
        """The batch in whatever shape the first stage can read it. The same
        question `_handed` asks, asked of the roots: with the first node of the
        net on another machine, a tensor wrapped to cross an edge here would not
        cross that one."""
        if _they_take_data(stage, stage.graph.roots()):
            return _data(input_)
        return _crossable(input_)

    def _handed(self, stage, producer, value, seams):
        """What to hand a stage for something an earlier one produced.

        Three shapes, one rule: a value crosses in whatever way the end that
        reads it can read it.

        - **data**, when what reads it runs elsewhere or trains itself — a live
          object does not cross a wire, and a node that learns lets go of the
          chain anyway;
        - **the tensor as it is**, when it still carries the chain that made it:
          no cut was crossed here, and passing it on keeps one backward pass
          doing the whole job;
        - **a leaf**, when it does not. That leaf is the seam, and its gradient
          is the first thing handed back to whoever produced it.
        """
        if _they_take_data(stage, stage.graph.successors(producer)):
            return _data(value)
        if torch.is_tensor(value) and value.grad_fn is not None:
            return Opaque(value)
        seams[producer] = seam = leaf(value)
        return Opaque(seam)

    def _as_the_output(self, produced, seams):
        """What the whole graph produced, shaped as `forward` would have given it
        and differentiable, which after a cut it is not: what came across one
        enters the loss as a leaf too."""
        out = {}
        for node_id in self.graph.leaves():
            value = produced[node_id]
            if torch.is_tensor(value) and value.grad_fn is not None:
                out[node_id] = value
            else:
                seams[node_id] = out[node_id] = leaf(value)
        return out[self.graph.leaves()[0]] if len(out) == 1 else out

    def _hand_the_gradients_back(self, produced, seams, closing=True):
        """The stages in reverse, each handed what it is owed.

        Two ways of owing it, and which applies is the **node's** and not the
        stage's: whoever trains itself is handed the gradient through the
        transposed stage, so that it arrives where the node runs; whoever does
        not gets it applied here, over the tensor it produced, and autograd
        carries on from there into whatever is above.

        Read node by node as we get there and not all at once at the start: a
        `backward` two stages down adds to a seam further up, and taking the
        gradients before that would be reading them a step early.

        `retain_graph` because two nodes of one stage can share what is above
        them, and the second `backward` would find it freed.
        """
        owed = {}
        for stage in reversed(self.stages):
            here = {}
            for node_id in stage.taps:
                seam = seams.get(node_id)
                got = _both(
                    owed.pop(node_id, None), None if seam is None else seam.grad
                )
                if got is not None:
                    here[node_id] = got
            learning = {
                node_id: got
                for node_id, got in here.items()
                if takes_a_gradient(self.running.implementation(node_id))
            }
            for node_id, got in here.items():
                if node_id not in learning:
                    landed = produced[node_id]
                    landed.backward(got.to(landed.device), retain_graph=True)
            if not learning:
                continue
            # Every hold, and an empty envelope for whoever is owed nothing: a
            # stage runs whole, and one learner of it having no gradient this
            # step is not the same as nobody having handed it anything.
            back = self.backs[stage.level]
            back.fill(
                {
                    node_id: envelope(learning.get(node_id), closing=closing)
                    for node_id in back.holds
                }
            )
            given = back.read(back.graph.forward(None, workers=self.workers))
            for producer, value in given.items():
                owed[producer] = _both(owed.get(producer), gradient(value))

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
        if self._checked or self.optimizer is None:
            return
        self._checked = True
        orphans = [
            parameter
            for group in self.optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
            and parameter.grad is None
            and id(parameter) not in self.theirs
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
            # A group the epoch ended in the middle of is still a group: with
            # `every=1` there is never one, and this costs nothing.
            self.update()
        return Result(history)

    def __repr__(self):
        kept = f", keeping in {self.store}" if self.store else ""
        return f"Trainer({len(parameters(self.graph))} parameters{kept})"


def _check_the_group(every):
    """That a group is a whole number of steps, and at least one of them."""
    if isinstance(every, bool) or not isinstance(every, int) or every < 1:
        raise ValueError(
            f"`every` is how many steps go into one update, so it is a whole "
            f"number and at least 1; `{every!r}` is not"
        )


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


def _they_take_data(stage, who):
    """Whether what these nodes read has to be plain data: one that runs
    elsewhere cannot be handed a live object.

    Nothing is asked about training here, and that is the simplification the
    trainer standing beside the node buys: whoever takes a tensor and has to let
    go of the chain does that itself, on arrival.
    """
    hosts = stage.graph.hosts()
    return any(hosts.get(node_id) for node_id in who)


def _both(one, other):
    """Two gradients for the same value add up, and either of them may not be
    there at all."""
    if one is None:
        return other
    if other is None:
        return one
    return one + other.to(one.device)


def _data(value):
    """A tensor with the chain behind it let go of, which is what crosses to a
    node that runs elsewhere.

    It used to be a list of floats, because that was the only thing that crossed.
    Now a codec writes a tensor down and it crosses as bytes: 44× faster and half
    the bytes on a batch that is not even large, and — worth more than that — the
    same node is handed the same shape wherever it runs, which is the whole
    argument of `.at()`.

    What stays is the `detach`: the graph does not cross a wire and never did, so
    letting go of it here is saying out loud what the wire does anyway.
    """
    return Opaque(value.detach()) if torch.is_tensor(value) else value


def _check_who_is_trained(graph, trains):
    """That every node `trains` names is in the graph and says what its
    parameters are, since that is what its trainer will build an optimizer
    over."""
    for node_id in trains:
        if node_id not in graph:
            raise ValueError(
                f"`trains` names `{node_id}`, which is not a node of this graph"
            )
        if getattr(graph.implementation(node_id), "parameters", None) is None:
            raise ValueError(
                f"`{node_id}` is to be trained where it runs and does not say "
                f"what its parameters are: whoever is trained answers "
                f"`parameters()`, the same duck the graph asks with"
            )


def _check_nobody_moves_them_twice(theirs, trains, optimizer):
    """That this optimizer does not hold weights somebody else is training.

    Where they run may well be here, and then both would move them every step —
    two updates for one gradient, and a loss that is merely worse instead of
    wrong. Refused, and not left to whoever notices.
    """
    if optimizer is None:
        return
    also = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
        if id(parameter) in theirs
    ]
    if also:
        raise ValueError(
            f"this optimizer holds {len(also)} parameter(s) of "
            f"{', '.join(f'`{node_id}`' for node_id in trains)}, which "
            f"`trains={{…}}` says are trained where they run: with the node here "
            f"that is two updates a step. Leave them out with "
            f"`parameters(g, without=trains)`"
        )


def _check_somebody_moves_them(mine, trains, optimizer):
    """That every weight in this graph has somebody who will update it.

    Said before the first step, because the symptom otherwise is the one this
    class exists to prevent: a loss that comes down while half the net stands
    still.
    """
    if not mine and not trains:
        raise ValueError(
            "this graph has no parameters: no node answers `.parameters()`, "
            "so training it would change nothing and the loss would come out "
            "flat"
        )
    if mine and optimizer is None:
        raise ValueError(
            f"this graph has {len(mine)} parameter(s) that nobody would move: "
            f"leaving `optimizer` out is only for a graph where **everything** "
            f"with weights is trained where it runs, and these are not"
        )
    if optimizer is None:
        return
    if not mine:
        raise ValueError(
            "everything with weights in this graph is trained where it runs, so "
            "an optimizer here has nothing to update: what each of them is "
            "trained with is what `trains={...}` says"
        )
    _check_they_talk(mine, optimizer)


def _check_nobody_is_settled_and_trained(graph, trains):
    """That nobody was declared settled **and** handed to a trainer.

    `.frozen()` says this node's state does not change while the graph runs, and
    training changes it every step. Both at once is a contradiction, and it is
    the kind that a cache would turn into the wrong tensor coming back.
    """
    both = [f"`{node_id}`" for node_id in trains if node_id in graph.frozen()]
    if both:
        raise ValueError(
            f"a node cannot be settled and trained at the same time, and these "
            f"are both: {', '.join(both)}. `.frozen()` says its state does not "
            f"change while the graph runs, and training is changing it every step"
        )


def _check_what_is_kept_is_at_the_front(stages_of_it):
    """That nothing beyond the first stage says `.cached()`.

    A root's key comes from the input it was handed, and after a cut the roots of
    a stage are holds, handed nothing: two different batches would name the same
    thing. What is kept is named by what came before it, and after a cut this
    side no longer knows what came before.
    """
    kept = [
        f"`{node_id}`"
        for stage in stages_of_it[1:]
        for node_id in stage.graph.cached()
    ]
    if kept:
        raise ValueError(
            f"{', '.join(kept)} is declared `.cached()` and is not in the first "
            f"stage of this graph: what is kept is named by what came before it, "
            f"and after a cut this side no longer knows what came before, so two "
            f"different batches would be kept under one name"
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
