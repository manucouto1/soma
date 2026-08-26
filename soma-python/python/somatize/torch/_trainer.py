"""Training **one** graph. Neither does the graph know this exists, nor does this
know there are other training runs.

| level | what it is | scale |
|---|---|---|
| the graph | a network | one `forward` |
| `Trainer` | one training run | an afternoon |
| a study | N training runs | an experiment |

The third **has no type** on purpose: N independent training runs are a Python
list, and modelling a list as a graph pays a DAG's price without using it.
Training is not a node for the same reason — a node's contract describes one
step, and the original that put `fit` in it has four crates implementing an
empty one to be able to exist.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Sequence

from somatize._typing import Fact

if TYPE_CHECKING:
    import torch as _torch

    from somatize._graph import Graph
    from somatize._remote import Broker
    from somatize._somatize import Store
    from somatize._stage import Stage
    from somatize.torch._learning import Learning

#: One example: what goes in, and what should come out.
Batch = tuple[Any, Any]

#: What a training run exported: the state of each node, by node id.
Weights = dict[str, dict[str, "_torch.Tensor"]]

#: What turns an output and a target into a number to minimise.
Objective = Callable[[Any, Any], "_torch.Tensor"]

import torch

from somatize import Opaque
from somatize._stage import around, stages, takes_a_gradient
from somatize.torch._freeze import freeze
from somatize.torch._learning import envelope, gradient, leaf
from somatize.torch._params import parameters


class NoGradient(Exception):
    """Something the optimizer holds never got a gradient. Its own type because
    it is worth catching: with a cut on purpose — split learning — this is what
    you expect, and you say so by taking those parameters out of the optimizer.
    """


class Result:
    """What a training run leaves behind: the loss, step by step."""

    def __init__(self, history: list[float]) -> None:
        self.history = history

    @property
    def loss(self) -> float | None:
        """The last loss, or `None` if not a single step was taken."""
        return self.history[-1] if self.history else None

    def __repr__(self) -> str:
        if not self.history:
            return "Result(no steps)"
        return (
            f"Result({len(self.history)} steps, "
            f"{self.history[0]:.4f} → {self.history[-1]:.4f})"
        )


class Trainer:
    """Trains a graph, without the graph finding out — no `g.fit(...)`, so the
    same graph can be trained three ways without touching it::

        t = Trainer(g, objective=cross_entropy,
                    optimizer=torch.optim.Adam(parameters(g), lr=1e-3))

    The optimizer is the caller's, which keeps a name registry
    (`optimizer="adam"`) out. `store` is a directory, and makes a settled
    `.cached()` prefix run once per batch. `broker` says who knows where each
    host is, as in `Graph.forward`.

    Training a graph with a slice on another machine is **not** training that
    slice: what crosses a wire is the value and not the graph that made it.
    `trains` is how that half gets trained anyway, said **here** because it is a
    fact of this training run and not of the graph::

        Trainer(g, objective=cross_entropy,
                optimizer=Adam(parameters(g), lr=1e-3),   # the half that is here
                trains={"body": Split(SGD, lr=0.1)},      # the half that is not
                broker=Broker.embedded({"gpu": Worker.at("node3:7000")}))

    That puts a trainer **beside** the node rather than inside it, so the node is
    never asked to know it is being trained. Those weights are that trainer's, so
    they come out of this optimizer — `parameters(g, without=trains)` — and
    holding both is refused rather than quietly updating them twice.

    `every` is how many steps go into one update and `micro` how many pieces one
    step is cut into; they **multiply** rather than compete. `watching` is told
    what happens and is handed on to every `forward`, so one stream carries both
    the engine's vocabulary and this level's `loss` and `updated`.
    """

    def __init__(
        self,
        graph: "Graph",
        *,
        objective: Objective,
        optimizer: Any = None,
        trains: dict[str, "Learning"] | None = None,
        every: int = 1,
        micro: int = 1,
        store: "Store | str | None" = None,
        broker: "Broker | None" = None,
        watching: Any = None,
        auditing: Any = None,
    ) -> None:
        trains = dict(trains or {})
        _check_the_group(every, micro)
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
        self.broker = broker
        # Kept twice on purpose: `watching` goes on to `forward`, which resolves
        # a list of watchers itself, and `_telling` is what **this** side calls.
        # They have to accept the same shapes or `watching=` means two things.
        self.watching = watching
        self._telling = _telling(watching)
        # Measuring is opt-in and it is not free: hooks on every node, a handful
        # of reductions a step, and an SVD on a cadence.
        self.audit = _auditing(auditing)
        if self.audit is not None:
            self.audit.watch(graph)
        self.trains = trains
        self.theirs = theirs
        self.every = every
        self.micro = micro
        # What a group is made of, counted in **pieces** and not in steps: with
        # `micro` there are several of them to a step, and the two multiply
        # rather than compete. `every` stays a number of steps because that is
        # what somebody writing a loop counts.
        self.pieces = every * micro
        self.seen = 0
        self._checked = False
        # Everything structural, decided once: where a graph is cut does not
        # change between steps. Driven stage by stage when something is trained
        # where it runs and **only** then — not "when the graph is cut": a lone
        # trained node is one stage and still needs its transpose, and a slice
        # with nobody training it needs none of this.
        self.by_stages = bool(trains)
        # Empty for a graph that is not cut, which is what the `else` below used
        # to say — said here instead, so there is one place they are declared.
        self.stages: list["Stage"] = []
        self.backs: list["Stage"] = []
        if self.by_stages:
            self.running, beside = around(
                graph,
                {
                    node_id: learning.accumulating(self.pieces)
                    .of(graph.implementation(node_id))
                    .beside()
                    for node_id, learning in trains.items()
                },
            )
            self.stages = stages(self.running, learns=beside)
            self.backs = [stage.transposed() for stage in self.stages]
            _check_what_is_kept_is_at_the_front(self.stages)
        else:
            self.running = graph
        # Whatever the expression declared settled has to **be** settled before
        # the first step, not after somebody notices the loss going flat where
        # it should not. Declaring is the graph's, obeying is torch's.
        freeze(graph)

    def step(self, batch: Batch) -> float:
        """One step: forward, loss, backward, and update when the group closes.
        Returns the loss **whole** — divided for the backward pass and not for
        whoever is reading, or a history would change shape with `every`.

        **The primitive**, and `fit` is sugar on top: whatever does not fit in an
        epoch loop is written as a `while` over this.
        """
        if self.micro == 1:
            return self._once(batch)
        # The pieces are steps in every way that matters — each one a forward, a
        # loss and a backward, each one counted — so the group closes on the last
        # of them and the optimizer moves once, which is the whole point.
        pieces = _in_pieces(batch, self.micro)
        return sum(self._once(piece) for piece in pieces) / len(pieces)

    def _once(self, batch: Batch) -> float:
        """One pass: a whole batch, or one piece of one."""
        input_, target = batch
        if self.by_stages:
            return self._over_the_stages(input_, target)
        if self._opens():
            self.optimizer.zero_grad()
        output = self.graph.forward(
            _crossable(input_),
            store=self.store,
            broker=self.broker,
            watching=self.watching,
        )
        loss = self.objective(output, _where_the_output_is(target, output))
        self._shared(loss).backward()
        self._check_the_gradient_arrived()
        if self._closes():
            self.optimizer.step()
            self._said("updated")
        self._audited()
        self._counted()
        return self._said_the_loss(loss)

    def _over_the_stages(self, input_: Any, target: Any) -> float:
        """One step of a graph that is cut, which is the same step with the
        stages in between: each one is handed what the ones before it produced,
        and the gradients go back the way the values came."""
        if self.optimizer is not None and self._opens():
            self.optimizer.zero_grad()
        produced: dict[str, Any] = {}
        seams: dict[str, Any] = {}
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
                        broker=self.broker,
                        watching=self.watching,
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
            self._said("updated")
        self._audited()
        self._counted()
        return self._said_the_loss(loss)

    def _said(self, kind: str, **fields: Any) -> None:
        """One fact of **this** level, out through the same door as the engine's.
        A loss is this object's arithmetic and the engine cannot see it, so level
        2 keeps its own vocabulary and the two meet in the record — arriving as
        the same `dict`, so nothing downstream has to know which level said it.
        """
        if self._telling is not None:
            self._telling({"fact": kind, **{k: str(v) for k, v in fields.items()}})

    def _audited(self) -> None:
        """What the audit saw this step, out through the same door as the loss.
        **After the optimizer moved**, which is the only moment this step's
        update exists: asking before would measure the previous one.
        """
        if self.audit is None:
            return
        for one in self.audit.observed(self.graph):
            self._said(one.pop("fact"), **one)

    def _said_the_loss(self, loss: Any) -> float:
        """The loss, said and then returned. Whole, as `step` promises: divided
        for the backward pass and not for whoever is reading."""
        whole: float = loss.item()
        self._said("loss", value=whole)
        return whole

    def _opens(self) -> bool:
        """Whether this step starts a group, which is where gradients are cleared
        rather than added to."""
        return self.seen == 0

    def _closes(self) -> bool:
        """Whether this step ends one, which is where the optimizer moves."""
        return self.seen + 1 >= self.pieces

    def _counted(self) -> None:
        """This step, gone by. The far side counts the same steps from the same
        start, which is what keeps the two groups the same group."""
        self.seen = 0 if self._closes() else self.seen + 1

    def _shared(self, loss: Any) -> Any:
        """The loss each step of a group is answerable for.

        `N` steps accumulated are meant to be one step of a batch `N` times as
        long, and an objective that takes the mean has to be divided for that to
        hold. Untouched with a group of one, so the graph a `backward` walks is
        the one it always was.
        """
        return loss if self.pieces == 1 else loss / self.pieces

    def update(self) -> bool:
        """Applies what has been accumulated and starts a new group, for the group
        a run ends in the middle of. Does nothing, and says so, if none is open.

        Across a cut it costs one pass over the transposed stages and not a step:
        what travels is the fact that the group is over, in an empty envelope.
        """
        if self.seen == 0:
            return False
        if self.by_stages:
            self._close_the_group()
        if self.optimizer is not None:
            self.optimizer.step()
        self.seen = 0
        return True

    def _close_the_group(self) -> None:
        """Tells whoever trains itself elsewhere that the group is over. Every
        hold of a transposed stage feeds a trainer directly, so an envelope
        carrying nothing reaches all of them and nothing else.
        """
        for back in reversed(self.backs):
            if not back.nodes:
                continue
            back.fill({node_id: envelope(None, closing=True) for node_id in back.holds})
            back.graph.forward(None, broker=self.broker)

    def _the_input(self, input_: Any, stage: "Stage") -> Any:
        """The batch in whatever shape the first stage can read it. The same
        question `_handed` asks, asked of the roots: with the first node of the
        net on another machine, a tensor wrapped to cross an edge here would not
        cross that one."""
        if _they_take_data(stage, stage.graph.roots()):
            return _data(input_)
        return _crossable(input_)

    def _handed(
        self,
        stage: "Stage",
        producer: str,
        value: Any,
        seams: dict[str, Any],
    ) -> Any:
        """What to hand a stage for something an earlier one produced.

        Three shapes, one rule: a value crosses in whatever way the end that
        reads it can read it — **data** when that end runs elsewhere or trains
        itself, **the tensor as it is** when it still carries the chain that made
        it, and **a leaf** when it does not. That leaf is the seam.
        """
        if _they_take_data(stage, stage.graph.successors(producer)):
            return _data(value)
        if torch.is_tensor(value) and value.grad_fn is not None:
            return Opaque(value)
        seams[producer] = seam = leaf(value)
        return Opaque(seam)

    def _as_the_output(self, produced: dict[str, Any], seams: dict[str, Any]) -> Any:
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

    def _hand_the_gradients_back(
        self,
        produced: dict[str, Any],
        seams: dict[str, Any],
        closing: bool = True,
    ) -> None:
        """The stages in reverse, each handed what it is owed.

        Which of the two ways applies is the **node's** and not the stage's:
        whoever trains itself gets it through the transposed stage, whoever does
        not gets it applied here and autograd carries on above. `retain_graph`
        because two nodes of one stage can share what is above them.
        """
        owed: dict[str, Any] = {}
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
            given = back.read(back.graph.forward(None, broker=self.broker))
            for producer, value in given.items():
                owed[producer] = _both(owed.get(producer), gradient(value))

    def _check_the_gradient_arrived(self) -> None:
        """That nothing about to be updated was left out of the backward pass.
        Once, on the first step, since what is structural does not change.

        Without it the run does not fail — it trains half the network, the loss
        comes down because the other half is learning, and nothing says so.
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

    def export(self) -> Weights:
        """What this training run learnt: its weights, node by node, as
        `{node_id: {key: tensor}}`, by the same two ducks everything here asks by.

        A **snapshot** and not a view: detached and copied, so the next step does
        not move it under whoever is holding it. The optimizer's state is not in
        it — momentum is this client's. Refused for a node trained **and**
        running elsewhere: the copy here never learnt anything.
        """
        self._check_they_are_here("exported")
        return {
            node_id: {key: value.detach().clone() for key, value in state}
            for node_id, state in _the_weights(self.graph)
        }

    def load(self, weights: Weights) -> None:
        """The mirror of `export`. Every node it names has to be here with the
        weights and shapes it says, and nothing is copied in until all of that is
        true, so a refusal leaves the net as it was rather than half loaded.
        """
        self._check_they_are_here("loaded")
        mine = dict(_the_weights(self.graph))
        putting = []
        for node_id, state in weights.items():
            if node_id not in mine:
                raise ValueError(
                    f"there is nothing called `{node_id}` with weights in this "
                    f"graph, and `{'`, `'.join(sorted(mine))}` is what there is"
                )
            here = dict(mine[node_id])
            for key, value in state.items():
                if key not in here:
                    raise ValueError(f"`{node_id}` has no `{key}` to load into")
                if here[key].shape != value.shape:
                    raise ValueError(
                        f"`{node_id}`'s `{key}` is {tuple(here[key].shape)} here "
                        f"and what arrived is {tuple(value.shape)}"
                    )
                putting.append((here[key], value))
        with torch.no_grad():
            for mine_, theirs in putting:
                mine_.copy_(theirs.to(mine_.device, mine_.dtype))

    def _check_they_are_here(self, what: str) -> None:
        """That nothing this is about to speak for is being trained on another
        machine."""
        hosts = self.graph.hosts()
        elsewhere = sorted(node_id for node_id in self.trains if hosts.get(node_id))
        if elsewhere:
            raise ValueError(
                f"`{'`, `'.join(elsewhere)}` is trained where it runs, which is "
                f"not here, so its weights cannot be {what} from this side: what "
                f"is here is the copy that was sent, and it never learnt anything"
            )

    def fit(self, data: Iterable[Batch], epochs: int = 1) -> Result:
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

    def __repr__(self) -> str:
        kept = f", keeping in {self.store}" if self.store else ""
        return f"Trainer({len(parameters(self.graph))} parameters{kept})"


def _the_weights(
    graph: "Graph",
) -> Iterator[tuple[str, list[tuple[str, "_torch.Tensor"]]]]:
    """Every node that has any, and what its weights are called — the same two
    ducks as `state_digest` and `Graph._check_it_was_obeyed`, or the project
    would ask two different questions about one state.

    **The keys are text**: an export gets written into stores and sent down
    wires, and the one thing a map that crosses anything here may key on is text.
    """
    for node_id in graph.nodes():
        implementation = graph.implementation(node_id)
        named = getattr(implementation, "state_dict", None)
        if named is not None:
            found: list[tuple[Any, Any]] = sorted(named().items())
        else:
            in_order = getattr(implementation, "parameters", None)
            found = list(enumerate(in_order())) if in_order is not None else []
        state = [(str(key), value) for key, value in found if torch.is_tensor(value)]
        if state:
            yield node_id, state


def _check_the_group(every: int, micro: int) -> None:
    """That a group is a whole number of pieces, and at least one of them."""
    for what, how_many in (("every", every), ("micro", micro)):
        if isinstance(how_many, bool) or not isinstance(how_many, int) or how_many < 1:
            raise ValueError(
                f"`{what}` is a count of {'steps' if what == 'every' else 'pieces'}"
                f", so it is a whole number and at least 1; `{how_many!r}` is not"
            )


def _telling(watching: Any) -> Callable[[Fact], None] | None:
    """Whatever `watching=` was given, as one callable — or `None`. This side
    calls it itself, so it has to understand the same shapes `Graph.forward`
    hands the engine: `watching=[recorder, live]` meaning two things depending on
    which door it went through is the trap this project exists not to build.
    """
    if watching is None or callable(watching):
        return watching
    if isinstance(watching, (list, tuple)):
        several = [one for one in (_telling(each) for each in watching) if one is not None]

        def all_of_them(fact: Fact) -> None:
            for one in several:
                one(fact)

        return all_of_them
    raise ValueError(
        "`watching` takes a Recorder, anything callable, or a list of them; "
        f"what arrived is a {type(watching).__name__}"
    )


def _auditing(auditing: Any) -> Any:
    """Whatever `auditing=` was given, as an `Audit` or `None`."""
    from somatize.torch._audit import Audit

    if auditing is None or auditing is False:
        return None
    if auditing is True:
        return Audit()
    if isinstance(auditing, Audit):
        return auditing
    raise ValueError(
        "`auditing` takes True, or an Audit if you want to choose a cadence; "
        f"what arrived is a {type(auditing).__name__}"
    )


def _in_pieces(batch: Batch, micro: int) -> list[Batch]:
    """One batch cut into `micro`, both halves the same way.

    **Who knows how to cut a batch is this module and nobody else**: at this
    level the batch is the caller's, so `torch.chunk` reaches it without the core
    learning what an item is.

    **It has to divide, and that is checked.** `chunk` gives *at most* what it is
    asked for, and a group counting four while three run never closes.
    """
    input_, target = batch
    if torch.is_tensor(input_) and torch.is_tensor(target) and len(input_) != len(target):
        raise ValueError(
            f"the input is {len(input_)} long and the target {len(target)}: they "
            f"have to line up along the batch dimension for a piece of one to go "
            f"with a piece of the other"
        )
    return list(zip(*(_cut(half, micro, which) for which, half in enumerate(batch))))


def _cut(half: Any, micro: int, which: int) -> Any:
    """One half of a batch — the input or the target — in `micro` equal pieces."""
    where = "input" if which == 0 else "target"
    if not torch.is_tensor(half):
        raise TypeError(
            f"`micro` cuts a batch into pieces and the {where} is a "
            f"`{type(half).__name__}`, which this does not know how to cut: a "
            f"tensor is what it cuts. Cut it yourself and take one step per "
            f"piece, which is what `every` is for"
        )
    if len(half) % micro:
        raise ValueError(
            f"`micro={micro}` cuts a batch into {micro} equal pieces and the "
            f"{where} is {len(half)} long, which does not divide. Drop the short "
            f"batch — `drop_last=True` on a `DataLoader` — or pick a `micro` "
            f"that divides"
        )
    return list(torch.chunk(half, micro))


def _whose(graph: "Graph", parameters: Iterable[Any]) -> str:
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


def _crossable(input_: Any) -> Any:
    """A tensor is wrapped to cross an edge; everything else passes as it is.

    The one place `Opaque` is not asked for by hand, because here a tensor is
    the case and not a surprise.
    """
    return Opaque(input_) if isinstance(input_, torch.Tensor) else input_


def _where_the_output_is(target: Any, output: Any) -> Any:
    """The target goes to meet the output wherever it ended up. The input crosses
    the graph and each node moves it; the target goes straight to the loss, which
    is the only one that sees both. The target and not the output, or the
    backward pass would be dragged back to the cpu at every step.
    """
    if torch.is_tensor(target) and torch.is_tensor(output):
        return target.to(output.device)
    return target


def _they_take_data(stage: "Stage", who: Iterable[str]) -> bool:
    """Whether what these nodes read has to be plain data: one that runs elsewhere
    cannot be handed a live object. Nothing is asked about training, which is the
    simplification a trainer standing beside the node buys.
    """
    hosts = stage.graph.hosts()
    return any(hosts.get(node_id) for node_id in who)


def _both(one: Any, other: Any) -> Any:
    """Two gradients for the same value add up, and either of them may not be
    there at all."""
    if one is None:
        return other
    if other is None:
        return one
    return one + other.to(one.device)


def _data(value: Any) -> Any:
    """A tensor with the chain behind it let go of, which is what crosses to a
    node that runs elsewhere.

    It used to be a list of floats; now a codec writes it down and it crosses as
    bytes — 44× faster, half the bytes, and the same node is handed the same
    shape wherever it runs. The `detach` stays: the graph never crossed a wire.
    """
    return Opaque(value.detach()) if torch.is_tensor(value) else value


def _check_who_is_trained(graph: "Graph", trains: dict[str, "Learning"]) -> None:
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


def _check_nobody_moves_them_twice(
    theirs: set[int],
    trains: dict[str, "Learning"],
    optimizer: Any,
) -> None:
    """That this optimizer does not hold weights somebody else is training. Where
    they run may be here, and then both would move them every step: two updates
    for one gradient, and a loss merely worse instead of wrong.
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


def _check_somebody_moves_them(
    mine: Sequence[Any],
    trains: dict[str, "Learning"],
    optimizer: Any,
) -> None:
    """That every weight in this graph has somebody who will update it. Said
    before the first step, because the symptom otherwise is the one this class
    exists to prevent: a loss coming down while half the net stands still.
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


def _check_nobody_is_settled_and_trained(
    graph: "Graph",
    trains: dict[str, "Learning"],
) -> None:
    """That nobody was declared settled **and** handed to a trainer. `.frozen()`
    says the state does not change while the graph runs and training changes it
    every step — a contradiction a cache would turn into the wrong tensor.
    """
    both = [f"`{node_id}`" for node_id in trains if node_id in graph.frozen()]
    if both:
        raise ValueError(
            f"a node cannot be settled and trained at the same time, and these "
            f"are both: {', '.join(both)}. `.frozen()` says its state does not "
            f"change while the graph runs, and training is changing it every step"
        )


def _check_what_is_kept_is_at_the_front(stages_of_it: Sequence["Stage"]) -> None:
    """That nothing beyond the first stage says `.cached()`. A root's key comes
    from the input it was handed, and after a cut the roots of a stage are holds
    handed nothing: two different batches would name the same thing.
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


def _check_they_talk(params: Sequence[Any], optimizer: Any) -> None:
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
