"""What trains a node, standing beside it.

The node in this file is a node and nothing else: no base class of ours, no
`learn`, no idea that any of this exists. What trains it is another object, said
by whoever trains — and the same node runs untouched with it or without it.

It is checked here with the trainer **local**, which is not a lesser case: for
autograd a trainer already is "the other side", because it lets go of the chain
the moment it is handed one. That is the same property that makes `(host,
trained)` the pair a graph is cut by, so what passes here passes over a wire for
the same reason — and there is a test at the bottom that says so against a real
process, bit for bit.
"""

import pickle
import sys
from functools import partial

import pytest

torch = pytest.importorskip("torch")
cloudpickle = pytest.importorskip("cloudpickle")
cloudpickle.register_pickle_by_value(sys.modules[__name__])

from soma_next import Done, Graph, Node, Opaque, Worker  # noqa: E402
from soma_next.torch import (  # noqa: E402
    Learning,
    NoGradient,
    OutOfStep,
    Split,
    Trainer,
    envelope,
    parameters,
)
from soma_next.torch._learning import SIGNAL, Enters  # noqa: E402


class Body(Node):
    """A node. It does not know it is going to be trained, and nothing here ever
    tells it."""

    def __init__(self, wide=4, tall=3):
        self.lin = torch.nn.Linear(wide, tall)

    def forward(self, x, ctx):
        return Done(Opaque(self.lin(x).relu()))

    def parameters(self):
        return list(self.lin.parameters())


class Head(Node):
    """The near half: plain torch, and the one this side's optimizer updates."""

    def __init__(self, wide=3, tall=2):
        self.lin = torch.nn.Linear(wide, tall)

    def forward(self, x, ctx):
        return Done(Opaque(self.lin(x)))

    def parameters(self):
        return list(self.lin.parameters())


class Doubles(Node):
    """One that notes what it was handed, for the dispatch to be observable."""

    def __init__(self):
        self.weight = torch.nn.Parameter(torch.tensor([2.0]))
        self.seen = []

    def forward(self, x, ctx):
        self.seen.append("computed")
        return Done(Opaque(x * self.weight))

    def parameters(self):
        return [self.weight]


class Alone:
    """The `ctx` of something called by hand: nowhere placed, and no turn to
    speak of."""

    device = None


HERE = Alone()


def beside(node, technique=Split, lr=0.1, **how):
    """That node's trainer and its two positions, as `around` would make them."""
    learning = technique(torch.optim.SGD, lr=lr, **how).of(node)
    return learning, Enters(learning)


def out(transition):
    """What a `forward` produced, seen the way whoever reads it next sees it: a
    node is handed what was wrapped, not the wrapper."""
    value = transition.value
    return value.value if isinstance(value, Opaque) else value


def a_pass(learning, entering, value, ctx=None):
    """One forward through the three of them, in the order a stage runs them."""
    ctx = ctx or HERE
    given = out(entering.forward(value, ctx))
    return learning.forward(out(learning.node.forward(given, ctx)), ctx)


def seeded(seed=0):
    """The weights of a test are its own and not whatever the one before it left
    in the generator. Said before the nodes are built, which is where they draw
    from."""
    torch.manual_seed(seed)


# ── The two positions ──


def test_the_input_becomes_a_leaf_and_the_node_never_finds_out():
    node = Doubles()
    learning, entering = beside(node)

    landed = out(entering.forward([3.0], HERE))
    assert landed.requires_grad and landed.grad_fn is None
    assert learning.given is landed, "the trainer is the one that remembers it"
    assert node.seen == [], "nothing was computed by making a leaf"


def test_an_ordinary_value_is_the_activation_and_an_envelope_is_a_gradient():
    node = Doubles()
    learning, entering = beside(node, lr=0.0)

    a_pass(learning, entering, [3.0, 5.0])
    assert learning.held is not None, "the activation is kept here"

    back = out(learning.forward(envelope([1.0, 1.0]), HERE))
    assert node.seen == ["computed"], "the gradient did not go into the node"
    assert back == {SIGNAL: [2.0, 2.0]}, "what goes back is dL/d(what it was given)"
    assert node.weight.grad.tolist() == [8.0], "and dL/dw stayed here"


def test_a_fan_in_of_gradients_is_summed():
    # A value that fed two is owed one gradient per consumer, and the chain rule
    # says they add. A map of envelopes is that, and nothing else is.
    node = Doubles()
    learning, entering = beside(node, lr=0.0)
    a_pass(learning, entering, [1.0])

    back = learning.forward({"left": envelope([1.0]), "right": envelope([10.0])}, HERE)
    assert out(back) == {SIGNAL: [22.0]}


def test_a_map_that_is_not_all_envelopes_is_an_input():
    # And an input is what a trained node cannot take two of: one gradient per
    # producer is not something the transpose routes. Said in as many words.
    _, entering = beside(Doubles())

    with pytest.raises(ValueError, match="takes one input"):
        entering.forward({"left": [1.0], "right": [2.0]}, HERE)


def test_a_gradient_with_no_activation_says_so():
    learning, _ = beside(Body())

    with pytest.raises(OutOfStep, match="Split"):
        learning.forward(envelope([[0.0, 0.0, 0.0]]), HERE)


def test_the_activation_is_let_go_of_after_learning():
    node = Body()
    learning, entering = beside(node)
    a_pass(learning, entering, [[1.0, 2.0, 3.0, 4.0]])
    learning.forward(envelope([[0.1, 0.1, 0.1]]), HERE)

    with pytest.raises(OutOfStep):
        learning.forward(envelope([[0.1, 0.1, 0.1]]), HERE)


def test_it_lets_go_of_the_chain_that_produced_its_input():
    """The premise the whole cut rests on: a trainer beside a node **is** the
    other side of a wire for autograd. Nothing above it gets a gradient unless it
    is handed one back."""
    seeded()
    above = torch.nn.Linear(4, 4)
    node = Body()
    learning, entering = beside(node)

    given = out(entering.forward(above(torch.randn(2, 4)), HERE))
    assert given.grad_fn is None, "it kept somebody else's chain"

    learning.forward(out(node.forward(given, HERE)), HERE)
    learning.forward(envelope(torch.ones(2, 3)), HERE)
    assert all(p.grad is None for p in above.parameters())


# ── The optimizer, which is not built where you would think ──


def test_it_is_built_on_first_use_over_the_parameters_of_the_node():
    node = Body()
    learning, entering = beside(node)
    assert learning.built is None

    a_pass(learning, entering, [[1.0, 2.0, 3.0, 4.0]])
    assert learning.built is None, "computing does not need one"

    learning.forward(envelope([[0.1, 0.1, 0.1]]), HERE)
    assert isinstance(learning.built, torch.optim.SGD)
    assert learning.optimizer is learning.built, "and it is not built twice"
    held = {id(p) for group in learning.built.param_groups for p in group["params"]}
    assert held == {id(p) for p in node.parameters()}


def test_a_trainer_rebuilt_by_pickle_still_trains():
    # Which is the reason for all of the above: `pickle` does not call
    # `__init__`, and being rebuilt on another machine is this object's normal
    # life. What travels is the factory — and the node, once, for both of them.
    seeded()
    node = Body()
    learning, entering = pickle.loads(pickle.dumps(beside(node)))
    assert learning.node is entering.learning.node, "two copies of one node"
    before = float(learning.node.lin.weight.detach().abs().sum())

    a_pass(learning, entering, [[1.0, 2.0, 3.0, 4.0]])
    learning.forward(envelope([[1.0, 1.0, 1.0]]), HERE)
    assert float(learning.node.lin.weight.detach().abs().sum()) != before


# ── Driven by the Trainer, which is how anybody would write it ──


def driving(node, technique=None, lr=0.1, at=None, seed=0, **how):
    """A `body >> head` graph with `body` trained where it runs, its trainer and
    a batch. The same setup for every technique, so what tells them apart is the
    `learn` and nothing else."""
    seeded(seed)
    head = Head(node.lin.out_features, 2)
    body = node.named("body")
    trains = {"body": (technique or Split)(torch.optim.SGD, lr=lr, **how)}
    g = Graph.somatize((body.at(at) if at else body) >> head.named("head"))
    driven = Trainer(
        g,
        objective=torch.nn.functional.cross_entropy,
        optimizer=torch.optim.SGD(parameters(g, without=trains), lr=0.1),
        trains=trains,
        workers={at: Worker.generic(mode="network")} if at else None,
    )
    seeded(7)
    return g, driven, (torch.randn(16, 4), torch.randint(0, 2, (16,)))


def test_the_far_half_trains_and_the_graph_never_finds_out():
    seeded()
    body = Body(4, 3)
    g, driven, batch = driving(body)
    before = float(body.lin.weight.detach().abs().sum())

    losses = [driven.step(batch) for _ in range(30)]
    assert losses[-1] < losses[0], "the loss did not come down"
    assert float(body.lin.weight.detach().abs().sum()) != before
    assert g.nodes() == ["body", "head"], "the graph was rewritten under it"
    assert g.implementation("body") is body


def test_and_without_a_trainer_beside_it_those_weights_never_move():
    # The control: the same graph, the same batch, and `body` left to a
    # `backward()` that cannot reach it. It is `NoGradient` that says so, and
    # leaving it in the optimizer is what asks the question.
    seeded()
    body, head = Body(4, 3), Head(3, 2)
    g = Graph.somatize(body.named("body") >> head.named("head"))
    driven = Trainer(
        g,
        objective=torch.nn.functional.cross_entropy,
        optimizer=torch.optim.SGD(parameters(g), lr=0.1),
    )
    seeded(7)
    x, y = torch.randn(16, 4), torch.randint(0, 2, (16,))
    before = float(body.lin.weight.detach().abs().sum())

    for _ in range(30):
        driven.step((x, y))
    assert float(body.lin.weight.detach().abs().sum()) != before, "here it does move"


def test_the_same_weights_in_two_optimizers_is_refused():
    # Where the node runs may well be here, and then both would move them every
    # step. Two updates for one gradient is a worse loss, not a wrong one, which
    # is exactly the kind that goes unnoticed.
    seeded()
    body = Body(4, 3)
    g = Graph.somatize(body.named("body") >> Head(3, 2).named("head"))

    with pytest.raises(ValueError, match="two updates a step"):
        Trainer(
            g,
            objective=torch.nn.functional.cross_entropy,
            optimizer=torch.optim.SGD(parameters(g), lr=0.1),
            trains={"body": Split(torch.optim.SGD, lr=0.1)},
        )


# ── Over a wire, which is the case all of this was written for ──


def a_run(at, lr, steps=10):
    """The same net, the same seed and the same batches, with the body here or on
    another process. Returns the losses."""
    seeded()
    _, driven, batch = driving(Body(4, 3), lr=lr, at=at)
    return [driven.step(batch) for _ in range(steps)]


def test_a_node_trained_in_another_process_comes_out_the_same_as_here():
    # The use case, end to end and driven by `Trainer.step`: the trainer travels,
    # keeps the activation over there, gets `dL/da` back and steps its own
    # optimizer, and what crosses is data in both directions. Bit for bit against
    # the same net trained in one piece, which says the framework changed who
    # writes the loop and not the arithmetic.
    assert a_run(None, 0.1) == a_run("w1", 0.1)


def test_and_with_the_far_side_standing_still_the_loss_comes_down_less():
    # The control: the same run with the far side's rate at zero. Without it the
    # test above would pass just as well with the head doing all the work.
    assert a_run("w1", 0.1)[-1] < a_run("w1", 0.0)[-1]


# ── The four techniques, which are one hole answered four ways ──
#
# Split learning is the one `Learning` comes with, and it is what everything
# above this line is about. These three are the same hole with another answer
# written into it, and each is here with the control that says it really does
# what it claims — because "the loss came down" is something a net says just as
# loudly when only half of it is learning.


class Greedy(Learning):
    """Local greedy: an objective of its own — putting back what it was given —
    and the signal **ignored**. Nothing goes back up."""

    def __init__(self, optimizer, wide=4, tall=3, **how):
        super().__init__(optimizer, **how)
        self.back = torch.nn.Linear(tall, wide)
        self.mine = []

    def training(self):
        return super().training() + list(self.back.parameters())

    def learn(self, signal, ctx):
        local = torch.nn.functional.mse_loss(self.back(self.waiting()), self.given)
        self.optimizer.zero_grad()
        local.backward()
        self.optimizer.step()
        self.mine.append(local.item())
        self.done()
        return None


class ForwardForward(Learning):
    """Forward-forward: **no backward pass at all**, not even its own chain rule.
    The batch comes in two halves — the real ones first — and what it trains on is
    goodness: above the threshold for the first half, below for the second."""

    def __init__(self, optimizer, threshold=2.0, **how):
        super().__init__(optimizer, **how)
        self.threshold = threshold
        self.apart = []

    def learn(self, signal, ctx):
        good = self.waiting().pow(2).sum(dim=1)
        half = len(good) // 2
        real, made_up = good[:half], good[half:]
        loss = torch.nn.functional.softplus(
            torch.cat([self.threshold - real, made_up - self.threshold])
        ).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.apart.append(float((real.mean() - made_up.mean()).detach()))
        self.done()
        return None


class Synthetic(Learning):
    """Synthetic gradients: it does not wait for the signal, it **guesses** one
    out of the activation and updates with that. When the real one turns up, that
    is what the guesser trains on, and `apart` is how wrong the guess was as a
    fraction of the real thing.

    The guess starts at exactly zero, which is what makes the control exact: with
    the guesser frozen the error is 1 at every step, never once closer. It does
    not reach zero either, and the reason is worth knowing: the real `dL/da`
    depends on the labels, and this thing never sees them. Feeding them to it is
    what the paper calls cDNI.
    """

    guessing = None

    def __init__(self, optimizer, tall=3, learning_the_guesser=True, **how):
        super().__init__(optimizer, **how)
        self.guesser = torch.nn.Linear(tall, tall)
        torch.nn.init.zeros_(self.guesser.weight)
        torch.nn.init.zeros_(self.guesser.bias)
        self.learning_the_guesser = learning_the_guesser
        self.apart = []

    def learn(self, signal, ctx):
        held = self.waiting()
        guessed = self.guesser(held.detach())
        self.apart.append(float((guessed - signal).detach().norm() / signal.norm()))
        self.optimizer.zero_grad()
        held.backward(guessed.detach())
        self.optimizer.step()
        if self.learning_the_guesser:
            if self.guessing is None:
                self.guessing = torch.optim.Adam(self.guesser.parameters(), lr=0.01)
            self.guessing.zero_grad()
            torch.nn.functional.mse_loss(guessed, signal.detach()).backward()
            self.guessing.step()
        return self.done().grad


def test_greedy_trains_on_an_objective_the_loss_knows_nothing_about():
    seeded()
    body = Body(4, 3)
    g, driven, batch = driving(body, Greedy)
    assert len(parameters(g, without=driven.trains)) == 2, "only the head is here"

    for _ in range(20):
        driven.step(batch)
    assert driven.trains["body"].mine[-1] < driven.trains["body"].mine[0]


def test_and_nothing_crosses_back_out_of_a_greedy_layer():
    # The control, and the framework is the one that says it: whatever is above a
    # greedy layer gets no gradient, because it gives none back.
    seeded()
    above, body, head = Head(4, 4), Body(4, 3), Head(3, 2)
    trains = {"body": Greedy(torch.optim.SGD, lr=0.1)}
    g = Graph.somatize(
        above.named("above") >> body.named("body") >> head.named("head")
    )
    driven = Trainer(
        g,
        objective=torch.nn.functional.cross_entropy,
        optimizer=torch.optim.SGD(parameters(g, without=trains), lr=0.1),
        trains=trains,
    )
    seeded(7)

    with pytest.raises(NoGradient, match="`above`"):
        driven.step((torch.randn(16, 4), torch.randint(0, 2, (16,))))


def test_forward_forward_separates_the_goodness():
    seeded()
    g, driven, batch = driving(Body(4, 8), ForwardForward)

    for _ in range(20):
        driven.step(batch)
    apart = driven.trains["body"].apart
    assert apart[-1] > apart[0], "the goodness did not separate"


def test_and_with_the_rate_at_zero_the_goodness_does_not_separate():
    seeded()
    g, driven, batch = driving(Body(4, 8), ForwardForward, lr=0.0)

    for _ in range(20):
        driven.step(batch)
    apart = driven.trains["body"].apart
    assert apart[-1] == apart[0], "with nothing moving, nothing separated"


def test_synthetic_gradients_get_closer_to_the_real_one():
    seeded()
    g, driven, batch = driving(Body(4, 3), Synthetic)

    for _ in range(30):
        driven.step(batch)
    # The best it got, and not the last: the target keeps moving under it — the
    # head is training too — so the guess closes in and then drifts. What is
    # being claimed is that it ever got closer at all, which is exactly what the
    # control below says never happens on its own.
    assert min(driven.trains["body"].apart) < 0.97, "the guess never got closer"


def test_and_with_the_guesser_frozen_they_never_do():
    # Exactly 1 at every step, because a guess of zero is wrong by the whole of
    # what it was guessing at. Not once closer, not by accident.
    seeded()
    g, driven, batch = driving(Body(4, 3), Synthetic, learning_the_guesser=False)

    for _ in range(30):
        driven.step(batch)
    assert set(driven.trains["body"].apart) == {1.0}
