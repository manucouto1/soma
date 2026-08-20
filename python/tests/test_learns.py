"""A node that trains itself.

The mixin's half is checked here — the dispatch, the leaf, what is held and what
is let go of — with a **local** learner, which is not a lesser case: for autograd
a `Learns` node already **is** "the other side", because it detaches what it was
given. That is the same property that makes `(host, learns)` the pair a graph is
cut by, so what passes here passes over a wire for the same reason.

The four techniques and their controls are in `tests/cluster`, against a real
container, and in the use-case docs.
"""

import pickle
from functools import partial

import pytest

torch = pytest.importorskip("torch")

from soma_next import Graph  # noqa: E402

from conftest import Add  # noqa: E402
from soma_next.torch import Learns, envelope  # noqa: E402
from soma_next.torch._learns import SIGNAL  # noqa: E402


class Body(Learns):
    """A learner, written the way a user writes one: torch inside, and the two
    ducks the rest of the project already asks for."""

    def __init__(self, wide=4, tall=3):
        self.lin = torch.nn.Linear(wide, tall)

    def compute(self, x, ctx):
        return self.lin(x).relu()

    def parameters(self):
        return list(self.lin.parameters())


class Doubles(Learns):
    """One that notes what it was handed, for the dispatch to be observable."""

    def __init__(self):
        self.weight = torch.nn.Parameter(torch.tensor([2.0]))
        self.seen = []

    def compute(self, x, ctx):
        self.seen.append("compute")
        return x * self.weight

    def parameters(self):
        return [self.weight]


class Alone:
    """The `ctx` of a node called by hand: nowhere placed, and no turn to speak
    of. A node that learns reads only the device."""

    device = None


HERE = Alone()


def learner(node, lr=0.1):
    """The same node, told what to build its optimizer with."""
    return node.learns_with(partial(torch.optim.SGD, lr=lr))


# ── The dispatch, which is the one contract filled in ──


def test_an_ordinary_input_computes_and_comes_back_as_data():
    body = learner(Doubles())
    g = Graph.somatize(body.named("body"))

    assert g.forward([1.0, 2.0]) == [2.0, 4.0]
    assert body.seen == ["compute"]


def test_an_envelope_learns_instead_of_computing():
    body = learner(Doubles())
    g = Graph.somatize(body.named("body"))
    g.forward([1.0, 2.0])

    back = g.forward(envelope([1.0, 1.0]))
    assert body.seen == ["compute"], "the gradient went in through `compute`"
    assert back == {SIGNAL: [2.0, 2.0]}, "what goes back is dL/d(what I was given)"


def test_what_it_gives_back_is_the_gradient_of_what_it_was_given():
    # `y = w·x` with `w = 2`, so `dL/dx = 2·dL/dy` and `dL/dw = Σ x·dL/dy`.
    body = learner(Doubles(), lr=0.0)
    g = Graph.somatize(body.named("body"))
    g.forward([3.0, 5.0])

    assert g.forward(envelope([1.0, 1.0]))[SIGNAL] == [2.0, 2.0]
    assert body.weight.grad.tolist() == [8.0]


def test_a_fan_in_of_gradients_is_summed():
    # A node that fed two is owed one gradient per consumer, and the chain rule
    # says they add. A map of envelopes is that, and nothing else is.
    body = learner(Doubles(), lr=0.0)
    g = Graph.somatize(body.named("body"))
    g.forward([1.0])

    back = g.forward({"left": envelope([1.0]), "right": envelope([10.0])})
    assert back == {SIGNAL: [22.0]}


def test_a_map_that_is_not_all_envelopes_is_an_input():
    # And an input is what it cannot take: one gradient per producer is not
    # something the transpose routes. Said in as many words, and not as
    # torch's "must be real number, not dict".
    body = learner(Doubles())
    g = Graph.somatize(body.named("body"))

    with pytest.raises(ValueError, match="takes one input"):
        g.forward({"left": [1.0], "right": envelope([1.0])})


# ── What is held, and what is let go of ──


def test_a_gradient_with_no_activation_says_so_by_name():
    from soma_next.torch import OutOfStep

    body = learner(Body())

    with pytest.raises(OutOfStep, match="Body"):
        body.forward(envelope([[0.0, 0.0, 0.0]]), HERE)


def test_the_activation_is_let_go_of_after_learning():
    # Two gradients for one forward is the failure CU12 wrote down against
    # itself: without this the second one would train on a stale activation.
    body = learner(Body())
    g = Graph.somatize(body.named("body"))
    g.forward([[1.0, 2.0, 3.0, 4.0]])
    g.forward(envelope([[0.1, 0.1, 0.1]]))

    with pytest.raises(ValueError, match="OutOfStep"):
        g.forward(envelope([[0.1, 0.1, 0.1]]))


def test_it_lets_go_of_the_chain_that_produced_its_input():
    """The premise the whole cut rests on: for autograd a learner **is** the
    other side of a wire, because it detaches what it was handed. Nothing above
    it gets a gradient unless it is handed one back."""
    above = torch.nn.Linear(4, 4)
    body = learner(Body(4, 3))

    body.forward(above(torch.randn(2, 4)), HERE)
    assert body.given.grad_fn is None, "it kept somebody else's chain"

    body.forward(envelope(torch.ones(2, 3)), HERE)
    assert all(p.grad is None for p in above.parameters())


# ── The optimizer, which is not built where you would think ──


def test_it_is_built_on_first_use_and_not_in_init():
    body = learner(Body())
    assert body.optimizer is None

    g = Graph.somatize(body.named("body"))
    g.forward([[1.0, 2.0, 3.0, 4.0]])
    assert body.optimizer is None, "computing does not need one"

    g.forward(envelope([[0.1, 0.1, 0.1]]))
    assert isinstance(body.optimizer, torch.optim.SGD)


def test_a_node_rebuilt_by_pickle_still_learns():
    # Which is the reason for all of the above: `pickle` does not call
    # `__init__`, and being rebuilt on another machine is this node's normal
    # life. What travels is the factory.
    body = pickle.loads(pickle.dumps(learner(Body())))
    before = float(body.lin.weight.detach().abs().sum())

    body.forward([[1.0, 2.0, 3.0, 4.0]], HERE)
    body.forward(envelope([[1.0, 1.0, 1.0]]), HERE)
    assert float(body.lin.weight.detach().abs().sum()) != before


def test_nobody_said_what_to_train_it_with():
    body = Body()
    body.forward([[1.0, 2.0, 3.0, 4.0]], HERE)

    with pytest.raises(ValueError, match="learns_with"):
        body.forward(envelope([[1.0, 1.0, 1.0]]), HERE)


def test_a_learner_that_does_not_say_what_its_parameters_are():
    class Mute(Learns):
        def compute(self, x, ctx):
            return x * 2

    mute = learner(Mute())
    mute.forward([1.0], HERE)

    with pytest.raises(ValueError, match="parameters"):
        mute.forward(envelope([1.0]), HERE)


# ── The whole point: the far half comes down on its own ──


def test_the_far_half_trains_itself_from_the_gradient_it_is_handed():
    torch.manual_seed(0)
    body = learner(Body(4, 3))
    g = Graph.somatize(body.named("body"))
    head = torch.nn.Linear(3, 2)
    head_optimizer = torch.optim.SGD(head.parameters(), lr=0.1)
    x, y = torch.randn(16, 4), torch.randint(0, 2, (16,))

    losses, weights = [], []
    for _ in range(30):
        seam = torch.tensor(g.forward(x.tolist()), requires_grad=True)
        loss = torch.nn.functional.cross_entropy(head(seam), y)
        head_optimizer.zero_grad()
        loss.backward()
        head_optimizer.step()
        g.forward(envelope(seam.grad))
        losses.append(loss.item())
        weights.append(float(body.lin.weight.detach().abs().sum()))

    assert losses[-1] < losses[0], "the loss did not come down"
    assert weights[0] != weights[-1], "the gradient never reached the far half"


def test_and_without_the_gradient_going_back_it_does_not():
    # The control, which is the only thing that makes the test above mean
    # anything: the same loop with nothing handed back leaves it where it was.
    torch.manual_seed(0)
    body = learner(Body(4, 3))
    g = Graph.somatize(body.named("body"))
    head = torch.nn.Linear(3, 2)
    head_optimizer = torch.optim.SGD(head.parameters(), lr=0.1)
    x, y = torch.randn(16, 4), torch.randint(0, 2, (16,))
    before = float(body.lin.weight.detach().abs().sum())

    for _ in range(30):
        seam = torch.tensor(g.forward(x.tolist()), requires_grad=True)
        loss = torch.nn.functional.cross_entropy(head(seam), y)
        head_optimizer.zero_grad()
        loss.backward()
        head_optimizer.step()

    assert float(body.lin.weight.detach().abs().sum()) == before


# ── And what the graph makes of it ──


def test_a_learner_cuts_the_graph_where_it_stands():
    # The same duck `_stage` asks with, so the two halves of this use case agree
    # without knowing about each other.
    from soma_next._stage import stages

    g = Graph()
    g.node("head", Add(1))
    g.node("body", learner(Body()))
    g.edge("head", "body")

    assert [stage.nodes for stage in stages(g)] == [("head",), ("body",)]
