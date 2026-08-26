"""What a training run exports, and putting several of them together.

Level 3, where N training runs are **a Python list**. Nothing in here is a node
and nothing in here is a graph: a federated round has no dependencies to declare,
so it is a `for`, and the arithmetic that averages the weights is a function.

The one that says any of this is worth doing is the last one in the file: three
clients that each only ever see one corner of the input space, and an average
that reads all of it better than any of them does. Everything above it is what
has to be true for that to mean anything.
"""

import sys

import pytest

torch = pytest.importorskip("torch")
cloudpickle = pytest.importorskip("cloudpickle")
cloudpickle.register_pickle_by_value(sys.modules[__name__])

from torch import nn  # noqa: E402

from somatize import Broker, Graph, Node, Opaque, Worker  # noqa: E402
from somatize.torch import Split, Trainer, fedavg, parameters  # noqa: E402

IN, MID, CLASSES = 6, 5, 3


class Layer(Node):
    def __init__(self, n_in, n_out):
        self.lin = nn.Linear(n_in, n_out)

    def forward(self, x, ctx):
        return Opaque(self.lin(x))

    def parameters(self):
        return list(self.lin.parameters())


class Named(Node):
    """The other duck: a node that says what its weights are **called**."""

    def __init__(self, n_in, n_out):
        self.lin = nn.Linear(n_in, n_out)

    def state_dict(self):
        return self.lin.state_dict()

    def parameters(self):
        return list(self.lin.parameters())

    def forward(self, x, ctx):
        return Opaque(self.lin(x))


class Counts(Node):
    """No weights at all. It does not stop being a node for it."""

    def forward(self, x, ctx):
        return x


def net(seed=0):
    torch.manual_seed(seed)
    return Graph.somatize(
        Layer(IN, MID).named("body") >> Layer(MID, CLASSES).named("head")
    )


def trainer(g, lr=0.1, **how):
    return Trainer(
        g,
        objective=nn.functional.cross_entropy,
        optimizer=torch.optim.SGD(
            parameters(g, without=how.get("trains", {})), lr=lr
        ),
        **how,
    )


def batch(n=8, seed=0):
    torch.manual_seed(seed)
    return torch.randn(n, IN), torch.randint(0, CLASSES, (n,))


def test_it_is_the_weights_node_by_node():
    g = net()
    exported = trainer(g).export()

    assert sorted(exported) == ["body", "head"]
    assert sorted(exported["body"]) == ["0", "1"]  # weight and bias, in order
    assert exported["body"]["0"].shape == (MID, IN)


def test_a_node_that_says_what_its_weights_are_called_is_asked_by_name():
    # The same two ducks the rest of the project asks by, and it has to be the
    # same two: a node that can be told to settle and cannot be exported would be
    # a node this asks two different questions about its state.
    g = Graph.somatize(Named(IN, CLASSES).named("named"))

    assert sorted(trainer(g).export()["named"]) == ["bias", "weight"]


def test_a_node_with_no_weights_is_simply_not_in_there():
    g = Graph.somatize(Counts().named("counts") >> Layer(IN, CLASSES).named("head"))

    assert sorted(trainer(g).export()) == ["head"]


def test_what_comes_out_is_a_snapshot_and_not_a_view():
    # The whole point of exporting one: the next step must not move it under
    # whoever is holding it.
    g = net()
    t = trainer(g)
    before = t.export()

    t.step(batch())

    assert not torch.equal(before["head"]["0"], g.implementation("head").lin.weight)


def test_it_goes_back_in_where_it_came_from():
    g = net()
    t = trainer(g)
    kept = t.export()

    t.step(batch())
    assert not torch.equal(kept["head"]["0"], g.implementation("head").lin.weight)

    t.load(kept)
    assert torch.equal(kept["head"]["0"], g.implementation("head").lin.weight)


def test_the_optimizer_state_is_not_in_it():
    # Momentum is this client's, and averaging it is not what averaging weights
    # means. Said by there being nowhere for it to be.
    g = net()
    exported = trainer(g).export()

    assert all(
        set(state) <= {"0", "1"} for state in exported.values()
    ), "something that is not a weight came out"


def test_loading_something_this_graph_does_not_have_is_refused_by_name():
    g = net()
    t = trainer(g)

    with pytest.raises(ValueError, match="nothing called `elsewhere`"):
        t.load({"elsewhere": {"0": torch.zeros(1)}})


def test_loading_a_weight_of_the_wrong_shape_is_refused_with_both():
    g = net()
    t = trainer(g)

    with pytest.raises(ValueError, match=r"\(1, 1\)"):
        t.load({"head": {"0": torch.zeros(1, 1)}})


def test_a_refusal_leaves_the_net_as_it_was_rather_than_half_loaded():
    # Nothing is copied in until all of it checks out, which is what makes a
    # refusal something you can carry on from.
    g = net()
    t = trainer(g)
    before = t.export()

    with pytest.raises(ValueError):
        t.load({"body": {"0": torch.zeros(MID, IN)}, "head": {"0": torch.zeros(1, 1)}})

    assert torch.equal(before["body"]["0"], g.implementation("body").lin.weight)


def test_what_is_trained_where_it_runs_cannot_be_exported_from_here():
    # The one that would have been silent: those weights are on the other
    # machine and the copy here is the one that was sent, so exporting it would
    # hand back a net that never learnt anything.
    body = Layer(IN, MID)
    g = Graph.somatize(
        body.named("body").at("w1") >> Layer(MID, CLASSES).named("head")
    )
    trains = {"body": Split(torch.optim.SGD, lr=0.1)}
    t = trainer(g, trains=trains, broker=Broker.embedded({"w1": Worker.generic(mode="network")}))

    with pytest.raises(ValueError, match="`body` is trained where it runs"):
        t.export()
    with pytest.raises(ValueError, match="`body` is trained where it runs"):
        t.load({})


def test_but_one_trained_here_is_exported_like_any_other():
    # `trains` on its own is not the problem — running elsewhere is. A trainer
    # standing beside a node in this process holds this graph's own node.
    g = net()
    trains = {"body": Split(torch.optim.SGD, lr=0.1)}
    t = trainer(g, trains=trains)

    assert sorted(t.export()) == ["body", "head"]


def test_the_average_of_one_is_that_one():
    g = net()
    only = trainer(g).export()

    averaged = fedavg([only])

    assert torch.equal(averaged["head"]["0"], only["head"]["0"])


def test_the_average_of_two_is_halfway_between_them():
    one, other = trainer(net(1)).export(), trainer(net(2)).export()

    averaged = fedavg([one, other])

    assert torch.allclose(
        averaged["head"]["0"], (one["head"]["0"] + other["head"]["0"]) / 2
    )


def test_sizes_are_what_it_weighs_by_and_ten_times_the_data_pulls_ten_times():
    one, other = trainer(net(1)).export(), trainer(net(2)).export()

    averaged = fedavg([one, other], sizes=[900, 100])

    assert torch.allclose(
        averaged["head"]["0"], one["head"]["0"] * 0.9 + other["head"]["0"] * 0.1
    )


def test_what_is_not_a_number_you_can_halve_is_not_halved():
    # A `num_batches_tracked` is a count, and the mean of two counts is not a
    # count. Every implementation of this does it and none says so out loud.
    counted = torch.tensor(7)
    one = {"n": {"seen": counted, "w": torch.ones(2)}}
    other = {"n": {"seen": torch.tensor(11), "w": torch.zeros(2)}}

    averaged = fedavg([one, other])

    assert averaged["n"]["seen"].dtype == counted.dtype
    assert int(averaged["n"]["seen"]) == 7
    assert torch.allclose(averaged["n"]["w"], torch.full((2,), 0.5))


def test_averaging_two_different_networks_is_refused_and_not_computed():
    one = trainer(net(1)).export()
    other = trainer(Graph.somatize(Layer(IN, CLASSES).named("body"))).export()

    with pytest.raises(ValueError, match="different nodes"):
        fedavg([one, other])


def test_averaging_the_same_node_with_a_different_shape_says_both():
    one = {"n": {"0": torch.ones(2, 3)}}
    other = {"n": {"0": torch.ones(2, 4)}}

    with pytest.raises(ValueError, match=r"\(2, 4\).*\(2, 3\)"):
        fedavg([one, other])


def test_averaging_nothing_is_refused():
    with pytest.raises(ValueError, match="nothing to average"):
        fedavg([])


def test_one_size_per_training_run_and_not_a_number_of_them():
    one, other = trainer(net(1)).export(), trainer(net(2)).export()

    with pytest.raises(ValueError, match="2 training runs .* 3 sizes"):
        fedavg([one, other], sizes=[1, 2, 3])


def rule():
    """The one thing every client is trying to learn, and none of them owns.

    A fixed teacher, so the task is **the same** everywhere and what differs is
    only where each client's inputs come from. With one class per client instead,
    cross-entropy simply pushes its own logit up for ever: each of them diverges
    on its own, and the average of three diverged nets is not a lesson about
    federated learning, it is a lesson about learning rates. Measured on the way
    here: a loss of 2.7e8 against 4.5 for the client that stayed home.
    """
    torch.manual_seed(1234)
    return torch.randn(IN, CLASSES)


def shard(which, n=64):
    """One client's slice of the world: the same rule, seen from one corner of
    the input space and never from the others."""
    torch.manual_seed(100 + which)
    corner = torch.zeros(IN)
    corner[which % IN] = 2.5
    x = torch.randn(n, IN) + corner
    return x, (x @ rule()).argmax(dim=1)


def everything():
    """The union, which no client ever sees."""
    xs, ys = zip(*(shard(which) for which in range(CLASSES)))
    return torch.cat(xs), torch.cat(ys)


def loss_on(g, data):
    x, y = data
    with torch.no_grad():
        return float(nn.functional.cross_entropy(g.forward(Opaque(x)), y))


def test_three_clients_that_each_see_a_corner_average_into_one_that_sees_it_all():
    # The use case, and the control that makes it mean something: a client
    # trained alone on its own corner learns the rule as it looks from there and
    # no better. What the rounds have to show is the average beating it **on the
    # union** — which is the whole claim of federated learning, and it is a `for`.
    rounds, epochs, lr = 12, 3, 0.1
    clients = [trainer(net(seed=0), lr=lr) for _ in range(CLASSES)]
    alone = trainer(net(seed=0), lr=lr)

    for _ in range(rounds):
        for which, client in enumerate(clients):
            client.fit([shard(which)], epochs=epochs)
        average = fedavg([client.export() for client in clients])
        for client in clients:
            client.load(average)
        # The same amount of training, on one corner: what tells the average
        # apart from having simply taken more steps.
        alone.fit([shard(0)], epochs=epochs)

    together = loss_on(clients[0].graph, everything())
    apart = loss_on(alone.graph, everything())

    assert together < apart, f"averaged {together}, alone {apart}"


def test_and_the_round_leaves_every_client_at_the_same_weights():
    # What `load` is for: after a round they are one model, not three that
    # happen to be close.
    clients = [trainer(net(seed=0), lr=0.3) for _ in range(CLASSES)]
    for which, client in enumerate(clients):
        client.fit([shard(which)])

    average = fedavg([client.export() for client in clients])
    for client in clients:
        client.load(average)

    first = clients[0].export()
    for client in clients[1:]:
        for node_id, state in first.items():
            for key, value in state.items():
                assert torch.equal(client.export()[node_id][key], value)
