"""Training a graph from outside the graph.

The two things this file is defending:

**The graph does not find out.** After training, its nodes, its edges, its plan
and its placement are identical. What changes are the weights, which live inside
the nodes and always did.

**Several training runs are a list, not a graph.** The hyperparameter search is
written down here as a list comprehension, without a single new type — and the
test that matters most of all is the one that checks that two runs from the same
factory **do not share weights**, because sharing them would give results that
look good and are not.
"""

from functools import partial

import pytest

from soma_next import Done, Graph, Node, Opaque
from soma_next.torch import Learns, Trainer, parameters

torch = pytest.importorskip("torch")
nn = torch.nn

IN, MID, CLASSES = 4, 3, 2
no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")


class Layer(Node):
    """A node with parameters that obeys its placement. The CU10 pattern."""

    def __init__(self, in_, out):
        self.lin = nn.Linear(in_, out)
        self.placed = None

    def forward(self, x, ctx):
        if ctx.device:
            if self.placed != ctx.device:
                self.lin.to(ctx.device)
                self.placed = ctx.device
            x = x.to(ctx.device)
        return Done(Opaque(self.lin(x)))

    def parameters(self):
        return list(self.lin.parameters())


class Alone(Learns):
    """A node that trains itself, wherever it runs. Its weights are not this
    graph's optimizer's business, and saying so is the `learn` it inherits."""

    def __init__(self, in_, out):
        self.lin = nn.Linear(in_, out)

    def compute(self, x, ctx):
        return self.lin(x)

    def parameters(self):
        return list(self.lin.parameters())


class Label(Node):
    """No parameters: not every node trains, and it does not stop being a node."""

    def forward(self, x, ctx):
        return Done(x)


def net(gpu=None):
    """The factory: one configuration → one freshly built trainable network."""
    encoder, head = Layer(IN, MID), Layer(MID, CLASSES)
    expression = encoder.named("encoder") >> head.named("head")
    if gpu:
        expression = expression.on(gpu)
    return Graph.somatize(expression)


def batches(n=4):
    torch.manual_seed(0)
    return [(torch.randn(6, IN), torch.randint(0, CLASSES, (6,))) for _ in range(n)]


def trainer(g, lr=0.05):
    return Trainer(
        g,
        objective=nn.functional.cross_entropy,
        optimizer=torch.optim.Adam(parameters(g), lr=lr),
    )


# ── A graph's parameters ──


def test_it_collects_those_of_every_node_that_has_any():
    g = net()
    assert len(parameters(g)) == 4  # weight and bias of each layer


def test_it_skips_the_nodes_without_parameters():
    g = Graph.somatize(Label().named("label") >> Layer(IN, MID).named("layer"))
    assert len(parameters(g)) == 2


def test_they_come_in_declaration_order_and_do_not_change_between_calls():
    g = net()
    assert [id(p) for p in parameters(g)] == [id(p) for p in parameters(g)]


def test_a_shared_module_does_not_come_out_twice():
    # Tied weights: two nodes, the same module. An optimizer with duplicates
    # warns or fails, depending on the torch version.
    shared = Layer(IN, IN)
    g = Graph()
    g.node("a", shared)
    g.node("b", shared)
    g.edge("a", "b")
    assert len(parameters(g)) == 2


def test_a_node_that_trains_itself_is_not_in_them():
    # Two optimizers over one tensor is what this avoids, and it is also what
    # keeps `NoGradient` meaning what it means.
    g = Graph.somatize(
        Layer(IN, MID).named("layer") >> Alone(MID, CLASSES).named("alone")
    )
    assert len(parameters(g)) == 2


# ── Building the Trainer: what gets rejected ──


def test_a_graph_without_parameters_fails_when_building_the_trainer():
    # And not with a flat loss twenty minutes later.
    g = Graph.somatize(Label().named("label"))
    with pytest.raises(ValueError, match="has no parameters"):
        Trainer(g, objective=nn.functional.cross_entropy, optimizer=None)


def test_an_optimizer_from_another_graph_fails():
    g, other = net(), net()
    with pytest.raises(ValueError, match="updates no parameter"):
        Trainer(
            g,
            objective=nn.functional.cross_entropy,
            optimizer=torch.optim.Adam(parameters(other), lr=0.1),
        )


def test_a_graph_where_everything_trains_itself_needs_no_optimizer():
    g = Graph.somatize(Alone(IN, MID).named("alone"))
    built = Trainer(g, objective=nn.functional.cross_entropy)
    assert repr(built) == "Trainer(0 parameters)"


def test_weights_here_and_no_optimizer_is_refused():
    with pytest.raises(ValueError, match="no optimizer to move them"):
        Trainer(net(), objective=nn.functional.cross_entropy)


def test_an_optimizer_over_a_graph_that_trains_itself_is_refused():
    alone = Alone(IN, MID)
    g = Graph.somatize(alone.named("alone"))
    with pytest.raises(ValueError, match="nothing to update"):
        Trainer(
            g,
            objective=nn.functional.cross_entropy,
            optimizer=torch.optim.Adam(alone.parameters(), lr=0.1),
        )


def test_settled_and_training_itself_is_a_contradiction():
    # One says its state does not change while the graph runs and the other
    # changes it every step. Before the first one, not after a cache gives back
    # the wrong tensor.
    g = Graph.somatize(
        Layer(IN, MID).named("layer") >> Alone(MID, CLASSES).named("alone").frozen()
    )
    with pytest.raises(ValueError, match="settled and train itself"):
        trainer(g)


def test_freezing_a_part_is_legitimate_and_passes():
    # Training only the head covers a part of the parameters, not all of them.
    g = net()
    head = g.implementation("head")
    trainable = Trainer(
        g,
        objective=nn.functional.cross_entropy,
        optimizer=torch.optim.Adam(head.parameters(), lr=0.1),
    )
    assert trainable.step(batches()[0]) > 0


# ── Training ──


def test_training_brings_the_loss_down():
    t = trainer(net())
    result = t.fit(batches(), epochs=10)

    assert len(result.history) == 40
    assert result.loss == result.history[-1]
    assert result.loss < result.history[0], f"it did not go down: {result!r}"


def test_fit_gives_the_same_as_the_hand_written_loop_over_step():
    torch.manual_seed(7)
    with_fit = trainer(net()).fit(batches(), epochs=2)

    torch.manual_seed(7)
    by_hand, history = trainer(net()), []
    for _ in range(2):
        for batch in batches():
            history.append(by_hand.step(batch))

    assert with_fit.history == pytest.approx(history)


def test_the_weights_the_optimizer_updates_are_the_ones_the_graph_uses():
    g = net()
    before = g.implementation("head").lin.weight.detach().clone()
    trainer(g, lr=0.1).step(batches()[0])
    assert not torch.allclose(before, g.implementation("head").lin.weight)


def test_training_does_not_change_the_graph():
    g = net()
    snapshot = (g.nodes(), g.edges(), g.plan(), g.devices())
    trainer(g).fit(batches())
    assert (g.nodes(), g.edges(), g.plan(), g.devices()) == snapshot


def test_an_input_that_is_not_a_tensor_crosses_as_always():
    # `_crossable` only wraps tensors; everything else takes the usual path.
    class Count(Node):
        def __init__(self):
            self.lin = nn.Linear(1, CLASSES)

        def forward(self, texts, ctx):
            lengths = torch.tensor([[float(len(t))] for t in texts])
            return Done(Opaque(self.lin(lengths)))

        def parameters(self):
            return list(self.lin.parameters())

    g = Graph.somatize(Count().named("count"))
    batch = (["hello", "goodbye"], torch.tensor([0, 1]))
    assert trainer(g).step(batch) > 0


# ── Several training runs: a list, not a graph ──


def test_two_nets_from_the_same_factory_do_not_share_weights():
    # The counterexample that ruled out "one graph, N catalogs": cloning a
    # catalog clones `Arc`s, i.e. the replicas would share the weights and all
    # train the same model. Each one has to be built.
    one, other = net(), net()
    assert {id(p) for p in parameters(one)}.isdisjoint({id(p) for p in parameters(other)})

    trainer(one, lr=0.5).fit(batches(), epochs=3)
    assert not torch.allclose(
        one.implementation("head").lin.weight,
        other.implementation("head").lin.weight,
    ), "training one moved the other's weights"


def test_the_hyperparameter_search_is_a_list_comprehension():
    # Without a new type, without branches in the graph, without waves. Two
    # networks, two training runs, and the best comes out of a `min`.
    data = batches()
    study = {lr: trainer(net(), lr=lr).fit(data, epochs=5) for lr in (1e-4, 1e-2)}

    best = min(study, key=lambda lr: study[lr].loss)
    assert best == 1e-2, {lr: r.loss for lr, r in study.items()}


# ── With a GPU: CU10 and CU11 at once ──


@no_cuda
def test_the_optimizer_still_points_at_the_weights_after_moving_to_the_gpu():
    # The node is placed **lazily**, on the first forward, i.e. after the
    # optimizer exists. `Module.to()` moves the parameters in place keeping the
    # same objects, so it should still hold — and that is exactly the kind of
    # "should" that has to be turned into a test.
    g = net(gpu="cuda:0")
    t = trainer(g, lr=0.1)
    assert g.implementation("head").lin.weight.device.type == "cpu", "not moved yet"

    before = g.implementation("head").lin.weight.detach().clone()
    t.step(batches()[0])

    weight = g.implementation("head").lin.weight
    assert weight.device.type == "cuda", "the node moved when it executed"
    assert not torch.allclose(before.cuda(), weight), "and the optimizer updated it"


@no_cuda
def test_training_with_the_two_layers_on_different_devices():
    encoder, head = Layer(IN, MID), Layer(MID, CLASSES)
    g = Graph.somatize(encoder.named("encoder").on("cuda:0") >> head.named("head").on("cpu"))
    result = trainer(g, lr=0.1).fit(batches(), epochs=5)
    assert result.loss < result.history[0]


@no_cuda
def test_the_target_goes_to_meet_the_output_on_its_device():
    # The asymmetry that only shows on a GPU: the input is moved by each node,
    # because it crosses the graph; the target does not enter the graph, so
    # nobody moved it and the loss blew up with "expected all tensors on the same
    # device". The only one that sees both sides fixes it.
    g = net(gpu="cuda:0")
    input_, target = batches()[0]
    assert target.device.type == "cpu"

    trainer(g).step((input_, target))  # the output ends up on cuda:0

    assert target.device.type == "cpu", "and the user's batch stays as it was"


# ── A graph that is cut: the Trainer drives ──


def cut_and_whole():
    """The same network twice, weight for weight: once in one piece, once with
    its first half training itself."""
    torch.manual_seed(0)
    whole = Graph.somatize(Layer(IN, MID).named("a") >> Layer(MID, CLASSES).named("b"))
    cut = Graph.somatize(Alone(IN, MID).named("a") >> Layer(MID, CLASSES).named("b"))
    for node_id in ("a", "b"):
        cut.implementation(node_id).lin.load_state_dict(
            whole.implementation(node_id).lin.state_dict()
        )
    return cut, whole


def test_a_cut_graph_trains_to_the_same_numbers_as_the_whole_one():
    # The best control there is: the framework did not change the arithmetic,
    # only who writes the loop. Bit for bit, because everything in between —
    # `tolist` on a float32, a detached leaf, an optimizer of its own with the
    # same rule — is the same operations in the same order.
    cut, whole = cut_and_whole()
    driven = Trainer(
        cut,
        objective=nn.functional.cross_entropy,
        optimizer=torch.optim.SGD(parameters(cut), lr=0.1),
        learns_with=partial(torch.optim.SGD, lr=0.1),
    )
    in_one_go = Trainer(
        whole,
        objective=nn.functional.cross_entropy,
        optimizer=torch.optim.SGD(parameters(whole), lr=0.1),
    )
    batch = batches(1)[0]

    assert [driven.step(batch) for _ in range(10)] == [
        in_one_go.step(batch) for _ in range(10)
    ]
    assert torch.equal(
        cut.implementation("a").lin.weight, whole.implementation("a").lin.weight
    )


def test_it_is_driven_in_stages_only_when_something_trains_itself():
    # A slice on another host with nobody learning is the step it always was:
    # the whole point of asking is to keep this out of everybody else's way.
    away = Graph.somatize(
        Layer(IN, MID).named("a") >> Layer(MID, CLASSES).named("b").at("w1")
    )
    assert not trainer(away).by_stages

    cut, _ = cut_and_whole()
    assert Trainer(
        cut,
        objective=nn.functional.cross_entropy,
        optimizer=torch.optim.SGD(parameters(cut), lr=0.1),
        learns_with=partial(torch.optim.SGD, lr=0.1),
    ).by_stages


def test_what_is_above_a_learner_gets_its_gradient_through_it():
    # Three stages, and the middle one breaks the chain by construction: `a` is
    # only trained if the gradient the learner gives back is applied here. And
    # if it were asked for too early, `NoGradient` would call `a` an orphan.
    torch.manual_seed(0)
    g = Graph.somatize(
        Layer(IN, MID).named("a")
        >> Alone(MID, MID).named("b")
        >> Layer(MID, CLASSES).named("c")
    )
    driven = Trainer(
        g,
        objective=nn.functional.cross_entropy,
        optimizer=torch.optim.SGD(parameters(g), lr=0.1),
        learns_with=partial(torch.optim.SGD, lr=0.1),
    )
    before = {i: _weights(g, i) for i in g.nodes()}
    losses = [driven.step(batches(1)[0]) for _ in range(20)]

    assert losses[-1] < losses[0]
    assert all(before[i] != _weights(g, i) for i in g.nodes()), "somebody stood still"


def test_a_graph_that_only_trains_itself_takes_no_optimizer_here():
    torch.manual_seed(0)
    alone = Alone(IN, CLASSES)
    g = Graph.somatize(alone.named("alone"))
    driven = Trainer(
        g,
        objective=nn.functional.cross_entropy,
        learns_with=partial(torch.optim.SGD, lr=0.1),
    )
    before = _weights(g, "alone")

    losses = [driven.step(batches(1)[0]) for _ in range(20)]
    assert losses[-1] < losses[0]
    assert _weights(g, "alone") != before


def test_kept_after_a_cut_is_refused():
    # A root's key comes from the input it was handed, and after a cut the roots
    # of a stage are holds handed nothing: two batches would name one thing.
    g = Graph.somatize(
        Alone(IN, MID).named("a") >> Layer(MID, CLASSES).named("b").cached()
    )
    with pytest.raises(ValueError, match="no longer knows what came before"):
        Trainer(
            g,
            objective=nn.functional.cross_entropy,
            optimizer=torch.optim.SGD(parameters(g), lr=0.1),
            learns_with=partial(torch.optim.SGD, lr=0.1),
        )


def test_learns_with_naming_somebody_who_does_not_learn():
    g = net()
    with pytest.raises(ValueError, match="do not train themselves"):
        Trainer(
            g,
            objective=nn.functional.cross_entropy,
            optimizer=torch.optim.Adam(parameters(g), lr=0.1),
            learns_with={"encoder": partial(torch.optim.SGD, lr=0.1)},
        )


def test_the_node_keeps_its_own_rule_and_a_named_one_overrides_it():
    said_by_the_node = partial(torch.optim.SGD, lr=0.5)
    mine, theirs = Alone(IN, MID), Alone(IN, MID)
    mine.learns_with(said_by_the_node)
    theirs.learns_with(said_by_the_node)
    g = Graph()
    g.node("mine", mine)
    g.node("theirs", theirs)
    named = partial(torch.optim.Adam, lr=0.1)

    Trainer(g, objective=nn.functional.cross_entropy, learns_with={"theirs": named})
    assert mine.learning is said_by_the_node, "nobody named it, nobody touched it"
    assert theirs.learning is named, "naming it is the override"


def test_a_factory_for_everybody_fills_in_only_whoever_said_nothing():
    said_by_the_node = partial(torch.optim.SGD, lr=0.5)
    mine, silent = Alone(IN, MID), Alone(IN, MID)
    mine.learns_with(said_by_the_node)
    g = Graph()
    g.node("mine", mine)
    g.node("silent", silent)
    for_everybody = partial(torch.optim.Adam, lr=0.1)

    Trainer(g, objective=nn.functional.cross_entropy, learns_with=for_everybody)
    assert mine.learning is said_by_the_node, "the rule lives in the node"
    assert silent.learning is for_everybody


def _weights(graph, node_id):
    return float(graph.implementation(node_id).lin.weight.detach().abs().sum())
