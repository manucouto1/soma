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

import pytest

from somatize import Graph, Node, Opaque
from somatize.torch import Split, Trainer, parameters

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
        return Opaque(self.lin(x))

    def parameters(self):
        return list(self.lin.parameters())


def split(lr=0.1):
    """A trainer to stand beside a node, with the technique this slice ships."""
    return Split(torch.optim.SGD, lr=lr)


class Label(Node):
    """No parameters: not every node trains, and it does not stop being a node."""

    def forward(self, x, ctx):
        return x


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


def test_whoever_is_trained_elsewhere_is_left_out_by_name():
    # Two optimizers over one tensor is what this avoids — and the graph is not
    # the one that knows, so it is told.
    g = Graph.somatize(
        Layer(IN, MID).named("layer") >> Layer(MID, CLASSES).named("alone")
    )
    assert len(parameters(g)) == 4
    assert len(parameters(g, without={"alone": split()})) == 2


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


def test_a_graph_where_everything_is_trained_elsewhere_needs_no_optimizer():
    g = Graph.somatize(Layer(IN, MID).named("alone"))
    built = Trainer(
        g, objective=nn.functional.cross_entropy, trains={"alone": split()}
    )
    assert repr(built) == "Trainer(2 parameters)"


def test_weights_here_and_no_optimizer_is_refused():
    with pytest.raises(ValueError, match="nobody would move"):
        Trainer(net(), objective=nn.functional.cross_entropy)


def test_an_optimizer_with_nothing_of_its_own_to_update_is_refused():
    alone = Layer(IN, MID)
    g = Graph.somatize(alone.named("alone"))
    with pytest.raises(ValueError, match="nothing to update"):
        Trainer(
            g,
            objective=nn.functional.cross_entropy,
            optimizer=torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=0.1),
            trains={"alone": split()},
        )


def test_trains_naming_somebody_who_is_not_there():
    with pytest.raises(ValueError, match="not a node of this graph"):
        Trainer(
            net(),
            objective=nn.functional.cross_entropy,
            optimizer=torch.optim.Adam(parameters(net()), lr=0.1),
            trains={"nobody": split()},
        )


def test_settled_and_trained_is_a_contradiction():
    # One says its state does not change while the graph runs and the other
    # changes it every step. Before the first one, not after a cache gives back
    # the wrong tensor.
    g = Graph.somatize(
        Layer(IN, MID).named("layer") >> Layer(MID, CLASSES).named("alone").frozen()
    )
    with pytest.raises(ValueError, match="settled and trained"):
        Trainer(
            g,
            objective=nn.functional.cross_entropy,
            optimizer=torch.optim.Adam(parameters(g, without={"alone"}), lr=0.1),
            trains={"alone": split()},
        )


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
            return Opaque(self.lin(lengths))

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


def two_of_them():
    """The same network twice, weight for weight."""
    torch.manual_seed(0)
    whole = net()
    other = net()
    for node_id in ("encoder", "head"):
        other.implementation(node_id).lin.load_state_dict(
            whole.implementation(node_id).lin.state_dict()
        )
    return other, whole


def test_a_cut_graph_trains_to_the_same_numbers_as_the_whole_one():
    # The best control there is: the framework did not change the arithmetic,
    # only who writes the loop. Bit for bit, because everything in between —
    # `tolist` on a float32, a detached leaf, an optimizer of its own with the
    # same rule — is the same operations in the same order.
    cut, whole = two_of_them()
    trains = {"encoder": split()}
    driven = Trainer(
        cut,
        objective=nn.functional.cross_entropy,
        optimizer=torch.optim.SGD(parameters(cut, without=trains), lr=0.1),
        trains=trains,
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
        cut.implementation("encoder").lin.weight,
        whole.implementation("encoder").lin.weight,
    )


def test_it_is_driven_in_stages_only_when_something_is_trained_where_it_runs():
    # A slice on another host with nobody training it is the step it always was:
    # the whole point of asking is to keep this out of everybody else's way.
    away = Graph.somatize(
        Layer(IN, MID).named("a") >> Layer(MID, CLASSES).named("b").at("w1")
    )
    assert not trainer(away).by_stages

    cut, _ = two_of_them()
    trains = {"encoder": split()}
    assert Trainer(
        cut,
        objective=nn.functional.cross_entropy,
        optimizer=torch.optim.SGD(parameters(cut, without=trains), lr=0.1),
        trains=trains,
    ).by_stages


def test_what_is_above_a_trained_node_gets_its_gradient_through_it():
    # Three stages, and the middle one breaks the chain by construction: `a` is
    # only trained if the gradient the trainer gives back is applied here. And if
    # it were asked for too early, `NoGradient` would call `a` an orphan.
    torch.manual_seed(0)
    g = Graph.somatize(
        Layer(IN, MID).named("a")
        >> Layer(MID, MID).named("b")
        >> Layer(MID, CLASSES).named("c")
    )
    trains = {"b": split()}
    driven = Trainer(
        g,
        objective=nn.functional.cross_entropy,
        optimizer=torch.optim.SGD(parameters(g, without=trains), lr=0.1),
        trains=trains,
    )
    before = {i: _weights(g, i) for i in g.nodes()}
    losses = [driven.step(batches(1)[0]) for _ in range(20)]

    assert losses[-1] < losses[0]
    assert all(before[i] != _weights(g, i) for i in g.nodes()), "somebody stood still"


def test_a_graph_where_everything_is_trained_elsewhere_takes_no_optimizer():
    torch.manual_seed(0)
    alone = Layer(IN, CLASSES)
    g = Graph.somatize(alone.named("alone"))
    driven = Trainer(
        g,
        objective=nn.functional.cross_entropy,
        trains={"alone": split()},
    )
    before = _weights(g, "alone")

    losses = [driven.step(batches(1)[0]) for _ in range(20)]
    assert losses[-1] < losses[0]
    assert _weights(g, "alone") != before


def test_kept_after_a_cut_is_refused():
    # A root's key comes from the input it was handed, and after a cut the roots
    # of a stage are holds handed nothing: two batches would name one thing.
    g = Graph.somatize(
        Layer(IN, MID).named("a") >> Layer(MID, CLASSES).named("b").cached()
    )
    trains = {"a": split()}
    with pytest.raises(ValueError, match="no longer knows what came before"):
        Trainer(
            g,
            objective=nn.functional.cross_entropy,
            optimizer=torch.optim.SGD(parameters(g, without=trains), lr=0.1),
            trains=trains,
        )


def _weights(graph, node_id):
    return float(graph.implementation(node_id).lin.weight.detach().abs().sum())


# ── A group of steps, and one update ──


def in_two_halves(batch):
    """One batch cut down the middle, so the two halves put back together are it.

    The control every one of these needs: accumulating over the halves has to
    come out where one step over the whole thing does, or `every` means something
    other than what it says.
    """
    x, y = batch
    half = len(x) // 2
    return (x[:half], y[:half]), (x[half:], y[half:])


def accumulating(g, every, lr=0.1, trains=None):
    return Trainer(
        g,
        objective=nn.functional.cross_entropy,
        optimizer=torch.optim.SGD(parameters(g, without=trains or {}), lr=lr),
        trains=trains,
        every=every,
    )


def test_a_group_of_steps_comes_out_where_one_step_over_all_of_them_does():
    # The whole claim, and the reason the loss is divided by the size of the
    # group: an objective that takes the mean of its batch has to be, or two
    # halves would each pull with the weight of a whole.
    torch.manual_seed(0)
    apart, together = net(), net()
    together.implementation("encoder").lin.load_state_dict(
        apart.implementation("encoder").lin.state_dict()
    )
    together.implementation("head").lin.load_state_dict(
        apart.implementation("head").lin.state_dict()
    )
    batch = batches(1)[0]

    by_halves = accumulating(apart, every=2)
    for half in in_two_halves(batch):
        by_halves.step(half)
    accumulating(together, every=1).step(batch)

    for node_id in ("encoder", "head"):
        assert torch.allclose(
            apart.implementation(node_id).lin.weight,
            together.implementation(node_id).lin.weight,
            atol=1e-6,
        ), node_id


def test_the_optimizer_moves_once_per_group_and_not_once_per_step():
    g = net()
    t = accumulating(g, every=3)
    before = g.implementation("head").lin.weight.clone()
    data = batches(3)

    t.step(data[0])
    assert torch.equal(g.implementation("head").lin.weight, before), "it moved early"
    t.step(data[1])
    assert torch.equal(g.implementation("head").lin.weight, before), "it moved early"
    t.step(data[2])
    assert not torch.equal(g.implementation("head").lin.weight, before)


def test_a_group_of_one_is_what_there_was_before_it_could_be_said():
    torch.manual_seed(0)
    said, unsaid = net(), net()
    unsaid.implementation("encoder").lin.load_state_dict(
        said.implementation("encoder").lin.state_dict()
    )
    unsaid.implementation("head").lin.load_state_dict(
        said.implementation("head").lin.state_dict()
    )
    data = batches(3)

    with_it = accumulating(said, every=1)
    without = Trainer(
        unsaid,
        objective=nn.functional.cross_entropy,
        optimizer=torch.optim.SGD(parameters(unsaid), lr=0.1),
    )

    assert [with_it.step(b) for b in data] == [without.step(b) for b in data]
    assert torch.equal(
        said.implementation("head").lin.weight,
        unsaid.implementation("head").lin.weight,
    )


def test_the_loss_it_gives_back_is_the_one_the_objective_said():
    # Divided for the backward pass, whole for whoever is reading: a history that
    # changed shape with the size of the group would be unreadable across runs.
    torch.manual_seed(0)
    g = net()
    t = accumulating(g, every=4)
    batch = batches(1)[0]

    said = t.step(batch)

    expected = nn.functional.cross_entropy(
        g.forward(Opaque(batch[0])), batch[1]
    ).item()
    assert said == pytest.approx(expected, rel=1e-5)


def test_a_group_the_epoch_ended_in_the_middle_of_is_still_a_group():
    # Three batches into groups of two: without closing it, the third would have
    # pulled nothing at all, and next epoch's first would have closed a group
    # made of two epochs.
    g = net()
    t = accumulating(g, every=2)
    before = g.implementation("head").lin.weight.clone()

    t.fit(batches(3))

    assert not torch.equal(g.implementation("head").lin.weight, before)
    assert t.seen == 0, "the epoch left a group open"


def test_closing_a_group_that_is_not_open_does_nothing_and_says_so():
    g = net()
    t = accumulating(g, every=2)

    assert t.update() is False, "nothing had been accumulated"
    t.step(batches(1)[0])
    assert t.update() is True
    assert t.update() is False


def test_a_group_is_a_whole_number_of_steps_and_at_least_one():
    for wrong in (0, -1, 1.5, True, "2"):
        with pytest.raises(ValueError, match="whole number"):
            accumulating(net(), every=wrong)


# ── And across a cut, where the far side has to count the same steps ──


def test_a_cut_graph_accumulates_in_step_with_the_whole_one():
    # The one that catches a far side out of phase: it counts its own `learn`
    # calls, so if the two groups were not the same group the weights over there
    # would move on a different step and this would not close.
    cut, whole = two_of_them()
    trains = {"encoder": split()}
    driven = accumulating(cut, every=2, trains=trains)
    in_one_go = accumulating(whole, every=2)
    data = batches(4)

    assert [driven.step(b) for b in data] == [in_one_go.step(b) for b in data]
    assert torch.equal(
        cut.implementation("encoder").lin.weight,
        whole.implementation("encoder").lin.weight,
    )


def test_the_far_side_does_not_move_until_the_group_closes_either():
    cut, _ = two_of_them()
    trains = {"encoder": split()}
    t = accumulating(cut, every=2, trains=trains)
    before = cut.implementation("encoder").lin.weight.clone()
    data = batches(2)

    t.step(data[0])
    assert torch.equal(
        cut.implementation("encoder").lin.weight, before
    ), "the one trained beside the node moved early"
    t.step(data[1])
    assert not torch.equal(cut.implementation("encoder").lin.weight, before)


def test_closing_a_group_across_a_cut_reaches_the_one_that_trains_itself():
    # `update` with a trained node elsewhere: no forward, no gradient, and the
    # fact that the group is over travels the road a gradient goes.
    cut, _ = two_of_them()
    trains = {"encoder": split()}
    t = accumulating(cut, every=4, trains=trains)
    before = cut.implementation("encoder").lin.weight.clone()

    t.step(batches(1)[0])
    assert torch.equal(cut.implementation("encoder").lin.weight, before)

    assert t.update() is True
    assert not torch.equal(cut.implementation("encoder").lin.weight, before)


def test_a_technique_that_names_its_own_group_wins_over_the_trainers():
    # Two numbers is a thing somebody meant, not a thing nobody noticed: the
    # trainer's is the default for whoever did not say.
    cut, _ = two_of_them()
    trains = {"encoder": Split(torch.optim.SGD, lr=0.1, every=3)}
    t = accumulating(cut, every=2, trains=trains)

    assert t.every == 2
    assert trains["encoder"].every == 3


# ── A batch that does not fit, cut into pieces ──


def with_micro(g, micro, every=1, lr=0.1, trains=None):
    return Trainer(
        g,
        objective=nn.functional.cross_entropy,
        optimizer=torch.optim.SGD(parameters(g, without=trains or {}), lr=lr),
        trains=trains,
        every=every,
        micro=micro,
    )


def test_a_batch_in_pieces_comes_out_where_the_whole_one_does():
    # The claim, and the same control as accumulation's: cutting a batch changes
    # how much has to fit at once and nothing else.
    torch.manual_seed(0)
    cut, whole = net(), net()
    for node_id in ("encoder", "head"):
        whole.implementation(node_id).lin.load_state_dict(
            cut.implementation(node_id).lin.state_dict()
        )
    batch = batches(1)[0]

    with_micro(cut, micro=2).step(batch)
    with_micro(whole, micro=1).step(batch)

    for node_id in ("encoder", "head"):
        assert torch.allclose(
            cut.implementation(node_id).lin.weight,
            whole.implementation(node_id).lin.weight,
            atol=1e-6,
        ), node_id


def test_the_optimizer_still_moves_once_a_step():
    g = net()
    t = with_micro(g, micro=3)  # six rows into three
    before = g.implementation("head").lin.weight.clone()

    t.step(batches(1)[0])

    assert not torch.equal(g.implementation("head").lin.weight, before)
    assert t.seen == 0, "the step left a group half open"


def test_the_two_multiply_rather_than_compete():
    # `every=2, micro=2` is eight halves of a batch to a group... no: four. The
    # point is that neither of them is ignored.
    g = net()
    t = with_micro(g, micro=2, every=2)
    before = g.implementation("head").lin.weight.clone()
    data = batches(2)

    assert t.pieces == 4
    t.step(data[0])
    assert torch.equal(g.implementation("head").lin.weight, before), "it moved early"
    t.step(data[1])
    assert not torch.equal(g.implementation("head").lin.weight, before)


def test_the_loss_it_gives_back_is_still_the_batch_it_was_handed():
    # The mean of what the pieces said, which for equal pieces is the number the
    # whole batch would have said: a history stays comparable across `micro`.
    torch.manual_seed(0)
    one = net()
    torch.manual_seed(0)
    other = net()
    batch = batches(1)[0]

    said = with_micro(one, micro=1).step(batch)
    in_pieces = with_micro(other, micro=3).step(batch)

    assert in_pieces == pytest.approx(said, rel=1e-5)


def test_a_batch_that_does_not_divide_is_refused_with_both_numbers():
    # Found by running it: `chunk` gives **at most** what it is asked for, so six
    # rows into four is three pieces — and a group that counts four while three
    # run never closes. Across a cut it is worse: the far side counts what it
    # sees and the two fall out of step in silence.
    g = net()
    t = with_micro(g, micro=4)

    with pytest.raises(ValueError) as e:
        t.step(batches(1)[0])  # six rows

    assert "micro=4" in str(e.value) and "6 long" in str(e.value)
    assert "drop_last" in str(e.value)


def test_something_it_cannot_cut_is_refused_with_its_type_and_which_half():
    g = net()
    t = with_micro(g, micro=2)

    with pytest.raises(TypeError) as e:
        t.step(("a batch of text", batches(1)[0][1]))

    assert "`str`" in str(e.value)
    assert "input" in str(e.value)


def test_an_input_and_a_target_of_different_lengths_are_refused():
    # Both of them divide by two and they still do not go together: a piece of
    # one has to be a piece of the other.
    g = net()
    t = with_micro(g, micro=2)

    with pytest.raises(ValueError, match="have to line up"):
        t.step((torch.randn(8, IN), torch.randint(0, CLASSES, (4,))))


def test_micro_is_a_whole_number_of_pieces_and_at_least_one():
    for wrong in (0, -1, 1.5, True, "2"):
        with pytest.raises(ValueError, match="`micro` is a count"):
            with_micro(net(), micro=wrong)


def test_and_across_a_cut_the_far_side_counts_the_pieces_too():
    # The phase argument again: a piece is a `learn` over there, so the far side
    # has to make its group out of pieces and not out of steps. Out by a factor
    # of `micro` and the two optimizers move on different steps.
    cut, whole = two_of_them()
    trains = {"encoder": split()}
    driven = with_micro(cut, micro=2, trains=trains)
    in_one_go = with_micro(whole, micro=2)
    data = batches(3)

    assert [driven.step(b) for b in data] == [in_one_go.step(b) for b in data]
    assert torch.equal(
        cut.implementation("encoder").lin.weight,
        whole.implementation("encoder").lin.weight,
    )
