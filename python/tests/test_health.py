"""Whether what happened was healthy — which is an **opinion** about the record.

The test that matters is the invariant CU19 wrote down and this is where it
stops being an aspiration:

> a diagnosis has to be reproducible from the stored record, without training
> again.

Everything else here is the taxonomy: each pathology is **built** and has to be
caught at default thresholds, plus a guard that says the detectors do not cry
wolf on a network that is fine. That shape is inherited from the original's
`test_pathologies.py`, which is the closest thing this project has to an
executable specification of what health means.
"""

import pytest

torch = pytest.importorskip("torch")

import soma_next.torch  # noqa: E402, F401
from soma_next import Graph, Node, Opaque, Recorder, Store  # noqa: E402
from soma_next.health import Thresholds, about, diagnose, history, seen  # noqa: E402
from soma_next.torch import Audit, Trainer, parameters  # noqa: E402

MACRO = ("VANISHING", "EXPLODING", "DEAD", "SATURATED", "NAN", "INF", "LEAKAGE")


class Block(Node):
    """One layer and a non-linearity, which is all any of these need."""

    def __init__(self, width=16, activation="relu", bias=None, gain=None):
        self.net = torch.nn.Linear(width, width)
        if bias is not None:
            torch.nn.init.constant_(self.net.bias, bias)
        if gain is not None:
            with torch.no_grad():
                self.net.weight *= gain
        self.after = {
            "relu": torch.nn.ReLU(),
            "sigmoid": torch.nn.Sigmoid(),
            "tanh": torch.nn.Tanh(),
            "none": torch.nn.Identity(),
        }[activation]

    def forward(self, x, ctx):
        return Opaque(self.after(self.net(x)))

    def parameters(self):
        return list(self.net.parameters())


def chain(blocks):
    """A graph of those blocks in a row, named `b0`, `b1`, ..."""
    named = [block.named(f"b{i}") for i, block in enumerate(blocks)]
    wired = named[0]
    for one in named[1:]:
        wired = wired >> one
    return Graph.somatize(wired), [f"b{i}" for i in range(len(named))]


def trained(g, store, *, run="a-run", steps=20, lr=0.05, width=16, auditing=True):
    """A short run, audited and written down."""
    t = Trainer(
        g,
        objective=torch.nn.functional.mse_loss,
        optimizer=torch.optim.SGD(parameters(g), lr=lr),
        auditing=auditing,
        watching=Recorder(store, run=run),
    )
    for _ in range(steps):
        t.step((torch.randn(32, width), torch.randn(32, width)))
    return t


@pytest.fixture
def store(tmp_path):
    return Store(str(tmp_path))


# ── The invariant ──


def test_a_diagnosis_is_taken_from_the_record_and_not_from_the_run(store):
    # No graph, no torch, no optimizer in the call: a store and a name. That is
    # what makes the third row of observability a separate thing from the
    # second, rather than a nicer way of printing it.
    torch.manual_seed(0)
    g, _ = chain([Block(activation="sigmoid") for _ in range(8)])
    trained(g, store, steps=15)
    del g

    said = diagnose(store, run="a-run")

    assert said, "a deep sigmoid stack is not healthy"


def test_the_same_record_answers_differently_under_other_thresholds(store):
    # The whole point of the split: an argument about a bound costs a scan
    # rather than an afternoon of GPU.
    torch.manual_seed(0)
    g, ids = chain([Block(activation="sigmoid") for _ in range(8)])
    trained(g, store, steps=15)

    strict = diagnose(store, run="a-run", thresholds=Thresholds(update_low=1e-2))
    lenient = diagnose(store, run="a-run", thresholds=Thresholds(update_low=1e-30))

    assert all("STALLED" in flags for flags in strict.values())
    assert not any("STALLED" in flags for flags in lenient.values())


def test_a_threshold_nobody_has_is_refused_by_name(store):
    # A bound quietly ignored is an argument somebody thinks they won.
    with pytest.raises(ValueError, match="not a threshold"):
        Thresholds(grad_lo=1e-9)


# ── The depth profile, which is what vanishing actually is ──


def test_a_deep_sigmoid_stack_starves_its_early_layers(store):
    # The classic pathology: sigma' <= 0.25 per layer, so with unit-gain init
    # the backpropagated signal shrinks geometrically with depth. It is a
    # **profile**, not a property — the last block still learns.
    torch.manual_seed(0)
    g, ids = chain([Block(activation="sigmoid") for _ in range(8)])
    trained(g, store, steps=15)

    got = seen(store, run="a-run")
    norms = [got[i]["grad_norm"] for i in ids]

    assert norms[0] < norms[-1] * 1e-4, f"no depth decay: {norms}"
    said = diagnose(store, run="a-run")
    assert "STALLED" in said[ids[0]], "the first block is not moving"
    assert ids[-1] not in said, "and the last one is fine"


def test_the_update_ratio_lands_a_healthy_layer_near_a_thousandth(store):
    # The number practice puts a healthy layer at, and the reason this was
    # worth adding: it orders the same depth profile the gradients do, but with
    # the learning rate already in it.
    torch.manual_seed(0)
    g, ids = chain([Block(activation="sigmoid") for _ in range(8)])
    trained(g, store, steps=15)

    got = seen(store, run="a-run")

    assert 1e-4 < got[ids[-1]]["update_ratio"] < 1e-2
    assert got[ids[0]]["update_ratio"] < 1e-8


# ── The other pathologies ──


def test_a_block_whose_relu_cuts_everything_off_is_dead(store):
    # A large negative bias puts every pre-activation under zero, so the layer
    # outputs nothing at all.
    torch.manual_seed(0)
    g, ids = chain([Block(activation="relu", bias=-50.0)])
    trained(g, store, steps=6)

    assert "DEAD" in diagnose(store, run="a-run")[ids[0]]


def test_a_block_pinned_at_the_far_end_of_its_range_is_saturated(store):
    torch.manual_seed(0)
    g, ids = chain([Block(activation="none", gain=200.0, bias=500.0)])
    trained(g, store, steps=6, lr=0.0)

    assert "SATURATED" in diagnose(store, run="a-run")[ids[0]]


def test_a_gradient_too_big_to_step_on_explodes(store):
    torch.manual_seed(0)
    g, ids = chain([Block(activation="none", gain=60.0) for _ in range(4)])
    trained(g, store, steps=4, lr=0.0)

    said = diagnose(store, run="a-run")
    assert any("EXPLODING" in flags for flags in said.values()), said


# ── The guard: they may not cry wolf ──


def test_a_healthy_shallow_stack_raises_nothing_macro(store):
    # tanh, three blocks, an ordinary rate. If the detectors fire here they are
    # noise, and noise is worse than nothing because somebody will turn them off.
    torch.manual_seed(0)
    g, ids = chain([Block(activation="tanh") for _ in range(3)])
    trained(g, store, steps=20, lr=0.05)

    said = diagnose(store, run="a-run")
    for node in ids:
        raised = [flag for flag in said.get(node, []) if flag.startswith(MACRO)]
        assert not raised, f"{node}: false positives {raised} — {seen(store, run='a-run')[node]}"


def test_a_node_with_no_weights_is_not_diagnosed_at_all(store):
    # Nothing to measure is not the same as nothing wrong, and it must not read
    # like a clean bill.
    class Passes(Node):
        def forward(self, x, ctx):
            # Wrapped again on the way out: what arrives unwrapped has to leave
            # wrapped, or the next edge refuses a bare tensor.
            return Opaque(x)

    torch.manual_seed(0)
    g = Graph.somatize(Block(activation="tanh").named("body") >> Passes().named("through"))
    trained(g, store, steps=6)

    assert "through" not in seen(store, run="a-run")


# ── What it costs, and what it does not change ──


def test_a_run_that_is_not_audited_says_nothing_about_health(store):
    torch.manual_seed(0)
    g, _ = chain([Block(activation="sigmoid") for _ in range(4)])
    trained(g, store, steps=6, auditing=None)

    assert seen(store, run="a-run") == {}
    assert diagnose(store, run="a-run") == {}


def test_a_cadence_measures_fewer_steps_and_says_the_same_kind_of_thing(store):
    torch.manual_seed(0)
    g, ids = chain([Block(activation="sigmoid") for _ in range(4)])
    trained(g, store, steps=12, auditing=Audit(every=4))

    drawn = history(store, run="a-run", node=ids[0], of="grad_norm")

    assert len(drawn) == 3, "twelve steps, measured every fourth"
    assert all(value > 0 for _, value in drawn)


def test_auditing_does_not_change_what_the_network_computes(store):
    # Hooks read; they do not write. Two runs from the same seed have to end at
    # the same weights whether or not anybody was looking.
    def weights(auditing):
        torch.manual_seed(0)
        g, _ = chain([Block(activation="tanh") for _ in range(3)])
        trained(g, Store(str(store)) if False else store, run=f"r{auditing}",
                steps=8, auditing=auditing)
        return torch.cat([p.detach().reshape(-1) for p in parameters(g)])

    assert torch.equal(weights(True), weights(None))


# ── What a flag says about itself ──


def test_every_flag_it_raises_says_what_to_do_about_it(store):
    torch.manual_seed(0)
    g, _ = chain([Block(activation="sigmoid") for _ in range(8)])
    trained(g, store, steps=15)

    for flags in diagnose(store, run="a-run").values():
        for flag in flags:
            assert len(about(flag)) > 20, flag


def test_something_that_is_not_a_flag_says_so():
    with pytest.raises(ValueError, match="not a flag"):
        about("SLIGHTLY_OFF")


# ── The architecture a node holds ──


def test_a_node_says_what_it_is_made_of():
    pytest.importorskip("plotly")
    from soma_next.torch import architecture

    torch.manual_seed(0)
    g, _ = chain([Block(activation="sigmoid") for _ in range(2)])

    made = architecture(g)

    # Everything and not only what has parameters: a picture of a sigmoid stack
    # that leaves out the sigmoids is a picture of something else.
    assert made["b0"] == [("net", "Linear"), ("after", "Sigmoid")]


def test_what_is_drawn_is_a_superset_of_what_is_measured(store):
    # Every layer that can carry a flag has a box. The other way round is fine —
    # a `Sigmoid` has no gradient of its own to report.
    pytest.importorskip("plotly")
    from soma_next.torch import architecture

    torch.manual_seed(0)
    g, _ = chain([Block(activation="sigmoid") for _ in range(3)])
    trained(g, store, steps=8, auditing=Audit(inside=True))

    drawn = {f"{node}.{path}" for node, made in architecture(g).items() for path, _ in made}
    measured = {one for one in seen(store, run="a-run") if "." in one}

    assert measured <= drawn, f"measured but never drawn: {measured - drawn}"


def test_the_overlay_marks_the_layer_inside_the_node(store):
    pytest.importorskip("plotly")
    from soma_next.health import overlaid

    torch.manual_seed(0)
    g, _ = chain([Block(activation="sigmoid") for _ in range(8)])
    trained(g, store, steps=15, auditing=Audit(inside=True))

    marked = overlaid(g, store, run="a-run")

    said = " ".join(n.text for n in marked.layout.annotations)
    assert "STALLED" in said
