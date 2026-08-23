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
from soma_next.record import forwards  # noqa: E402
from soma_next.torch import Audit, Trainer, architecture, parameters, probe  # noqa: E402

MACRO = ("VANISHING", "EXPLODING", "DEAD", "SATURATED", "NAN", "INF", "LEAKAGE")


class Block(Node):
    """One layer and a non-linearity, which is all any of these need."""

    def __init__(self, width=16, activation="relu", bias=None, gain=None, norm=False):
        #: Before the layer and not after it, which is where a normalisation
        #: resets the scale a probe measures against.
        self.norm = torch.nn.LayerNorm(width) if norm else None
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
        return Opaque(self.after(self.net(self.norm(x) if self.norm else x)))

    def parameters(self):
        held = list(self.net.parameters())
        return held + list(self.norm.parameters()) if self.norm else held


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

    made = architecture(g, Opaque(torch.randn(4, 16)))

    # Everything and not only what has parameters: a picture of a sigmoid stack
    # that leaves out the sigmoids is a picture of something else.
    kinds = [one.kind for one in made["b0"].layers]
    assert "learned" in kinds and "activation" in kinds


def test_a_skip_connection_is_an_edge_and_not_an_order():
    # The thing a list of children cannot show, and the reason the inside is
    # traced rather than listed.
    pytest.importorskip("plotly")
    from soma_next.torch._inside import traced

    class Residual(torch.nn.Module):
        def __init__(self, width=16):
            super().__init__()
            self.lin = torch.nn.Linear(width, width)

        def forward(self, x):
            return x + self.lin(x)

    inside = traced(Residual(), torch.randn(2, 16))

    joins = [b for _, b in inside.edges if b.endswith("add")]
    assert len(joins) == 2, f"a residual joins two paths: {inside.edges}"


def test_a_bottleneck_is_visible_in_the_shapes():
    pytest.importorskip("plotly")
    from soma_next.torch._inside import traced

    class Squeeze(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.down = torch.nn.Linear(64, 4)
            self.up = torch.nn.Linear(4, 64)

        def forward(self, x):
            return self.up(torch.relu(self.down(x)))

    inside = traced(Squeeze(), torch.randn(2, 64))

    shapes = [one.shape for one in inside.layers if one.shape]
    assert any(one.endswith("×4") for one in shapes), shapes
    assert any(one.endswith("×64") for one in shapes), shapes


def test_a_module_fx_cannot_trace_is_still_drawn_and_says_how():
    # A residual that is missing looks exactly like a residual that is not
    # there, so which path answered has to be on the figure.
    pytest.importorskip("plotly")
    from soma_next.torch._inside import traced

    class Loopy(torch.nn.Module):
        """Control flow that depends on the values, which `fx` cannot follow."""

        def __init__(self):
            super().__init__()
            self.cell = torch.nn.Linear(8, 8)

        def forward(self, x):
            for _ in range(int(x.shape[1]) // 8):
                x = self.cell(x)
            return x

    inside = traced(Loopy(), torch.randn(2, 8))

    assert inside.how == "traced"
    assert inside.why, "it has to say why the symbolic path was not used"


def test_what_is_drawn_is_a_superset_of_what_is_measured(store):
    # Every layer that can carry a flag has a box. The other way round is fine —
    # a `Sigmoid` has no gradient of its own to report.
    pytest.importorskip("plotly")
    from soma_next.torch import architecture

    torch.manual_seed(0)
    g, _ = chain([Block(activation="sigmoid") for _ in range(3)])
    trained(g, store, steps=8, auditing=Audit(inside=True))

    drawn = {
        f"{node}.{one.path}" for node, made in architecture(g, Opaque(torch.randn(4, 16))).items()
        for one in made.layers
    }
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


def test_a_composite_everybody_recognises_is_one_box():
    # A `TransformerEncoderLayer` read as its fourteen leaves is fourteen things
    # and a diagram nobody looks at twice.
    pytest.importorskip("plotly")
    from soma_next.torch._inside import _worth_drawing

    block = torch.nn.TransformerEncoderLayer(16, 4, 32, batch_first=True)

    drawn = _worth_drawing(block)

    assert [path for path, _ in drawn] == [""], drawn
    assert len(_worth_drawing(block, depth=1)) > 1, "and `depth=` opens it"


def test_blocks_that_are_the_same_block_collapse_to_one_and_a_count():
    # Twelve identical layers drawn twelve times is a figure nobody reads.
    pytest.importorskip("plotly")
    from soma_next.torch import architecture

    class Deep(Node):
        def __init__(self, width=16, how_many=5):
            self.body = torch.nn.Sequential(
                *[
                    m
                    for _ in range(how_many)
                    for m in (torch.nn.Linear(width, width), torch.nn.ReLU())
                ]
            )

        def forward(self, x, ctx):
            return Opaque(self.body(x))

        def parameters(self):
            return list(self.body.parameters())

    g = Graph.somatize(Deep().named("deep"))

    made = architecture(g, Opaque(torch.randn(2, 16)))

    said = [one.label for one in made["deep"].layers]
    assert any("×5" in one for one in said), said
    assert len(made["deep"].layers) < 10, "five pairs, drawn once"


def test_what_comes_after_a_stack_is_not_adopted_by_its_last_block():
    # Found by looking at a picture: the `Linear` after four encoder layers was
    # being pulled into the fourth, so four identical blocks came out as three
    # and an odd one.
    pytest.importorskip("plotly")
    from soma_next.torch import architecture

    class Stack(Node):
        def __init__(self, width=16):
            self.body = torch.nn.Sequential(
                *[torch.nn.Linear(width, width) for _ in range(4)]
            )
            self.out = torch.nn.Linear(width, 2)

        def forward(self, x, ctx):
            return Opaque(self.out(self.body(x)))

        def parameters(self):
            return list(self.body.parameters()) + list(self.out.parameters())

    g = Graph.somatize(Stack().named("net"))

    made = architecture(g, Opaque(torch.randn(2, 16)))

    said = [one.label for one in made["net"].layers]
    assert said == ["Linear  ×4", "Linear"], said


def test_a_tensor_nobody_holds_cannot_invent_an_edge():
    # CPython reuses an id the moment the object behind it is freed, and an
    # intermediate nobody holds is freed at once. A later tensor landing on a
    # dead one's id draws an edge that never existed — worse than a missing one,
    # because a missing edge looks like a missing edge.
    pytest.importorskip("plotly")
    from soma_next.torch import architecture

    class Apart(Node):
        """Two halves with something functional between them, so the second's
        input has no producer any hook saw."""

        def __init__(self, width=16):
            self.first = torch.nn.Linear(width, width)
            self.second = torch.nn.Linear(width, 2)

        def forward(self, x, ctx):
            return Opaque(self.second(self.first(x).relu().mean(0, keepdim=True)))

        def parameters(self):
            return list(self.first.parameters()) + list(self.second.parameters())

    g = Graph.somatize(Apart().named("net"))

    made = architecture(g, Opaque(torch.randn(4, 16)))

    assert made["net"].edges == [("first", "second")], made["net"].edges


def test_a_shape_says_what_each_of_its_numbers_is():
    # Three numbers and no way to tell which is the batch, which is time and
    # which is the width is what makes a shape useless at a glance.
    pytest.importorskip("plotly")
    from soma_next.torch import architecture

    class Conv(Node):
        def __init__(self):
            self.stem = torch.nn.Conv1d(1, 8, 3, padding=1)
            self.act = torch.nn.GELU()

        def forward(self, said, ctx):
            return Opaque(self.act(self.stem(said)))

        def parameters(self):
            return list(self.stem.parameters())

    g = Graph.somatize(Conv().named("audio"))

    made = architecture(g, Opaque(torch.randn(4, 1, 16)))

    stem = next(one for one in made["audio"].layers if one.label.startswith("Conv"))
    assert stem.dims == ("batch", "ch", "len")


def test_something_that_did_not_change_the_shape_keeps_the_names():
    # A `BatchNorm1d` in a convolutional trunk produces `(batch, channels,
    # length)` because that is what it was handed; naming it by its own kind
    # gets the right words for the wrong tensor.
    pytest.importorskip("plotly")
    from soma_next.torch import architecture

    class Trunk(Node):
        def __init__(self):
            self.body = torch.nn.Sequential(
                torch.nn.Conv1d(1, 8, 3, padding=1),
                torch.nn.BatchNorm1d(8),
                torch.nn.AdaptiveAvgPool1d(1),
            )

        def forward(self, said, ctx):
            return Opaque(self.body(said))

        def parameters(self):
            return list(self.body.parameters())

    g = Graph.somatize(Trunk().named("audio"))

    made = architecture(g, Opaque(torch.randn(4, 1, 16)))
    said = {one.label.split()[0]: one.dims for one in made["audio"].layers}

    assert said["BatchNorm1d"] == ("batch", "ch", "len")
    # And a pooling layer keeps them too: it changes what the numbers are, not
    # what they mean.
    assert said["AdaptiveAvgPool1d"] == ("batch", "ch", "len")


def test_a_recurrent_cell_says_its_output_and_not_its_hidden_state():
    # `(output, h_n)` reversed shows the hidden state where the output belongs,
    # which is a wrong number written confidently on a figure.
    pytest.importorskip("plotly")
    from soma_next.torch import architecture

    class Cell(Node):
        def __init__(self):
            self.gru = torch.nn.GRU(4, 8, batch_first=True)

        def forward(self, said, ctx):
            return Opaque(self.gru(said)[0][:, -1])

        def parameters(self):
            return list(self.gru.parameters())

    g = Graph.somatize(Cell().named("vitals"))

    (one,) = architecture(g, Opaque(torch.randn(4, 6, 4)))["vitals"].layers

    assert one.shape == "4×6×8", "the output, not the 1×4×8 hidden state"
    assert one.dims == ("batch", "steps", "dim")


def test_depth_counts_composites_opened_and_not_names():
    # A `TransformerEncoderLayer` sits three names deep inside a
    # `TransformerEncoder`, and asking for one level of detail should not have
    # to know that.
    pytest.importorskip("plotly")
    from soma_next.torch import architecture

    class Stack(Node):
        def __init__(self):
            self.body = torch.nn.TransformerEncoder(
                torch.nn.TransformerEncoderLayer(8, 2, 16, batch_first=True), num_layers=2
            )

        def forward(self, said, ctx):
            return Opaque(self.body(said))

        def parameters(self):
            return list(self.body.parameters())

    g = Graph.somatize(Stack().named("text"))
    x = Opaque(torch.randn(4, 5, 8))

    whole = architecture(g, x)["text"]
    opened = architecture(g, x, depth=1)["text"]

    assert len(whole.layers) == 1, "one box, and it says what is in it"
    assert whole.layers[0].made_of
    assert len(opened.layers) > 4, "and `depth=1` opens it"


# ── Before a step is taken ──


def test_a_probe_is_one_forward_that_was_recorded_and_never_trained(store):
    # Not a metaphor for the record's benefit: literally `run/<id>/0`, which is
    # why `diagnose`, `seen`, `profile` and `overlaid` read a probe without
    # knowing one exists. Nothing was added to the record's shape.
    torch.manual_seed(0)
    g, _ = chain([Block(gain=4.0) for _ in range(10)])

    probe(g, torch.randn(32, 16), watching=Recorder(store, run="before"))

    assert [one["forward"] for one in forwards(store, run="before")] == [0]
    assert diagnose(store, run="before"), "a stack this hot is not healthy"


def test_nothing_is_trained_and_no_weight_moves(store):
    torch.manual_seed(0)
    g, _ = chain([Block(gain=4.0) for _ in range(4)])
    before = [p.detach().clone() for p in parameters(g)]

    probe(g, torch.randn(32, 16), watching=Recorder(store, run="before"))

    assert all(torch.equal(a, b) for a, b in zip(before, parameters(g)))


def test_a_signal_growing_where_nothing_normalises_it_is_found_before_a_step(store):
    torch.manual_seed(0)
    g, _ = chain([Block(gain=4.0) for _ in range(10)])

    probe(g, torch.randn(32, 16), watching=Recorder(store, run="before"))
    said = diagnose(store, run="before")

    assert any("MISSING_NORMALISATION" in flags for flags in said.values())


def test_and_the_same_stack_normalised_says_nothing_about_it(store):
    # The conjunction, and both halves are load-bearing. Structure alone —
    # "there is no norm layer in this stretch" — would have flagged the stack
    # above and every healthy one beside it.
    torch.manual_seed(0)
    g, _ = chain([Block(gain=4.0, norm=True) for _ in range(10)])

    probe(g, torch.randn(32, 16), watching=Recorder(store, run="before"))
    said = diagnose(store, run="before")

    assert not any("MISSING_NORMALISATION" in flags for flags in said.values())


def test_a_signal_that_shrank_says_nothing_because_that_is_what_was_measured(store):
    # A plain stack whose output arrives a fraction of the size it went in
    # trains as well as a healthy one: Adam is scale-invariant per parameter.
    # The flag has one side and the measurement is `health/tests/normalisation.py`.
    # The bias has to go, or it is the floor: `Wx * 0.2 + b` stops shrinking as
    # soon as `b` is the bigger half, and what would be measured is the bias.
    torch.manual_seed(0)
    g, _ = chain([Block(gain=0.2, bias=0.0) for _ in range(10)])

    read = probe(g, torch.randn(32, 16), watching=Recorder(store, run="before"))
    said = diagnose(store, run="before")

    assert min(one["signal_gain"] for one in read.values()) < 1e-9
    assert not any("MISSING_NORMALISATION" in flags for flags in said.values())


def test_everything_a_probe_measures_has_a_box(store):
    # The same invariant the audit keeps — *every layer that can carry a flag
    # has a box* — and the probe gets it by construction, because it takes its
    # scope from what the figure will draw rather than walking the modules
    # itself. Those are not the same walk: at `depth=1` a module walk opens a
    # composite the figure keeps whole, and a finding on a layer with no box
    # lands nowhere.
    pytest.importorskip("plotly")

    class Held(Node):
        def __init__(self, width=16):
            self.body = torch.nn.Sequential(
                torch.nn.TransformerEncoderLayer(width, 2, 32, batch_first=True),
                torch.nn.Linear(width, width))

        def forward(self, x, ctx):
            return Opaque(self.body(x))

        def parameters(self):
            return list(self.body.parameters())

    torch.manual_seed(0)
    g = Graph.somatize(Held().named("enc"))
    x = torch.randn(4, 6, 16)

    for depth in (0, 1):
        measured = set(probe(g, x, depth=depth))
        made = architecture(g, Opaque(x), depth=depth)
        drawn = {f"{node}.{one.path}" for node, inside in made.items()
                 for one in inside.layers}
        folded = {f"{node}.{path}" for node, inside in made.items()
                  for path in inside.folded}

        assert measured, f"depth={depth} measured nothing"
        assert measured <= drawn | folded, measured - drawn - folded


def test_the_backward_signal_falls_away_with_depth_before_an_optimizer_exists():
    # The vanishing picture with no loss, no target and no step: `jacobian_gain`
    # is the factor a gradient at the output arrives by, so it is a ratio and it
    # means the same thing at every depth.
    torch.manual_seed(0)
    g, _ = chain([Block(activation="sigmoid") for _ in range(10)])

    read = probe(g, torch.randn(32, 16))

    assert read["b0.net"]["jacobian_gain"] < read["b9.net"]["jacobian_gain"] / 100


def test_every_flag_a_probe_raises_says_what_to_do_about_it(store):
    # The same guard the run has, and it is here because a name goes back to its
    # variant through a **list** rather than a `match`: the compiler does not
    # keep that one, so a flag only a probe can raise is a flag `about` can be
    # missing without anything failing to compile. It was, once.
    torch.manual_seed(0)
    g, _ = chain([Block(gain=4.0) for _ in range(10)])
    probe(g, torch.randn(32, 16), watching=Recorder(store, run="before"))

    raised = {flag for flags in diagnose(store, run="before").values() for flag in flags}

    assert raised
    assert all(about(flag) for flag in raised)


def test_a_node_holding_no_modules_is_not_probed():
    class Doubles(Node):
        def forward(self, x, ctx):
            return Opaque(x * 2)

    g = Graph.somatize(Doubles().named("twice") >> Block().named("b0"))

    read = probe(g, torch.randn(8, 16))

    assert not any(where.startswith("twice") for where in read)


def test_a_normalisation_resets_what_the_gain_is_measured_from(store):
    # The half of the conjunction that lives in the measurement. Fed data whose
    # scale is nowhere near one, a stack measured from the **input** reads a
    # thousandfold drift and cries about a normalisation that is right there.
    # Measuring from the last norm reads one, which is what is true.
    torch.manual_seed(0)
    g, _ = chain([Block(norm=True) for _ in range(6)])

    probe(g, torch.randn(32, 16) * 1e-3, watching=Recorder(store, run="before"))
    said = diagnose(store, run="before")

    assert not any("MISSING_NORMALISATION" in flags for flags in said.values()), said


def test_a_node_whose_modules_never_ran_is_said_out_loud():
    # A node quietly absent from a diagnosis reads exactly like a healthy one,
    # which is the mistake `Seen` spends its whole docstring avoiding. The case
    # that matters is a slice on another machine: the hooks are registered here
    # and its modules run over there, so it contributes nothing. It is
    # `architecture` that says so, because the probe now takes its scope from
    # there — one warning for one fact, and it arrives either way.
    class Holds(Node):
        def __init__(self):
            self.unused = torch.nn.Linear(16, 16)

        def forward(self, x, ctx):
            return Opaque(x)

    g = Graph.somatize(Block().named("b0") >> Holds().named("idle"))

    with pytest.warns(UserWarning, match="idle"):
        read = probe(g, torch.randn(8, 16))

    assert not any(where.startswith("idle") for where in read)


def test_a_repeated_block_of_several_layers_puts_its_count_on_the_block():
    # Four encoder layers opened up are eight boxes each saying `×4`: the count
    # said eight times and the block itself said none.
    pytest.importorskip("plotly")

    class Stack(Node):
        def __init__(self, width=16, layers=4):
            self.body = torch.nn.TransformerEncoder(
                torch.nn.TransformerEncoderLayer(width, 2, 32, batch_first=True), layers
            )

        def forward(self, x, ctx):
            return Opaque(self.body(x))

        def parameters(self):
            return list(self.body.parameters())

    g = Graph.somatize(Stack().named("enc"))

    made = architecture(g, Opaque(torch.randn(2, 5, 16)), depth=1)["enc"]

    assert made.groups, "an opened composite that repeats is a block"
    assert [count for _, count in made.groups.values()] == [4]
    assert not any("×" in one.label for one in made.layers), [o.label for o in made.layers]
    assert all(one.block in made.groups for one in made.layers)


def test_and_a_block_that_is_one_layer_keeps_its_count_inline():
    # A frame around one box is a frame saying nothing a word could not.
    pytest.importorskip("plotly")

    class Deep(Node):
        def __init__(self, width=16, how_many=5):
            self.body = torch.nn.Sequential(
                *[torch.nn.Linear(width, width) for _ in range(how_many)]
            )

        def forward(self, x, ctx):
            return Opaque(self.body(x))

        def parameters(self):
            return list(self.body.parameters())

    g = Graph.somatize(Deep().named("deep"))

    made = architecture(g, Opaque(torch.randn(2, 16)))["deep"]

    assert not made.groups
    assert any("×5" in one.label for one in made.layers), [o.label for o in made.layers]


def test_how_many_lanes_a_layer_runs_is_read_and_never_inferred():
    # Torch packs the heads of a `MultiheadAttention` into one `in_proj_weight`
    # and a reshape, so there is no second module to find. What is drawn is the
    # count on the one box that exists, and four boxes wired together would be a
    # graph nobody built.
    pytest.importorskip("plotly")

    class Attends(Node):
        def __init__(self, width=16, heads=4):
            self.attn = torch.nn.MultiheadAttention(width, heads, batch_first=True)

        def forward(self, x, ctx):
            return Opaque(self.attn(x, x, x)[0])

        def parameters(self):
            return list(self.attn.parameters())

    g = Graph.somatize(Attends().named("a"))

    made = architecture(g, Opaque(torch.randn(2, 5, 16)))["a"]
    one = made.layers[0]

    assert one.parallel == 4
    assert one.made_of == "4 heads", one.made_of
    assert len(made.layers) == 1, "one module, one box"


def test_something_that_runs_one_lane_says_nothing_about_lanes():
    pytest.importorskip("plotly")
    g, _ = chain([Block() for _ in range(2)])

    made = architecture(g, Opaque(torch.randn(2, 16)))

    assert all(one.parallel is None for inside in made.values() for one in inside.layers)
