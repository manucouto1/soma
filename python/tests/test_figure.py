"""The graph drawn: where the boxes land, and what the arrows are for.

Almost everything here is about `boxes`, which is pure and knows nothing about
plotly. That is on purpose: the geometry is the part that can be wrong in a way
a person would not notice, and it is the part that can be asserted.
"""

import json

import pytest

from conftest import Add, Identity
from soma_next import Graph, Node
from soma_next import _figure, _theme


def placed(g):
    """The boxes of a graph, by kind, for the assertions below."""
    return _figure.boxes(json.loads(g.plan_json()))


def named(boxes):
    """The node boxes, by id."""
    return {box.node: box for box in boxes if box.kind == "node"}


def test_a_wave_puts_its_branches_side_by_side(g):
    for name in ("input", "left", "right", "join"):
        g.node(name, Identity())
    for source, target in (("input", "left"), ("input", "right"), ("left", "join"), ("right", "join")):
        g.edge(source, target)

    where = named(placed(g))
    assert where["left"].y == where["right"].y
    assert where["left"].x != where["right"].x


def test_a_sequence_stacks_its_steps(g):
    g.node("one", Identity())
    g.node("two", Identity())
    g.node("three", Identity())
    g.edge("one", "two")
    g.edge("two", "three")

    where = named(placed(g))
    assert where["one"].y < where["two"].y < where["three"].y
    assert where["one"].cx == where["two"].cx == where["three"].cx


def test_a_wave_is_framed_and_a_sequence_is_not(g):
    for name in ("input", "left", "right", "join"):
        g.node(name, Identity())
    for source, target in (("input", "left"), ("input", "right"), ("left", "join"), ("right", "join")):
        g.edge(source, target)

    kinds = [box.kind for box in placed(g)]
    assert kinds.count("wave") == 1
    assert "sequence" not in kinds


def test_a_remote_frames_its_slice_and_says_the_host(g):
    g.node("here", Identity())
    g.node("away", Identity())
    g.edge("here", "away")
    g.place_at("away", "worker1")

    frames = [box for box in placed(g) if box.kind == "remote"]
    assert [frame.label for frame in frames] == ["worker1"]

    inside = named(placed(g))["away"]
    frame = frames[0]
    assert frame.x <= inside.x and inside.x + inside.w <= frame.x + frame.w
    assert frame.y <= inside.y and inside.y + inside.h <= frame.y + frame.h


def test_the_arrows_are_the_only_truth_when_the_graph_is_not_series_parallel(g):
    """The `N`: `a→c`, `a→d`, `b→d`.

    It has no series cut, so `decompose` falls back to a flat `Sequence` and the
    nesting stops saying who feeds whom — `a` and `b` come out one under the
    other although neither reads the other. Every edge still has to be drawn, and
    that is what keeps the figure honest.
    """
    for name in "abcd":
        g.node(name, Identity())
    g.edge("a", "c")
    g.edge("a", "d")
    g.edge("b", "d")

    plan = json.loads(g.plan_json())
    assert list(plan) == ["Sequence"], "the N is not series-parallel; this is the fallback"
    assert [box.kind for box in placed(g)] == ["node"] * 4, "no wave, so no frame"

    drawn = {(source, node) for node, comes_from in _figure.steps(plan) for source in comes_from}
    assert drawn == {("a", "c"), ("a", "d"), ("b", "d")}
    assert drawn == set(g.edges())


def test_every_edge_of_a_dsl_graph_is_drawn_too(g):
    a, b, c, d = (Identity().named(name) for name in ("a", "b", "c", "d"))
    built = Graph.somatize(a >> (b | c) >> d)

    plan = json.loads(built.plan_json())
    drawn = {(source, node) for node, comes_from in _figure.steps(plan) for source in comes_from}
    assert drawn == set(built.edges())


def test_an_arrow_crosses_into_a_remote_slice(g):
    g.node("here", Identity())
    g.node("away", Identity())
    g.edge("here", "away")
    g.place_at("away", "worker1")

    plan = json.loads(g.plan_json())
    assert [step for step in _figure.steps(plan)] == [("here", []), ("away", ["here"])]


def test_an_empty_graph_is_a_statement_and_not_an_exception(g):
    figure = g.figure()
    assert figure.layout.shapes == ()
    assert "empty graph" in figure.layout.annotations[0].text


def test_drawing_runs_nothing(g):
    """A node that would blow up if it ever ran, drawn without complaint."""

    class Explodes(Node):
        def forward(self, x, ctx):
            raise AssertionError("drawing must not execute anything")

    g.node("boom", Explodes())
    g.node("after", Explodes())
    g.edge("boom", "after")

    assert len(g.figure().layout.shapes) == 2


def test_a_node_named_like_a_script_tag_is_escaped(g):
    g.node("<script>alert(1)</script>", Identity())

    figure = g.figure()
    written = [note.text for note in figure.layout.annotations]
    hovered = list(figure.data[0].hovertext)
    assert all("<script>" not in text for text in written + hovered)
    assert any("&lt;script&gt;" in text for text in written)


def test_the_decisions_are_on_the_figure(g):
    g.node("one", Add(1))
    g.node("two", Identity())
    g.edge("one", "two")
    g.place("one", "cuda:0")
    g.cache("one", "salty")
    g.mapped("two")

    hovered = " ".join(g.figure().data[0].hovertext)
    assert "device: cuda:0" in hovered
    assert "salt salty" in hovered
    assert "mapped over its items" in hovered


def test_a_mapped_node_is_marked_in_its_box(g):
    g.node("one", Identity())
    g.mapped("one")

    written = " ".join(note.text for note in g.figure().layout.annotations)
    assert "mapped" in written


def test_without_plotly_the_notebook_falls_back_to_text(g, monkeypatch):
    g.node("one", Identity())

    def no_plotly():
        raise RuntimeError("drawing needs plotly")

    # On `_theme` and not on `_figure`: since there are two figures there is one
    # place that reaches for plotly, and this is it.
    monkeypatch.setattr(_theme, "plotly", no_plotly)
    assert g._repr_mimebundle_() is None
    assert repr(g) == "Graph(1 nodes, 0 edges)"

    with pytest.raises(RuntimeError, match="needs plotly"):
        g.figure()


def test_a_graph_too_big_to_read_is_not_drawn_on_its_own(g):
    for i in range(_figure.TOO_MANY + 1):
        g.node(f"n{i}", Identity())

    assert g._repr_mimebundle_() is None
    assert g.figure() is not None, "asked for by hand, it still obeys"


def test_a_small_graph_reaches_the_cell_as_a_figure(g, monkeypatch):
    """With a renderer configured — which is what a notebook has and a test run
    does not — the cell gets plotly's own mime type and not a `repr`."""
    plotly_io = pytest.importorskip("plotly.io")
    monkeypatch.setattr(plotly_io.renderers, "default", "plotly_mimetype")
    g.node("one", Identity())

    bundle = g._repr_mimebundle_()
    assert bundle is not None
    assert any("plotly" in mime for mime in bundle)


def test_a_figure_with_nothing_to_show_reads_as_nothing(g, monkeypatch):
    """Plotly answers `{}` when no renderer is configured, which is what happens
    outside a notebook. An empty bundle has to come back as `None`, or the cell
    would show neither a figure nor a `repr`."""
    g.node("one", Identity())
    monkeypatch.setattr(
        _figure,
        "figure",
        lambda graph, overlay=None, inside=None: type(
            "Blank", (), {"_repr_mimebundle_": lambda s, **kw: {}}
        )(),
    )

    assert g._repr_mimebundle_() is None


# ── An edge that would cross a node goes around it ──


def _paths(figure):
    """The routed edges: shapes drawn as an SVG path rather than a rectangle."""
    return [s for s in figure.layout.shapes if s.type == "path"]


def _n_graph():
    """`a→c`, `a→d`, `b→d` — not series-parallel, so it falls back to a flat
    sequence and `a→c` has `b` sitting between its two ends."""
    g = Graph()
    for who in ("a", "b", "c", "d"):
        g.node(who, Identity())
    g.edge("a", "c")
    g.edge("a", "d")
    g.edge("b", "d")
    return g


def test_an_edge_that_would_cross_a_node_is_routed_around_it():
    # An edge drawn over a node reads as an edge **into** that node, which is
    # the figure saying something that is not true.
    figure = _n_graph().figure()

    assert len(_paths(figure)) == 3, "all three have something in the way"


def test_and_an_edge_with_nothing_in_the_way_is_still_a_straight_arrow():
    g = Graph.somatize(Identity().named("a") >> Identity().named("b"))

    figure = g.figure()

    assert _paths(figure) == []
    assert len([n for n in figure.layout.annotations if n.showarrow]) == 1


def test_three_routed_edges_do_not_share_one_lane():
    # Without a lane each they draw over one another and the figure stops
    # saying there are three.
    figure = _n_graph().figure()

    lanes = set()
    for shape in _paths(figure):
        # `M x,y C ...` — the second point of the first curve is the lane.
        lanes.add(shape.path.split("C")[1].split()[1].split(",")[0])
    assert len(lanes) == 3


def test_a_routed_edge_runs_outside_every_box():
    boxes = _figure.boxes(json.loads(_n_graph().plan_json()), {})
    left = min(box.x for box in boxes)

    for shape in _paths(_n_graph().figure()):
        lane = float(shape.path.split("C")[1].split()[1].split(",")[0])
        assert lane < left, "a lane threaded between boxes is one that will cross a third"


def test_a_segment_is_tested_against_a_box_exactly():
    # Sampling the segment would miss a thin box, and a figure that is *usually*
    # honest is the kind of thing nobody ever finds.
    box = _figure.Box(kind="node", x=10.0, y=10.0, w=1.0, h=100.0)

    assert _figure._hits(0.0, 50.0, 20.0, 50.0, box), "straight through it"
    assert not _figure._hits(0.0, 5.0, 20.0, 5.0, box), "above it"
    assert not _figure._hits(0.0, 50.0, 9.0, 50.0, box), "stops short of it"


# ── What happened, laid over what was declared ──


def test_an_empty_overlay_draws_exactly_what_no_overlay_draws():
    # The property that lets the declaration stay drawable by somebody who has
    # never run anything, and the original left it written down.
    g = Graph.somatize(Identity().named("a") >> Identity().named("b"))

    plain, empty = g.figure(), g.figure(overlay={})

    assert plain.to_json() == empty.to_json()


def test_an_overlay_marks_the_node_without_taking_the_fill():
    # Two facts, two channels: the fill goes on saying where a node runs and
    # health is the outline. Recolouring the fill would have let *is this
    # unhealthy* eat *where does this run*.
    g = Graph.somatize(Identity().named("a") >> Identity().named("b"))

    plain = {s.fillcolor for s in g.figure().layout.shapes}
    marked = g.figure(overlay={"a": ["VANISHING"]})

    assert {s.fillcolor for s in marked.layout.shapes} == plain
    outlines = [s.line.color for s in marked.layout.shapes]
    assert _theme.SERIES["alarm"] in outlines
    assert sum(colour == _theme.SERIES["alarm"] for colour in outlines) == 1


def test_the_flags_are_written_on_the_box_and_not_counted():
    # `2 findings` is a number somebody has to go and look up, and the whole
    # point of putting it on the box is not having to.
    g = Graph.somatize(Identity().named("a"))

    said = " ".join(n.text for n in g.figure(overlay={"a": ["DEAD_CHANNELS(7)"]}).layout.annotations)

    assert "DEAD_CHANNELS" in said
    assert "(7)" not in said, "the count is on the hover, where there is room"


# ── A node opened up ──


def _inside(layers, edges=()):
    """What `architecture` answers with, built by hand: this file has no torch."""
    from soma_next.torch._inside import Inside, Layer

    return Inside([Layer(*one) for one in layers], edges, "symbolic")


def test_a_node_with_an_inside_becomes_a_frame_around_it():
    # The shape a `Wave` and a `Remote` already are, which is the sign the
    # layout was right: nothing new had to be invented for it.
    plan = json.loads(Graph.somatize(Identity().named("a")).plan_json())
    made = _inside(
        [("0", "learned", "Linear"), ("1", "activation", "ReLU")], [("0", "1")]
    )

    plain = _figure.boxes(plan)
    opened = _figure.boxes(plan, inside={"a": made})

    assert [b.kind for b in plain] == ["node"]
    assert [b.kind for b in opened] == ["node", "layer", "layer"]
    assert opened[0].h > plain[0].h, "it has to have grown to hold them"


def test_a_layer_is_drawn_by_what_it_is_and_not_by_its_name():
    # A `Linear` and a `Sigmoid` are not the same kind of thing, and drawing
    # them the same says they are.
    plan = json.loads(Graph.somatize(Identity().named("a")).plan_json())
    made = _inside([("0", "learned", "Linear"), ("1", "activation", "ReLU")])

    _, learned, activation = _figure.boxes(plan, inside={"a": made})

    assert learned.mark == "learned"
    assert activation.mark == "activation"
    assert activation.h < learned.h, "a non-linearity has nothing to live in it"


def test_what_feeds_what_decides_the_rows_and_a_skip_jumps_one():
    # A stack cannot show a skip connection. The rank is the longest way down,
    # so a skip's two ends land on rows that are not adjacent — and that gap is
    # the whole reason the picture is worth drawing.
    plan = json.loads(Graph.somatize(Identity().named("a")).plan_json())
    made = _inside(
        [
            ("x", "shaping", "input"),
            ("lin", "learned", "Linear"),
            ("add", "shaping", "+"),
        ],
        [("x", "lin"), ("lin", "add"), ("x", "add")],
    )

    _, x, lin, add = _figure.boxes(plan, inside={"a": made})

    assert x.y < lin.y < add.y, "three rows, in the order the data flows"


def test_every_layer_sits_inside_the_box_it_belongs_to():
    plan = json.loads(Graph.somatize(Identity().named("a")).plan_json())
    made = _inside([("0", "learned", "Linear"), ("1", "activation", "ReLU")])

    node, *layers = _figure.boxes(plan, inside={"a": made})

    for one in layers:
        assert node.x <= one.x and one.x + one.w <= node.x + node.w
        assert node.y <= one.y and one.y + one.h <= node.y + node.h


def test_a_layer_is_named_after_the_node_it_is_in():
    # `encoder.2`, which is the same key the audit measures under — so a layer
    # with a flag on it is a layer that has a box.
    plan = json.loads(Graph.somatize(Identity().named("encoder")).plan_json())

    _, first = _figure.boxes(plan, inside={"encoder": _inside([("2", "learned", "Linear")])})

    assert first.node == "encoder.2"


def test_the_shape_is_written_on_the_layer_because_a_bottleneck_is_shapes():
    # `512 → 8 → 512` is a picture; `Linear · Linear · Linear` is not.
    plan = json.loads(Graph.somatize(Identity().named("a")).plan_json())
    made = _inside([("0", "learned", "Linear", "2×8")])

    figure = Graph.somatize(Identity().named("a")).figure(inside={"a": made})

    assert any("2×8" in n.text for n in figure.layout.annotations)
    del plan


def test_a_flag_on_a_layer_marks_that_layer_and_not_the_others():
    g = Graph.somatize(Identity().named("encoder"))
    made = _inside(
        [("0", "learned", "Linear"), ("1", "activation", "Sigmoid"), ("2", "learned", "Linear")]
    )

    figure = g.figure(inside={"encoder": made}, overlay={"encoder.2": ["STALLED"]})

    marked = [s for s in figure.layout.shapes if s.line.color == _theme.SERIES["alarm"]]
    assert len(marked) == 1, "one layer, not the node and not its neighbours"


def test_a_node_with_no_inside_is_drawn_exactly_as_before():
    g = Graph.somatize(Identity().named("a") >> Identity().named("b"))

    assert g.figure().to_json() == g.figure(inside={}).to_json()


def test_a_branch_of_a_wave_can_be_opened_too():
    # Found by looking at a picture: the wave branch had a local called `inside`
    # of its own, so the argument never reached its children and every node in a
    # wave drew as a bare box.
    g = Graph.somatize(Identity().named("a") >> (Identity().named("l") | Identity().named("r")))
    made = _inside([("0", "learned", "Linear")])

    placed = _figure.boxes(json.loads(g.plan_json()), inside={"l": made, "r": made})

    assert [b.node for b in placed if b.kind == "layer"] == ["l.0", "r.0"]
