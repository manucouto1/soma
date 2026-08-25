"""A run, drawn — while it happens and after it is over.

The property everything here rests on is that those two are **one figure**: a
fact read back off a store is the very dict a watcher was handed, so `Live` and
`progress` fill the same drawing rather than slowly stopping to agree.
"""

import math

import pytest

from soma_next import Graph, Node, Recorder, Store, _theme
from soma_next import _figure as graph_figure
from soma_next.record import Live, progress, spent
from soma_next.record import _figure as drawn

pytest.importorskip("plotly")


class Add(Node):
    def __init__(self, how_much):
        self.how_much = how_much

    def forward(self, x, ctx):
        return x + self.how_much


class Boom(Node):
    def forward(self, x, ctx):
        raise ValueError("I broke")


@pytest.fixture
def g():
    return Graph.somatize(Add(1).named("a") >> Add(10).named("b"))


def series(figure, named):
    (trace,) = [one for one in figure.data if one.name == named]
    return list(trace.y)


def run(g, store, live=None, steps=20):
    """A run written down and, if there is one, watched live at the same time."""
    recorder = Recorder(store, run="tuesday", summarising=["loss"])
    for step in range(steps):
        g.forward(0.0, watching=[recorder, live] if live else recorder)
        loss = {"fact": "loss", "value": math.exp(-step / 6)}
        recorder(loss)
        if live:
            live(loss)
    return recorder


# ── One figure, two sources ──


def test_live_and_read_back_draw_the_same_thing(g, tmp_path):
    store, live = Store(str(tmp_path)), Live(title="tuesday")
    run(g, store, live)

    there = progress(store, run="tuesday")
    here = live.figure()
    for named in ("loss", "loss, smoothed", "took"):
        assert series(here, named) == pytest.approx(series(there, named))


def test_live_keeps_one_row_per_forward_and_not_one_per_fact(g, tmp_path):
    # Watching a run for an afternoon must not grow with the facts, only with
    # the steps: a `forward` of a hundred nodes is one row either way.
    live = Live()
    run(g, Store(str(tmp_path)), live, steps=12)

    assert len(live.rows) == 12


# ── The smoothing, which is where a figure could start lying ──


def test_the_smoothed_line_stays_inside_what_was_measured():
    # A mean cannot leave the range of what it averages. A spline through the
    # points could, and a loss curve dipping below a minimum that never happened
    # is exactly the kind of lie this project's figures may not tell.
    raw = [2.0, 0.1, 1.7, 0.3, 1.5, 0.2, 1.1]

    smoothed = drawn._smoothed(raw, window=3)

    assert min(smoothed) >= min(raw)
    assert max(smoothed) <= max(raw)


def test_the_mean_is_centred_and_not_trailing():
    # A trailing mean is the same curve shifted right, and drawn over the raw
    # series that shift reads as the smoothing disagreeing with the measurement.
    # Nothing is being predicted — every point is already in hand.
    raw = [0.0] * 5 + [1.0] * 5

    smoothed = drawn._smoothed(raw, window=5)

    assert smoothed[4] == pytest.approx(0.4), "two of the five ahead are ones"
    assert smoothed[5] == pytest.approx(0.6), "and symmetric across the step"


def test_asking_for_no_smoothing_gives_back_what_was_measured():
    raw = [3.0, 1.0, 2.0]

    assert drawn._smoothed(raw, window=0) == raw


def test_a_forward_with_no_loss_is_a_gap_and_not_a_zero(g, tmp_path):
    # `None` is a break in the line; zero would be a step down to nothing, which
    # on a loss curve reads as the best result of the run.
    store = Store(str(tmp_path))
    recorder = Recorder(store, run="tuesday", summarising=["loss"])
    g.forward(0.0, watching=recorder)
    recorder({"fact": "loss", "value": 0.5})
    g.forward(0.0, watching=recorder)

    assert series(progress(store, run="tuesday"), "loss") == [0.5, None]


# ── What the figure says happened ──


def test_a_forward_that_broke_is_marked_and_only_then_is_it_in_the_legend(tmp_path):
    fine, store = Graph.somatize(Add(1).named("a")), Store(str(tmp_path))
    run(fine, store, steps=2)
    (quiet,) = [one for one in progress(store, run="tuesday").data if one.name == "broke"]
    assert quiet.showlegend is False, "nothing broke, so it is not in the legend"

    broken = Graph.somatize(Add(1).named("a") >> Boom().named("boom"))
    with pytest.raises(Exception):
        broken.forward(0.0, watching=Recorder(store, run="wednesday"))

    (marked,) = [one for one in progress(store, run="wednesday").data if one.name == "broke"]
    assert list(marked.x) == [0]
    assert marked.showlegend is True


def test_the_title_says_what_the_figure_is_showing(g, tmp_path):
    store = Store(str(tmp_path))
    run(g, store, steps=3)

    assert "3 forwards" in progress(store, run="tuesday").layout.title.text


def test_a_node_is_coloured_by_where_it_ran_and_by_nothing_else(tmp_path):
    # The same rule the graph's fill obeys, out of the same table: a device is
    # green, another machine is orange. Not fast-or-slow, not good-or-bad.
    on_device = Graph.somatize(Add(1).named("a").on("cuda:0") >> Add(2).named("b"))
    store = Store(str(tmp_path))
    on_device.forward(0.0, watching=Recorder(store, run="tuesday"))

    bars = spent(store, run="tuesday").data[0]
    where = dict(zip(bars.y, bars.marker.color))
    assert where["a"] == _theme.PALETTE["cuda"][0]
    assert where["b"] == _theme.PALETTE["cpu"][0]


# ── One product ──


def test_the_graph_and_the_run_are_drawn_from_the_same_table():
    # A library whose graph is light and whose curves are dark is two libraries,
    # and the rule CU19 wrote about one table applied one level up the moment
    # there was a second figure.
    assert graph_figure.PALETTE is _theme.PALETTE


def test_without_plotly_drawing_says_how_to_get_it(g, tmp_path, monkeypatch):
    store = Store(str(tmp_path))
    run(g, store, steps=1)

    def no_plotly():
        raise RuntimeError("drawing needs plotly")

    monkeypatch.setattr(_theme, "plotly", no_plotly)
    with pytest.raises(RuntimeError, match="needs plotly"):
        progress(store, run="tuesday")


def test_a_live_view_outside_a_notebook_reads_as_nothing(g, tmp_path):
    # The same wall CU19 hit: plotly answers `{}` with no renderer configured,
    # and an empty bundle has to become `None` or the cell shows neither a
    # figure nor a `repr`.
    live = Live()
    run(g, Store(str(tmp_path)), live, steps=2)

    assert live._repr_mimebundle_() is None
