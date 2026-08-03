"""soma.viz figures and dataframes: asserts on figure *data* (trace
counts, arrays, monotonicity), never on rendered images."""

from __future__ import annotations

import pytest

pytest.importorskip("plotly")

import soma
from soma import Filter, Graph, Study


class _Plain(Filter):
    # `tag` gives each node its own identity. Two nodes of the same class
    # with the same config learn the same state and — `forward` being the
    # identity — see the same input, so the second one's output key equals
    # the first one's and it is served from cache. Correct behaviour, but
    # it turns a two-node run into one node timing.
    def __init__(self, tag="x"):
        super().__init__(tag=tag)

    def fit(self, x, y=None):
        return {}

    def forward(self, x, state):
        return x


def _tracked_fit(tmp_path, name="viz-run"):
    # In-memory cache so each test's run is cold: fitting consults the
    # output cache now, and these tests share one session SOMA_CACHE_DIR.
    g = Graph(cache="memory")
    g.node("a", _Plain("a"))
    g.node("b", _Plain("b"))
    g.connect("a", "b")
    with g.track_run(name, root=str(tmp_path), kind="fit") as run:
        g.fit([1.0, 2.0])
        for step in range(4):
            run.log("loss", 1.0 - 0.2 * step, step=step)
            run.log("val_f1", 0.6 + 0.1 * step, step=step)
    return soma.RunView(run.dir)


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    root = tmp_path_factory.mktemp("viz-study")

    def objective(trial):
        base = trial["x"]
        for step in range(4):
            trial.report("f1", base + 0.05 * step, step)
        return None

    study = Study(
        "viz-hpo",
        search_space=[
            {"type": "float", "name": "x", "low": 0.1, "high": 0.9},
            {"type": "categorical", "name": "opt", "choices": ["adam", "sgd"]},
        ],
        strategy="random",
        n_trials=6,
        objectives=[("f1", "maximize")],
        seed=11,
        root=str(root),
    )
    study.run(objective)
    return study


# ── study figures ───────────────────────────────────────────────────


def test_objectives_getter(study):
    assert study.objectives == [("f1", "maximize")]
    assert study.name == "viz-hpo"


def test_optimization_history_best_so_far_is_monotone(study):
    fig = study.plot_optimization_history()
    assert len(fig.data) == 2
    points, best = fig.data
    assert points.mode == "markers"
    assert len(points.y) == 6
    best_vals = list(best.y)
    assert best_vals == sorted(best_vals), "maximize → best-so-far non-decreasing"
    assert best_vals[-1] == max(points.y)
    assert fig.layout.yaxis.title.text == "f1"


def test_intermediate_values_highlights_best(study):
    fig = study.plot_intermediate_values()
    assert len(fig.data) == 6, "one curve per trial"
    widths = [t.line.width for t in fig.data]
    assert widths.count(3) == 1, "exactly one highlighted best trial"
    for trace in fig.data:
        assert list(trace.x) == [0, 1, 2, 3]


def test_parallel_coordinate_dimensions(study):
    fig = study.plot_parallel_coordinate()
    dims = fig.data[0].dimensions
    labels = [d.label for d in dims]
    assert labels[-1] == "f1", "objective is the last axis"
    assert "x" in labels
    assert "opt" in labels
    opt_dim = next(d for d in dims if d.label == "opt")
    assert set(opt_dim.ticktext) <= {"adam", "sgd"}, "categorical axis labeled"


def test_param_importances_spearman(study):
    fig = study.plot_param_importances()
    bars = fig.data[0]
    assert "x" in bars.y, "numeric param x is ranked"
    assert "opt" not in bars.y, "non-numeric params are excluded from rank corr"
    # f1 = x + step noise → |rho| for x should be maximal (1.0).
    x_rho = abs(bars.customdata[list(bars.y).index("x")])
    assert x_rho > 0.95
    assert all(0 <= v <= 1 for v in bars.x)


def test_timeline_states_and_axis(study):
    fig = study.plot_timeline()
    assert len(fig.data) == 6
    assert fig.layout.xaxis.type == "date"
    assert {t.name for t in fig.data} == {"completed"}


def test_pareto_front_requires_multiobjective(study):
    with pytest.raises(ValueError, match="two objectives"):
        study.plot_pareto_front()


def test_pareto_front_multiobjective(tmp_path):
    def objective(trial):
        trial.report("f1", trial["x"], 0)
        trial.report("latency", 1.0 - trial["x"] * 0.5 + (0.3 if trial["x"] > 0.8 else 0), 0)
        return None

    study = Study(
        "pareto",
        search_space=[{"type": "float", "name": "x", "low": 0.0, "high": 1.0}],
        strategy="random",
        n_trials=8,
        objectives=[("f1", "maximize"), ("latency", "minimize")],
        seed=3,
        root=str(tmp_path),
    )
    study.run(objective)
    fig = study.plot_pareto_front()
    front = next(t for t in fig.data if t.name == "pareto front")
    assert len(front.x) >= 1
    # Front sorted by x → f1 increasing along the line.
    assert list(front.x) == sorted(front.x)


# ── run figures ─────────────────────────────────────────────────────


def test_plot_metrics_series(tmp_path):
    view = _tracked_fit(tmp_path)
    fig = view.plot_metrics()
    assert {t.name for t in fig.data} == {"loss", "val_f1"}
    loss = next(t for t in fig.data if t.name == "loss")
    assert list(loss.x) == [0, 1, 2, 3]
    assert loss.y[0] == pytest.approx(1.0)
    assert fig.layout.showlegend

    only = view.plot_metrics(names=["loss"])
    assert len(only.data) == 1


def test_plot_gantt_outcomes(tmp_path):
    view = _tracked_fit(tmp_path)
    fig = view.plot_gantt()
    assert len(fig.data) == 2, "one span per node"
    assert {t.name for t in fig.data} == {"completed"}
    assert fig.layout.xaxis.type == "date"


def test_plot_metrics_empty_run_raises(tmp_path):
    g = Graph()
    g.node("a", _Plain())
    with g.track_run("empty", root=str(tmp_path)) as run:
        pass
    with pytest.raises(ValueError, match="no metrics"):
        soma.RunView(run.dir).plot_metrics()


# ── dataframes ──────────────────────────────────────────────────────


def test_trials_dataframe(study):
    pytest.importorskip("pandas")
    df = study.trials_dataframe()
    assert len(df) == 6
    assert "param_x" in df.columns
    assert "param_opt" in df.columns
    assert "metric_f1" in df.columns
    assert (df["state"] == "completed").all()
    assert str(df["started_at"].dtype).startswith("datetime64")


def test_metrics_dataframe(tmp_path):
    pytest.importorskip("pandas")
    view = _tracked_fit(tmp_path)
    df = view.metrics_dataframe()
    assert set(df["name"]) == {"loss", "val_f1"}
    assert list(df.columns) == ["ts", "name", "value", "step", "trial_id", "node_id"]
    only = view.metrics_dataframe("loss")
    assert set(only["name"]) == {"loss"}


def test_experiments_dataframe(tmp_path):
    pytest.importorskip("pandas")
    _tracked_fit(tmp_path, "exp-df")
    df = soma.experiments_dataframe(str(tmp_path))
    assert len(df) == 1
    assert df.iloc[0]["name"] == "exp-df"
    assert "metric_val_f1" in df.columns

    empty = soma.experiments_dataframe(str(tmp_path / "nowhere"))
    assert empty.empty


def test_parallel_coordinate_visual_upgrade(tmp_path):
    """Viridis gradient, auto-log axes for decade-spanning params,
    dimmed unselected lines."""

    def objective(trial):
        trial.report("f1", 0.5 + trial["x"], 0)
        return None

    study = Study(
        "parcoords-style",
        search_space=[
            {"type": "float", "name": "lr", "low": 1e-4, "high": 1e-1, "scale": "log"},
            {"type": "float", "name": "x", "low": 0.1, "high": 0.4},
        ],
        strategy="random",
        n_trials=6,
        objectives=[("f1", "maximize")],
        seed=9,
        root=str(tmp_path),
    )
    study.run(objective)

    fig = study.plot_parallel_coordinate()
    pc = fig.data[0]
    assert pc.line.colorscale is not None
    assert pc.unselected.line.opacity == pytest.approx(0.35)

    dims = {d.label: d for d in pc.dimensions}
    # lr spans ≥ 2 decades → log₁₀ axis with 10^e tick labels.
    lr = dims["lr"]
    assert all(v <= 0 for v in lr.values), "log10 of values < 1"
    assert any(t.startswith("10") for t in lr.ticktext)
    # x spans < 2 decades → linear, untouched.
    assert dims["x"].ticktext is None
    assert 0.1 <= min(dims["x"].values)
