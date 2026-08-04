"""Read-side API: soma.runs(), RunView aggregations, the local
RunStarted/RunCompleted bracket, and un-flattened trial series."""

from __future__ import annotations

import json
import pathlib

import pytest

import soma
from soma import Filter, Graph, Study
from soma._cache_cli import main as cli_main


class _Plain(Filter):
    # `tag` exists only to give each node its own identity. Without it the
    # two nodes below are the same class with the same config, learn the
    # same (empty) state, and — since `forward` is the identity — see the
    # same input, so the second one's output key equals the first one's
    # and it is legitimately served from cache. That is the content-
    # addressed "early cutoff" working, but it leaves a two-node run with
    # one node timing, which is not what these tests are about.
    def __init__(self, tag="x"):
        super().__init__(tag=tag)

    def fit(self, x, y=None):
        return {}

    def forward(self, x, state):
        return x


class _Boom(Filter):
    def fit(self, x, y=None):
        raise ValueError("boom")

    def forward(self, x, state):
        return x


def _graph():
    # An in-memory cache, so each test's run is cold. Fitting goes through
    # the same execution site as running now, which means it consults the
    # output cache — and these tests share one session-scoped
    # SOMA_CACHE_DIR, so with the persistent default the second test to
    # fit this graph would read the first one's results and record cache
    # hits instead of the node timings it is here to inspect.
    g = Graph(cache="memory")
    g.node("a", _Plain("a"))
    g.node("b", _Plain("b"))
    g.connect("a", "b")
    return g


def _tracked_fit(tmp_path, name="fit-run"):
    g = _graph()
    with g.track_run(name, root=str(tmp_path), kind="fit") as run:
        g.fit([1.0, 2.0, 3.0])
        run.log("val_f1", 0.9, step=1)
        run.log("val_f1", 0.95, step=2)
    return run


# ── run bracket ─────────────────────────────────────────────────────


def test_local_fit_emits_run_bracket_with_matching_ids(tmp_path):
    run = _tracked_fit(tmp_path)
    events = soma.RunView(run.dir).events()
    types = [e["event_type"] for e in events]
    assert "RunStarted" in types
    assert "RunCompleted" in types

    started = next(e for e in events if e["event_type"] == "RunStarted")
    completed = next(e for e in events if e["event_type"] == "RunCompleted")
    node_events = [e for e in events if e["event_type"] == "NodeStarted"]
    assert started["run_id"] == completed["run_id"]
    assert len(node_events) == 2
    assert all(e["run_id"] == started["run_id"] for e in node_events)
    assert started["plan_summary"]["total_nodes"] == 2
    assert types.index("RunStarted") < types.index("NodeStarted")


def test_failed_fit_emits_run_failed(tmp_path):
    g = Graph()
    g.node("bad", _Boom())
    with pytest.raises(Exception):
        with g.track_run("failing", root=str(tmp_path), kind="fit"):
            g.fit([1.0])

    run_dir = next((tmp_path / "runs").iterdir())
    events = soma.RunView(str(run_dir)).events()
    types = [e["event_type"] for e in events]
    assert "RunStarted" in types
    assert "RunFailed" in types
    assert "RunCompleted" not in types
    failed = next(e for e in events if e["event_type"] == "RunFailed")
    assert "boom" in failed["error"]


# ── soma.runs() / RunView ───────────────────────────────────────────


def test_runs_lists_tracked_runs_newest_first(tmp_path):
    _tracked_fit(tmp_path, "first")
    _tracked_fit(tmp_path, "second")

    runs = soma.runs(str(tmp_path))
    assert len(runs) == 2
    assert {r.name for r in runs} == {"first", "second"}
    assert all(r.state == "completed" for r in runs)
    assert all(r.kind == "fit" for r in runs)
    created = [r.info["created_at"] for r in runs]
    assert created == sorted(created, reverse=True), "newest first"


def test_runs_empty_root(tmp_path):
    assert soma.runs(str(tmp_path)) == []


def test_runview_aggregations(tmp_path):
    run = _tracked_fit(tmp_path)
    view = soma.RunView(run.dir)

    # identity
    assert view.id == run.id
    assert view.state == "completed"
    assert view.dir == run.dir
    assert "RunView(" in repr(view)

    # manifest passthrough
    assert view.manifest()["graph"]["n_nodes"] == 2

    # envelopes carry seq + ts (wall clock lives in the envelope)
    events = view.events()
    assert events[0]["seq"] == 0
    assert all("ts" in e for e in events)

    # metric series (from metrics.jsonl)
    series = view.metric_series("val_f1")
    assert [p["value"] for p in series] == [0.9, 0.95]
    assert [p["step"] for p in series] == [1, 2]
    assert view.metric_series("nope") == []
    assert len(view.metric_series()) == 2

    # node timings: one completed span per node with wall timestamps
    spans = view.node_timings()
    assert [s["node_id"] for s in spans] == ["a", "b"]
    assert all(s["outcome"] == "completed" for s in spans)
    assert all(s["started_ts"] is not None for s in spans)
    assert all(s["duration_ms"] is not None for s in spans)

    # A cold fit is all misses, one per node. Fitting used to have an
    # execution walk of its own that never consulted the output cache, so
    # a training run reported no cache activity at all and the report's
    # cache panel was blank for exactly the runs that take longest.
    activity = view.cache_activity()
    assert activity["hits"] == 0
    assert activity["misses"] == 2

    # no health flags, no trials
    assert view.health_flags() == []
    assert view.trial_timeline() == []


def test_runview_health_flags(tmp_path):
    g = _graph()
    with g.track_run("flagged", root=str(tmp_path)) as run:
        g.emit_event(
            {
                "event_type": "HealthFlag",
                "run_id": run.id,
                "node_id": "a",
                "step": 4,
                "flag": "DEAD_CHANNELS",
                "detail": "3/64 dead",
            }
        )
    flags = soma.RunView(run.dir).health_flags()
    assert len(flags) == 1
    assert flags[0]["flag"] == "DEAD_CHANNELS"
    assert flags[0]["node_id"] == "a"
    assert flags[0]["step"] == 4
    assert flags[0]["ts"]


def test_runview_rejects_non_run_dir(tmp_path):
    with pytest.raises(RuntimeError, match="open run dir"):
        soma.RunView(str(tmp_path))


# ── study runs: trial timeline + un-flattened series ────────────────


def _study(tmp_path, n_trials=3):
    def objective(trial):
        for step in range(3):
            trial.report("score", 0.5 + 0.1 * step, step)
        return None

    study = Study(
        "reader-hpo",
        search_space=[
            {"type": "float", "name": "lr", "low": 0.001, "high": 0.1, "scale": "log"},
        ],
        strategy="random",
        n_trials=n_trials,
        objectives=[("score", "maximize")],
        root=str(tmp_path),
        seed=7,
    )
    study.run(objective)
    return study


def test_trials_expose_series_and_timestamps(tmp_path):
    study = _study(tmp_path)
    trial = study.trials[0]

    # Back-compat: flattened last-value dict still present.
    assert isinstance(trial["metrics"], dict)

    # New: full series with step + timestamp per record.
    series = [m for m in trial["series"] if m["name"] == "score"]
    assert len(series) >= 3
    assert [m["step"] for m in series[:3]] == [0, 1, 2]
    assert all("timestamp" in m for m in series)
    values = [m["value"] for m in series[:3]]
    assert values == sorted(values), "reported curve is increasing"
    assert values[0] == pytest.approx(0.5)

    # New: trial wall-clock bounds.
    assert trial["started_at"] is not None
    assert trial["finished_at"] is not None
    assert trial["started_at"] <= trial["finished_at"]


def test_study_run_dir_has_trial_timeline(tmp_path):
    study = _study(tmp_path)
    view = soma.RunView(study.run_dir)
    timeline = view.trial_timeline()
    assert len(timeline) == 3
    assert all(t["state"] == "completed" for t in timeline)
    assert all(t["started_at"] is not None for t in timeline)
    assert all(t["duration_ms"] is not None for t in timeline)

    listed = soma.runs(str(tmp_path))
    assert listed and listed[0].kind == "study"


# ── CLI ─────────────────────────────────────────────────────────────


def test_cli_soma_runs_table_and_json(tmp_path, capsys):
    _tracked_fit(tmp_path, "cli-run")

    assert cli_main(["runs", "--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "cli-run" in out
    assert "completed" in out
    assert "RUN ID" in out

    assert cli_main(["runs", "--root", str(tmp_path), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data[0]["name"] == "cli-run"


def test_cli_soma_runs_empty(tmp_path, capsys):
    assert cli_main(["runs", "--root", str(tmp_path)]) == 0
    assert "no runs" in capsys.readouterr().out


# ── overlays (architecture + efficiency rendering) ──────────────────


def test_graph_to_mermaid_accepts_overlay_kwarg(tmp_path):
    g = _graph()
    plain = g.to_mermaid()
    assert "classDef" not in plain

    annotated = g.to_mermaid(
        overlay={
            "nodes": {
                "a": {"status": "completed", "duration_ms": 1200},
                "b": {"status": "cached", "cache_tier": "memory", "duration_ms": 3},
            }
        }
    )
    assert 'a["a<br/>1.2s"]' in annotated
    assert 'b["b<br/>3ms · mem hit"]' in annotated
    assert "class a soma_completed" in annotated
    assert "class b soma_cached" in annotated

    dot = g.to_graphviz(overlay={"nodes": {"a": {"status": "failed"}}})
    assert "fillcolor" in dot

    with pytest.raises(RuntimeError, match="invalid overlay"):
        g.to_mermaid(overlay={"nodes": {"a": {"status": "not-a-status"}}})


def test_runview_overlay_and_annotated_rendering(tmp_path):
    run = _tracked_fit(tmp_path)
    view = soma.RunView(run.dir)

    overlay = view.overlay()
    assert set(overlay["nodes"]) == {"a", "b"}
    assert overlay["nodes"]["a"]["status"] == "completed"
    assert overlay["nodes"]["a"]["duration_ms"] is not None

    mermaid = view.to_mermaid()
    assert "class a soma_completed" in mermaid
    assert "class b soma_completed" in mermaid

    plain = view.to_mermaid(overlay=False)
    assert "classDef" not in plain
    assert plain == _graph().to_mermaid()

    dot = view.to_graphviz()
    assert "fillcolor" in dot

    # The overlay dict round-trips through the Graph kwarg path too.
    assert "class a soma_completed" in _graph().to_mermaid(overlay=overlay)


def test_cli_soma_graph(tmp_path, capsys):
    run = _tracked_fit(tmp_path, "graph-cli")
    run_id = run.id

    assert cli_main(["graph", run_id, "--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("graph LR")
    assert "class a soma_completed" in out

    assert cli_main(["graph", run.dir, "--format", "dot"]) == 0
    assert capsys.readouterr().out.startswith("digraph G {")

    assert cli_main(["graph", run_id, "--root", str(tmp_path), "--no-overlay"]) == 0
    assert "classDef" not in capsys.readouterr().out

    assert cli_main(["graph", "nope", "--root", str(tmp_path)]) == 1
    assert "no run" in capsys.readouterr().err


# ── notebook/terminal UX: HTML reprs, rich table, progress bar ──────


def test_runs_returns_runlist_with_html_repr(tmp_path):
    from soma._runs import RunList

    _tracked_fit(tmp_path, "pretty-run")
    listed = soma.runs(str(tmp_path))
    assert isinstance(listed, RunList)

    table = listed._repr_html_()
    assert "<table" in table
    assert "pretty-run" in table
    assert "completed" in table
    assert listed[0].id in table

    card = listed[0]._repr_html_()
    assert listed[0].id in card
    assert "completed" in card

    assert soma.runs(str(tmp_path / "empty"))._repr_html_() == "<i>no runs</i>"


def test_cli_runs_rich_and_plain(tmp_path, capsys):
    pytest.importorskip("rich")
    run = _tracked_fit(tmp_path, "rich-run")

    assert cli_main(["runs", "--root", str(tmp_path)]) == 0
    rich_out = capsys.readouterr().out
    assert "rich-run" in rich_out
    assert run.id in rich_out

    assert cli_main(["runs", "--root", str(tmp_path), "--plain"]) == 0
    plain_out = capsys.readouterr().out
    assert plain_out.startswith("RUN ID")
    assert run.id in plain_out


def test_study_run_progress_bar(tmp_path):
    pytest.importorskip("tqdm")
    seen_events = []

    def objective(trial):
        trial.report("score", trial["lr"], 0)
        return None

    study = Study(
        "progress-hpo",
        search_space=[
            {"type": "float", "name": "lr", "low": 0.001, "high": 0.1},
        ],
        strategy="random",
        n_trials=4,
        objectives=[("score", "maximize")],
        root=str(tmp_path),
        seed=3,
    )
    # progress=True draws the bar AND still chains the user callback.
    study.run(objective, on_event=seen_events.append, progress=True)
    assert study.n_trials == 4
    assert study.best_trial is not None
    assert any(e["event_type"] == "TrialCompleted" for e in seen_events)


# ── inline SVG diagrams (notebook reprs, offline reports) ───────────


def test_graph_to_svg_and_repr_html(tmp_path):
    g = _graph()
    svg = g.to_svg()
    assert svg.startswith("<svg xmlns=")
    assert ">a</text>" in svg and ">b</text>" in svg
    assert svg.count("marker-end") == 1, "one edge a→b"

    # Evaluating a Graph in a notebook shows the diagram.
    assert g.to_svg() == g._repr_html_()
    assert "empty graph" in Graph()._repr_html_()

    annotated = g.to_svg(
        overlay={"nodes": {"a": {"status": "completed", "duration_ms": 1200}}}
    )
    assert "#e8f5e9" in annotated and ">1.2s</text>" in annotated


def test_runview_to_svg(tmp_path):
    run = _tracked_fit(tmp_path)
    view = soma.RunView(run.dir)

    svg = view.to_svg()
    assert svg.startswith("<svg xmlns=")
    assert "#e8f5e9" in svg, "completed nodes colored"

    plain = view.to_svg(overlay=False)
    assert "#e8f5e9" not in plain

    with pytest.raises(ValueError, match="no inner-architecture"):
        view.to_svg(node="ghost")


# ── Agent activity aggregates ──


def test_runview_agentic_activity_and_timeline(tmp_path):
    """A tracked run of a step graph lands its agent events in
    events.jsonl, and RunView aggregates them: per-node activity and the
    per-effect timeline (the read side of the agentic layer)."""
    from soma.agentic import Await, Done, Sleep

    class Napper:
        _cache_version = "1"

        def poll(self, ctx):
            if ctx.turn == 0:
                return Await([Sleep(0.001)])
            return Done("rested")

    g = Graph(cache="memory")
    g.node("napper", Napper())
    with g.track_run("agentic-run", root=str(tmp_path), kind="forward") as run:
        assert g.forward("x") == "rested"

    view = soma.RunView(run.dir)

    activity = view.agentic_activity()
    napper = activity["by_node"]["napper"]
    assert napper["turns"] == 2
    assert napper["effects"] == 1
    assert list(napper["effects_by_label"]) == ["sleep:1ms"]
    assert napper["completions"] == 1
    assert activity["turns"] == 2
    assert activity["steps_completed"] == 1

    timeline = view.agentic_timeline()
    assert len(timeline) == 1
    span = timeline[0]
    assert span["node_id"] == "napper"
    assert span["effect"] == "sleep:1ms"
    assert span["outcome"] == "completed"
    assert not span["replayed"]

    # The step's node span knows it was a step.
    napper_span = next(s for s in view.node_timings() if s["node_id"] == "napper")
    assert napper_span["effectful"] is True

    # And a filter-only run reports no agent activity at all.
    plain = soma.RunView(_tracked_fit(tmp_path, name="no-agents").dir)
    assert plain.agentic_activity()["by_node"] == {}
    assert plain.agentic_timeline() == []
