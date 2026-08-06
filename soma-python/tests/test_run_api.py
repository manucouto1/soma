"""Run/tracking API surface that needs no torch: PyRun methods,
Graph.emit_event, Graph.graph_json, and the track_run lifecycle."""

from __future__ import annotations

import json
import pathlib

import pytest

import soma
from soma import Filter, Graph


class _Plain(Filter):
    def fit(self, x, y=None):
        return {}

    def forward(self, x, state):
        return x


def _graph():
    g = Graph()
    g.node("a", _Plain())
    g.node("b", _Plain())
    g.edge("a", "b")
    return g


def _events(run_dir):
    return [
        json.loads(l)
        for l in (pathlib.Path(run_dir) / "events.jsonl").read_text().splitlines()
    ]


# ── PyRun ───────────────────────────────────────────────────────────


def test_log_epoch_emits_event_and_heartbeats(tmp_path):
    g = Graph()
    run = g.begin_run("epochs", root=str(tmp_path))
    before = json.loads(
        (pathlib.Path(run.dir) / "status.json").read_text()
    )["heartbeat_at"]
    import time

    time.sleep(0.01)
    run.log_epoch(3, total=10)
    run.finish()

    events = _events(run.dir)
    epoch = next(e for e in events if e["event_type"] == "EpochStarted")
    assert epoch["epoch"] == 3
    assert epoch["total_epochs"] == 10

    after = json.loads(
        (pathlib.Path(run.dir) / "status.json").read_text()
    )["heartbeat_at"]
    assert after > before, "log_epoch refreshes the heartbeat"


def test_log_epoch_completed_carries_metrics_and_updates_summary(tmp_path):
    g = Graph()
    run = g.begin_run("epoch-end", root=str(tmp_path))
    run.log_epoch_completed(2, {"val_f1": 0.81, "loss": 0.3})
    run.finish()

    events = _events(run.dir)
    done = next(e for e in events if e["event_type"] == "EpochCompleted")
    assert done["epoch"] == 2
    metrics = {m["name"]: m["value"] for m in done["metrics"]}
    assert metrics == {"val_f1": 0.81, "loss": 0.3}
    assert all(m["step"] == 2 for m in done["metrics"])

    # Epoch metrics reach the run summary → the experiments journal.
    rec = soma.experiments(str(tmp_path))[0]
    assert rec["metrics"]["val_f1"] == 0.81


def test_step_completed_direct_with_epoch(tmp_path):
    g = Graph()
    run = g.begin_run("steps", root=str(tmp_path))
    run.step_completed(7, epoch=1)
    run.step_completed(8)
    run.finish()

    steps = [e for e in _events(run.dir) if e["event_type"] == "StepCompleted"]
    assert [(s["step"], s["epoch"]) for s in steps] == [(7, 1), (8, None)]


def test_heartbeat_direct(tmp_path):
    g = Graph()
    run = g.begin_run("hb", root=str(tmp_path))
    status_path = pathlib.Path(run.dir) / "status.json"
    before = json.loads(status_path.read_text())["heartbeat_at"]
    import time

    time.sleep(0.01)
    run.heartbeat()
    assert json.loads(status_path.read_text())["heartbeat_at"] > before
    run.finish()


def test_finish_failed_skips_journal_and_unknown_status_pins_completed(tmp_path):
    g = Graph()
    run = g.begin_run("fails", root=str(tmp_path))
    run.log("f1", 0.9)
    run.finish("failed")
    assert soma.experiments(str(tmp_path)) == []
    assert json.loads(
        (pathlib.Path(run.dir) / "status.json").read_text()
    )["state"] == "failed"

    # CONTRACT (pinned): any unrecognized status falls back to
    # "completed" — only "failed" is terminal-failure.
    run2 = g.begin_run("weird", root=str(tmp_path))
    run2.finish("weird")
    assert json.loads(
        (pathlib.Path(run2.dir) / "status.json").read_text()
    )["state"] == "completed"


def test_finish_detaches_the_sink(tmp_path):
    g = Graph()
    run = g.begin_run("detach", root=str(tmp_path))
    run.log("before", 1.0)
    run.finish()

    # Events emitted after finish must NOT bleed into the closed run.
    g.emit_event(
        {
            "event_type": "StepCompleted",
            "run_id": "someone-else",
            "step": 0,
            "epoch": None,
        }
    )
    types = [e["event_type"] for e in _events(run.dir)]
    assert "StepCompleted" not in types


def test_log_with_node_scopes_the_metric(tmp_path):
    g = Graph()
    run = g.begin_run("scoped", root=str(tmp_path))
    run.log("grad_norm", 0.02, step=5, node="encoder")
    run.finish()

    metrics = [
        json.loads(l)
        for l in (pathlib.Path(run.dir) / "metrics.jsonl").read_text().splitlines()
    ]
    assert metrics[0]["node_id"] == "encoder"
    assert metrics[0]["step"] == 5


def test_begin_run_kinds_and_graph_summary(tmp_path):
    g = _graph()
    for kind, expected in [
        ("fit", "fit"),
        ("train", "train"),
        ("study", "study"),
        ("trial", "trial"),
        ("whatever", "other"),
    ]:
        run = g.begin_run(f"k-{kind}", root=str(tmp_path), kind=kind)
        manifest = json.loads((pathlib.Path(run.dir) / "manifest.json").read_text())
        assert manifest["kind"] == expected, kind
        run.finish("failed")  # keep the journal clean

    manifest = json.loads((pathlib.Path(run.dir) / "manifest.json").read_text())
    assert manifest["graph"]["n_nodes"] == 2
    assert manifest["graph"]["node_ids"] == ["a", "b"]
    assert manifest["graph"]["graph_path"] == "graph.json"


# ── Graph.emit_event ────────────────────────────────────────────────


def test_emit_event_epoch_variants(tmp_path):
    g = Graph()
    run = g.begin_run("emit", root=str(tmp_path))
    g.emit_event(
        {"event_type": "EpochStarted", "run_id": run.id, "epoch": 0, "total_epochs": None}
    )
    g.emit_event(
        {
            "event_type": "EpochCompleted",
            "run_id": run.id,
            "epoch": 0,
            "metrics": [
                {"name": "loss", "value": 0.5, "step": 0, "timestamp": "2026-07-26T10:00:00Z"}
            ],
        }
    )
    run.finish()
    types = [e["event_type"] for e in _events(run.dir)]
    assert types == ["EpochStarted", "EpochCompleted"]


def test_emit_event_error_paths():
    g = Graph()
    # Known type, missing required field.
    with pytest.raises(RuntimeError, match="unknown or malformed"):
        g.emit_event({"event_type": "MetricReported", "run_id": "r"})
    # Known type, wrong field type.
    with pytest.raises(RuntimeError, match="unknown or malformed"):
        g.emit_event(
            {
                "event_type": "HealthFlag",
                "run_id": "r",
                "node_id": "n",
                "step": "not-a-number",
                "flag": "X",
                "detail": "",
            }
        )
    # Unserializable payload dies in json.dumps.
    with pytest.raises(TypeError):
        g.emit_event({"event_type": "StepCompleted", "run_id": object()})


def test_emit_event_without_any_run_is_a_quiet_noop():
    g = Graph()
    g.emit_event(
        {"event_type": "StepCompleted", "run_id": "r", "step": 0, "epoch": None}
    )  # no sink attached — must not raise


# ── Graph.graph_json ────────────────────────────────────────────────


def test_graph_json_includes_edges_and_handles_empty():
    g = _graph()
    data = json.loads(g.graph_json())
    assert [n["id"] for n in data["nodes"]] == ["a", "b"]
    edges = [(e["source"], e["target"]) for e in data["edges"]]
    assert ("a", "b") in edges

    empty = json.loads(Graph().graph_json())
    assert empty["nodes"] == []
    assert empty["edges"] == []


# ── track_run lifecycle (torch-free) ────────────────────────────────


def test_track_run_writes_topology_and_manages_py_state(tmp_path):
    g = _graph()
    with g.track_run("lifecycle", root=str(tmp_path), kind="fit", tags=["t1"]) as run:
        assert g.py_state["active_run"] is run
        assert g.py_state["train_step"] == 0
        run_dir = pathlib.Path(run.dir)
        assert json.loads((run_dir / "graph.json").read_text())["nodes"]
        assert (run_dir / "graph.mmd").read_text().startswith("graph")

    assert "active_run" not in g.py_state
    assert "train_step" not in g.py_state
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["kind"] == "fit"
    assert manifest["tags"] == ["t1"]
    assert json.loads((run_dir / "status.json").read_text())["state"] == "completed"


def test_track_run_cleans_py_state_on_exception(tmp_path):
    g = _graph()
    with pytest.raises(KeyboardInterrupt):
        with g.track_run("interrupted", root=str(tmp_path)):
            raise KeyboardInterrupt  # BaseException path
    assert "active_run" not in g.py_state
    assert "train_step" not in g.py_state
    run_dir = next((tmp_path / "runs").iterdir())
    assert json.loads((run_dir / "status.json").read_text())["state"] == "failed"


def test_nested_track_run_is_pinned_as_clobbering(tmp_path):
    """CONTRACT (pinned): track_run is not reentrant — the inner run
    replaces py_state['active_run'], and the outer exit clears it.
    Diagnostics inside the inner block persist to the INNER run."""
    g = _graph()
    with g.track_run("outer", root=str(tmp_path)) as outer:
        with g.track_run("inner", root=str(tmp_path)) as inner:
            assert g.py_state["active_run"] is inner
        # After the inner exit the slot is gone, not restored to outer.
        assert "active_run" not in g.py_state
    assert outer.dir != inner.dir
