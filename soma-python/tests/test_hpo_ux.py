"""New HPO UX: Trial handle, pruning, composite objective, tracking,
resume, graph-level search spaces, and event emission."""

from __future__ import annotations

import json
import pathlib

import pytest

import soma
from soma import Filter, Graph, Study, search


# ── Trial handle & legacy compatibility ─────────────────────────────


def test_trial_handle_mapping_protocol(tmp_path):
    seen = {}

    def executor(trial):
        seen["x"] = trial["x"]
        seen["get"] = trial.get("missing", 1.23)
        seen["contains"] = "x" in trial
        seen["keys"] = trial.keys()
        seen["id"] = trial.id
        return {"f1": trial["x"]}

    study = Study(
        "handle",
        search_space=[{"type": "float", "name": "x", "low": 0.0, "high": 1.0}],
        strategy="random",
        n_trials=1,
        objectives=[("f1", "maximize")],
        seed=1,
        root=str(tmp_path),
    )
    study.run(executor)

    assert 0.0 <= seen["x"] <= 1.0
    assert seen["get"] == 1.23
    assert seen["contains"] is True
    assert seen["keys"] == ["x"]
    assert seen["id"] == "trial_0000"


def test_legacy_params_style_still_works(tmp_path):
    """Old executors used params.get(...) and returned a dict."""

    def legacy(params):
        x = params.get("x", 0.5)
        return {"f1": max(0.0, 1.0 - abs(x - 0.5) * 2)}

    study = Study(
        "legacy",
        search_space=[{"type": "float", "name": "x", "low": 0.0, "high": 1.0}],
        strategy="random",
        n_trials=5,
        objectives=[("f1", "maximize")],
        seed=42,
        root=str(tmp_path),
    )
    study.run(legacy)
    assert study.n_trials == 5
    assert study.best_trial["metrics"]["f1"] > 0.0


def test_float_return_becomes_score(tmp_path):
    study = Study(
        "float-return",
        search_space=[{"type": "float", "name": "x", "low": 0.0, "high": 1.0}],
        strategy="random",
        n_trials=3,
        seed=1,
        direction="maximize",
        objectives=[("score", "maximize")],
        root=str(tmp_path),
    )
    study.run(lambda trial: trial["x"])
    assert study.best_trial["metrics"]["score"] >= 0.0


# ── Pruning through trial.report ────────────────────────────────────


def test_median_pruner_stops_bad_trials(tmp_path):
    calls = {"n": 0}

    def train(trial):
        calls["n"] += 1
        good = calls["n"] == 1
        for step in range(10):
            value = 0.5 + step * 0.05 if good else 0.01
            if trial.report("f1", value, step):
                return None  # pruned
        return None

    study = Study(
        "pruned",
        search_space=[{"type": "float", "name": "x", "low": 0.0, "high": 1.0}],
        strategy="random",
        n_trials=4,
        objectives=[("f1", "maximize")],
        pruning=("median", 2),
        seed=7,
        root=str(tmp_path),
    )
    study.run(train)

    states = [t["state"] for t in study.trials]
    assert states.count("pruned") == 3, states
    assert states.count("completed") == 1


# ── Composite objective ─────────────────────────────────────────────


def test_objective_callable(tmp_path):
    study = Study(
        "composite",
        search_space=[{"type": "float", "name": "x", "low": 0.0, "high": 1.0}],
        strategy="grid",
        n_trials=5,
        objective=lambda m: m["a"] - m["b"],
        direction="maximize",
        root=str(tmp_path),
    )
    # a - b = x - x² → maximum at x = 0.5
    study.run(lambda trial: {"a": trial["x"], "b": trial["x"] ** 2})

    best = study.best_trial
    assert abs(best["params"]["x"] - 0.5) < 1e-9
    assert "score" in best["metrics"]


# ── Tracking, run dir, events, resume ───────────────────────────────


def test_run_dir_contains_manifest_events_study(tmp_path):
    study = Study(
        "tracked",
        search_space=[{"type": "float", "name": "x", "low": 0.0, "high": 1.0}],
        strategy="random",
        n_trials=3,
        objectives=[("f1", "maximize")],
        seed=3,
        root=str(tmp_path),
    )
    study.run(lambda trial: {"f1": trial["x"]})

    run_dir = pathlib.Path(study.run_dir)
    assert run_dir.exists()

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["kind"] == "study"
    assert manifest["schema_version"] == 1

    status = json.loads((run_dir / "status.json").read_text())
    assert status["state"] == "completed"

    events = [json.loads(l) for l in (run_dir / "events.jsonl").read_text().splitlines()]
    types = [e["event_type"] for e in events]
    assert "StudyStarted" in types
    assert types.count("TrialCompleted") == 3
    assert "StudyCompleted" in types
    # seq is monotonically increasing from 0
    assert [e["seq"] for e in events] == list(range(len(events)))

    saved = json.loads((run_dir / "study.json").read_text())
    assert len(saved["trials"]) == 3


def test_tracking_disabled_writes_nothing(tmp_path):
    study = Study(
        "untracked",
        search_space=[{"type": "float", "name": "x", "low": 0.0, "high": 1.0}],
        strategy="random",
        n_trials=2,
        objectives=[("f1", "maximize")],
        seed=1,
        tracking=False,
        root=str(tmp_path),
    )
    study.run(lambda trial: {"f1": 0.5})
    assert study.run_dir is None
    assert not (tmp_path / "runs").exists()


def test_on_event_receives_trial_events(tmp_path):
    got = []

    study = Study(
        "events",
        search_space=[{"type": "float", "name": "x", "low": 0.0, "high": 1.0}],
        strategy="random",
        n_trials=3,
        objectives=[("f1", "maximize")],
        seed=1,
        root=str(tmp_path),
    )
    study.run(lambda trial: {"f1": 0.5}, on_event=lambda e: got.append(e["event_type"]))

    # The callback thread may lag slightly behind run() returning.
    import time

    for _ in range(50):
        if "StudyCompleted" in got:
            break
        time.sleep(0.05)
    assert "TrialStarted" in got
    assert "StudyCompleted" in got


def test_load_and_resume_without_duplicates(tmp_path):
    space = [{"type": "float", "name": "x", "low": 0.0, "high": 1.0}]
    full = Study(
        "reference",
        search_space=space,
        strategy="grid",
        n_trials=6,
        objectives=[("f1", "maximize")],
        root=str(tmp_path),
    )
    full.run(lambda trial: {"f1": trial["x"]})
    all_params = [t["params"]["x"] for t in full.trials]
    assert len(all_params) == 6

    # Simulate an interrupted study: rewrite study.json with 3 trials.
    run_dir = pathlib.Path(full.run_dir)
    partial = json.loads((run_dir / "study.json").read_text())
    partial["trials"] = partial["trials"][:3]
    (run_dir / "study.json").write_text(json.dumps(partial))

    resumed = Study.load(str(run_dir))
    assert resumed.n_trials == 3
    assert resumed.progress == 0.5  # planned_trials survives save/load (grid)
    resumed.run(lambda trial: {"f1": trial["x"]}, resume=True)

    assert resumed.n_trials == 6
    resumed_params = [t["params"]["x"] for t in resumed.trials]
    assert resumed_params == all_params, "resume must not repeat or skip grid points"
    assert [t["id"] for t in resumed.trials][3:] == [
        "trial_0003",
        "trial_0004",
        "trial_0005",
    ]


# ── Graph-level search space ────────────────────────────────────────


class _Embed(Filter):
    dim: int = search(choices=[16, 32])

    def fit(self, x, y=None):
        return {}

    def forward(self, x, state):
        return x


class _Encoder(Filter):
    lr: float = search(0.001, 0.1, scale="log")

    def fit(self, x, y=None):
        return {}

    def forward(self, x, state):
        return x


def _two_node_graph():
    g = Graph()
    g.node("embed", _Embed())
    g.node("encoder", _Encoder())
    g.connect("embed", "encoder")
    return g


def test_graph_search_space_prefixes_node_ids():
    g = _two_node_graph()
    space = g.search_space()
    names = sorted(d["name"] for d in space)
    assert names == ["embed.dim", "encoder.lr"]


def test_apply_params_sets_live_filters():
    g = _two_node_graph()
    g.apply_params({"embed.dim": 32, "encoder.lr": 0.05})
    filters = dict(g.filters())
    assert filters["embed"].dim == 32
    assert filters["encoder"].lr == 0.05

    # Unambiguous bare name works; unknown raises.
    g.apply_params({"lr": 0.01})
    assert dict(g.filters())["encoder"].lr == 0.01
    with pytest.raises(KeyError):
        g.apply_params({"nope": 1})


def test_graph_study_runs_over_aggregated_space(tmp_path):
    g = _two_node_graph()
    study = g.study(
        "graph-study",
        strategy="grid",
        n_trials=2,
        objectives=[("f1", "maximize")],
        root=str(tmp_path),
    )

    def train(trial):
        g.apply_params(trial.params)
        f = dict(g.filters())
        return {"f1": 1.0 if f["embed"].dim == 32 else 0.1}

    study.run(train)
    assert study.n_trials == 4  # 2 dims × 2 values
    assert study.best_trial["params"]["embed.dim"] == 32


# ── Graph event emission & run tracking ─────────────────────────────


def test_emit_event_reaches_run_dir(tmp_path):
    g = Graph()
    run = g.begin_run("emit-test", root=str(tmp_path), tags=["t1"])
    g.emit_event(
        {
            "event_type": "MetricReported",
            "run_id": run.id,
            "metric": {
                "name": "val_f1",
                "value": 0.9,
                "step": 3,
                "timestamp": "2026-07-26T10:00:00Z",
            },
            "node_id": None,
            "trial_id": None,
        }
    )
    run.log("loss", 0.42, step=7)
    run.finish()

    run_dir = pathlib.Path(run.dir)
    events = [json.loads(l) for l in (run_dir / "events.jsonl").read_text().splitlines()]
    types = [e["event_type"] for e in events]
    assert types == ["MetricReported", "MetricReported"]

    metrics = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert {m["name"] for m in metrics} == {"val_f1", "loss"}

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["tags"] == ["t1"]
    assert manifest["python_version"].startswith("3.")

    status = json.loads((run_dir / "status.json").read_text())
    assert status["state"] == "completed"


def test_emit_event_rejects_unknown_type():
    g = Graph()
    with pytest.raises(RuntimeError):
        g.emit_event({"event_type": "NotAnEvent"})


def test_graph_json_serializes_topology():
    g = _two_node_graph()
    data = json.loads(g.graph_json())
    ids = [n["id"] for n in data["nodes"]]
    assert ids == ["embed", "encoder"]
