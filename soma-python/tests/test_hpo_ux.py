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


# ── Error handling & robustness of the trial protocol ───────────────


def test_executor_exception_marks_trial_failed_and_study_continues(tmp_path):
    """CONTRACT: a raising executor fails THAT TRIAL, not the study.
    The study finishes 'completed' and the journal is still written."""
    calls = {"n": 0}

    def flaky(trial):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("cuda out of memory")
        return {"f1": trial["x"]}

    study = Study(
        "flaky",
        search_space=[{"type": "float", "name": "x", "low": 0.0, "high": 1.0}],
        strategy="random",
        n_trials=3,
        objectives=[("f1", "maximize")],
        seed=1,
        root=str(tmp_path),
    )
    study.run(flaky)  # must not raise

    states = [t["state"] for t in study.trials]
    assert states == ["completed", "failed", "completed"]
    assert study.best_trial is not None

    run_dir = pathlib.Path(study.run_dir)
    events = [json.loads(l) for l in (run_dir / "events.jsonl").read_text().splitlines()]
    failed = [e for e in events if e["event_type"] == "TrialFailed"]
    assert len(failed) == 1
    assert "cuda out of memory" in failed[0]["error"]
    status = json.loads((run_dir / "status.json").read_text())
    assert status["state"] == "completed", "a failed trial is not a failed study"
    assert len(soma.experiments(str(tmp_path))) == 1


def test_objective_callable_errors_fail_the_trial(tmp_path):
    study = Study(
        "bad-objective",
        search_space=[{"type": "float", "name": "x", "low": 0.0, "high": 1.0}],
        strategy="random",
        n_trials=2,
        objective=lambda m: m["missing_key"],
        direction="maximize",
        seed=1,
        root=str(tmp_path),
    )
    study.run(lambda trial: {"f1": 0.5})
    assert all(t["state"] == "failed" for t in study.trials)

    # Non-numeric objective return is also a trial failure.
    study2 = Study(
        "nonnumeric-objective",
        search_space=[{"type": "float", "name": "x", "low": 0.0, "high": 1.0}],
        strategy="random",
        n_trials=1,
        objective=lambda m: "not a number",
        direction="maximize",
        seed=1,
        root=str(tmp_path),
    )
    study2.run(lambda trial: {"f1": 0.5})
    assert study2.trials[0]["state"] == "failed"


def test_bad_executor_return_type_fails_the_trial(tmp_path):
    study = Study(
        "bad-return",
        search_space=[{"type": "float", "name": "x", "low": 0.0, "high": 1.0}],
        strategy="random",
        n_trials=1,
        objectives=[("f1", "maximize")],
        seed=1,
        root=str(tmp_path),
    )
    study.run(lambda trial: ["not", "a", "dict"])
    assert study.trials[0]["state"] == "failed"


def test_trial_handle_edges(tmp_path):
    seen = {}

    def executor(trial):
        with pytest.raises(KeyError):
            trial["nope"]
        seen["default_none"] = trial.get("nope")
        seen["keys"] = sorted(trial.keys())
        return {"f1": 0.5}

    study = Study(
        "edges",
        search_space=[
            {"type": "float", "name": "x", "low": 0.0, "high": 1.0},
            {"type": "float", "name": "y", "low": 0.0, "high": 1.0},
        ],
        strategy="random",
        n_trials=1,
        objectives=[("f1", "maximize")],
        seed=1,
        root=str(tmp_path),
    )
    study.run(executor)
    assert seen["default_none"] is None
    assert seen["keys"] == ["x", "y"]


def test_int_dims_stay_python_ints_end_to_end(tmp_path):
    seen = {}

    def executor(trial):
        seen["n_layers"] = trial["n_layers"]
        seen["width"] = trial["width"]
        return {"f1": float(trial["n_layers"])}

    study = Study(
        "ints",
        search_space=[
            {"type": "int", "name": "n_layers", "low": 1, "high": 4},
            {"type": "categorical", "name": "width", "choices": [16, 32]},
        ],
        strategy="grid",
        n_trials=4,
        objectives=[("f1", "maximize")],
        root=str(tmp_path),
    )
    study.run(executor)

    assert isinstance(seen["n_layers"], int), type(seen["n_layers"])
    assert isinstance(seen["width"], int), type(seen["width"])
    best = study.best_trial
    assert isinstance(best["params"]["n_layers"], int)
    assert best["params"]["n_layers"] == 4


def test_bayesian_through_the_new_ux(tmp_path):
    study = Study(
        "bayes-ux",
        search_space=[{"type": "float", "name": "x", "low": 0.0, "high": 1.0}],
        strategy="bayesian",
        n_trials=12,
        objectives=[("score", "maximize")],
        seed=42,
        root=str(tmp_path),
    )
    study.run(lambda trial: {"score": 1.0 - abs(trial["x"] - 0.7)})
    assert study.n_trials == 12
    assert study.progress == 1.0
    assert (pathlib.Path(study.run_dir) / "study.json").exists()


def test_constructor_validation():
    space = [{"type": "float", "name": "x", "low": 0.0, "high": 1.0}]
    with pytest.raises(RuntimeError, match="Unknown strategy"):
        Study("s", search_space=space, strategy="tpe", n_trials=2)
    # A typo must not silently maximize (it used to).
    with pytest.raises(RuntimeError, match="unknown direction"):
        Study("s", search_space=space, strategy="random", n_trials=2,
              objective=lambda m: 0.0, direction="minimise")
    with pytest.raises(RuntimeError, match="unknown direction"):
        Study("s", search_space=space, strategy="random", n_trials=2,
              objectives=[("f1", "MAX")])
    with pytest.raises(RuntimeError, match="pruning"):
        Study("s", search_space=space, strategy="random", n_trials=2,
              pruning=12345)
    with pytest.raises(RuntimeError, match="unknown pruning"):
        Study("s", search_space=space, strategy="random", n_trials=2,
              pruning="hyperband")


def test_pruning_string_and_percentile_forms(tmp_path):
    calls = {"n": 0}

    def train(trial):
        calls["n"] += 1
        good = calls["n"] == 1
        for step in range(10):
            value = 0.5 + step * 0.05 if good else 0.01
            if trial.report("f1", value, step):
                return None
        return None

    # "median" string form → warmup 0: bad trials die at step 0.
    study = Study(
        "median-string",
        search_space=[{"type": "float", "name": "x", "low": 0.0, "high": 1.0}],
        strategy="random",
        n_trials=3,
        objectives=[("f1", "maximize")],
        pruning="median",
        seed=7,
        root=str(tmp_path),
    )
    study.run(train)
    assert [t["state"] for t in study.trials].count("pruned") == 2

    calls["n"] = 0
    study2 = Study(
        "percentile",
        search_space=[{"type": "float", "name": "x", "low": 0.0, "high": 1.0}],
        strategy="random",
        n_trials=3,
        objectives=[("f1", "maximize")],
        pruning=("percentile", 50.0, 2),
        seed=7,
        root=str(tmp_path),
    )
    study2.run(train)
    assert [t["state"] for t in study2.trials].count("pruned") == 2


def test_objective_callable_takes_precedence_over_objectives(tmp_path):
    """CONTRACT: when both are passed, objective= wins and the study
    scores on 'score'; the objectives list is ignored."""
    study = Study(
        "precedence",
        search_space=[{"type": "float", "name": "x", "low": 0.0, "high": 1.0}],
        strategy="grid",
        n_trials=3,
        objective=lambda m: -m["loss"],
        objectives=[("loss", "minimize")],
        direction="maximize",
        root=str(tmp_path),
    )
    study.run(lambda trial: {"loss": abs(trial["x"] - 0.5)})
    best = study.best_trial
    assert "score" in best["metrics"]
    assert abs(best["params"]["x"] - 0.5) < 1e-9


def test_seed_reproducibility(tmp_path):
    def run_with(seed, name):
        study = Study(
            name,
            search_space=[{"type": "float", "name": "x", "low": 0.0, "high": 1.0}],
            strategy="random",
            n_trials=4,
            objectives=[("f1", "maximize")],
            seed=seed,
            tracking=False,
        )
        study.run(lambda trial: {"f1": trial["x"]})
        return [t["params"]["x"] for t in study.trials]

    assert run_with(42, "a") == run_with(42, "b")
    assert run_with(42, "c") != run_with(99, "d")


def test_empty_search_space_runs_one_trial_under_grid(tmp_path):
    """CONTRACT (pinned): grid over zero dimensions is a single empty
    configuration; random with no dims runs n_trials empty configs."""
    study = Study(
        "empty-grid",
        strategy="grid",
        n_trials=3,
        objectives=[("f1", "maximize")],
        root=str(tmp_path),
    )
    study.run(lambda trial: {"f1": 0.5})
    assert study.n_trials == 1
    assert study.trials[0]["params"] == {}


def test_study_save_explicit_path_and_error(tmp_path):
    study = Study(
        "save",
        search_space=[{"type": "float", "name": "x", "low": 0.0, "high": 1.0}],
        strategy="random",
        n_trials=1,
        objectives=[("f1", "maximize")],
        seed=1,
        tracking=False,
    )
    with pytest.raises(RuntimeError, match="no path given"):
        study.save()
    study.run(lambda trial: {"f1": 0.5})
    out = tmp_path / "exported.json"
    study.save(str(out))
    assert json.loads(out.read_text())["name"] == "save"


def test_resume_false_on_loaded_study_creates_new_run_dir(tmp_path):
    study = Study(
        "fresh-dir",
        search_space=[{"type": "float", "name": "x", "low": 0.0, "high": 1.0}],
        strategy="random",
        n_trials=2,
        objectives=[("f1", "maximize")],
        seed=1,
        root=str(tmp_path),
    )
    study.run(lambda trial: {"f1": trial["x"]})
    old_dir = pathlib.Path(study.run_dir)
    old_events = (old_dir / "events.jsonl").read_text()

    loaded = Study.load(str(old_dir))
    loaded.run(lambda trial: {"f1": trial["x"]})  # resume=False (default)

    new_dir = pathlib.Path(loaded.run_dir)
    assert new_dir != old_dir, "default run() mints a new run directory"
    # Old trials carried over into the new study.json; old dir untouched.
    saved = json.loads((new_dir / "study.json").read_text())
    assert len(saved["trials"]) == 2  # random n_trials=2, already done → no new ones
    assert (old_dir / "events.jsonl").read_text() == old_events


def test_events_seq_stays_contiguous_across_resume(tmp_path):
    space = [{"type": "float", "name": "x", "low": 0.0, "high": 1.0}]
    full = Study(
        "seq",
        search_space=space,
        strategy="grid",
        n_trials=4,
        objectives=[("f1", "maximize")],
        root=str(tmp_path),
    )
    full.run(lambda trial: {"f1": trial["x"]})
    run_dir = pathlib.Path(full.run_dir)

    partial = json.loads((run_dir / "study.json").read_text())
    partial["trials"] = partial["trials"][:2]
    (run_dir / "study.json").write_text(json.dumps(partial))

    resumed = Study.load(str(run_dir))
    resumed.run(lambda trial: {"f1": trial["x"]}, resume=True)

    events = [json.loads(l) for l in (run_dir / "events.jsonl").read_text().splitlines()]
    seqs = [e["seq"] for e in events]
    assert seqs == list(range(len(seqs))), "no gaps or duplicates across resume"


def test_objective_callable_survives_load_when_repassed(tmp_path):
    objective = lambda m: m["a"] - m["b"]  # noqa: E731
    executor = lambda trial: {"a": trial["x"], "b": trial["x"] ** 2}  # noqa: E731

    study = Study(
        "reload-objective",
        search_space=[{"type": "float", "name": "x", "low": 0.0, "high": 1.0}],
        strategy="grid",
        n_trials=4,
        objective=objective,
        direction="maximize",
        root=str(tmp_path),
    )
    study.run(executor)
    run_dir = pathlib.Path(study.run_dir)

    partial = json.loads((run_dir / "study.json").read_text())
    partial["trials"] = partial["trials"][:2]
    (run_dir / "study.json").write_text(json.dumps(partial))

    # Re-passing the callable on load keeps scoring alive.
    resumed = Study.load(str(run_dir), objective=objective)
    resumed.run(executor, resume=True)
    assert all("score" in t["metrics"] for t in resumed.trials)

    # Without it, a warning fires (and new trials would lack "score").
    partial["trials"] = partial["trials"][:1]
    (run_dir / "study.json").write_text(json.dumps(partial))
    naked = Study.load(str(run_dir))
    with pytest.warns(UserWarning, match="objective="):
        naked.run(executor, resume=True)


def test_load_errors(tmp_path):
    with pytest.raises(RuntimeError):
        Study.load(str(tmp_path / "does-not-exist"))
    bad = tmp_path / "bad-run"
    bad.mkdir()
    (bad / "study.json").write_text("{nope")
    with pytest.raises(RuntimeError):
        Study.load(str(bad))


def test_best_trial_none_when_nothing_completed(tmp_path):
    study = Study(
        "no-best",
        search_space=[{"type": "float", "name": "x", "low": 0.0, "high": 1.0}],
        strategy="random",
        n_trials=2,
        objectives=[("f1", "maximize")],
        seed=1,
        tracking=False,
    )
    assert study.best_trial is None
    study.run(lambda trial: (_ for _ in ()).throw(RuntimeError("boom")))
    assert study.best_trial is None


def test_study_manifest_is_enriched(tmp_path):
    study = Study(
        "manifest",
        search_space=[{"type": "float", "name": "x", "low": 0.0, "high": 1.0}],
        strategy="random",
        n_trials=1,
        objectives=[("f1", "maximize")],
        seed=1,
        root=str(tmp_path),
        tags=["mos", "suite"],
    )
    study.run(lambda trial: {"f1": 0.5})
    manifest = json.loads(
        (pathlib.Path(study.run_dir) / "manifest.json").read_text()
    )
    assert manifest["tags"] == ["mos", "suite"]
    assert manifest["python_version"].startswith("3.")


def test_on_event_fires_during_the_run_and_swallows_callback_errors(tmp_path):
    import time

    callback_times = []
    executor_entries = []

    def slow_executor(trial):
        executor_entries.append(time.monotonic())
        time.sleep(0.15)
        return {"f1": 0.5}

    def chatty_callback(event):
        callback_times.append(time.monotonic())
        raise ValueError("callbacks may be buggy")  # must be swallowed

    study = Study(
        "live-events",
        search_space=[{"type": "float", "name": "x", "low": 0.0, "high": 1.0}],
        strategy="random",
        n_trials=3,
        objectives=[("f1", "maximize")],
        seed=1,
        root=str(tmp_path),
    )
    study.run(slow_executor, on_event=chatty_callback)  # must not raise

    # GIL release check: some callback ran BEFORE the last trial began,
    # i.e. the callback thread interleaved with the running study.
    for _ in range(50):
        if callback_times:
            break
        time.sleep(0.05)
    assert callback_times, "callback never fired"
    assert min(callback_times) < executor_entries[-1], (
        "events must be delivered while the study is still running"
    )
