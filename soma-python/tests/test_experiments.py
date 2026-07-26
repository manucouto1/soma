"""Experiments journal: automatic records from studies and runs."""

from __future__ import annotations

import json
import pathlib

from soma import Graph, Study, experiments


def test_completed_study_appends_experiment(tmp_path):
    study = Study(
        "journal-study",
        search_space=[{"type": "float", "name": "x", "low": 0.0, "high": 1.0}],
        strategy="random",
        n_trials=3,
        objectives=[("f1", "maximize")],
        seed=5,
        root=str(tmp_path),
        tags=["mos"],
    )
    study.run(lambda trial: {"f1": trial["x"]})

    recs = experiments(str(tmp_path))
    assert len(recs) == 1
    rec = recs[0]
    assert rec["name"] == "journal-study"
    assert "f1" in rec["metrics"]
    assert "mos" in rec["tags"]
    run_tag = next(t for t in rec["tags"] if t.startswith("run:"))
    assert run_tag[len("run:") :] == pathlib.Path(study.run_dir).name


def test_finished_run_appends_experiment_with_summary(tmp_path):
    g = Graph()
    run = g.begin_run("journal-run", root=str(tmp_path), tags=["baseline"])
    run.log("val_f1", 0.7, step=0)
    run.log("val_f1", 0.85, step=1)  # summary keeps the last value
    run.log("loss", 0.3, step=1)
    run.finish()
    run.finish()  # idempotent — must not double-append

    recs = experiments(str(tmp_path))
    assert len(recs) == 1
    rec = recs[0]
    assert rec["name"] == "journal-run"
    assert rec["metrics"]["val_f1"] == 0.85
    assert rec["metrics"]["loss"] == 0.3
    assert f"run:{run.id}" in rec["tags"]


def test_experiments_tolerates_torn_tail(tmp_path):
    g = Graph()
    run = g.begin_run("r", root=str(tmp_path))
    run.log("f1", 0.5)
    run.finish()
    path = tmp_path / "experiments.jsonl"
    with path.open("a") as fh:
        fh.write('{"id": "torn')
    assert len(experiments(str(tmp_path))) == 1


def test_experiments_empty_when_missing(tmp_path):
    assert experiments(str(tmp_path)) == []


def test_failed_run_records_nothing(tmp_path):
    g = Graph()
    import pytest

    with pytest.raises(RuntimeError):
        with g.track_run("boom", root=str(tmp_path)):
            raise RuntimeError("no")
    assert experiments(str(tmp_path)) == []
    # but the run dir itself is preserved for debugging
    status = json.loads(
        (next((tmp_path / "runs").iterdir()) / "status.json").read_text()
    )
    assert status["state"] == "failed"


def test_study_record_content_is_complete(tmp_path):
    study = Study(
        "content",
        search_space=[{"type": "float", "name": "x", "low": 0.0, "high": 1.0}],
        strategy="grid",
        n_trials=3,
        objectives=[("f1", "maximize")],
        root=str(tmp_path),
    )
    study.run(lambda trial: {"f1": trial["x"]})

    rec = experiments(str(tmp_path))[0]
    assert rec["id"].startswith("study_")
    assert rec["pipeline_summary"] == "study over 3 trials"
    # Params are the BEST trial's params (grid max of f1=x is x=1.0).
    assert rec["params"]["x"] == 1.0
    assert rec["metrics"]["f1"] == 1.0
    assert "duration" in rec


def test_all_failed_study_records_nothing(tmp_path):
    study = Study(
        "all-fail",
        search_space=[{"type": "float", "name": "x", "low": 0.0, "high": 1.0}],
        strategy="random",
        n_trials=2,
        objectives=[("f1", "maximize")],
        seed=1,
        root=str(tmp_path),
    )
    study.run(lambda trial: (_ for _ in ()).throw(RuntimeError("boom")))
    assert experiments(str(tmp_path)) == []


def test_journal_preserves_append_order(tmp_path):
    g = Graph()
    for name in ["first", "second", "third"]:
        run = g.begin_run(name, root=str(tmp_path))
        run.log("f1", 0.5)
        run.finish()
    names = [r["name"] for r in experiments(str(tmp_path))]
    assert names == ["first", "second", "third"]


def test_corruption_in_the_middle_is_loud(tmp_path):
    g = Graph()
    run = g.begin_run("r", root=str(tmp_path))
    run.log("f1", 0.5)
    run.finish()

    path = tmp_path / "experiments.jsonl"
    content = path.read_text()
    path.write_text("corrupt line\n" + content)
    import pytest

    with pytest.raises(ValueError):
        experiments(str(tmp_path))


def test_blank_lines_are_tolerated(tmp_path):
    g = Graph()
    run = g.begin_run("r", root=str(tmp_path))
    run.log("f1", 0.5)
    run.finish()

    path = tmp_path / "experiments.jsonl"
    path.write_text("\n" + path.read_text() + "\n\n")
    assert len(experiments(str(tmp_path))) == 1
