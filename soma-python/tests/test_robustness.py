"""Robustness suite: real crash-recovery (SIGKILL), concurrent runs,
and a statistical check that TPE beats random search.

The expensive tests are marked `slow` and excluded from the default
run (`pytest -m slow` executes them; CI has a dedicated job)."""

from __future__ import annotations

import json
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time

import pytest

import soma
from soma import Graph, Study

CHILD_SCRIPT = """
import sys, time
from soma import Study

root = sys.argv[1]
study = Study(
    "crashable",
    search_space=[{"type": "float", "name": "x", "low": 0.0, "high": 1.0}],
    strategy="grid",
    n_trials=6,          # 6 points per dim -> 6 trials
    objectives=[("f1", "maximize")],
    root=root,
)

def train(trial):
    time.sleep(0.4)
    return {"f1": trial["x"]}

study.run(train)
"""


def _study_dir(root: pathlib.Path) -> pathlib.Path | None:
    runs = root / "runs"
    if not runs.exists():
        return None
    dirs = list(runs.iterdir())
    return dirs[0] if dirs else None


@pytest.mark.slow
def test_sigkill_mid_study_then_resume(tmp_path):
    """The durability story end-to-end: kill -9 a study mid-flight,
    then recover entirely from disk."""
    script = tmp_path / "child.py"
    script.write_text(CHILD_SCRIPT)

    proc = subprocess.Popen(
        [sys.executable, str(script), str(tmp_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        # Wait until at least 2 trials are durably recorded.
        deadline = time.monotonic() + 30
        run_dir = None
        while time.monotonic() < deadline:
            run_dir = _study_dir(tmp_path)
            if run_dir and (run_dir / "study.json").exists():
                trials = json.loads((run_dir / "study.json").read_text())["trials"]
                if len(trials) >= 2:
                    break
            if proc.poll() is not None:
                pytest.fail(f"child exited early: {proc.stderr.read().decode()}")
            time.sleep(0.05)
        else:
            pytest.fail("child never recorded 2 trials")

        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()

    # The corpse: status still says running (nobody finalized it) —
    # exactly how an external reader detects a crash (stale heartbeat).
    status = json.loads((run_dir / "status.json").read_text())
    assert status["state"] == "running"

    # study.json is a COMPLETE document despite the kill (atomic
    # tmp+rename), and events.jsonl may at worst have a torn tail,
    # which open() repairs.
    interrupted = Study.load(str(run_dir))
    n_before = interrupted.n_trials
    assert 2 <= n_before < 6

    interrupted.run(lambda trial: {"f1": trial["x"]}, resume=True)

    assert interrupted.n_trials == 6
    params = [t["params"]["x"] for t in interrupted.trials]
    assert len(set(params)) == 6, "no grid point repeated or skipped"
    assert [t["id"] for t in interrupted.trials] == [
        f"trial_{i:04}" for i in range(6)
    ]

    events = [
        json.loads(l) for l in (run_dir / "events.jsonl").read_text().splitlines()
    ]  # every line parses — the torn tail (if any) was repaired
    seqs = [e["seq"] for e in events]
    assert seqs == list(range(len(seqs))), "seq contiguous across the crash"
    assert json.loads((run_dir / "status.json").read_text())["state"] == "completed"


def test_concurrent_studies_share_a_root_safely(tmp_path):
    """Two studies running simultaneously against the same root must
    not collide: distinct run dirs, and a journal whose every line is
    intact (O_APPEND whole-line writes)."""
    errors = []

    def run_study(name, seed):
        try:
            study = Study(
                name,
                search_space=[{"type": "float", "name": "x", "low": 0.0, "high": 1.0}],
                strategy="random",
                n_trials=5,
                objectives=[("f1", "maximize")],
                seed=seed,
                root=str(tmp_path),
            )
            study.run(lambda trial: {"f1": trial["x"]})
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [
        threading.Thread(target=run_study, args=(f"concurrent-{i}", i))
        for i in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    run_dirs = list((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 2, "each study got its own directory"

    records = soma.experiments(str(tmp_path))  # every line parses
    assert sorted(r["name"] for r in records) == ["concurrent-0", "concurrent-1"]
    for run_dir in run_dirs:
        events = [
            json.loads(l)
            for l in (run_dir / "events.jsonl").read_text().splitlines()
        ]
        assert [e["seq"] for e in events] == list(range(len(events)))


def test_two_runs_on_one_graph_pinned_semantics(tmp_path):
    """CONTRACT (pinned): begin_run attaches a sink to the graph's
    shared bus, so two overlapping runs BOTH record every event emitted
    on that graph. finish() detaches only its own sink."""
    g = Graph()
    run_a = g.begin_run("overlap-a", root=str(tmp_path))
    run_b = g.begin_run("overlap-b", root=str(tmp_path))

    run_a.log("m", 1.0)  # emitted on the shared bus → both sinks
    run_a.finish()

    run_b.log("m", 2.0)  # a is detached; only b records this
    run_b.finish()

    def types(run):
        return [
            json.loads(l)["event_type"]
            for l in (pathlib.Path(run.dir) / "events.jsonl").read_text().splitlines()
        ]

    assert types(run_a) == ["MetricReported"]
    assert types(run_b) == ["MetricReported", "MetricReported"]


@pytest.mark.slow
def test_tpe_beats_random_statistically():
    """Regression guard for the ask/tell wiring: with the same budget,
    bayesian must beat random on a unimodal objective for a majority of
    seeds (2/3). This is exactly the behavior that was silently broken
    before record_result existed."""

    def best_of_tail(strategy, seed):
        study = Study(
            f"{strategy}-{seed}",
            search_space=[{"type": "float", "name": "x", "low": 0.0, "high": 1.0}],
            strategy=strategy,
            n_trials=40,
            objectives=[("score", "maximize")],
            seed=seed,
            tracking=False,
        )
        study.run(lambda trial: {"score": 1.0 - abs(trial["x"] - 0.7)})
        tail = [t["metrics"]["score"] for t in study.trials[-10:]]
        return sum(tail) / len(tail)

    wins = sum(
        best_of_tail("bayesian", seed) > best_of_tail("random", seed)
        for seed in (11, 23, 47)
    )
    assert wins >= 2, f"bayesian won only {wins}/3 seeds"
