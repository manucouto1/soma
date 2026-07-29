"""First-class experiment seeds (Phase 4): Study(seeds=[...]) crosses
every sampled config with every seed; each seed is an independent trial
and an independent cache line."""

from __future__ import annotations

import json
import os
import pathlib

from soma import Graph, Study


def test_study_seeds_cross_configs(tmp_path):
    study = Study(
        "seeded",
        search_space=[{"type": "float", "name": "x", "low": 0.0, "high": 1.0}],
        strategy="grid",
        n_trials=2,  # 2 grid points
        objectives=[("f1", "maximize")],
        seeds=[11, 22, 33],
        root=str(tmp_path),
    )

    seen: list[tuple[float, int]] = []

    def train(trial):
        seen.append((trial["x"], trial["seed"]))
        return {"f1": trial["x"]}

    study.run(train)

    # 2 configs × 3 seeds, config-major.
    assert study.n_trials == 6
    configs = sorted({x for x, _ in seen})
    assert len(configs) == 2
    for x in configs:
        assert sorted(s for cx, s in seen if cx == x) == [11, 22, 33]

    # Seeds recorded in the manifest.
    run_dir = pathlib.Path(study.run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["seeds"] == {"seed_0": 11, "seed_1": 22, "seed_2": 33}

    # Trials carry the seed in their persisted params (resume-safe).
    trials = json.loads((run_dir / "study.json").read_text())["trials"]
    assert all("seed" in t["params"] for t in trials)


def test_study_seeds_resume_mid_block(tmp_path):
    """Killing between seeds of one config must resume with the same
    config for the remaining seeds (recovered from persisted trials)."""
    calls: list[int] = []

    study = Study(
        "seeded-resume",
        search_space=[{"type": "float", "name": "x", "low": 0.0, "high": 1.0}],
        strategy="grid",
        n_trials=2,
        objectives=[("f1", "maximize")],
        seeds=[1, 2],
        root=str(tmp_path),
    )

    class StopEarly(Exception):
        pass

    def train_then_die(trial):
        calls.append(trial["seed"])
        if len(calls) == 3:  # die inside config 2, seed 1 done
            raise KeyboardInterrupt
        return {"f1": trial["x"]}

    try:
        study.run(train_then_die)
    except KeyboardInterrupt:
        pass

    resumed = Study.load(study.run_dir)
    resumed.run(lambda trial: {"f1": trial["x"]}, resume=True)
    assert resumed.n_trials >= 4
    params = [(t["params"]["x"], t["params"]["seed"]) for t in resumed.trials]
    assert len(set(params)) == len(params), "no (config, seed) pair repeated"
    xs = {x for x, _ in params}
    for x in xs:
        assert sorted(s for px, s in params if px == x) == [1, 2]


def test_graph_seed_creates_independent_cache_lines(tmp_path, monkeypatch):
    # monkeypatch RESTORES the previous values on teardown. A plain
    # `os.environ.pop` here used to delete conftest's session-scoped
    # SOMA_CACHE_DIR, silently pointing every later test at the
    # developer's real ~/.soma/cache (order-dependent failures).
    counters = tmp_path / "counters.txt"
    monkeypatch.setenv("SOMA_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("SOMA_TEST_SEED_COUNTERS", str(counters))

    class Trainer:
        _differentiable = False
        _cache_version = "test-v1"

        def fit(self, x, y=None):
            with open(os.environ["SOMA_TEST_SEED_COUNTERS"], "a") as f:
                f.write("fit\n")
            return {"n": len(x)}

        def forward(self, x, state):
            return x

    def run(seed):
        g = Graph()
        g.node("trainer", Trainer())
        g.fit([1.0, 2.0], seed=seed)

    run(seed=1)
    run(seed=2)  # different seed → its own cache line → refit
    run(seed=1)  # same seed → hit
    run(seed=2)  # same seed → hit

    fits = counters.read_text().count("fit")
    assert fits == 2, f"expected one fit per seed, got {fits}"
