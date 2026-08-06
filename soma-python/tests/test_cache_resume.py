"""The motivating scenario for the persistent cache: a multi-trial study
killed mid-flight (SIGKILL) must not lose completed pipeline work.

On re-run: completed trials are skipped (existing study resume), and —
new with the persistent cache — filter-level fit work inside re-executed
pipelines is served from `$SOMA_CACHE_DIR` instead of recomputed.

Both the crashed process and the resuming process import the SAME
filter module (as real experiments do): since Phase 2 the filter's
source code is part of its cache identity, so defining "the same"
filter with different source text is a legitimately different filter.

Execution counts are tracked via append-only side-effect files written
by the filters themselves, so they survive the kill and span processes.
"""

from __future__ import annotations

import json
import os
import pathlib
import signal
import subprocess
import sys
import textwrap
import time

import pytest

# Shared filter module: written once, imported by child AND resumer.
FILTERS_MODULE = textwrap.dedent(
    """
    import os, time

    COUNTERS = os.environ["SOMA_TEST_COUNTERS"]


    class Prep:
        _differentiable = False

        def fit(self, x, y=None):
            with open(COUNTERS, "a") as f:
                f.write("prep_fit\\n")
            time.sleep(0.05)
            return {"mean": sum(x) / len(x)}

        def forward(self, x, state):
            return [v - state["mean"] for v in x]


    class Model:
        _differentiable = False

        def __init__(self, scale=1.0):
            self.scale = scale

        def fit(self, x, y=None):
            with open(COUNTERS, "a") as f:
                f.write("model_fit\\n")
            time.sleep(0.4)
            return {"w": self.scale}

        def forward(self, x, state):
            return [v * state["w"] for v in x]


    X = [1.0, 2.0, 3.0, 4.0]


    def train_factory(Graph):
        def train(trial):
            g = Graph()  # default: persistent tiered cache at $SOMA_CACHE_DIR
            g.node("prep", Prep())
            g.node("model", Model(scale=trial["scale"]))
            g.edge("prep", "model")
            g.fit(X)
            return {"f1": trial["scale"]}

        return train
    """
)

CHILD_SCRIPT = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, sys.argv[2])
    from soma import Graph, Study
    import exp_filters

    study = Study(
        "cache-crashable",
        search_space=[{"type": "float", "name": "scale", "low": 0.0, "high": 1.0}],
        strategy="grid",
        n_trials=6,
        objectives=[("f1", "maximize")],
        root=sys.argv[1],
    )
    study.run(exp_filters.train_factory(Graph))
    """
)

RESUME_SCRIPT = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, sys.argv[2])
    from soma import Graph, Study
    import exp_filters

    study = Study.load(sys.argv[1])
    study.run(exp_filters.train_factory(Graph), resume=True)
    """
)


def _study_dir(root: pathlib.Path) -> pathlib.Path | None:
    runs = root / "runs"
    if not runs.exists():
        return None
    dirs = list(runs.iterdir())
    return dirs[0] if dirs else None


def _counts(counters: pathlib.Path) -> dict[str, int]:
    out: dict[str, int] = {}
    if counters.exists():
        for line in counters.read_text().splitlines():
            out[line] = out.get(line, 0) + 1
    return out


@pytest.mark.slow
def test_sigkill_mid_study_resumes_from_cache(tmp_path):
    """Kill -9 a study mid-flight; the re-run must reuse every completed
    fit via the persistent cache instead of recomputing it."""
    cache_dir = tmp_path / "cache"
    counters = tmp_path / "counters.txt"
    mod_dir = tmp_path / "mod"
    mod_dir.mkdir()
    (mod_dir / "exp_filters.py").write_text(FILTERS_MODULE)
    child = tmp_path / "child.py"
    child.write_text(CHILD_SCRIPT)
    resumer = tmp_path / "resume.py"
    resumer.write_text(RESUME_SCRIPT)

    env = dict(
        os.environ,
        SOMA_CACHE_DIR=str(cache_dir),
        SOMA_TEST_COUNTERS=str(counters),
    )
    proc = subprocess.Popen(
        [sys.executable, str(child), str(tmp_path), str(mod_dir)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env=env,
    )
    try:
        # Wait until at least 2 trials are durably recorded, then kill.
        deadline = time.monotonic() + 60
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

    before = _counts(counters)
    n_completed = len(json.loads((run_dir / "study.json").read_text())["trials"])
    # Prep has a fixed config and fixed input: computed once EVER, all
    # later trials hit the cache (persistent tier) inside the child.
    assert before.get("prep_fit") == 1

    # Resume in a fresh process (a "new machine day"): completed trials
    # are skipped, and the re-executed remainder finds states in cache.
    proc2 = subprocess.run(
        [sys.executable, str(resumer), str(run_dir), str(mod_dir)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc2.returncode == 0, proc2.stderr

    after = _counts(counters)

    # The shared stage was NEVER refit — not even once across the crash.
    assert after.get("prep_fit") == 1, f"prep was refit after the crash: {after}"

    # Every distinct model config is fit at most once ever, except the
    # single in-flight trial killed between fit() and cache.put().
    assert 6 <= after.get("model_fit", 0) <= 7, (
        f"model fits not bounded by cache reuse: {after} "
        f"(completed before kill: {n_completed})"
    )

    # And the study itself completed exactly.
    trials = json.loads((run_dir / "study.json").read_text())["trials"]
    assert len(trials) == 6
    assert len({t["params"]["scale"] for t in trials}) == 6


def test_rerunning_identical_fit_uses_persistent_cache(tmp_path):
    """The cross-investigation case, no crash involved: two independent
    processes fitting the same pipeline on the same data — the second
    must do zero fit work."""
    cache_dir = tmp_path / "cache"
    counters = tmp_path / "counters.txt"
    mod_dir = tmp_path / "mod"
    mod_dir.mkdir()
    (mod_dir / "exp_filters.py").write_text(FILTERS_MODULE)
    script = tmp_path / "fit_once.py"
    script.write_text(
        textwrap.dedent(
            """
            import sys
            sys.path.insert(0, sys.argv[1])
            from soma import Graph
            import exp_filters

            g = Graph()
            g.node("prep", exp_filters.Prep())
            g.node("model", exp_filters.Model(scale=3.0))
            g.edge("prep", "model")
            g.fit(exp_filters.X)
            """
        )
    )

    env = dict(
        os.environ,
        SOMA_CACHE_DIR=str(cache_dir),
        SOMA_TEST_COUNTERS=str(counters),
    )
    for _ in range(2):
        proc = subprocess.run(
            [sys.executable, str(script), str(mod_dir)],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, proc.stderr

    counts = _counts(counters)
    assert counts.get("prep_fit") == 1 and counts.get("model_fit") == 1, (
        "the second process must reuse the first process's fit states "
        f"from the persistent cache, got: {counts}"
    )


def test_memory_cache_opt_out_disables_persistence(tmp_path):
    """`cache='memory'` must restore the old behavior: nothing persists."""
    counters = tmp_path / "counters.txt"
    script = tmp_path / "fit_mem.py"
    script.write_text(
        textwrap.dedent(
            """
            import os, sys
            from soma import Graph

            counters = os.environ["SOMA_TEST_COUNTERS"]


            class Scaler:
                _differentiable = False

                def fit(self, x, y=None):
                    with open(counters, "a") as f:
                        f.write("fit\\n")
                    return {"m": 1.0}

                def forward(self, x, state):
                    return x


            g = Graph(cache="memory")
            g.node("scaler", Scaler())
            g.fit([1.0, 2.0])
            """
        )
    )
    env = dict(
        os.environ,
        SOMA_CACHE_DIR=str(tmp_path / "cache"),
        SOMA_TEST_COUNTERS=str(counters),
    )
    for _ in range(2):
        proc = subprocess.run(
            [sys.executable, str(script)],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, proc.stderr

    assert _counts(counters).get("fit") == 2


# ── one lever: "memory", or where to keep it ────────────────


def test_cache_accepts_a_path_and_persists_there(tmp_path):
    """`cache=<dir>` is the whole answer to "where do I want this kept".

    It used to take three arguments — a kind (`"local"` / `"tiered"`) and
    a separate `cache_path` — to say this, and `"local"` asked for a disk
    store with no LRU in front, which is strictly worse than the default
    and was never what anyone wanted.
    """
    import soma

    class Counting(soma.Filter):
        _kind = "stateless"
        _cache_version = "1"
        calls = 0

        def forward(self, x, state):
            type(self).calls += 1
            return [v + 1 for v in x]

    store = tmp_path / "store"

    g1 = soma.Graph(cache=str(store))
    g1.node("n", Counting())
    assert g1.forward([1.0]) == [2.0]
    assert Counting.calls == 1
    assert store.exists(), "the directory was created and written to"

    # A different Graph over the same directory reuses the work.
    g2 = soma.Graph(cache=str(store))
    g2.node("n", Counting())
    assert g2.forward([1.0]) == [2.0]
    assert Counting.calls == 1, "served from the store on disk"


def test_cache_memory_does_not_touch_the_disk(tmp_path, monkeypatch):
    import soma

    monkeypatch.setenv("SOMA_CACHE_DIR", str(tmp_path / "unused"))
    g = soma.Graph(cache="memory")
    g.node("n", soma.Filter())
    assert not (tmp_path / "unused").exists()
