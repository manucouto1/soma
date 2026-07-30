"""The experiment pool, end to end: run → conclusion → lineage → move.

These tests exercise the whole capture path against real run
directories, because that is the only place the join actually happens:
``begin_run`` writes the topology snapshot, ``finish`` summarizes the
directory, and the journal line has to come out with a conclusion, a
parent and a derivation on it.
"""

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


class _Model(Filter):
    def __init__(self, depth=1):
        self.depth = depth

    def fit(self, x, y=None):
        return {}

    def forward(self, x, state):
        return x


def _graph(depth=1):
    g = Graph()
    g.node("scaler", _Plain())
    g.node("model", _Model(depth=depth))
    g.connect("scaler", "model")
    return g


def _journal(root):
    path = pathlib.Path(root) / "experiments.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _run(root, name, *, params=None, depth=1, f1=0.5, parent=None):
    g = _graph(depth=depth)
    with g.track_run(name, root=str(root), params=params, parent=parent) as run:
        run.log("val_f1", f1, step=0)
    return run.id


# ── Capture ─────────────────────────────────────────────────────────


def test_begin_run_snapshots_the_topology(tmp_path):
    g = _graph()
    run = g.begin_run("snapshot", root=str(tmp_path))
    run.finish()

    run_dir = pathlib.Path(run.dir)
    fingerprint = json.loads((run_dir / "fingerprint.json").read_text())
    assert fingerprint["n_nodes"] == 2
    assert fingerprint["n_edges"] == 1
    assert fingerprint["nodes"] == {
        "model": "filter:_Model",
        "scaler": "filter:_Plain",
    }
    assert len(fingerprint["digest"]) == 64
    # Every node's filter identity is stamped in — the seam that makes
    # a NodeReconfigured diff possible later.
    assert set(fingerprint["node_config"]) == {"scaler", "model"}

    # graph.json/.mmd come from the same single writer.
    assert (run_dir / "graph.json").exists()
    assert (run_dir / "graph.mmd").read_text().startswith("graph")


def test_the_journal_line_carries_a_real_conclusion(tmp_path):
    run_id = _run(tmp_path, "baseline", params={"lr": 0.01}, f1=0.81)

    (record,) = _journal(tmp_path)
    assert record["id"] == run_id
    assert record["schema_version"] == 2
    assert record["kind"] == "experiment"
    assert record["run_dir"].endswith(run_id)

    # The old constant is gone: this is read off the run directory.
    assert record["pipeline_summary"] == "scaler(_Plain) → model(_Model)"
    assert record["pipeline_summary"] != "tracked run"

    conclusion = record["conclusion"]
    assert conclusion["outcome"] == "completed"
    assert conclusion["headline"].startswith("completed in ")
    assert "val_f1=0.81" in conclusion["headline"]
    assert record["metrics"]["val_f1"] == 0.81
    assert record["params"]["lr"] == 0.01
    assert record["architecture"]["n_nodes"] == 2
    assert record["research_line"] == "baseline"


def test_a_failed_run_is_not_recorded_and_does_not_move_head(tmp_path):
    g = _graph()
    with pytest.raises(RuntimeError):
        with g.track_run("doomed", root=str(tmp_path)):
            raise RuntimeError("boom")

    assert _journal(tmp_path) == []
    assert soma.head(root=str(tmp_path)) is None


# ── Lineage ─────────────────────────────────────────────────────────


def test_head_advances_only_on_success(tmp_path):
    assert soma.head(root=str(tmp_path)) is None
    first = _run(tmp_path, "first")
    assert soma.head(root=str(tmp_path)) == first

    g = _graph()
    with pytest.raises(RuntimeError):
        with g.track_run("crash", root=str(tmp_path)):
            raise RuntimeError("boom")
    assert soma.head(root=str(tmp_path)) == first, "a crash must not become a parent"


def test_a_variant_records_its_move_with_a_signed_delta(tmp_path):
    baseline = _run(tmp_path, "baseline", params={"lr": 0.01}, f1=0.81)
    variant = _run(tmp_path, "wider", params={"lr": 0.05}, f1=0.87)

    records = {r["id"]: r for r in _journal(tmp_path)}
    child = records[variant]
    assert child["parent"] == baseline
    # A variant stays in the line it branched from, whatever it is called.
    assert child["research_line"] == records[baseline]["research_line"] == "baseline"

    derivation = child["derivation"]
    assert derivation["from"] == baseline
    assert derivation["to"] == variant
    assert derivation["changes"] == [
        {"change": "ParamChanged", "key": "lr", "from": 0.01, "to": 0.05}
    ]
    delta = derivation["metric_delta"]["val_f1"]
    assert delta["before"] == 0.81
    assert delta["after"] == 0.87
    assert delta["delta"] == pytest.approx(0.06)
    assert "lr: 0.01 → 0.05" in derivation["summary"]
    assert "val_f1 +0.06" in derivation["summary"]


def test_checkout_rewinds_so_the_next_run_is_a_sibling(tmp_path):
    baseline = _run(tmp_path, "baseline", f1=0.81)
    variant = _run(tmp_path, "wider", f1=0.87)

    soma.checkout(baseline, root=str(tmp_path))
    assert soma.head(root=str(tmp_path)) == baseline
    sibling = _run(tmp_path, "deeper", depth=4, f1=0.79)

    records = {r["id"]: r for r in _journal(tmp_path)}
    assert records[variant]["parent"] == baseline
    assert records[sibling]["parent"] == baseline, "a sibling, not a grandchild"

    # Reconfiguring a filter is visible without any params at all.
    changes = records[sibling]["derivation"]["changes"]
    assert [c["change"] for c in changes] == ["NodeReconfigured"]
    assert changes[0]["node"] == "model"


def test_an_explicit_parent_beats_head(tmp_path):
    baseline = _run(tmp_path, "baseline")
    _run(tmp_path, "middle")
    child = _run(tmp_path, "explicit", parent=baseline)

    records = {r["id"]: r for r in _journal(tmp_path)}
    assert records[child]["parent"] == baseline


def test_checkout_refuses_an_unknown_run(tmp_path):
    _run(tmp_path, "baseline")
    with pytest.raises(Exception, match="no run 'nope'"):
        soma.checkout("nope", root=str(tmp_path))


def test_detach_starts_a_new_line(tmp_path):
    _run(tmp_path, "baseline")
    soma.detach(root=str(tmp_path))
    assert soma.head(root=str(tmp_path)) is None

    orphan = _run(tmp_path, "fresh-start")
    records = {r["id"]: r for r in _journal(tmp_path)}
    assert records[orphan]["parent"] is None
    assert records[orphan]["research_line"] == "fresh-start"


# ── Recovery ────────────────────────────────────────────────────────


def test_reindex_rebuilds_the_journal_from_the_run_dirs(tmp_path):
    baseline = _run(tmp_path, "baseline", params={"lr": 0.01}, f1=0.81)
    variant = _run(tmp_path, "wider", params={"lr": 0.05}, f1=0.87)
    before = _journal(tmp_path)

    (pathlib.Path(tmp_path) / "experiments.jsonl").unlink()
    assert soma.reindex(root=str(tmp_path)) == 2

    after = _journal(tmp_path)
    assert [r["id"] for r in after] == [baseline, variant]
    assert after[1]["parent"] == baseline
    assert after[1]["derivation"]["summary"] == before[1]["derivation"]["summary"]


def test_run_summary_works_on_a_run_recorded_before_the_pool(tmp_path):
    # A run directory with nothing but a manifest — what a pre-pool run
    # or a hard crash leaves behind. It must still summarize.
    run_dir = pathlib.Path(tmp_path) / "runs" / "run_ancient"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "run_ancient",
                "kind": "train",
                "name": "ancient",
                "created_at": "2026-01-01T00:00:00Z",
            }
        )
    )

    summary = json.loads(soma._soma.run_summary_json(str(run_dir)))
    assert summary["run_id"] == "run_ancient"
    assert summary["pipeline_summary"] == ""
    assert summary["architecture"] is None
    assert summary["conclusion"]["headline"]
    assert soma.reindex(root=str(tmp_path)) == 1
