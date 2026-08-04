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


def _run(root, name, *, params=None, depth=1, f1=0.5, parent=None, hypothesis=None):
    g = _graph(depth=depth)
    with g.track_run(
        name, root=str(root), params=params, parent=parent, hypothesis=hypothesis
    ) as run:
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


def _mcp_binary():
    """The debug-built MCP server, if this checkout has one."""
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "target" / "debug" / "somatize-mcp"
        if candidate.is_file():
            return candidate
    return None


@pytest.mark.skipif(
    _mcp_binary() is None,
    reason="needs `cargo build -p somatize-mcp` (skipped rather than built: too slow here)",
)
def test_runs_trained_here_are_queryable_over_mcp(tmp_path):
    """The two halves, joined: Python writes the pool, MCP reads it.

    Everything else tests one side or the other. This is the only test
    that proves a record produced by a real ``track_run`` is one the MCP
    tools can actually rank, follow and render.
    """
    import json as _json
    import subprocess

    # The realistic layout: the MCP server is pointed at a project, and
    # finds the pool at <project>/.soma/experiments.jsonl.
    root = pathlib.Path(tmp_path) / ".soma"
    baseline = _run(root, "mos-baseline", params={"lr": 0.01}, f1=0.81)
    variant = _run(root, "mos-wider", params={"lr": 0.05}, f1=0.87)
    assert (root / "experiments.jsonl").exists()

    proc = subprocess.Popen(
        [str(_mcp_binary()), str(tmp_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    counter = iter(range(1, 100))

    def rpc(method, params=None):
        proc.stdin.write(
            _json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": next(counter),
                    "method": method,
                    "params": params or {},
                }
            )
            + "\n"
        )
        proc.stdin.flush()
        return _json.loads(proc.stdout.readline())

    def call(tool, arguments):
        result = rpc("tools/call", {"name": tool, "arguments": arguments})["result"]
        assert not result.get("isError"), result["content"][0]["text"]
        return result["content"][0]["text"]

    try:
        rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}})
        names = [t["name"] for t in rpc("tools/list")["result"]["tools"]]
        assert "kb_find_similar" in names

        text = call("kb_find_similar", {"query": "wider", "limit": 5})
        assert "mos-wider" in text
        assert "lr: 0.01 → 0.05" in text, text
        assert "val_f1 +0.06" in text, text
        assert variant in text
        assert f"next: kb_lineage(id=\"{variant}\")" in text, text

        text = call("kb_lineage", {"id": variant})
        assert baseline in text and variant in text
        assert "1 ancestor, 0 descendants." in text, text
        assert "← lr: 0.01 → 0.05 ⇒ val_f1 +0.06" in text, text

        text = call("kb_diff", {"a": baseline, "b": variant})
        assert "- val_f1: 0.81 → 0.87 (+0.06)" in text, text
    finally:
        proc.stdin.close()
        proc.wait(timeout=10)


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


# ── Retain and retrieve ─────────────────────────────────────────────


def test_a_hypothesis_declared_at_run_start_reaches_the_journal(tmp_path):
    run_id = _run(tmp_path, "wider", hypothesis="two branches beat one")
    (record,) = _journal(tmp_path)
    assert record["id"] == run_id
    assert record["hypothesis"] == "two branches beat one"


def test_find_similar_ranks_by_relevance(tmp_path):
    _run(tmp_path, "dropout-sweep", f1=0.81)
    _run(tmp_path, "batch-norm-study", f1=0.79)

    hits = soma.find_similar("dropout", root=str(tmp_path))
    assert hits, "the pool is not empty, so something must rank"
    assert hits[0]["record"]["name"] == "dropout-sweep"
    assert hits[0]["score"] > 0
    assert set(hits[0]["components"]) == {
        "lexical",
        "structural",
        "recency",
        "importance",
    }
    assert hits[0]["why"].startswith("score ")

    assert len(soma.find_similar("dropout", limit=1, root=str(tmp_path))) == 1


def test_find_similar_needs_something_to_go_on(tmp_path):
    _run(tmp_path, "baseline")
    with pytest.raises(Exception, match="needs a query"):
        soma.find_similar(root=str(tmp_path))


def test_find_similar_can_match_on_architecture(tmp_path):
    baseline = _run(tmp_path, "baseline", depth=1)
    _run(tmp_path, "unrelated", depth=1)

    hits = soma.find_similar(like_run=baseline, root=str(tmp_path))
    assert hits
    assert all(h["components"]["structural"] > 0 for h in hits), (
        "matching on architecture must actually use the structural term"
    )


def test_a_conclusion_is_appended_and_becomes_retrievable(tmp_path):
    run_id = _run(tmp_path, "deeper", f1=0.79)
    before = _journal(tmp_path)

    amendment_id = soma.record_conclusion(
        run_id,
        "depth past 3 vanishes; the trunk needs residuals, not layers",
        tags=["dead-end"],
        root=str(tmp_path),
    )
    assert amendment_id.startswith("amend_")

    after = _journal(tmp_path)
    assert len(after) == len(before) + 1
    assert after[: len(before)] == before, "existing records are never rewritten"
    assert after[-1]["kind"] == "amendment"
    assert after[-1]["amends"] == run_id
    assert after[-1]["tags"] == ["dead-end"]

    # And the point of retaining it: it comes back.
    hits = soma.find_similar("residuals trunk", root=str(tmp_path))
    assert any("residuals" in (h["record"]["notes"] or "") for h in hits)


def test_record_conclusion_rejects_an_unknown_run(tmp_path):
    _run(tmp_path, "baseline")
    with pytest.raises(Exception, match="no experiment 'ghost'"):
        soma.record_conclusion("ghost", "...", root=str(tmp_path))


def test_lineage_walks_up_and_down(tmp_path):
    root_id = _run(tmp_path, "baseline")
    mid = _run(tmp_path, "wider")
    leaf = _run(tmp_path, "wider-deeper")

    tree = soma.lineage(mid, root=str(tmp_path))
    assert tree["focus"]["id"] == mid
    assert [a["id"] for a in tree["ancestors"]] == [root_id]
    assert [(d["record"]["id"], d["depth"]) for d in tree["descendants"]] == [(leaf, 1)]
    assert soma.lineage("nope", root=str(tmp_path)) is None


def test_diff_compares_two_siblings_that_never_met(tmp_path):
    baseline = _run(tmp_path, "baseline", params={"lr": 0.01}, f1=0.81)
    a = _run(tmp_path, "wider", params={"lr": 0.05}, f1=0.87)
    soma.checkout(baseline, root=str(tmp_path))
    b = _run(tmp_path, "deeper", params={"lr": 0.01, "depth": 4}, f1=0.79)

    # Neither is the other's parent, so no recorded derivation links them.
    records = {r["id"]: r for r in _journal(tmp_path)}
    assert records[a]["parent"] == records[b]["parent"] == baseline

    move = soma.diff(a, b, root=str(tmp_path))
    assert move["from"] == a and move["to"] == b
    changes = {c["change"] for c in move["changes"]}
    assert "ParamChanged" in changes
    assert move["metric_delta"]["val_f1"]["delta"] == pytest.approx(-0.08)
