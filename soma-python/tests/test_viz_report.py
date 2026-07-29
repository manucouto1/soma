"""HTML report: embedded JSON data blobs (the front-end contract),
figure specs, section structure, and the offline --inline mode."""

from __future__ import annotations

import json
import re

import pytest

pytest.importorskip("plotly")

import soma
from soma import Filter, Graph, Study
from soma._cache_cli import main as cli_main

# Stable blob ids — the contract a future front-end reads. Keep in sync
# with docs/src/content/docs/design/tracking.md.
DATA_BLOB_IDS = [
    "soma-data-info",
    "soma-data-manifest",
    "soma-data-overlay",
    "soma-data-node-timings",
    "soma-data-cache",
    "soma-data-metrics",
    "soma-data-health-flags",
    "soma-data-trial-timeline",
]


class _Plain(Filter):
    def fit(self, x, y=None):
        return {}

    def forward(self, x, state):
        return x


def _tracked_fit(tmp_path):
    g = Graph()
    g.node("a", _Plain())
    g.node("b", _Plain())
    g.connect("a", "b")
    with g.track_run("report-run", root=str(tmp_path), kind="fit") as run:
        g.fit([1.0, 2.0])
        run.log("val_f1", 0.9, step=1)
    return soma.RunView(run.dir)


def _blob(doc: str, blob_id: str):
    m = re.search(
        f'<script type="application/json" id="{blob_id}">(.*?)</script>', doc, re.S
    )
    assert m, f"missing blob {blob_id}"
    return json.loads(m.group(1))


def test_report_embeds_all_data_blobs(tmp_path):
    view = _tracked_fit(tmp_path)
    doc = view.to_html()

    assert doc.startswith("<!doctype html>")
    assert "<title>soma report — report-run</title>" in doc
    for blob_id in DATA_BLOB_IDS:
        _blob(doc, blob_id)

    info = _blob(doc, "soma-data-info")
    assert info["state"] == "completed"
    timings = _blob(doc, "soma-data-node-timings")
    assert [s["node_id"] for s in timings] == ["a", "b"]
    metrics = _blob(doc, "soma-data-metrics")
    assert metrics[0]["name"] == "val_f1"

    # Rendered sections: architecture (mermaid), gantt + metrics figures.
    assert '<pre class="mermaid">' in doc
    assert 'id="soma-fig-gantt"' in doc
    assert 'id="soma-fig-metrics"' in doc
    assert 'id="soma-fig-health"' in doc
    # Figure specs parse as plotly JSON.
    gantt = _blob(doc, "soma-fig-gantt")
    assert gantt["data"], "gantt has traces"


def _external_refs(doc: str) -> list[str]:
    """src/href attributes of actual script/link TAGS that point at a
    network URL. (Never scan the raw document: the inlined plotly.js
    bundle itself contains 'src=\"https' as string literals.)"""
    return [
        url
        for tag, url in re.findall(
            r"<(script|link)\b[^>]*?(?:src|href)=\"(https?://[^\"]+)\"", doc
        )
    ]


def test_report_inline_needs_no_network(tmp_path):
    view = _tracked_fit(tmp_path)
    doc = view.to_html(inline=True)
    assert _external_refs(doc) == [], "no external scripts in --inline mode"
    assert "Plotly" in doc, "plotly.js embedded"
    # DAG falls back to source form offline.
    assert 'class="mermaid-src"' in doc
    assert '<pre class="mermaid">' not in doc

    # The CDN variant, by contrast, references plotly (and mermaid).
    online = view.to_html()
    assert any("plot.ly" in u or "plotly" in u for u in _external_refs(online))


def test_report_writes_file(tmp_path):
    view = _tracked_fit(tmp_path)
    out = tmp_path / "report.html"
    returned = view.to_html(path=str(out))
    assert out.read_text() == returned


def test_study_report_has_hpo_sections(tmp_path):
    def objective(trial):
        for step in range(3):
            trial.report("f1", trial["x"] + 0.05 * step, step)
        return None

    study = Study(
        "report-hpo",
        search_space=[{"type": "float", "name": "x", "low": 0.1, "high": 0.9}],
        strategy="random",
        n_trials=4,
        objectives=[("f1", "maximize")],
        seed=5,
        root=str(tmp_path),
    )
    study.run(objective)

    doc = study.to_html()
    assert 'id="soma-fig-history"' in doc
    assert 'id="soma-fig-intermediate"' in doc
    assert 'id="soma-fig-timeline"' in doc
    assert 'id="soma-fig-importances"' in doc
    assert "<h2>Trials</h2>" in doc
    assert doc.count("trial_") >= 4, "all trials in the table"
    timeline = _blob(doc, "soma-data-trial-timeline")
    assert len(timeline) == 4


def test_study_without_tracking_raises():
    study = Study(
        "untracked",
        search_space=[{"type": "float", "name": "x", "low": 0.0, "high": 1.0}],
        strategy="random",
        n_trials=1,
        objectives=[("f1", "maximize")],
        tracking=False,
    )
    with pytest.raises(ValueError, match="no run directory"):
        study.to_html()


def test_cli_soma_report(tmp_path, capsys, monkeypatch):
    view = _tracked_fit(tmp_path)
    monkeypatch.chdir(tmp_path)

    assert cli_main(["report", view.id, "--root", str(tmp_path)]) == 0
    out_msg = capsys.readouterr().out
    assert "report written to" in out_msg
    produced = tmp_path / f"{view.id}.html"
    assert produced.exists()
    assert "soma-data-info" in produced.read_text()

    out = tmp_path / "custom.html"
    assert cli_main(["report", view.dir, "-o", str(out), "--inline"]) == 0
    assert out.exists()
    assert _external_refs(out.read_text()) == []

    assert cli_main(["report", "missing-run", "--root", str(tmp_path)]) == 1
