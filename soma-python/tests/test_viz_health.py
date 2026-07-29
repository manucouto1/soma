"""Health visualization: flag scatter, audit time series, channel
snapshots. Fixtures fabricate the diagnostics files the audit system
writes, so these tests need no torch."""

from __future__ import annotations

import json
import pathlib

import pytest

pytest.importorskip("plotly")

import soma
from soma import Filter, Graph


class _Plain(Filter):
    def fit(self, x, y=None):
        return {}

    def forward(self, x, state):
        return x


def _run_with_diagnostics(tmp_path, *, flags=(), audit_steps=(), snapshots=()):
    g = Graph()
    g.node("enc", _Plain())
    g.node("head", _Plain())
    g.connect("enc", "head")
    with g.track_run("diag", root=str(tmp_path)) as run:
        g.fit([1.0])
        for f in flags:
            g.emit_event({"event_type": "HealthFlag", "run_id": run.id, **f})
        run_dir = pathlib.Path(run.dir)
        diag = run_dir / "diagnostics"
        if audit_steps:
            diag.mkdir(exist_ok=True)
            with (diag / "audit_steps.jsonl").open("w") as fh:
                for line in audit_steps:
                    fh.write(json.dumps(line) + "\n")
        if snapshots:
            import numpy as np
            from safetensors.numpy import save_file

            chan = diag / "channels"
            with_index = []
            for snap in snapshots:
                fdir = chan / snap["filter"]
                fdir.mkdir(parents=True, exist_ok=True)
                fname = f"step_{snap['step']:06}.safetensors"
                save_file(
                    {
                        "corr": np.array(snap["corr"], dtype=np.float32),
                        "act_zero_frac": np.array(snap["zero_frac"], dtype=np.float32),
                    },
                    str(fdir / fname),
                )
                with_index.append(
                    {
                        "filter": snap["filter"],
                        "step": snap["step"],
                        "file": f"{snap['filter']}/{fname}",
                        "keys": ["corr", "act_zero_frac"],
                        "eff_rank": snap.get("eff_rank"),
                        "cka": snap.get("cka", {}),
                    }
                )
            with (chan / "index.jsonl").open("w") as fh:
                for line in with_index:
                    fh.write(json.dumps(line) + "\n")
    return soma.RunView(run.dir)


def _audit_line(fid, step, norm, zero_frac=0.0):
    return {
        "filter": fid,
        "step": step,
        "ts": 0.0,
        "act": {"abs_mean": 0.5, "zero_frac": zero_frac},
        "out_grad": {"norm": norm, "max": norm * 2},
        "param": {"grad_norm": norm / 2, "grad_param_ratio": 1e-3},
    }


def test_plot_health_marks_flags_by_family(tmp_path):
    view = _run_with_diagnostics(
        tmp_path,
        flags=[
            {"node_id": "enc", "step": 3, "flag": "DEAD_CHANNELS(4)", "detail": "4/64"},
            {"node_id": "enc", "step": 5, "flag": "DEAD_CHANNELS(6)", "detail": "6/64"},
            {"node_id": "head", "step": 5, "flag": "LEAKAGE", "detail": "cka=0.98"},
        ],
    )
    fig = view.plot_health()
    assert {t.name for t in fig.data} == {"DEAD_CHANNELS", "LEAKAGE"}
    dead = next(t for t in fig.data if t.name == "DEAD_CHANNELS")
    assert list(dead.x) == [3, 5]
    assert list(dead.y) == ["enc", "enc"]


def test_plot_health_empty_is_a_statement(tmp_path):
    view = _run_with_diagnostics(tmp_path)
    fig = view.plot_health()
    assert len(fig.data) == 0
    assert "No health flags" in fig.layout.annotations[0].text


def test_plot_audit_series_and_log_axis(tmp_path):
    view = _run_with_diagnostics(
        tmp_path,
        audit_steps=[_audit_line(f, s, norm=10.0 ** -s) for f in ("enc", "head") for s in range(5)],
    )
    fig = view.plot_audit("out_grad.norm")
    assert {t.name for t in fig.data} == {"enc", "head"}
    assert fig.layout.yaxis.type == "log", "norms default to log scale"
    enc = next(t for t in fig.data if t.name == "enc")
    assert list(enc.x) == [0, 1, 2, 3, 4]

    linear = view.plot_audit("act.zero_frac")
    assert linear.layout.yaxis.type == "linear"

    with pytest.raises(ValueError, match="not present"):
        view.plot_audit("act.nonexistent")


def test_plot_audit_without_diagnostics_raises(tmp_path):
    view = _run_with_diagnostics(tmp_path)
    with pytest.raises(ValueError, match="audit_steps"):
        view.plot_audit()


def test_plot_channels_heatmap_and_dead_marks(tmp_path):
    pytest.importorskip("safetensors")
    corr = [[1.0, 0.2, 0.0], [0.2, 1.0, -0.4], [0.0, -0.4, 1.0]]
    view = _run_with_diagnostics(
        tmp_path,
        snapshots=[
            {
                "filter": "enc",
                "step": 10,
                "corr": corr,
                "zero_frac": [0.0, 0.99, 0.1],
                "eff_rank": 2.5,
                "cka": {"g0|g1": 0.4},
            }
        ],
    )
    fig = view.plot_channels("enc")
    z = fig.data[0].z
    assert z.shape == (3, 3)
    assert fig.data[0].zmin == -1 and fig.data[0].zmax == 1
    assert "ch1 †" in fig.data[0].x, "dead channel marked"
    assert "1 dead channel" in fig.layout.title.text
    assert "eff. rank 2.5" in fig.layout.title.text

    with pytest.raises(ValueError, match="no channel snapshots"):
        view.plot_channels("missing")


def test_plot_channel_evolution(tmp_path):
    pytest.importorskip("safetensors")
    eye = [[1.0, 0.0], [0.0, 1.0]]
    view = _run_with_diagnostics(
        tmp_path,
        snapshots=[
            {"filter": "enc", "step": 0, "corr": eye, "zero_frac": [0, 0],
             "eff_rank": 2.0, "cka": {"a|b": 0.2}},
            {"filter": "enc", "step": 10, "corr": eye, "zero_frac": [0, 0],
             "eff_rank": 1.2, "cka": {"a|b": 0.9}},
        ],
    )
    fig = view.plot_channel_evolution("enc")
    names = {t.name for t in fig.data}
    assert names == {"enc eff. rank", "enc max CKA"}
    rank = next(t for t in fig.data if "rank" in t.name)
    assert list(rank.y) == [2.0, 1.2], "collapse trajectory visible"
    cka = next(t for t in fig.data if "CKA" in t.name)
    assert list(cka.y) == [0.2, 0.9]
