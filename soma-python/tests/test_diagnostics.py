"""Channel-level diagnostics + tracked-run persistence.

Covers: dead channels, ignored (gradient-starved) channels, group CKA
leakage, snapshot cadence, per-step time series, audit_steps.jsonl,
safetensors channel snapshots, report.json, HealthFlag events, and
track_run lifecycle.
"""

from __future__ import annotations

import json
import pathlib

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

import soma  # noqa: E402
from soma import ChannelConfig, DifferentiableFilter, Graph  # noqa: E402


# ── Test filters ────────────────────────────────────────────────────


class _Shape(nn.Module):
    """Zeroes `dead` channels / mirrors channel ranges INSIDE the
    module, so audit hooks observe the shaped output."""

    def __init__(self, dead, duplicate):
        super().__init__()
        self.dead = tuple(dead)
        self.duplicate = duplicate  # (src_idx_list, dst_idx_list)

    def forward(self, x):
        if self.dead:
            mask = torch.ones(x.shape[1], device=x.device)
            for c in self.dead:
                mask[c] = 0.0
            x = x * mask
        if self.duplicate:
            src, dst = self.duplicate
            y = x.clone()
            y[:, dst] = x[:, src]
            x = y
        return x


class ChannelBlock(DifferentiableFilter):
    """Linear block whose (B, C) output can have forced-dead channels
    or duplicated channel ranges."""

    def __init__(self, out_dim=8, dead=(), detached=(), duplicate=None, **kw):
        super().__init__(**kw)
        self.out_dim = out_dim
        self._dead = tuple(dead)
        self._detached = tuple(detached)  # excluded from the loss only
        self._duplicate = duplicate

    def build_module(self, input_shape):
        return nn.Sequential(
            nn.Linear(input_shape[-1], self.out_dim),
            _Shape(self._dead, self._duplicate),
        )

    def output_shape(self, input_shape):
        return (*input_shape[:-1], self.out_dim)


def _setup(g, batch=16):
    """Materialize and prepare the optimizer BEFORE entering the audit
    context (hooks install only on already-materialized modules)."""
    x = torch.randn(batch, 8)
    g.materialize(x)
    g.train()
    g.make_optimizer(lr=0.01)
    return x


def _train_steps(g, block, x, n_steps=6):
    """Drive the native loop; the loss uses only non-dead, non-detached
    channels so 'detached' channels stay alive forward but starved."""
    module = dict(g.filters())["block"]._module

    for _ in range(n_steps):
        with g.context() as ctx:
            g.zero_grad()
            out = module(x)
            keep = [
                c
                for c in range(out.shape[1])
                if c not in block._dead and c not in block._detached
            ]
            loss = (out[:, keep] ** 2).mean()
            g.backward(ctx, loss)
        g.step(ctx)


def _one_block_graph(**block_kw):
    g = Graph()
    block = ChannelBlock(**block_kw)
    g.node("block", block)
    return g, block


# ── Channel flags ───────────────────────────────────────────────────


def test_dead_channels_flagged():
    g, block = _one_block_graph(dead=(2, 5))
    x = _setup(g)
    with g.gradient_audit(channels=ChannelConfig(snapshot_every=2)) as audit:
        _train_steps(g, block, x)
    rep = audit.report().by_id()["block"]
    assert rep.metrics["dead_channels"] >= 2
    assert any(f.startswith("DEAD_CHANNELS") for f in rep.flags), rep.flags


def test_ignored_channels_flagged():
    g, block = _one_block_graph(detached=(1, 3))
    x = _setup(g)
    with g.gradient_audit(channels=True) as audit:
        _train_steps(g, block, x)
    rep = audit.report().by_id()["block"]
    assert rep.metrics["ignored_channels"] >= 1
    assert any(f.startswith("IGNORED_CHANNELS") for f in rep.flags), rep.flags


def test_healthy_block_has_no_channel_flags():
    g, block = _one_block_graph()
    x = _setup(g)
    with g.gradient_audit(channels=True) as audit:
        _train_steps(g, block, x)
    rep = audit.report().by_id()["block"]
    assert rep.metrics["dead_channels"] == 0
    assert not [f for f in rep.flags if f.startswith(("DEAD_CHANNELS", "IGNORED"))]


def test_duplicated_group_flags_leakage():
    # Channels 4-7 mirror channels 0-3 inside the module → the two
    # declared groups share all information: cross-group CKA ≈ 1.
    g, block = _one_block_graph(
        duplicate=(list(range(0, 4)), list(range(4, 8)))
    )
    x = _setup(g, batch=32)
    cfg = ChannelConfig(
        snapshot_every=1, groups={"block": {"a": range(0, 4), "b": range(4, 8)}}
    )
    with g.gradient_audit(channels=cfg) as audit:
        _train_steps(g, block, x, n_steps=3)

    rep = audit.report().by_id()["block"]
    assert rep.metrics["max_group_cka"] > 0.95
    assert "LEAKAGE" in rep.flags


# ── Time series & cadence ───────────────────────────────────────────


def test_records_expose_per_step_series():
    g, block = _one_block_graph()
    x = _setup(g)
    with g.gradient_audit() as audit:
        _train_steps(g, block, x, n_steps=4)
    series = audit.timeseries("block")
    assert len(series) == 4
    assert [r["step"] for r in series] == [0, 1, 2, 3]
    assert all("act" in r and "param" in r and "out_grad" in r for r in series)
    # records() returns the raw dataclasses
    recs = audit.records()["block"]
    assert recs[0].step == 0 and recs[0].ts > 0


def test_snapshot_cadence(tmp_path):
    g, block = _one_block_graph()
    x = _setup(g)
    with g.track_run("cadence", root=str(tmp_path)):
        with g.gradient_audit(channels=ChannelConfig(snapshot_every=3)) as audit:
            _train_steps(g, block, x, n_steps=7)

    run_dir = next((tmp_path / "runs").iterdir())
    index = [
        json.loads(l)
        for l in (run_dir / "diagnostics/channels/index.jsonl").read_text().splitlines()
    ]
    # snapshots at steps 0, 3, 6
    assert [e["step"] for e in index] == [0, 3, 6]
    for entry in index:
        assert (run_dir / "diagnostics/channels" / entry["file"]).exists()
        assert "corr" in entry["keys"]
        assert entry["eff_rank"] is None or entry["eff_rank"] > 0


# ── Persistence inside a tracked run ────────────────────────────────


def test_tracked_audit_persists_everything(tmp_path):
    g, block = _one_block_graph(dead=(2,))
    x = _setup(g)
    with g.track_run("full", root=str(tmp_path), tags=["mos"]) as run:
        with g.gradient_audit(channels=ChannelConfig(snapshot_every=2)) as audit:
            _train_steps(g, block, x, n_steps=4)
        run.log("val_f1", 0.8, step=0)

    run_dir = pathlib.Path(run.dir)

    # graph topology snapshot
    graph = json.loads((run_dir / "graph.json").read_text())
    assert [n["id"] for n in graph["nodes"]] == ["block"]
    assert (run_dir / "graph.mmd").read_text().startswith("graph")

    # per-step scalars: one line per filter per step, strict JSON
    steps = [
        json.loads(l)
        for l in (run_dir / "diagnostics/audit_steps.jsonl").read_text().splitlines()
    ]
    assert len(steps) == 4
    assert all(s["filter"] == "block" for s in steps)
    assert steps[0]["act"]["zero_frac"] is not None

    # aggregate report with flags
    report = json.loads((run_dir / "diagnostics/report.json").read_text())
    block_report = next(f for f in report["filters"] if f["filter"] == "block")
    assert any(fl.startswith("DEAD_CHANNELS") for fl in block_report["flags"])

    # HealthFlag events reached the run's event log
    events = [json.loads(l) for l in (run_dir / "events.jsonl").read_text().splitlines()]
    types = [e["event_type"] for e in events]
    assert "StepCompleted" in types  # emitted by graph.backward
    assert "MetricReported" in types
    health = [e for e in events if e["event_type"] == "HealthFlag"]
    assert any(e["flag"].startswith("DEAD_CHANNELS") for e in health)

    # safetensors snapshots load with numpy
    from safetensors.numpy import load_file

    snap_files = list((run_dir / "diagnostics/channels/block").glob("*.safetensors"))
    assert snap_files
    tensors = load_file(str(snap_files[0]))
    assert tensors["corr"].shape == (8, 8)
    assert tensors["act_abs_mean"].shape == (8,)

    status = json.loads((run_dir / "status.json").read_text())
    assert status["state"] == "completed"


def test_track_run_marks_failed_on_exception(tmp_path):
    g = Graph()
    with pytest.raises(ValueError):
        with g.track_run("boom", root=str(tmp_path)) as run:
            raise ValueError("training exploded")
    status = json.loads((pathlib.Path(run.dir) / "status.json").read_text())
    assert status["state"] == "failed"
    assert "active_run" not in g.py_state


def test_untracked_audit_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    g, block = _one_block_graph()
    x = _setup(g)
    with g.gradient_audit(channels=True) as audit:
        _train_steps(g, block, x, n_steps=2)
    assert audit.report().by_id()["block"].n_steps == 2
    assert not (tmp_path / ".soma").exists()
