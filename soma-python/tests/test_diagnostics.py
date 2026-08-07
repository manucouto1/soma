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
safetensors = pytest.importorskip("safetensors")
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


# ── ChannelConfig knobs ─────────────────────────────────────────────


def test_channel_dim_two_handles_btc_tensors():
    class TimeBlock(DifferentiableFilter):
        def build_module(self, input_shape):
            return nn.Linear(input_shape[-1], 6)

        def output_shape(self, input_shape):
            return (*input_shape[:-1], 6)

    g = Graph()
    g.node("tb", TimeBlock())
    x = torch.randn(4, 5, 8)  # (B, T, C_in) → output (B, T, 6)
    g.materialize(x)
    g.train()
    g.make_optimizer(lr=0.01)
    module = dict(g.filters())["tb"]._module

    cfg = ChannelConfig(channel_dim=2, snapshot_every=1)
    with g.gradient_audit(channels=cfg) as audit:
        for _ in range(2):
            with g.context() as ctx:
                g.zero_grad()
                loss = (module(x) ** 2).mean()
                g.backward(ctx, loss)
            g.step(ctx)

    rep = audit.report().by_id()["tb"]
    assert rep.metrics["n_channels"] == 6.0
    snap = audit._chan_snapshots["tb"]
    assert tuple(snap["corr"].shape) == (6, 6)


def test_ignored_grad_eps_knob_changes_the_verdict():
    """A channel with a VANISHING (but nonzero) gradient sits exactly on
    the knob: tiny eps says alive, generous eps says ignored. (A channel
    fully sliced out of the loss has gradient exactly 0.0 and is flagged
    at any eps — that case is test_ignored_channels_flagged.)"""

    def run_case(eps):
        g, _block = _one_block_graph()
        x = _setup(g)
        module = dict(g.filters())["block"]._module
        with g.gradient_audit(channels=ChannelConfig(ignored_grad_eps=eps)) as audit:
            for _ in range(4):
                with g.context() as ctx:
                    g.zero_grad()
                    out = module(x)
                    keep = [c for c in range(out.shape[1]) if c != 1]
                    # Channel 1: alive forward, gradient ~1e-10 — starved.
                    loss = (out[:, keep] ** 2).mean() + 1e-9 * (out[:, 1] ** 2).mean()
                    g.backward(ctx, loss)
                g.step(ctx)
        return audit.report().by_id()["block"].metrics["ignored_channels"]

    assert run_case(1e-30) == 0, "tiny eps: a 1e-10 gradient still counts as alive"
    assert run_case(1e-3) >= 1, "generous eps: the starved channel is flagged"


def test_corr_threshold_knob_suppresses_leakage():
    g, block = _one_block_graph(duplicate=(list(range(0, 4)), list(range(4, 8))))
    x = _setup(g, batch=32)
    cfg = ChannelConfig(
        snapshot_every=1,
        corr_threshold=1.01,  # unreachable → no flag even for clones
        groups={"block": {"a": range(0, 4), "b": range(4, 8)}},
    )
    with g.gradient_audit(channels=cfg) as audit:
        _train_steps(g, block, x, n_steps=2)
    rep = audit.report().by_id()["block"]
    assert rep.metrics["max_group_cka"] > 0.95
    assert "LEAKAGE" not in rep.flags


def test_dead_channel_frac_knob():
    g, block = _one_block_graph(dead=(2,))
    x = _setup(g)
    # Impossible fraction → nothing counts as dead.
    with g.gradient_audit(
        channels=ChannelConfig(dead_channel_frac=1.5)
    ) as audit:
        _train_steps(g, block, x)
    assert audit.report().by_id()["block"].metrics["dead_channels"] == 0


def test_dormancy_metrics_and_all_dead_edge():
    g, block = _one_block_graph()
    x = _setup(g)
    with g.gradient_audit(channels=True) as audit:
        _train_steps(g, block, x)
    rep = audit.report().by_id()["block"]
    assert 0.0 <= rep.metrics["dormancy_frac"] <= 1.0

    # Every channel dead → layer mean 0 → dormancy_frac pinned to 1.0.
    g2, block2 = _one_block_graph(dead=tuple(range(8)))
    x2 = _setup(g2)
    with g2.gradient_audit(channels=True) as audit2:
        # Loss over the (all-zero) outputs — keep keep-list non-empty.
        module = dict(g2.filters())["block"]._module
        for _ in range(3):
            with g2.context() as ctx:
                g2.zero_grad()
                loss = (module(x2) ** 2).mean() + 0.0 * sum(
                    p.sum() for p in module.parameters()
                )
                g2.backward(ctx, loss)
            g2.step(ctx)
    rep2 = audit2.report().by_id()["block"]
    assert rep2.metrics["dormancy_frac"] == 1.0
    assert rep2.metrics["dead_channels"] == 8


def test_eff_rank_reflects_channel_redundancy():
    # Rank-1-ish: second half clones the first, halving effective rank.
    g, block = _one_block_graph(duplicate=(list(range(0, 4)), list(range(4, 8))))
    x = _setup(g, batch=64)
    with g.gradient_audit(channels=ChannelConfig(snapshot_every=1)) as audit:
        _train_steps(g, block, x, n_steps=1)
    dup_rank = audit.report().by_id()["block"].metrics["eff_rank"]

    g2, block2 = _one_block_graph()
    x2 = _setup(g2, batch=64)
    with g2.gradient_audit(channels=ChannelConfig(snapshot_every=1)) as audit2:
        _train_steps(g2, block2, x2, n_steps=1)
    full_rank = audit2.report().by_id()["block"].metrics["eff_rank"]

    assert dup_rank < full_rank, f"duplication must shrink eff_rank ({dup_rank} vs {full_rank})"
    assert dup_rank <= 4.5


# ── Multi-node, shape edges, standalone ─────────────────────────────


def test_channel_diagnostics_across_two_nodes(tmp_path):
    g = Graph()
    a = ChannelBlock(out_dim=8)
    b = ChannelBlock(out_dim=6, dead=(1,))
    g.node("first", a)
    g.node("second", b)
    g.edge("first", "second")
    x = torch.randn(16, 8)
    g.materialize(x)
    g.train()
    g.make_optimizer(lr=0.01)
    filters = dict(g.filters())

    with g.track_run("two-nodes", root=str(tmp_path)):
        with g.gradient_audit(channels=ChannelConfig(snapshot_every=2)) as audit:
            for _ in range(4):
                with g.context() as ctx:
                    g.zero_grad()
                    out = filters["second"]._module(filters["first"]._module(x))
                    keep = [c for c in range(out.shape[1]) if c != 1]
                    loss = (out[:, keep] ** 2).mean()
                    g.backward(ctx, loss)
                g.step(ctx)

    by_id = audit.report().by_id()
    assert by_id["first"].metrics["n_channels"] == 8.0
    assert by_id["second"].metrics["n_channels"] == 6.0
    assert any(f.startswith("DEAD_CHANNELS") for f in by_id["second"].flags)
    assert not any(f.startswith("DEAD_CHANNELS") for f in by_id["first"].flags)

    run_dir = next((tmp_path / "runs").iterdir())
    steps = [
        json.loads(l)
        for l in (run_dir / "diagnostics/audit_steps.jsonl").read_text().splitlines()
    ]
    assert len(steps) == 4 * 2, "one line per filter per step"
    assert {s["filter"] for s in steps} == {"first", "second"}
    index = [
        json.loads(l)
        for l in (run_dir / "diagnostics/channels/index.jsonl").read_text().splitlines()
    ]
    assert {e["filter"] for e in index} == {"first", "second"}
    assert all("cka" in e for e in index)
    assert (run_dir / "diagnostics/channels/first").is_dir()
    assert (run_dir / "diagnostics/channels/second").is_dir()


def test_one_dimensional_output_skips_channel_stats():
    class Scalarizer(DifferentiableFilter):
        def build_module(self, input_shape):
            class M(nn.Module):
                def __init__(self, d):
                    super().__init__()
                    self.lin = nn.Linear(d, 1)

                def forward(self, x):
                    return self.lin(x).sum(dim=0).squeeze()  # 0-dim-ish output

            return M(input_shape[-1])

        def output_shape(self, input_shape):
            return (1,)

    g = Graph()
    g.node("s", Scalarizer())
    x = torch.randn(8, 4)
    g.materialize(x)
    g.train()
    g.make_optimizer(lr=0.01)
    module = dict(g.filters())["s"]._module

    with g.gradient_audit(channels=True) as audit:
        with g.context() as ctx:
            g.zero_grad()
            loss = module(x) ** 2
            g.backward(ctx, loss)
        g.step(ctx)

    rep = audit.report().by_id()["s"]
    assert "n_channels" not in rep.metrics, "no channel axis → no channel stats"
    assert rep.n_steps == 1  # scalar audit still ran, no crash


def test_audit_modules_standalone_with_channels(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # prove nothing is persisted anywhere
    from soma import audit_modules

    module = nn.Sequential(nn.Linear(8, 8), _Shape(dead=(0, 5), duplicate=None))
    x = torch.randn(16, 8)
    with audit_modules([("standalone", module)], channels=True) as audit:
        for _ in range(3):
            out = module(x)
            loss = (out**2).mean()
            loss.backward()
            audit._snapshot_after_backward()
            module.zero_grad()

    rep = audit.report().by_id()["standalone"]
    assert rep.metrics["dead_channels"] >= 2
    assert not (tmp_path / ".soma").exists()
    assert not (tmp_path / "diagnostics").exists()


def test_no_data_when_no_backward_ran_and_flag_is_not_emitted(tmp_path):
    """NO_DATA means the audit saw zero committed steps for a filter —
    which happens when the with-block runs no backward at all (e.g. a
    forward-only evaluation). Note: a materialized-but-unforwarded
    filter still gets records (its param stats are snapshotted on every
    backward), so it is NOT NO_DATA — that behavior is pinned here too."""
    g, block = _one_block_graph()
    x = _setup(g)
    module = dict(g.filters())["block"]._module

    with g.track_run("nodata", root=str(tmp_path)):
        with g.gradient_audit() as audit:
            _ = module(x)  # forward only, never backward

    rep = audit.report().by_id()["block"]
    assert rep.flags == ["NO_DATA"]
    assert rep.n_steps == 0
    with pytest.raises(soma.GradientHealthError):
        audit.assert_healthy()

    # NO_DATA never becomes a HealthFlag event.
    run_dir = next((tmp_path / "runs").iterdir())
    events = [json.loads(l) for l in (run_dir / "events.jsonl").read_text().splitlines()]
    assert not any(
        e["event_type"] == "HealthFlag" and e["flag"] == "NO_DATA" for e in events
    )

    # Pinned: an unforwarded filter in a graph that DOES backward gets
    # param-only records, so it reports data (not NO_DATA).
    g2 = Graph()
    g2.node("used", ChannelBlock(out_dim=8))
    g2.node("unused", ChannelBlock(out_dim=8))
    x2 = torch.randn(8, 8)
    g2.materialize(x2)
    g2.train()
    g2.make_optimizer(lr=0.01)
    used_module = dict(g2.filters())["used"]._module
    with g2.gradient_audit() as audit2:
        with g2.context() as ctx:
            g2.zero_grad()
            g2.backward(ctx, (used_module(x2) ** 2).mean())
        g2.step(ctx)
    unused_rep = audit2.report().by_id()["unused"]
    assert unused_rep.n_steps == 1
    assert "NO_DATA" not in unused_rep.flags


# ── Schema contracts ────────────────────────────────────────────────

REPORT_SCALAR_KEYS = {
    "act_mean_abs",
    "act_std",
    "act_zero_frac",
    "act_zero_frac_max",
    "act_sat_frac_max",
    "out_grad_norm",
    "out_grad_max",
    "param_grad_norm",
    "param_grad_max",
    "param_norm",
    "grad_param_ratio",
    "param_grad_zero_frac",
}

CHANNEL_KEYS = {"n_channels", "dead_channels", "dormancy_frac", "ignored_channels", "eff_rank"}


def test_report_metric_key_contract():
    """A rename of any report key silently breaks report.json consumers
    — this test is the contract."""
    g, block = _one_block_graph()
    x = _setup(g)
    with g.gradient_audit(channels=ChannelConfig(snapshot_every=1)) as audit:
        _train_steps(g, block, x, n_steps=2)
    metrics = audit.report().by_id()["block"].metrics
    assert REPORT_SCALAR_KEYS <= set(metrics), sorted(metrics)
    assert CHANNEL_KEYS <= set(metrics), sorted(metrics)


def test_step_record_to_dict_key_contract():
    from soma import StepRecord

    rec = StepRecord()
    d = rec.to_dict()
    assert set(d) == {"step", "ts", "act", "out_grad", "param"}
    assert set(d["act"]) == {
        "abs_mean", "std", "min", "max", "zero_frac", "sat_frac", "nan", "inf",
    }
    assert set(d["out_grad"]) == {"norm", "max", "nan", "inf"}
    assert set(d["param"]) == {
        "grad_norm", "grad_max", "grad_zero_frac", "norm", "grad_param_ratio", "nan", "inf",
    }
    # Public name and the pre-0.4 private alias are the same class.
    import soma._audit as _audit

    assert _audit._StepRecord is StepRecord


def test_audit_files_are_strict_json_with_nulls_for_nan(tmp_path):
    class ParamFree(DifferentiableFilter):
        def build_module(self, input_shape):
            return nn.ReLU()  # no parameters → param stats stay NaN

        def output_shape(self, input_shape):
            return input_shape

    g = Graph()
    g.node("relu", ParamFree())
    x = torch.randn(8, 4, requires_grad=True)
    g.materialize(x)
    g.train()

    def strict_loads(line):
        # Bare NaN/Infinity in the file must be a hard error.
        def reject(_):
            raise ValueError("non-finite constant leaked into JSONL")

        return json.loads(line, parse_constant=reject)

    with g.track_run("strict", root=str(tmp_path)):
        with g.gradient_audit() as audit:
            module = dict(g.filters())["relu"]._module
            with g.context() as ctx:
                loss = (module(x) ** 2).mean()
                g.backward(ctx, loss)

    run_dir = next((tmp_path / "runs").iterdir())
    steps = [
        strict_loads(l)
        for l in (run_dir / "diagnostics/audit_steps.jsonl").read_text().splitlines()
    ]
    assert steps[0]["param"]["grad_norm"] is None, "NaN must serialize as null"
    strict_loads((run_dir / "diagnostics/report.json").read_text())
    assert audit.report().by_id()["relu"].n_steps == 1


def test_timeseries_unknown_filter_is_empty():
    g, block = _one_block_graph()
    x = _setup(g)
    with g.gradient_audit() as audit:
        _train_steps(g, block, x, n_steps=1)
    assert audit.timeseries("does-not-exist") == []


def test_audit_without_torch_raises_cleanly(monkeypatch):
    import soma._audit as _audit

    monkeypatch.setattr(_audit, "torch", None)
    with pytest.raises(RuntimeError, match="needs torch"):
        _audit.Audit([], _audit._DEFAULT_THRESHOLDS)
    # The config dataclass stays usable without torch.
    assert ChannelConfig(snapshot_every=5).snapshot_every == 5


def test_backward_requires_a_loss():
    g, block = _one_block_graph()
    _setup(g)
    with g.context() as ctx:
        with pytest.raises(ValueError, match="loss is None"):
            g.backward(ctx, None)


def test_step_completed_events_are_exact_and_scoped_to_track_run(tmp_path):
    g, block = _one_block_graph()
    x = _setup(g)

    # Outside track_run: no StepCompleted at all.
    with g.context() as ctx:
        g.zero_grad()
        module = dict(g.filters())["block"]._module
        g.backward(ctx, (module(x) ** 2).mean())
    g.step(ctx)

    with g.track_run("exact-steps", root=str(tmp_path)):
        _train_steps(g, block, x, n_steps=3)

    run_dir = next((tmp_path / "runs").iterdir())
    events = [json.loads(l) for l in (run_dir / "events.jsonl").read_text().splitlines()]
    steps = [e["step"] for e in events if e["event_type"] == "StepCompleted"]
    assert steps == [0, 1, 2], "exact per-backward steps, none from outside the run"
