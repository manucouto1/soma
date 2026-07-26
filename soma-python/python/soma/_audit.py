"""Built-in gradient health audit for ``Graph`` training loops.

Records per-filter activation and gradient statistics during a training
pass so users can locate vanishing/exploding gradients, dead units,
NaN/Inf, and saturation by node id without writing manual hooks.

Usage::

    with g.gradient_audit() as audit:
        for x, y in batches:
            with g.context() as ctx:
                g.zero_grad()
                out, aux = g.forward(x)
                loss = my_loss(out, y, aux)
                g.backward(ctx, loss)
            g.step(ctx)

    print(audit.report().pretty())
    audit.assert_healthy()       # raises GradientHealthError on any flag

Hooks are installed on every live ``DifferentiableFilter._module`` on
context entry and removed on exit, even on exception. The audit
aggregates across all forward/backward calls inside the with-block
(audit a whole epoch).

For users not driving training through ``Graph``, the same primitives
work standalone via :func:`audit_modules`.

Design (RPC-ready): everything lives in Python and reads
``filter._module`` directly. Once filters move to remote workers, the
hook installation will move worker-side and the aggregator will gather
records via ``rpc.rpc_sync``; the public API does not change.
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
import sys
from collections import defaultdict
from typing import Any, Iterable, Iterator

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None     # type: ignore[assignment]

from soma._soma import Graph as _RustGraph


# ── Errors ──────────────────────────────────────────────────


class GradientHealthError(RuntimeError):
    """Raised by :meth:`Audit.assert_healthy` when any flag is set."""


# ── Thresholds ──────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class Thresholds:
    """Bounds used to flag per-filter records.

    Tuned for typical CV/NLP scales (LayerNorm-ish activations, Adam
    steps). Override per call: ``g.gradient_audit(thresholds=Thresholds(
    grad_lo=1e-10, ...))``.
    """

    # Param-grad L2 norm bounds
    grad_lo: float = 1e-7      # < this → VANISHING
    grad_hi: float = 1e3       # > this → EXPLODING

    # Activation-saturation: |x| above this counts as saturated
    activation_saturation: float = 50.0
    saturation_frac: float = 0.5  # > this fraction saturated → SATURATED

    # Dead-unit fraction: |activation| < dead_eps counts as dead
    dead_eps: float = 1e-7
    dead_frac: float = 0.95    # > this fraction dead → DEAD


_DEFAULT_THRESHOLDS = Thresholds()


# ── Records ─────────────────────────────────────────────────


@dataclasses.dataclass
class StepRecord:
    """One forward+backward sample for one filter.

    Public: per-step time series are exposed via :meth:`Audit.records`
    / :meth:`Audit.timeseries` and persisted to
    ``diagnostics/audit_steps.jsonl`` inside a tracked run.
    """

    # Position in the audited sequence
    step: int = -1
    ts: float = math.nan

    # Activations (output of the module, before grad)
    act_abs_mean: float = math.nan
    act_std: float = math.nan
    act_min: float = math.nan
    act_max: float = math.nan
    act_zero_frac: float = math.nan
    act_nan: bool = False
    act_inf: bool = False
    act_sat_frac: float = math.nan

    # Gradient on output (input grad to the next filter)
    out_grad_norm: float = math.nan
    out_grad_max: float = math.nan
    out_grad_nan: bool = False
    out_grad_inf: bool = False

    # Gradient on parameters
    param_grad_norm: float = math.nan
    param_grad_max: float = math.nan
    param_norm: float = math.nan
    grad_param_ratio: float = math.nan
    param_grad_zero_frac: float = math.nan
    param_grad_nan: bool = False
    param_grad_inf: bool = False

    def to_dict(self) -> dict:
        """Nested dict shaped like an ``audit_steps.jsonl`` line."""
        return {
            "step": self.step,
            "ts": self.ts,
            "act": {
                "abs_mean": self.act_abs_mean,
                "std": self.act_std,
                "min": self.act_min,
                "max": self.act_max,
                "zero_frac": self.act_zero_frac,
                "sat_frac": self.act_sat_frac,
                "nan": self.act_nan,
                "inf": self.act_inf,
            },
            "out_grad": {
                "norm": self.out_grad_norm,
                "max": self.out_grad_max,
                "nan": self.out_grad_nan,
                "inf": self.out_grad_inf,
            },
            "param": {
                "grad_norm": self.param_grad_norm,
                "grad_max": self.param_grad_max,
                "grad_zero_frac": self.param_grad_zero_frac,
                "norm": self.param_norm,
                "grad_param_ratio": self.grad_param_ratio,
                "nan": self.param_grad_nan,
                "inf": self.param_grad_inf,
            },
        }


# Backwards-compatible private alias (pre-0.4 name).
_StepRecord = StepRecord


@dataclasses.dataclass(frozen=True)
class ChannelConfig:
    """Opt-in per-channel diagnostics (``channels=`` of
    :meth:`Graph.gradient_audit`).

    Channel axis is ``channel_dim`` of each ``(B, C, ...)`` activation.
    Cheap O(numel) per-channel reductions run every step; the expensive
    channel×channel correlation / CKA / effective-rank snapshot runs
    every ``snapshot_every`` steps and is persisted as a safetensors
    file inside the run's ``diagnostics/channels/`` directory.

    ``groups`` declares intended channel partitions per filter, e.g.
    ``{"encoder": {"audio": range(0, 64), "text": range(64, 128)}}`` —
    linear CKA above ``corr_threshold`` between two groups flags
    ``LEAKAGE`` (information shared across a boundary the architecture
    intends to keep separate).
    """

    channel_dim: int = 1
    snapshot_every: int = 50
    corr_threshold: float = 0.95
    dead_channel_frac: float = 0.95
    dormancy_tau: float = 0.1
    ignored_grad_eps: float = 1e-9
    groups: dict | None = None


@dataclasses.dataclass
class FilterReport:
    """Aggregated metrics + flags for one filter across all audited steps."""

    filter_id: str
    n_steps: int
    metrics: dict[str, float]
    flags: list[str]

    def is_healthy(self) -> bool:
        return not self.flags


@dataclasses.dataclass
class AuditReport:
    filters: list[FilterReport]
    n_steps: int

    def is_healthy(self) -> bool:
        return all(f.is_healthy() for f in self.filters)

    def by_id(self) -> dict[str, FilterReport]:
        return {f.filter_id: f for f in self.filters}

    def pretty(self) -> str:
        """Tabular CLI report sorted by filter order; flags at the end."""
        if not self.filters:
            return "(no filters audited)"
        cols = [
            ("filter", lambda r: r.filter_id, 24),
            ("steps", lambda r: r.n_steps, 6),
            ("act|μ|", lambda r: r.metrics.get("act_mean_abs", math.nan), 10),
            ("act σ", lambda r: r.metrics.get("act_std", math.nan), 10),
            ("|out∂|", lambda r: r.metrics.get("out_grad_norm", math.nan), 10),
            ("|θ∂|", lambda r: r.metrics.get("param_grad_norm", math.nan), 10),
            ("|θ|", lambda r: r.metrics.get("param_norm", math.nan), 10),
            ("∂/θ", lambda r: r.metrics.get("grad_param_ratio", math.nan), 10),
            ("flags", lambda r: ",".join(r.flags) or "HEALTHY", 24),
        ]
        head = "  ".join(f"{name:>{w}}" for name, _, w in cols)
        rows = [head, "-" * len(head)]
        for r in self.filters:
            cells = []
            for _, getter, w in cols:
                v = getter(r)
                if isinstance(v, float):
                    cells.append(f"{v:>{w}.3e}" if not math.isnan(v) else f"{'':>{w}}")
                else:
                    cells.append(f"{str(v):>{w}}")
            rows.append("  ".join(cells))
        return "\n".join(rows)

    def dataframe(self):
        """Return a pandas DataFrame of per-filter metrics (one row per filter)."""
        try:
            import pandas as pd
        except ImportError as e:
            raise RuntimeError("AuditReport.dataframe() needs pandas") from e
        rows = []
        for r in self.filters:
            rows.append({
                "filter": r.filter_id,
                "steps": r.n_steps,
                "flags": ",".join(r.flags) or "HEALTHY",
                **r.metrics,
            })
        return pd.DataFrame(rows)


# ── Helpers ─────────────────────────────────────────────────


def _act_record(t: "torch.Tensor", thr: Thresholds) -> dict[str, float]:
    if t.numel() == 0:
        return {}
    nan = bool(torch.isnan(t).any().item())
    inf = bool(torch.isinf(t).any().item())
    f = t.detach().float()
    finite_mask = torch.isfinite(f)
    finite = f[finite_mask] if not finite_mask.all() else f
    if finite.numel() == 0:
        return {"nan": nan, "inf": inf}  # type: ignore[dict-item]
    abs_f = finite.abs()
    return {
        "mean": float(finite.mean().item()),
        "std": float(finite.std(unbiased=False).item()) if finite.numel() > 1 else 0.0,
        "min": float(finite.min().item()),
        "max": float(finite.max().item()),
        "abs_mean": float(abs_f.mean().item()),
        "zero_frac": float((abs_f < thr.dead_eps).float().mean().item()),
        "sat_frac": float((abs_f > thr.activation_saturation).float().mean().item()),
        "nan": nan,           # type: ignore[dict-item]
        "inf": inf,           # type: ignore[dict-item]
    }


def _flatten_grad(t: "torch.Tensor") -> "torch.Tensor":
    return t.detach().reshape(-1)


def _linear_cka(x: "torch.Tensor", y: "torch.Tensor") -> float:
    """Minibatch linear CKA between two (N, d) feature matrices
    (Kornblith et al. 2019): ‖YᵀX‖²_F / (‖XᵀX‖_F · ‖YᵀY‖_F)."""
    with torch.no_grad():
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)
        xty = (y.T @ x).norm() ** 2
        denom = (x.T @ x).norm() * (y.T @ y).norm()
        if denom <= 0:
            return math.nan
        return float((xty / denom).item())


def _json_safe(obj):
    """Replace non-finite floats with None so lines stay strict JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


# ── Audit ───────────────────────────────────────────────────


class Audit:
    """Collected records for a graph training pass.

    Acts as the value yielded by :meth:`Graph.gradient_audit`. Use
    :meth:`report` to get an aggregated :class:`AuditReport`,
    :meth:`assert_healthy` to fail-fast in tests.
    """

    def __init__(
        self,
        modules: list[tuple[str, "nn.Module"]],
        thresholds: Thresholds = _DEFAULT_THRESHOLDS,
        channels: "ChannelConfig | None" = None,
    ):
        if torch is None:
            raise RuntimeError("gradient_audit needs torch")
        self.modules = modules
        self.thresholds = thresholds
        self.channels = channels
        self._records: dict[str, list[StepRecord]] = defaultdict(list)
        self._handles: list[Any] = []
        # Per-step transient buffer: filter_id → in-progress record. The
        # forward hook initialises an entry; the backward hook completes
        # it; committed by _snapshot_after_backward.
        self._pending: dict[str, StepRecord] = {}
        # Committed steps so far (index of the record being built).
        self._step = 0

        # ── Per-channel state (only when channels is set) ──
        # Running sums over steps, per filter: act |mean| per channel,
        # zero-fraction per channel, |out-grad| mean per channel.
        self._chan_act_sum: dict[str, "torch.Tensor"] = {}
        self._chan_zero_sum: dict[str, "torch.Tensor"] = {}
        self._chan_grad_sum: dict[str, "torch.Tensor"] = {}
        self._chan_n: dict[str, int] = defaultdict(int)
        self._chan_grad_n: dict[str, int] = defaultdict(int)
        # Transient per-step channel vectors + stashed activation matrix
        # for snapshot steps.
        self._chan_pending: dict[str, dict[str, "torch.Tensor"]] = {}
        # Snapshot outputs of the last snapshot step, per filter:
        # {"corr": Tensor(C,C), "eff_rank": float, "cka": {pair: float}}
        self._chan_snapshots: dict[str, dict[str, Any]] = {}
        # Persistence (bound lazily when a tracked run is active).
        self._graph: "_RustGraph | None" = None
        self._steps_file = None

    # ── Hook installation ──

    def _install(self) -> None:
        for fid, mod in self.modules:
            self._handles.append(
                mod.register_forward_hook(self._make_fwd_hook(fid))
            )
            self._handles.append(
                mod.register_full_backward_hook(self._make_bwd_hook(fid))
            )

    def _remove(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()

    # ── Hook closures ──

    def _make_fwd_hook(self, fid: str):
        thr = self.thresholds

        def hook(module, inputs, output):
            t = output if isinstance(output, torch.Tensor) else None
            if t is None and isinstance(output, (tuple, list)) and output:
                cand = output[0]
                if isinstance(cand, torch.Tensor):
                    t = cand
            rec = StepRecord()
            if t is not None:
                stats = _act_record(t, thr)
                rec.act_abs_mean = stats.get("abs_mean", math.nan)
                rec.act_std = stats.get("std", math.nan)
                rec.act_min = stats.get("min", math.nan)
                rec.act_max = stats.get("max", math.nan)
                rec.act_zero_frac = stats.get("zero_frac", math.nan)
                rec.act_sat_frac = stats.get("sat_frac", math.nan)
                rec.act_nan = bool(stats.get("nan", False))
                rec.act_inf = bool(stats.get("inf", False))
                if self.channels is not None:
                    self._channel_forward(fid, t)
            self._pending[fid] = rec

        return hook

    # ── Per-channel capture ──

    def _channel_axes(self, t: "torch.Tensor") -> "tuple[int, list[int]] | None":
        """(channel_dim, other_dims) if the tensor has a channel axis."""
        cfg = self.channels
        if cfg is None or t.dim() <= cfg.channel_dim:
            return None
        other = [d for d in range(t.dim()) if d != cfg.channel_dim]
        return cfg.channel_dim, other

    def _channel_forward(self, fid: str, t: "torch.Tensor") -> None:
        axes = self._channel_axes(t)
        if axes is None:
            return
        cdim, other = axes
        cfg = self.channels
        f = t.detach().float()
        abs_f = f.abs()
        act_abs = abs_f.mean(dim=other) if other else abs_f
        zero = (abs_f < self.thresholds.dead_eps).float()
        zero_frac = zero.mean(dim=other) if other else zero

        pend = self._chan_pending.setdefault(fid, {})
        pend["act_abs_mean"] = act_abs.cpu()
        pend["act_zero_frac"] = zero_frac.cpu()

        # Stash the (C, N) activation matrix on snapshot steps for the
        # correlation / CKA / effective-rank pass at commit time.
        if self._step % cfg.snapshot_every == 0:
            pend["act_matrix"] = f.movedim(cdim, 0).reshape(f.shape[cdim], -1).cpu()

        acc = self._chan_act_sum.get(fid)
        if acc is None or acc.shape != act_abs.shape:
            self._chan_act_sum[fid] = act_abs.cpu().clone()
            self._chan_zero_sum[fid] = zero_frac.cpu().clone()
            self._chan_n[fid] = 1
        else:
            self._chan_act_sum[fid] += act_abs.cpu()
            self._chan_zero_sum[fid] += zero_frac.cpu()
            self._chan_n[fid] += 1

    def _channel_backward(self, fid: str, g_out: "torch.Tensor") -> None:
        axes = self._channel_axes(g_out)
        if axes is None:
            return
        _, other = axes
        g = g_out.detach().float().abs()
        grad_abs = g.mean(dim=other) if other else g
        pend = self._chan_pending.setdefault(fid, {})
        pend["out_grad_abs_mean"] = grad_abs.cpu()

        acc = self._chan_grad_sum.get(fid)
        if acc is None or acc.shape != grad_abs.shape:
            self._chan_grad_sum[fid] = grad_abs.cpu().clone()
            self._chan_grad_n[fid] = 1
        else:
            self._chan_grad_sum[fid] += grad_abs.cpu()
            self._chan_grad_n[fid] += 1

    def _make_bwd_hook(self, fid: str):
        """Backward hook captures only the output grad.

        Param grads are NOT read here — at the time this hook fires for
        an early module, ``p.grad`` accumulation has not necessarily
        run yet. Param grads are snapshotted in :meth:`_snapshot_after_backward`,
        called by ``Graph.backward`` once ``loss.backward()`` returns.
        """
        def hook(module, grad_input, grad_output):
            rec = self._pending.get(fid)
            if rec is None:
                rec = StepRecord()
                self._pending[fid] = rec
            g_out = None
            if grad_output and isinstance(grad_output, (tuple, list)):
                cand = grad_output[0]
                if isinstance(cand, torch.Tensor):
                    g_out = cand
            if g_out is not None and g_out.numel() > 0:
                gf = _flatten_grad(g_out)
                rec.out_grad_nan = bool(torch.isnan(gf).any().item())
                rec.out_grad_inf = bool(torch.isinf(gf).any().item())
                finite = gf[torch.isfinite(gf)]
                if finite.numel() > 0:
                    rec.out_grad_norm = float(finite.norm(2).item())
                    rec.out_grad_max = float(finite.abs().max().item())
                if self.channels is not None:
                    self._channel_backward(fid, g_out)

        return hook

    def _snapshot_after_backward(self) -> None:
        """Capture ``p.grad`` for every audited module and commit pending records.

        Called by ``Graph.backward`` after ``loss.backward()`` returns,
        when ``p.grad`` has been fully accumulated for every parameter.
        """
        import time as _time

        now = _time.time()
        snapshot_step = (
            self.channels is not None and self._step % self.channels.snapshot_every == 0
        )
        for fid, mod in self.modules:
            rec = self._pending.pop(fid, None) or StepRecord()
            rec.step = self._step
            rec.ts = now
            grads, params = [], []
            for p in mod.parameters():
                if p.grad is not None:
                    grads.append(_flatten_grad(p.grad))
                params.append(_flatten_grad(p.detach()))
            if grads:
                gcat = torch.cat(grads)
                rec.param_grad_nan = bool(torch.isnan(gcat).any().item())
                rec.param_grad_inf = bool(torch.isinf(gcat).any().item())
                finite = gcat[torch.isfinite(gcat)]
                if finite.numel() > 0:
                    rec.param_grad_norm = float(finite.norm(2).item())
                    rec.param_grad_max = float(finite.abs().max().item())
                    rec.param_grad_zero_frac = float(
                        (finite.abs() < self.thresholds.dead_eps).float().mean().item()
                    )
            if params:
                pcat = torch.cat(params)
                finite_p = pcat[torch.isfinite(pcat)]
                if finite_p.numel() > 0:
                    rec.param_norm = float(finite_p.norm(2).item())
                    if rec.param_norm > 0 and not math.isnan(rec.param_grad_norm):
                        rec.grad_param_ratio = rec.param_grad_norm / rec.param_norm
            self._records[fid].append(rec)

            if snapshot_step:
                self._channel_snapshot(fid)
            self._persist_step(fid, rec)
        self._step += 1

    # ── Channel snapshots (correlation / CKA / effective rank) ──

    def _channel_snapshot(self, fid: str) -> None:
        """Expensive per-cadence pass over the stashed activation matrix."""
        pend = self._chan_pending.get(fid) or {}
        mat = pend.pop("act_matrix", None)
        if mat is None or mat.shape[0] < 2:
            return
        cfg = self.channels
        snap: dict[str, Any] = {"step": self._step}

        # Channel × channel Pearson correlation.
        with torch.no_grad():
            corr = torch.corrcoef(mat)
            corr = torch.nan_to_num(corr, nan=0.0)
        snap["corr"] = corr

        # Effective rank (Roy & Vetterli 2007) of the (N, C) feature
        # matrix: exp of the entropy of normalized singular values.
        with torch.no_grad():
            try:
                s = torch.linalg.svdvals(mat)
                p = s / s.sum().clamp_min(1e-12)
                p = p[p > 0]
                snap["eff_rank"] = float(torch.exp(-(p * p.log()).sum()).item())
            except Exception:
                snap["eff_rank"] = math.nan

        # Linear CKA between declared channel groups (minibatch
        # estimator, Kornblith et al. 2019 / Nguyen et al. 2021).
        groups = (cfg.groups or {}).get(fid)
        cka: dict[str, float] = {}
        if groups:
            names = list(groups)
            feats = {}
            for gname in names:
                idx = torch.as_tensor(list(groups[gname]), dtype=torch.long)
                feats[gname] = mat.index_select(0, idx).T  # (N, |g|)
            for i, a in enumerate(names):
                for b in names[i + 1 :]:
                    cka[f"{a}|{b}"] = _linear_cka(feats[a], feats[b])
        snap["cka"] = cka
        self._chan_snapshots[fid] = snap
        self._persist_snapshot(fid, snap, pend)

    # ── Persistence into an active tracked run ──

    def _active_run(self):
        graph = self._graph
        if graph is None:
            return None
        try:
            return graph.py_state.get("active_run")
        except Exception:
            return None

    def _persist_step(self, fid: str, rec: StepRecord) -> None:
        run = self._active_run()
        if run is None:
            return
        import json
        import pathlib

        if self._steps_file is None:
            diag = pathlib.Path(run.dir) / "diagnostics"
            diag.mkdir(parents=True, exist_ok=True)
            self._steps_file = (diag / "audit_steps.jsonl").open("a")
        line = {"filter": fid, **rec.to_dict()}
        self._steps_file.write(json.dumps(_json_safe(line)) + "\n")

    def _persist_snapshot(self, fid: str, snap: dict, pend: dict) -> None:
        run = self._active_run()
        if run is None:
            return
        import json
        import pathlib

        try:
            from safetensors.numpy import save_file
        except ImportError:
            return  # tensor snapshots need safetensors; scalars still logged

        chan_dir = pathlib.Path(run.dir) / "diagnostics" / "channels"
        fdir = chan_dir / fid
        fdir.mkdir(parents=True, exist_ok=True)
        fname = f"step_{snap['step']:06}.safetensors"

        tensors = {"corr": snap["corr"].numpy()}
        for key in ("act_abs_mean", "act_zero_frac", "out_grad_abs_mean"):
            if key in pend:
                tensors[key] = pend[key].numpy()
        save_file(tensors, str(fdir / fname))

        index_line = {
            "filter": fid,
            "step": snap["step"],
            "file": f"{fid}/{fname}",
            "keys": sorted(tensors),
            "eff_rank": snap.get("eff_rank"),
            "cka": snap.get("cka", {}),
        }
        with (chan_dir / "index.jsonl").open("a") as fh:
            fh.write(json.dumps(_json_safe(index_line)) + "\n")

    def _close_files(self) -> None:
        if self._steps_file is not None:
            try:
                self._steps_file.close()
            except Exception:
                pass
            self._steps_file = None

    # ── Public time series ──

    def records(self) -> dict[str, list[StepRecord]]:
        """Per-filter, per-step records collected so far."""
        return dict(self._records)

    def timeseries(self, filter_id: str) -> list[dict]:
        """One dict per audited step for `filter_id` (jsonl-shaped)."""
        return [r.to_dict() for r in self._records.get(filter_id, [])]

    # ── Aggregation ──

    @staticmethod
    def _mean(values: list[float]) -> float:
        v = [x for x in values if not math.isnan(x)]
        return sum(v) / len(v) if v else math.nan

    @staticmethod
    def _max(values: list[float]) -> float:
        v = [x for x in values if not math.isnan(x)]
        return max(v) if v else math.nan

    def _flag(self, agg: dict[str, float], any_nan: bool, any_inf: bool) -> list[str]:
        thr = self.thresholds
        flags: list[str] = []
        if any_nan:
            flags.append("NAN")
        if any_inf:
            flags.append("INF")
        gn = agg.get("param_grad_norm", math.nan)
        if not math.isnan(gn):
            if gn < thr.grad_lo:
                flags.append("VANISHING")
            elif gn > thr.grad_hi:
                flags.append("EXPLODING")
        if agg.get("act_zero_frac_max", math.nan) > thr.dead_frac:
            flags.append("DEAD")
        if agg.get("act_sat_frac_max", math.nan) > thr.saturation_frac:
            flags.append("SATURATED")
        return flags

    def _channel_metrics(self, fid: str) -> tuple[dict[str, float], list[str]]:
        """Aggregate per-channel accumulators into report metrics/flags.

        - dead channels: mean zero-fraction above ``dead_channel_frac``
          (dying-ReLU criterion).
        - dormancy fraction β_τ (Sokar et al., ICML 2023): channels
          whose normalized mean |activation| is ≤ τ.
        - ignored channels: alive in forward (not dormant) but whose
          mean |output-gradient| stays below ``ignored_grad_eps`` —
          the gradient-starvation signature (the network computes the
          region and never asks for it).
        - LEAKAGE: any declared group pair with CKA above threshold in
          the latest snapshot.
        """
        cfg = self.channels
        metrics: dict[str, float] = {}
        flags: list[str] = []
        n = self._chan_n.get(fid, 0)
        if cfg is None or n == 0:
            return metrics, flags

        act_mean = self._chan_act_sum[fid] / n
        zero_mean = self._chan_zero_sum[fid] / n
        n_channels = act_mean.numel()
        metrics["n_channels"] = float(n_channels)

        dead = int((zero_mean > cfg.dead_channel_frac).sum().item())
        metrics["dead_channels"] = float(dead)

        layer_mean = float(act_mean.mean().item())
        if layer_mean > 0:
            dormancy = act_mean / layer_mean
            dormant_mask = dormancy <= cfg.dormancy_tau
            metrics["dormancy_frac"] = float(dormant_mask.float().mean().item())
        else:
            dormant_mask = torch.ones_like(act_mean, dtype=torch.bool)
            metrics["dormancy_frac"] = 1.0

        ignored = 0
        gn = self._chan_grad_n.get(fid, 0)
        if gn > 0 and self._chan_grad_sum[fid].shape == act_mean.shape:
            grad_mean = self._chan_grad_sum[fid] / gn
            ignored_mask = (~dormant_mask) & (grad_mean < cfg.ignored_grad_eps)
            ignored = int(ignored_mask.sum().item())
        metrics["ignored_channels"] = float(ignored)

        snap = self._chan_snapshots.get(fid, {})
        if not math.isnan(snap.get("eff_rank", math.nan)):
            metrics["eff_rank"] = snap["eff_rank"]
        max_cka = max(snap.get("cka", {}).values(), default=math.nan)
        if not math.isnan(max_cka):
            metrics["max_group_cka"] = max_cka

        if dead > 0:
            flags.append(f"DEAD_CHANNELS({dead})")
        if ignored > 0:
            flags.append(f"IGNORED_CHANNELS({ignored})")
        if not math.isnan(max_cka) and max_cka > cfg.corr_threshold:
            flags.append("LEAKAGE")
        return metrics, flags

    def report(self) -> AuditReport:
        """Aggregate per-filter records and assign flags."""
        filters: list[FilterReport] = []
        # Walk modules in original order so the report matches topology.
        seen = {fid for fid, _ in self.modules}
        for fid, _ in self.modules:
            recs = self._records.get(fid, [])
            n = len(recs)
            if n == 0:
                filters.append(
                    FilterReport(filter_id=fid, n_steps=0, metrics={}, flags=["NO_DATA"])
                )
                continue

            agg = {
                "act_mean_abs": self._mean([r.act_abs_mean for r in recs]),
                "act_std": self._mean([r.act_std for r in recs]),
                "act_zero_frac": self._mean([r.act_zero_frac for r in recs]),
                "act_zero_frac_max": self._max([r.act_zero_frac for r in recs]),
                "act_sat_frac_max": self._max([r.act_sat_frac for r in recs]),
                "out_grad_norm": self._mean([r.out_grad_norm for r in recs]),
                "out_grad_max": self._max([r.out_grad_max for r in recs]),
                "param_grad_norm": self._mean([r.param_grad_norm for r in recs]),
                "param_grad_max": self._max([r.param_grad_max for r in recs]),
                "param_norm": self._mean([r.param_norm for r in recs]),
                "grad_param_ratio": self._mean([r.grad_param_ratio for r in recs]),
                "param_grad_zero_frac": self._mean(
                    [r.param_grad_zero_frac for r in recs]
                ),
            }
            any_nan = any(
                r.act_nan or r.out_grad_nan or r.param_grad_nan for r in recs
            )
            any_inf = any(
                r.act_inf or r.out_grad_inf or r.param_grad_inf for r in recs
            )
            flags = self._flag(agg, any_nan, any_inf)
            chan_metrics, chan_flags = self._channel_metrics(fid)
            agg.update(chan_metrics)
            flags.extend(chan_flags)
            filters.append(FilterReport(filter_id=fid, n_steps=n, metrics=agg, flags=flags))

        return AuditReport(
            filters=filters,
            n_steps=max((len(self._records.get(fid, [])) for fid in seen), default=0),
        )

    def assert_healthy(self) -> None:
        """Raise :class:`GradientHealthError` if any filter has flags set."""
        rep = self.report()
        if rep.is_healthy():
            return
        bad = [f for f in rep.filters if f.flags]
        worst = bad[0]
        msg_lines = [
            f"gradient audit failed: {len(bad)} filter(s) unhealthy",
            f"  worst node: {worst.filter_id} flags={worst.flags}",
        ]
        for f in bad[1:5]:
            msg_lines.append(f"  also: {f.filter_id} flags={f.flags}")
        raise GradientHealthError("\n".join(msg_lines))


# ── Public entry points ──────────────────────────────────────


@contextlib.contextmanager
def audit_modules(
    modules: Iterable[tuple[str, "nn.Module"]],
    thresholds: Thresholds = _DEFAULT_THRESHOLDS,
    channels: "ChannelConfig | bool | None" = None,
) -> Iterator[Audit]:
    """Standalone context manager for users not driving training through ``Graph``.

    Pass ``[(name, module), ...]``; the same audit semantics apply.
    """
    audit = Audit(list(modules), thresholds, _resolve_channels(channels))
    audit._install()
    try:
        yield audit
    finally:
        audit._remove()
        audit._close_files()


def _resolve_channels(channels: "ChannelConfig | bool | None") -> "ChannelConfig | None":
    if channels is True:
        return ChannelConfig()
    if channels is False or channels is None:
        return None
    return channels


def _emit_health_flags(graph: _RustGraph, run, report: AuditReport) -> None:
    """Publish each flagged filter as a HealthFlag event on the bus."""
    for f in report.filters:
        for flag in f.flags:
            if flag == "NO_DATA":
                continue
            try:
                graph.emit_event(
                    {
                        "event_type": "HealthFlag",
                        "run_id": run.id,
                        "node_id": f.filter_id,
                        "step": report.n_steps,
                        "flag": flag,
                        "detail": ",".join(
                            f"{k}={v:.3e}" for k, v in sorted(f.metrics.items())
                        )[:500],
                    }
                )
            except Exception:
                pass


@contextlib.contextmanager
def _gradient_audit(
    self: _RustGraph,
    thresholds: Thresholds | None = None,
    channels: "ChannelConfig | bool | None" = None,
) -> Iterator[Audit]:
    """Install hooks on every live ``DifferentiableFilter._module``.

    Yields an :class:`Audit`. On exit (including exceptions) hooks are
    removed and the audit is unregistered from ``graph.py_state``.
    Filters whose ``_module`` is ``None`` (not yet materialised) are
    silently skipped — call ``g.materialize(sample)`` first or train
    inside the context to lazy-build them.

    ``channels=True`` (or a :class:`ChannelConfig`) enables per-channel
    diagnostics: dead/dormant channels, ignored regions (alive forward,
    starved gradient), and channel-correlation / group-CKA snapshots.

    Inside an active ``graph.track_run(...)`` the audit persists per-step
    scalars to ``diagnostics/audit_steps.jsonl``, channel snapshots to
    ``diagnostics/channels/`` (safetensors), the final aggregate to
    ``diagnostics/report.json``, and emits a ``HealthFlag`` event per
    flagged filter.

    Registers itself as ``py_state['active_audit']`` so
    ``Graph.backward`` can call back into it after ``loss.backward()``
    to snapshot ``p.grad`` reliably (param grad accumulation finishes
    after the per-module backward hook fires).
    """
    pairs: list[tuple[str, "nn.Module"]] = []
    for fid, f in self.filters():
        mod = getattr(f, "_module", None)
        if mod is not None:
            pairs.append((fid, mod))
    audit = Audit(pairs, thresholds or _DEFAULT_THRESHOLDS, _resolve_channels(channels))
    audit._graph = self
    audit._install()
    self.py_state["active_audit"] = audit
    try:
        yield audit
    finally:
        audit._remove()
        audit._close_files()
        if self.py_state.get("active_audit") is audit:
            del self.py_state["active_audit"]
        run = self.py_state.get("active_run")
        if run is not None:
            import json
            import pathlib

            report = audit.report()
            diag = pathlib.Path(run.dir) / "diagnostics"
            diag.mkdir(parents=True, exist_ok=True)
            (diag / "report.json").write_text(
                json.dumps(
                    _json_safe(
                        {
                            "n_steps": report.n_steps,
                            "filters": [
                                {
                                    "filter": f.filter_id,
                                    "n_steps": f.n_steps,
                                    "metrics": f.metrics,
                                    "flags": f.flags,
                                }
                                for f in report.filters
                            ],
                        }
                    ),
                    indent=2,
                )
            )
            _emit_health_flags(self, run, report)


# ── Install on Graph ─────────────────────────────────────────


def _install() -> None:
    _RustGraph.gradient_audit = _gradient_audit


_install()
