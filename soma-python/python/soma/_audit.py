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
class _StepRecord:
    """One forward+backward sample for one filter."""

    # Activations (output of the module, before grad)
    act_mean: float = math.nan
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


def _tensor_stats(t: "torch.Tensor") -> dict[str, float]:
    if t.numel() == 0:
        return {}
    nan = bool(torch.isnan(t).any().item())
    inf = bool(torch.isinf(t).any().item())
    finite = t[torch.isfinite(t)] if (nan or inf) else t
    if finite.numel() == 0:
        return {"nan": nan, "inf": inf}  # type: ignore[dict-item]
    f = finite.detach().float()
    return {
        "mean": float(f.mean().item()),
        "std": float(f.std(unbiased=False).item()) if f.numel() > 1 else 0.0,
        "min": float(f.min().item()),
        "max": float(f.max().item()),
        "abs_mean": float(f.abs().mean().item()),
        "abs_max": float(f.abs().max().item()),
        "zero_frac": float((f.abs() < 0.0).float().mean().item()),  # placeholder; overridden below
        "nan": nan,           # type: ignore[dict-item]
        "inf": inf,           # type: ignore[dict-item]
    }


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
    ):
        if torch is None:
            raise RuntimeError("gradient_audit needs torch")
        self.modules = modules
        self.thresholds = thresholds
        self._records: dict[str, list[_StepRecord]] = defaultdict(list)
        self._handles: list[Any] = []
        # Per-step transient buffer: filter_id → in-progress record. The
        # forward hook initialises an entry; the backward hook completes
        # it; on optimiser step the user can call _flush() if they want
        # finer-grained per-step records (default: aggregated).
        self._pending: dict[str, _StepRecord] = {}

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
            rec = _StepRecord()
            if t is not None:
                stats = _act_record(t, thr)
                rec.act_mean = stats.get("abs_mean", math.nan)
                rec.act_std = stats.get("std", math.nan)
                rec.act_min = stats.get("min", math.nan)
                rec.act_max = stats.get("max", math.nan)
                rec.act_zero_frac = stats.get("zero_frac", math.nan)
                rec.act_sat_frac = stats.get("sat_frac", math.nan)
                rec.act_nan = bool(stats.get("nan", False))
                rec.act_inf = bool(stats.get("inf", False))
            self._pending[fid] = rec

        return hook

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
                rec = _StepRecord()
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

        return hook

    def _snapshot_after_backward(self) -> None:
        """Capture ``p.grad`` for every audited module and commit pending records.

        Called by ``Graph.backward`` after ``loss.backward()`` returns,
        when ``p.grad`` has been fully accumulated for every parameter.
        """
        for fid, mod in self.modules:
            rec = self._pending.pop(fid, _StepRecord())
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
                "act_mean_abs": self._mean([r.act_mean for r in recs]),
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
) -> Iterator[Audit]:
    """Standalone context manager for users not driving training through ``Graph``.

    Pass ``[(name, module), ...]``; the same audit semantics apply.
    """
    audit = Audit(list(modules), thresholds)
    audit._install()
    try:
        yield audit
    finally:
        audit._remove()


@contextlib.contextmanager
def _gradient_audit(
    self: _RustGraph,
    thresholds: Thresholds | None = None,
) -> Iterator[Audit]:
    """Install hooks on every live ``DifferentiableFilter._module``.

    Yields an :class:`Audit`. On exit (including exceptions) hooks are
    removed and the audit is unregistered from ``graph.py_state``.
    Filters whose ``_module`` is ``None`` (not yet materialised) are
    silently skipped — call ``g.materialize(sample)`` first or train
    inside the context to lazy-build them.

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
    audit = Audit(pairs, thresholds or _DEFAULT_THRESHOLDS)
    audit._install()
    self.py_state["active_audit"] = audit
    try:
        yield audit
    finally:
        audit._remove()
        if self.py_state.get("active_audit") is audit:
            del self.py_state["active_audit"]


# ── Install on Graph ─────────────────────────────────────────


def _install() -> None:
    _RustGraph.gradient_audit = _gradient_audit


_install()
