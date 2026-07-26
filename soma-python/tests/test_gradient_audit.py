"""Tests for ``Graph.gradient_audit`` and the standalone ``audit_modules``."""

from __future__ import annotations

import warnings

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from soma import (
    DifferentiableFilter,
    Graph,
    GradientHealthError,
    Thresholds,
    audit_modules,
)


class Dense(DifferentiableFilter):
    def __init__(self, out_dim, lr=1e-2):
        super().__init__(out_dim=out_dim, lr=lr)

    def build_module(self, input_shape):
        return nn.Linear(input_shape[-1], self.out_dim)

    def output_shape(self, input_shape):
        return (self.out_dim,)


def _build():
    torch.manual_seed(0)
    g = Graph()
    a, b = Dense(out_dim=8), Dense(out_dim=2)
    g.node("a", a)
    g.node("b", b)
    g.connect("a", "b")
    W = torch.randn(2, 4)
    x = torch.randn(64, 4)
    y = x @ W.T
    return g, a, b, x, y


def _train_step(g, x, y):
    with g.context() as ctx:
        g.zero_grad()
        out, _ = g.forward(x)
        loss = nn.functional.mse_loss(out, y)
        g.backward(ctx, loss)
    g.step(ctx)


# Suppress the benign PyTorch warning about backward hooks on leaf inputs.
@pytest.fixture(autouse=True)
def _silence_torch_hook_warning():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Full backward hook is firing when gradients are computed",
        )
        yield


# ── 4.1 / 4.2 Healthy training is reported as healthy ────────


def test_healthy_two_filter_graph():
    g, a, b, x, y = _build()
    g.materialize(x)
    g.train()
    g.make_optimizer(torch.optim.Adam, lr=1e-2)

    with g.gradient_audit() as audit:
        for _ in range(5):
            _train_step(g, x, y)

    rep = audit.report()
    assert [f.filter_id for f in rep.filters] == ["a", "b"]
    assert all(f.n_steps == 5 for f in rep.filters)
    assert rep.is_healthy(), [(f.filter_id, f.flags) for f in rep.filters]
    audit.assert_healthy()


def test_param_grads_captured_for_every_filter():
    """Earlier filters must show non-NaN param_grad_norm too — the
    timing fix (snapshot after backward) is what lets this work."""
    g, _, _, x, y = _build()
    g.materialize(x)
    g.train()
    g.make_optimizer(torch.optim.Adam, lr=1e-2)
    with g.gradient_audit() as audit:
        _train_step(g, x, y)
    for f in audit.report().filters:
        assert f.metrics["param_grad_norm"] > 0, f"{f.filter_id}: missing param grad"
        assert f.metrics["param_norm"] > 0


# ── 4.3 Threshold flags ─────────────────────────────────────


def test_vanishing_flag_via_aggressive_grad_lo():
    g, _, _, x, y = _build()
    g.materialize(x)
    g.train()
    g.make_optimizer(torch.optim.Adam, lr=1e-2)

    with g.gradient_audit(thresholds=Thresholds(grad_lo=1.0, grad_hi=1e9)) as audit:
        # Tiny loss → tiny grads → both filters flag VANISHING under
        # the aggressive threshold.
        with g.context() as ctx:
            g.zero_grad()
            out, _ = g.forward(x)
            loss = nn.functional.mse_loss(out, y) * 1e-9
            g.backward(ctx, loss)
        g.step(ctx)

    rep = audit.report()
    assert all("VANISHING" in f.flags for f in rep.filters)
    with pytest.raises(GradientHealthError, match="VANISHING"):
        audit.assert_healthy()


def test_exploding_flag_via_aggressive_grad_hi():
    g, _, _, x, y = _build()
    g.materialize(x)
    g.train()
    g.make_optimizer(torch.optim.Adam, lr=1e-2)

    with g.gradient_audit(thresholds=Thresholds(grad_lo=0.0, grad_hi=1e-3)) as audit:
        _train_step(g, x, y)
    rep = audit.report()
    flagged = [f.filter_id for f in rep.filters if "EXPLODING" in f.flags]
    assert flagged, "EXPLODING should fire when grad_hi is below normal grad norm"


def test_nan_flag_when_param_set_to_nan():
    g, a, _, x, y = _build()
    g.materialize(x)
    g.train()
    with torch.no_grad():
        a._module.weight[0, 0] = float("nan")
    g.make_optimizer(torch.optim.Adam, lr=1e-2)

    with g.gradient_audit() as audit:
        with g.context() as ctx:
            g.zero_grad()
            out, _ = g.forward(x)
            loss = nn.functional.mse_loss(out, y)
            g.backward(ctx, loss)
        # Skip optimizer step (would propagate NaN further).

    rep = audit.report()
    flagged = [f.filter_id for f in rep.filters if "NAN" in f.flags]
    assert flagged, f"expected NAN flag, got {[(f.filter_id, f.flags) for f in rep.filters]}"
    with pytest.raises(GradientHealthError, match="NAN"):
        audit.assert_healthy()


# ── 4.4 Lifecycle: hooks are removed on exit ────────────────


def test_hooks_removed_after_context_exits():
    g, a, b, x, y = _build()
    g.materialize(x)
    g.train()
    g.make_optimizer(torch.optim.Adam, lr=1e-2)

    with g.gradient_audit() as audit:
        _train_step(g, x, y)
    assert audit._handles == [], "all hook handles should be removed"

    # active_audit must be unregistered from py_state.
    keys = list(g.py_state.keys()) if hasattr(g.py_state, "keys") else []
    assert "active_audit" not in keys

    # No new records when no audit is active.
    n_before = audit.report().filters[0].n_steps
    _train_step(g, x, y)
    n_after = audit.report().filters[0].n_steps
    assert n_after == n_before


def test_hooks_removed_on_exception_inside_context():
    g, _, _, x, y = _build()
    g.materialize(x)
    g.train()
    g.make_optimizer(torch.optim.Adam, lr=1e-2)

    with pytest.raises(ValueError, match="boom"):
        with g.gradient_audit() as audit:
            _train_step(g, x, y)
            raise ValueError("boom")
    assert audit._handles == []


# ── 4.5 Pretty + DataFrame ──────────────────────────────────


def test_pretty_contains_node_id_and_flag():
    g, _, _, x, y = _build()
    g.materialize(x)
    g.train()
    g.make_optimizer(torch.optim.Adam, lr=1e-2)
    with g.gradient_audit() as audit:
        _train_step(g, x, y)
    text = audit.report().pretty()
    assert "a" in text and "b" in text
    assert "HEALTHY" in text


def test_dataframe_shape_and_columns():
    pd = pytest.importorskip("pandas")
    g, _, _, x, y = _build()
    g.materialize(x)
    g.train()
    g.make_optimizer(torch.optim.Adam, lr=1e-2)
    with g.gradient_audit() as audit:
        _train_step(g, x, y)
    df = audit.report().dataframe()
    assert list(df["filter"]) == ["a", "b"]
    for col in ["param_grad_norm", "param_norm", "grad_param_ratio", "flags"]:
        assert col in df.columns


# ── Standalone audit_modules (no Graph required) ────────────


def test_audit_modules_standalone():
    torch.manual_seed(0)
    a = nn.Linear(4, 8)
    b = nn.Linear(8, 2)
    x = torch.randn(16, 4)
    y = torch.randn(16, 2)
    opt = torch.optim.Adam(list(a.parameters()) + list(b.parameters()), lr=1e-2)

    # Use a manual Audit wrapper since standalone audit_modules has no
    # Graph to call snapshot_after_backward. We call it ourselves.
    with audit_modules([("a", a), ("b", b)]) as audit:
        for _ in range(3):
            opt.zero_grad()
            out = b(a(x))
            loss = nn.functional.mse_loss(out, y)
            loss.backward()
            audit._snapshot_after_backward()
            opt.step()

    rep = audit.report()
    assert rep.is_healthy()
    assert all(f.n_steps == 3 for f in rep.filters)
