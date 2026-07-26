"""Literature-grounded pathological architectures, detected at DEFAULT
thresholds.

Unlike the threshold-tweaking tests in test_gradient_audit.py, every
network here genuinely exhibits its pathology:

- vanishing gradients in a deep sigmoid stack (Hochreiter 1991;
  Bengio et al. 1994; Glorot & Bengio 2010)
- exploding gradients from high-gain init (Pascanu et al. 2013)
- dying ReLU from a large negative bias (Lu et al. 2020)
- saturation from unbounded high-scale activations (Glorot & Bengio)
- NaN/Inf from genuine numeric overflow (Micikevicius et al. 2018)
- rank collapse through a width-1 bottleneck (Dong et al. 2021;
  effective rank per Roy & Vetterli 2007)
- dormant-but-alive channels from scale imbalance (Sokar et al. 2023 —
  dormancy is NOT death)
- leakage between tied parallel branches (CKA, Kornblith et al. 2019)
- plus a healthy deep control that must raise NO flags (false-positive
  guard).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

import soma  # noqa: E402
from soma import ChannelConfig, DifferentiableFilter, Graph  # noqa: E402


class Block(DifferentiableFilter):
    """One Linear(+activation) stage with controllable pathology knobs."""

    def __init__(
        self,
        width=16,
        activation="sigmoid",
        gain=1.0,
        bias=None,
        channel_scale=None,  # {channel_index: factor} applied after the activation
        **kw,
    ):
        super().__init__(**kw)
        self.width = width
        self.activation = activation
        self.gain = gain
        self.bias = bias
        self.channel_scale = channel_scale or {}

    def build_module(self, input_shape):
        lin = nn.Linear(input_shape[-1], self.width)
        with torch.no_grad():
            lin.weight.mul_(self.gain)
            if self.bias is not None:
                lin.bias.fill_(self.bias)
        act = {
            "sigmoid": nn.Sigmoid(),
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "none": nn.Identity(),
        }[self.activation]
        layers = [lin, act]
        if self.channel_scale:
            scale = torch.ones(self.width)
            for c, factor in self.channel_scale.items():
                scale[c] = factor
            class _Scale(nn.Module):
                def __init__(self, s):
                    super().__init__()
                    self.register_buffer("s", s)

                def forward(self, x):
                    return x * self.s

            layers.append(_Scale(scale))
        return nn.Sequential(*layers)

    def output_shape(self, input_shape):
        return (*input_shape[:-1], self.width)


def chain(blocks):
    g = Graph()
    ids = []
    for i, b in enumerate(blocks):
        node_id = f"block{i}"
        g.node(node_id, b)
        if ids:
            g.connect(ids[-1], node_id)
        ids.append(node_id)
    return g, ids


def train(g, ids, x, n_steps=3, lr=1e-3, channels=None, loss_fn=None):
    """Drive the chained modules through the native loop under an audit
    with DEFAULT thresholds."""
    g.materialize(x)
    g.train()
    g.make_optimizer(lr=lr)
    modules = [dict(g.filters())[i]._module for i in ids]

    with g.gradient_audit(channels=channels) as audit:
        for _ in range(n_steps):
            with g.context() as ctx:
                g.zero_grad()
                out = x
                for m in modules:
                    out = m(out)
                loss = (loss_fn or (lambda o: (o**2).mean()))(out)
                g.backward(ctx, loss)
            g.step(ctx)
    return audit.report().by_id()


# ── Vanishing gradients ─────────────────────────────────────────────


def test_deep_sigmoid_stack_vanishes_at_default_thresholds():
    """σ'(z) ≤ 0.25 per layer: with unit-gain init the backpropagated
    signal shrinks geometrically with depth — the classic pathology of
    Hochreiter/Bengio. The EARLY layers must fall under the default
    grad_lo and flag VANISHING; depth ordering must be visible in the
    per-node gradient norms."""
    torch.manual_seed(0)
    depth = 10
    g, ids = chain([Block(width=16, activation="sigmoid") for _ in range(depth)])
    report = train(g, ids, torch.randn(32, 16))

    norms = [report[i].metrics["param_grad_norm"] for i in ids]
    # Monotone-ish decay front-to-back: first ≪ last.
    assert norms[0] < norms[-1] * 1e-4, f"no depth decay: {norms}"
    assert "VANISHING" in report[ids[0]].flags, (
        f"first block grad {norms[0]:.2e} should be under the default "
        f"threshold; flags={report[ids[0]].flags}"
    )
    # The last block still learns — vanishing is a DEPTH profile, not a
    # global property.
    assert "VANISHING" not in report[ids[-1]].flags


# ── Exploding gradients ─────────────────────────────────────────────


def test_high_gain_relu_stack_explodes_at_default_thresholds():
    """Weight gain g > 1 compounds through depth (Pascanu et al.): the
    gradient norm must exceed the default grad_hi in at least the early
    blocks (gradients grow backwards) without any threshold tuning."""
    torch.manual_seed(0)
    depth = 8
    g, ids = chain([Block(width=16, activation="relu", gain=5.0) for _ in range(depth)])
    report = train(
        g, ids, torch.randn(32, 16), n_steps=1, lr=1e-12,
        loss_fn=lambda o: (o**2).sum(),
    )

    exploded = [i for i in ids if "EXPLODING" in report[i].flags]
    assert exploded, {i: report[i].metrics["param_grad_norm"] for i in ids}
    # Backward amplification shows in the OUTPUT gradient (the pure
    # backpropagated signal δ). Weight gradients are δ⊗activations and
    # activations grow FORWARD, so they need not be monotone.
    first = report[ids[0]].metrics["out_grad_norm"]
    last = report[ids[-1]].metrics["out_grad_norm"]
    assert first > last * 10, f"expected backward amplification: first={first}, last={last}"


# ── Dying ReLU ──────────────────────────────────────────────────────


def test_dying_relu_block_is_dead_at_default_thresholds():
    """A strongly negative bias puts every pre-activation below zero on
    the whole data distribution (Lu et al. 2020): output is identically
    zero → DEAD, and with channels enabled every channel is dead."""
    torch.manual_seed(0)
    g, ids = chain(
        [
            Block(width=16, activation="relu"),
            Block(width=16, activation="relu", bias=-25.0),
        ]
    )
    report = train(g, ids, torch.randn(32, 16), channels=True)

    dead = report["block1"]
    assert "DEAD" in dead.flags, dead.flags
    assert dead.metrics["dead_channels"] == 16.0
    # The healthy upstream block is not dead.
    assert "DEAD" not in report["block0"].flags


# ── Saturation ──────────────────────────────────────────────────────


def test_high_scale_activations_saturate_at_default_thresholds():
    """Activations with |x| beyond the default saturation bound (50.0)
    on most units — the pre-activation blow-up Glorot & Bengio tracked
    as saturation fraction."""
    torch.manual_seed(0)
    g, ids = chain([Block(width=16, activation="none", gain=400.0)])
    report = train(g, ids, torch.randn(32, 16), n_steps=1, lr=1e-12)
    rep = report["block0"]
    assert "SATURATED" in rep.flags, rep.flags
    assert rep.metrics["act_sat_frac_max"] > 0.5


# ── NaN / Inf from genuine overflow ─────────────────────────────────


def test_numeric_overflow_produces_nan_inf_flags():
    """float32 overflows past ~3.4e38: two 1e30-scale stages overflow
    the forward pass to inf, and backward turns to NaN — both flags
    must fire and assert_healthy must raise."""

    class Overflow(DifferentiableFilter):
        def build_module(self, input_shape):
            lin = nn.Linear(input_shape[-1], 8)
            with torch.no_grad():
                lin.weight.fill_(1e30)
                lin.bias.zero_()
            return lin

        def output_shape(self, input_shape):
            return (*input_shape[:-1], 8)

    torch.manual_seed(0)
    g, ids = chain([Overflow(), Overflow()])
    report = train(g, ids, torch.randn(8, 8), n_steps=1, lr=0.0)

    flags = set(report["block1"].flags) | set(report["block0"].flags)
    assert "INF" in flags or "NAN" in flags, flags

    # assert_healthy is the fail-fast entry point for CI training jobs.
    g2, ids2 = chain([Overflow(), Overflow()])
    g2.materialize(torch.randn(8, 8))
    g2.train()
    g2.make_optimizer(lr=0.0)
    modules = [dict(g2.filters())[i]._module for i in ids2]
    with g2.gradient_audit() as audit:
        with g2.context() as ctx:
            g2.zero_grad()
            out = torch.randn(8, 8)
            for m in modules:
                out = m(out)
            g2.backward(ctx, (out**2).mean())
    with pytest.raises(soma.GradientHealthError):
        audit.assert_healthy()


# ── Rank collapse ───────────────────────────────────────────────────


def test_width_one_bottleneck_collapses_effective_rank():
    """Everything downstream of a width-1 bottleneck is an affine
    function of ONE variable: the (uncentered) activation matrix spans
    at most two directions (weight + bias), so effective rank (Roy &
    Vetterli) collapses to ≲2 while a full-width control stays high
    (Dong et al.'s collapse signature)."""
    torch.manual_seed(0)
    g, ids = chain(
        [
            Block(width=16, activation="relu"),
            Block(width=1, activation="none"),
            Block(width=16, activation="none"),
        ]
    )
    report = train(
        g, ids, torch.randn(64, 16), channels=ChannelConfig(snapshot_every=1)
    )
    collapsed = report["block2"].metrics["eff_rank"]
    assert collapsed < 2.2, f"post-bottleneck eff_rank {collapsed} should be ≲2"

    torch.manual_seed(1)
    g2, ids2 = chain(
        [
            Block(width=16, activation="relu"),
            Block(width=16, activation="none"),
            Block(width=16, activation="none"),
        ]
    )
    report2 = train(
        g2, ids2, torch.randn(64, 16), channels=ChannelConfig(snapshot_every=1)
    )
    healthy = report2["block2"].metrics["eff_rank"]
    assert healthy > 3 * collapsed, f"control rank {healthy} vs collapsed {collapsed}"


# ── Dormancy is not death (Sokar et al.) ────────────────────────────


def test_scale_imbalanced_channels_are_dormant_but_not_dead():
    """Channels attenuated ×1e-4 are ALIVE (nonzero activations) but
    dormant on Sokar's normalized score — the audit must separate the
    two concepts: dormancy_frac > 0 while dead_channels == 0."""
    torch.manual_seed(0)
    g, ids = chain(
        [Block(width=16, activation="none", channel_scale={0: 1e-4, 1: 1e-4, 2: 1e-4})]
    )
    report = train(g, ids, torch.randn(32, 16), channels=True)
    rep = report["block0"]
    assert rep.metrics["dead_channels"] == 0, "attenuated ≠ zero"
    assert rep.metrics["dormancy_frac"] >= 3 / 16 - 1e-9, rep.metrics


# ── Architectural leakage: tied parallel branches ───────────────────


def test_tied_parallel_branches_flag_leakage():
    """Two 'independent' branches whose weights are tied at init compute
    the same function of the input: cross-branch CKA ≈ 1 → LEAKAGE.
    This is the branch-redundancy case (vs. the raw channel-duplication
    case in test_diagnostics)."""

    class TiedBranches(DifferentiableFilter):
        def build_module(self, input_shape):
            class M(nn.Module):
                def __init__(self, d):
                    super().__init__()
                    self.a = nn.Linear(d, 4)
                    self.b = nn.Linear(d, 4)
                    with torch.no_grad():
                        self.b.weight.copy_(self.a.weight)
                        self.b.bias.copy_(self.a.bias)

                def forward(self, x):
                    return torch.cat([self.a(x), self.b(x)], dim=1)

            return M(input_shape[-1])

        def output_shape(self, input_shape):
            return (*input_shape[:-1], 8)

    torch.manual_seed(0)
    g = Graph()
    g.node("branches", TiedBranches())
    x = torch.randn(64, 8)
    g.materialize(x)
    g.train()
    g.make_optimizer(lr=1e-3)
    module = dict(g.filters())["branches"]._module

    cfg = ChannelConfig(
        snapshot_every=1, groups={"branches": {"a": range(0, 4), "b": range(4, 8)}}
    )
    with g.gradient_audit(channels=cfg) as audit:
        for _ in range(2):
            with g.context() as ctx:
                g.zero_grad()
                g.backward(ctx, (module(x) ** 2).mean())
            g.step(ctx)

    rep = audit.report().by_id()["branches"]
    assert rep.metrics["max_group_cka"] > 0.95
    assert "LEAKAGE" in rep.flags


# ── False-positive guard ────────────────────────────────────────────


def test_healthy_deep_relu_stack_raises_no_flags():
    """Conventionally initialized deep stacks must be clean at default
    thresholds — the pathology detectors may not cry wolf.

    Two controls: a tanh stack (no exact zeros → strictly zero dead
    channels) asserts full cleanliness; a ReLU stack asserts no MACRO
    flags while tolerating a handful of dead channels — deep default-
    init ReLU nets genuinely lose units at init (Lu et al. 2020), so
    those detections are true positives, not noise."""
    torch.manual_seed(0)
    depth = 6

    g, ids = chain([Block(width=16, activation="tanh") for _ in range(depth)])
    report = train(g, ids, torch.randn(32, 16), channels=True)
    for i in ids:
        pathological = [
            f
            for f in report[i].flags
            if f.startswith(
                ("VANISHING", "EXPLODING", "DEAD", "SATURATED", "NAN", "INF", "LEAKAGE", "IGNORED")
            )
        ]
        assert not pathological, f"{i}: false positives {pathological}"
        assert report[i].metrics["dead_channels"] == 0

    torch.manual_seed(0)
    g2, ids2 = chain([Block(width=16, activation="relu") for _ in range(depth)])
    report2 = train(g2, ids2, torch.randn(32, 16), channels=True)
    for i in ids2:
        macro = [
            f
            for f in report2[i].flags
            if f.startswith(
                ("VANISHING", "EXPLODING", "SATURATED", "NAN", "INF", "LEAKAGE", "IGNORED")
            )
        ]
        assert not macro, f"{i}: false positives {macro} — {report2[i].metrics}"
    # Init-time dead ReLU units are expected — and INCREASINGLY so with
    # depth as activations correlate (Lu et al. 2020 predict massive
    # unit death in deep default-init ReLU nets). Only the first block,
    # fed by uncorrelated inputs, is held to a chance-level bound; the
    # deep blocks' detections are true positives the macro flags must
    # nevertheless stay quiet about.
    assert report2[ids2[0]].metrics["dead_channels"] <= 2
