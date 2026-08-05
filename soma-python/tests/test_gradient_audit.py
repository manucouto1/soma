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


# ── inside= (scoped submodule auditing) ─────────────────────────────

from soma import AuditScope  # noqa: E402
from soma._audit import _coerce_scope, _iter_scoped_modules  # noqa: E402


class Deep(DifferentiableFilter):
    """12 small tanh layers — a textbook vanishing-gradient stack."""

    def __init__(self, width=4, layers=12):
        super().__init__(width=width, layers=layers)

    def build_module(self, input_shape):
        mods = []
        for _ in range(self.layers):
            lin = nn.Linear(input_shape[-1], input_shape[-1])
            with torch.no_grad():
                lin.weight.mul_(0.3)
            mods.append(lin)
            mods.append(nn.Tanh())
        return nn.Sequential(*mods)

    def output_shape(self, input_shape):
        return input_shape


def _deep_graph(layers=12):
    torch.manual_seed(0)
    g = Graph()
    g.node("deep", Deep(layers=layers))
    x = torch.randn(32, 4)
    y = torch.randn(32, 4)
    g.materialize(x)
    g.train()
    g.make_optimizer(torch.optim.Adam, lr=1e-2)
    return g, x, y


def test_coerce_scope_duck_typing():
    assert _coerce_scope(None, where="w") is None
    assert _coerce_scope(False, where="w") is None
    assert _coerce_scope(True, where="w") == AuditScope()
    assert _coerce_scope("auto", where="w") == AuditScope()
    assert _coerce_scope(3, where="w") == AuditScope(depth=3)
    assert _coerce_scope(["a.*", "b"], where="w") == AuditScope(patterns=("a.*", "b"))
    scope = AuditScope(depth=2, sample_every=5)
    assert _coerce_scope(scope, where="w") is scope
    with pytest.raises(TypeError, match=r"inside\['enc'\]"):
        _coerce_scope(3.5, where="inside['enc']")


def test_iter_scoped_modules_selection():
    root = nn.Sequential(
        nn.Linear(4, 4),
        nn.Tanh(),
        nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)),
    )

    # depth=1: top-level entries with params ("0" and "2"), not tanh.
    names = [n for n, _ in _iter_scoped_modules(root, AuditScope(depth=1), node_id="x")]
    assert names == ["0", "2"]

    # depth=2 adds the inner linears.
    names = [n for n, _ in _iter_scoped_modules(root, AuditScope(depth=2), node_id="x")]
    assert names == ["0", "2", "2.0", "2.2"]

    # patterns: parameterless modules ARE included when named.
    names = [
        n
        for n, _ in _iter_scoped_modules(
            root, AuditScope(patterns=("1", "2.*")), node_id="x"
        )
    ]
    assert names == ["1", "2.0", "2.1", "2.2"]

    with pytest.warns(UserWarning, match="matched no submodule"):
        _iter_scoped_modules(root, AuditScope(patterns=("nope.*",)), node_id="x")

    # auto: direct children with params.
    names = [n for n, _ in _iter_scoped_modules(root, AuditScope(), node_id="x")]
    assert names == ["0", "2"]

    # auto descends a single wrapper child one level.
    wrapped = nn.Sequential(nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2)))
    names = [n for n, _ in _iter_scoped_modules(wrapped, AuditScope(), node_id="x")]
    assert names == ["0.0", "0.1"]

    # cap warns and truncates.
    with pytest.warns(UserWarning, match="dropping"):
        names = [
            n
            for n, _ in _iter_scoped_modules(
                root, AuditScope(depth=2, max_modules=2), node_id="x"
            )
        ]
    assert names == ["0", "2"]


def test_inside_true_hierarchical_ids_and_backcompat():
    g, x, y = _deep_graph(layers=3)

    # Default: byte-identical to today — root ids only.
    with g.gradient_audit() as audit:
        _train_step(g, x, y)
    assert list(audit.records()) == ["deep"]

    with g.gradient_audit(inside=True) as audit:
        _train_step(g, x, y)
    ids = list(audit.records())
    assert ids[0] == "deep", "root always audited, grouped first"
    assert ids[1:] == ["deep/0", "deep/2", "deep/4"]
    # Every child recorded a real gradient.
    for cid in ids[1:]:
        rec = audit.records()[cid][0]
        assert rec.out_grad_norm == rec.out_grad_norm, f"{cid} has NaN grad"
    # Execution order captured (ascending through the Sequential).
    assert audit._fwd_order["deep"] == ["0", "2", "4"]


def test_inside_precedence_class_vs_callsite():
    class Declared(Deep):
        _audit_scope = ["0"]  # only the first linear

    torch.manual_seed(0)
    g = Graph()
    g.node("deep", Declared(layers=3))
    x, y = torch.randn(8, 4), torch.randn(8, 4)
    g.materialize(x)
    g.train()
    g.make_optimizer(torch.optim.Adam, lr=1e-2)

    # inside=True → class declaration wins over auto.
    with g.gradient_audit(inside=True) as audit:
        _train_step(g, x, y)
    assert list(audit.records()) == ["deep", "deep/0"]

    # Call-site overrides the class declaration.
    with g.gradient_audit(inside={"deep": ["4"]}) as audit:
        _train_step(g, x, y)
    assert list(audit.records()) == ["deep", "deep/4"]

    # Default (inside not passed): declaration is inert.
    with g.gradient_audit() as audit:
        _train_step(g, x, y)
    assert list(audit.records()) == ["deep"]


def test_inside_sample_every(tmp_path):
    g, x, y = _deep_graph(layers=2)
    with g.track_run("sampled", root=str(tmp_path)):
        with g.gradient_audit(
            inside={"deep": AuditScope(depth=1, sample_every=2)}
        ) as audit:
            for _ in range(3):
                _train_step(g, x, y)

    recs = audit.records()
    assert len(recs["deep"]) == 3, "root records every step"
    assert [r.step for r in recs["deep/0"]] == [0, 2], "children sampled"

    import json
    import pathlib

    run_dir = next((tmp_path / "runs").iterdir())
    rows = [
        json.loads(line)
        for line in (run_dir / "diagnostics" / "audit_steps.jsonl")
        .read_text()
        .splitlines()
    ]
    by_fid = {}
    for row in rows:
        by_fid.setdefault(row["filter"], []).append(row["step"])
    assert by_fid["deep"] == [0, 1, 2]
    assert by_fid["deep/0"] == [0, 2], "no jsonl rows on sampled-out steps"


def test_inside_errors():
    g, x, y = _deep_graph(layers=2)

    with pytest.raises(ValueError, match="unknown node id"):
        with g.gradient_audit(inside={"nope": True}):
            pass

    class Plain:  # not differentiable: no _module at all
        def fit(self, x, y=None):
            return {}

        def forward(self, x, state):
            return x

    g2 = Graph()
    g2.node("plain", Plain())
    with pytest.raises(ValueError, match="not a differentiable filter"):
        with g2.gradient_audit(inside={"plain": True}):
            pass

    # Explicitly named but unmaterialized → warn + skip.
    g3 = Graph()
    g3.node("late", Deep(layers=2))
    with pytest.warns(UserWarning, match="not materialized"):
        with g3.gradient_audit(inside={"late": True}) as audit:
            pass
    assert audit.modules == []

    # A node id containing "/" colliding with a scoped child id.
    torch.manual_seed(0)
    g4 = Graph()
    g4.node("a", Deep(layers=1))
    g4.node("a/0", Dense(out_dim=4))
    g4.connect("a", "a/0")
    g4.materialize(torch.randn(4, 4))
    with pytest.raises(ValueError, match="collision"):
        with g4.gradient_audit(inside={"a": ["0"]}):
            pass


def test_inside_vanishing_deep_mlp():
    g, x, y = _deep_graph(layers=12)
    with g.gradient_audit(inside=True) as audit:
        for _ in range(2):
            _train_step(g, x, y)

    order = audit._fwd_order["deep"]
    assert len(order) == 12

    def mean_grad(path):
        recs = audit.records()[f"deep/{path}"]
        vals = [r.out_grad_norm for r in recs if r.out_grad_norm == r.out_grad_norm]
        return sum(vals) / len(vals)

    grads = [mean_grad(p) for p in order]
    # Backprop through 12 contractive tanh layers: early layers see far
    # smaller output-gradients than late ones — the staircase
    # plot_module_flow draws.
    assert grads[0] < grads[-1] / 10, f"expected decay toward input: {grads}"
    increasing = sum(1 for a, b in zip(grads, grads[1:]) if b > a)
    assert increasing >= 8, f"trend should rise along depth: {grads}"


def test_module_tree_persisted_and_flags_rolled_up(tmp_path):
    import json
    import pathlib

    g, x, y = _deep_graph(layers=3)
    # Aggressive threshold: every layer flags VANISHING.
    with g.track_run("tree", root=str(tmp_path)):
        with g.gradient_audit(
            thresholds=Thresholds(grad_lo=1e9, grad_hi=1e12), inside=True
        ):
            _train_step(g, x, y)

    run_dir = next((tmp_path / "runs").iterdir())
    tree_path = run_dir / "diagnostics" / "modules" / "deep.json"
    assert tree_path.exists()
    tree = json.loads(tree_path.read_text())
    assert tree["node"] == "deep"
    assert tree["order"] == ["0", "2", "4"], "execution order of the Sequential"
    assert tree["ids"] == {p: f"deep/{p}" for p in ("0", "2", "4")}
    # mermaid ids sanitized (digit-leading), labels raw.
    assert tree["mermaid_ids"]["0"] == "n_0"
    node0 = next(n for n in tree["graph"]["nodes"] if n["label"] == "0")
    assert node0["id"] == "n_0"
    assert node0["kind"] == {"type": "Filter", "filter_name": "Linear"}
    assert all(tree["params"][p] > 0 for p in tree["order"])
    # Edges chain the execution order.
    edges = [(e["source"], e["target"]) for e in tree["graph"]["edges"]]
    assert edges == [("n_0", "n_2"), ("n_2", "n_4")]

    # HealthFlags: one per flagged child + exactly one rolled-up parent
    # flag per family, whose detail names the submodules.
    events = [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text().splitlines()
    ]
    flags = [e for e in events if e["event_type"] == "HealthFlag"]
    child_vanishing = [
        f for f in flags if f["flag"] == "VANISHING" and f["node_id"].startswith("deep/")
    ]
    assert len(child_vanishing) == 3
    parent_vanishing = [
        f for f in flags if f["flag"] == "VANISHING" and f["node_id"] == "deep"
    ]
    rolled = [f for f in parent_vanishing if f["detail"].startswith("in: ")]
    assert len(rolled) == 1, f"exactly one rollup: {parent_vanishing}"
    assert rolled[0]["detail"] == "in: 0, 2, 4"


def test_differentiable_filter_repr_html_shows_architecture():
    enc = Deep(layers=3)
    # Before materialize: an informative note, no diagram.
    note = enc._repr_html_()
    assert "Deep" in note
    assert "not materialized" in note
    assert "<svg" not in note

    enc.materialize((4,))
    html = enc._repr_html_()
    assert "<svg xmlns=" in html
    assert "Linear" in html, "submodule class names in the diagram"
    assert "θ" in html, "parameter counts as sublabels"
    assert html.count("marker-end") == 2, "3 children chained by 2 edges"


def test_complex_multimodal_health_e2e(tmp_path):
    """Engineered pathologies on a branched model must all fire at
    DEFAULT thresholds: dying-ReLU channels, weight-collapsed branches
    (CKA leakage), and a gradient-starved branch — with rollup to the
    parent node. Mirrors notebooks/09."""
    import soma
    from soma import ChannelConfig

    class MultiModal(DifferentiableFilter):
        def build_module(self, input_shape):
            branches = nn.ModuleDict(
                {
                    "audio": nn.Sequential(nn.Linear(16, 16), nn.ReLU()),
                    "text": nn.Sequential(nn.Linear(16, 16), nn.ReLU()),
                }
            )
            with torch.no_grad():
                # leakage: branches collapsed to identical weights
                branches["text"][0].weight.copy_(branches["audio"][0].weight)
                branches["text"][0].bias.copy_(branches["audio"][0].bias)
            post = nn.Sequential(nn.Linear(32, 32), nn.ReLU())
            with torch.no_grad():
                post[0].bias[-4:] = -6.0  # dying ReLU after the fusion
            return nn.ModuleDict(
                {
                    "branches": branches,
                    "mix": nn.Identity(),
                    "post": post,
                    "ctx": nn.Linear(32, 4),  # gated to zero in forward
                    "head": nn.Linear(32, 4),
                }
            )

        def output_shape(self, input_shape):
            return (4,)

        def forward(self, x, state=None):
            x_t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
            self.materialize(tuple(x_t.shape[1:]))
            m = self._module
            a = m["branches"]["audio"](x_t[:, :16])
            t = m["branches"]["text"](x_t[:, 16:])
            mixed = m["mix"](torch.cat([a, t], dim=1))
            out = m["head"](m["post"](mixed)) + 0.0 * m["ctx"](mixed)
            return (out, {}) if self.training else (out.detach().tolist(), {})

    torch.manual_seed(0)
    g = Graph()
    g.node("encoder", MultiModal())
    audio = torch.randn(64, 16)
    x = torch.cat([audio, audio * 0.97 + 0.05 * torch.randn(64, 16)], dim=1)
    y = torch.randn(64, 4)
    g.materialize(x)
    g.train()
    g.make_optimizer(torch.optim.Adam, lr=1e-3)

    cfg = ChannelConfig(
        snapshot_every=1,
        groups={"encoder/mix": {"audio": range(0, 16), "text": range(16, 32)}},
    )
    with g.track_run("mm-health", root=str(tmp_path)):
        with g.gradient_audit(
            inside={"encoder": ["branches.audio", "branches.text", "mix", "post", "ctx"]},
            channels=cfg,
        ) as audit:
            for _ in range(3):
                _train_step(g, x, y)

    by_id = {f.filter_id: f for f in audit.report().filters}
    post_flags = by_id["encoder/post"].flags
    assert any(f.startswith("DEAD_CHANNELS") for f in post_flags), post_flags
    assert by_id["encoder/post"].metrics["dead_channels"] >= 4
    assert "LEAKAGE" in by_id["encoder/mix"].flags
    assert by_id["encoder/mix"].metrics["max_group_cka"] > 0.95
    assert any(f.startswith("IGNORED_CHANNELS") for f in by_id["encoder/ctx"].flags)

    # Rollup: the parent DAG node carries every family, naming layers.
    view = soma.RunView(soma.runs(str(tmp_path))[0].dir)
    parent = {
        f["flag"]: f["detail"]
        for f in view.health_flags()
        if f["node_id"] == "encoder" and f["detail"].startswith("in: ")
    }
    assert "DEAD_CHANNELS" in parent and "post" in parent["DEAD_CHANNELS"]
    assert "LEAKAGE" in parent and "mix" in parent["LEAKAGE"]
    assert "IGNORED_CHANNELS" in parent
    # And the outer SVG shows the flagged node.
    assert "#fff3e0" in view.to_svg(), "encoder painted as flagged"


def test_channel_snapshots_without_safetensors_warn_instead_of_vanishing(
    monkeypatch, tmp_path
):
    """A missing optional dependency has to say so.

    ``_persist_snapshot`` used to ``return`` on ImportError. The run
    finished, the health flags were correct, and nothing was written — so
    ``plot_channels`` reported "no channel snapshots (available: [])", a
    message about the filter rather than about the missing package.
    ``graph.save`` has always raised a clear error for the same import.

    Found by running an example against a clean install of the published
    package: every environment this was developed in happened to have
    ``safetensors`` pulled in by something else.
    """
    import builtins

    import soma
    from soma._audit import Audit

    real_import = builtins.__import__

    def no_safetensors(name, *args, **kwargs):
        if name.startswith("safetensors"):
            raise ImportError("no safetensors for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_safetensors)
    # Warn-once is per process by design, so a previous test must not
    # silence this one.
    monkeypatch.setattr(Audit, "_warned_no_safetensors", False)

    class Enc(soma.DifferentiableFilter):
        _cache_version = "chan-warn-v1"

        def __init__(self, out_dim=4, **kw):
            super().__init__(out_dim=out_dim, **kw)
            self.out_dim = out_dim

        def build_module(self, input_shape):
            torch.manual_seed(0)
            return nn.Sequential(
                nn.Linear(int(input_shape[-1]), self.out_dim), nn.ReLU()
            )

        def output_shape(self, input_shape):
            return (input_shape[0], self.out_dim)

    monkeypatch.chdir(tmp_path)
    g = soma.Graph(cache="memory")
    g.node("enc", Enc(4))
    x = torch.randn(8, 3)
    y = torch.randn(8, 4)
    g.materialize(x)
    g.train()
    g.make_optimizer(torch.optim.Adam, lr=1e-3)

    with pytest.warns(RuntimeWarning, match="safetensors"):
        with g.track_run("chan-warn"):
            with g.gradient_audit(channels=True):
                with g.context() as ctx:
                    g.zero_grad()
                    out, _ = g.forward(x)
                    g.backward(ctx, torch.nn.functional.mse_loss(out, y))
                g.step(ctx)


def test_plot_channels_on_an_empty_index_blames_the_run_not_the_filter():
    """An empty index is the run recording none, not this filter."""
    plotly = pytest.importorskip("plotly")  # noqa: F401
    from soma.viz._health import plot_channels

    class NoChannels:
        dir = "/nonexistent-run-dir-for-this-test"

    with pytest.raises(ValueError, match="no channel snapshots in this run at all"):
        plot_channels(NoChannels(), "encoder/fuse")
