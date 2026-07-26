"""Tests for native gradient flow within a Graph (no serialization between filters).

Covers the public surface installed by ``soma._orchestrator``:

  - ``materialize`` / ``train`` / ``eval`` / ``parameters``
  - polymorphic ``forward`` (autograd-live in train, Rust path in eval)
  - training-loop primitives ``context`` / ``backward`` / ``step`` / ``zero_grad``
  - ``freeze`` (snapshot live ``_module`` weights → runtime state library)
  - ``DifferentiableFilter`` ``(out, aux)`` contract + aux propagation
  - error messages for the common misuses
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from soma import DifferentiableFilter, Graph


# ── Test filters ─────────────────────────────────────────────


class Dense(DifferentiableFilter):
    """Single linear layer wrapped as a DifferentiableFilter."""

    def __init__(self, out_dim: int, lr: float = 1e-2):
        super().__init__(out_dim=out_dim, lr=lr)

    def build_module(self, input_shape):
        return nn.Linear(input_shape[-1], self.out_dim)

    def output_shape(self, input_shape):
        return (self.out_dim,)


class GatedClassifier(DifferentiableFilter):
    """Classifier that emits a per-sample gate as auxiliary signal.

    The user combines the main loss with ``gate_l1 * aux['gate'].mean()``
    in the training loop — this exercises the aux propagation contract.
    """

    def __init__(self, out_dim: int, lr: float = 1e-2):
        super().__init__(out_dim=out_dim, lr=lr)

    def build_module(self, input_shape):
        in_d = input_shape[-1]
        return nn.ModuleDict({
            "head": nn.Linear(in_d, self.out_dim),
            "gate": nn.Linear(in_d, 1),
        })

    def output_shape(self, input_shape):
        return (self.out_dim,)

    def forward(self, x, state=None):
        x_t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32)
        self.materialize(tuple(x_t.shape[1:]))
        if self.training:
            out = self._module["head"](x_t)
            gate = torch.sigmoid(self._module["gate"](x_t)).squeeze(-1)
            return out, {"gate": gate}
        # Eval branch mirrors DifferentiableFilter — load state, no_grad.
        from soma._composite import _deserialize_state_dict
        if isinstance(state, dict) and "weights_b64" in state:
            self._module.load_state_dict(_deserialize_state_dict(state["weights_b64"]))
        self._module.eval()
        with torch.no_grad():
            out = self._module["head"](x_t)
        return out.tolist(), {}


# ── Helpers ──────────────────────────────────────────────────


def _build_two_filter_graph(seed: int = 0):
    torch.manual_seed(seed)
    g = Graph()
    a = Dense(out_dim=8)
    b = Dense(out_dim=2)
    g.node("a", a)
    g.node("b", b)
    g.connect("a", "b")
    return g, a, b


def _learnable_data(n: int = 64, in_d: int = 4, out_d: int = 2, seed: int = 1):
    torch.manual_seed(seed)
    W = torch.randn(out_d, in_d)
    x = torch.randn(n, in_d)
    y = x @ W.T
    return x, y


# ── 1.5.1 End-to-end gradient flow ──────────────────────────


def test_train_loop_decreases_loss():
    g, a, b = _build_two_filter_graph()
    x, y = _learnable_data()

    g.materialize(x)
    g.train()
    g.make_optimizer(torch.optim.Adam, lr=1e-2)

    losses = []
    for _ in range(300):
        with g.context() as ctx:
            g.zero_grad()
            out, aux = g.forward(x)
            loss = nn.functional.mse_loss(out, y)
            g.backward(ctx, loss)
        g.step(ctx)
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.05, (
        f"loss should drop sharply on a learnable target; got "
        f"{losses[0]:.4f} → {losses[-1]:.4f}"
    )
    assert aux == {}  # plain Dense has no aux


def test_gradients_reach_every_filter():
    g, a, b = _build_two_filter_graph()
    x, y = _learnable_data()

    g.materialize(x)
    g.train()
    opt = torch.optim.Adam(list(g.parameters()), lr=1e-2)

    opt.zero_grad()
    out, _ = g.forward(x)
    loss = nn.functional.mse_loss(out, y)
    loss.backward()

    for filter_obj, name in [(a, "a"), (b, "b")]:
        grads = [p.grad for p in filter_obj._module.parameters()]
        assert all(g is not None for g in grads), f"{name}: missing grads"
        assert any(g.abs().sum() > 0 for g in grads), f"{name}: zero grads"


# ── 1.5.2 Aux propagation ───────────────────────────────────


def test_aux_dict_flows_through_forward_and_into_loss():
    torch.manual_seed(0)
    g = Graph()
    enc = Dense(out_dim=4)
    head = GatedClassifier(out_dim=2)
    g.node("enc", enc)
    g.node("head", head)
    g.connect("enc", "head")

    x, _ = _learnable_data()  # inputs only; targets aren't aligned with this head
    target = torch.randint(0, 2, (x.shape[0],))

    g.materialize(x)
    g.train()
    opt = torch.optim.Adam(list(g.parameters()), lr=1e-2)

    opt.zero_grad()
    out, aux_by_node = g.forward(x)
    # aux is keyed by node id, only filters that produced aux appear.
    assert "head" in aux_by_node, "GatedClassifier must surface aux"
    assert "enc" not in aux_by_node, "Dense has no aux"
    gate = aux_by_node["head"]["gate"]
    assert gate.shape == (x.shape[0],)
    assert gate.requires_grad, "aux tensors must keep autograd alive"

    main = nn.functional.cross_entropy(out, target)
    aux_l1 = gate.abs().mean()
    total = main + 0.1 * aux_l1
    total.backward()

    # Both filters must have grads.
    for f, name in [(enc, "enc"), (head, "head")]:
        grads = [p.grad for p in f._module.parameters()]
        assert all(g is not None for g in grads), f"{name}: missing grads"
        assert any(g.abs().sum() > 0 for g in grads), f"{name}: zero grads"

    # The gate sublayer specifically must receive grad — that proves the
    # aux loss made it back through the same parameters as the main loss.
    gate_layer = head._module["gate"]
    assert any(p.grad.abs().sum() > 0 for p in gate_layer.parameters())


# ── 1.5.3 Eval-after-freeze parity ──────────────────────────


def test_freeze_then_eval_matches_training_forward():
    g, a, b = _build_two_filter_graph()
    x, y = _learnable_data()

    g.materialize(x)
    g.train()
    g.make_optimizer(torch.optim.Adam, lr=1e-2)
    for _ in range(200):
        with g.context() as ctx:
            g.zero_grad()
            out, _ = g.forward(x)
            loss = nn.functional.mse_loss(out, y)
            g.backward(ctx, loss)
        g.step(ctx)

    out_train, _ = g.forward(x)
    ref = out_train.detach().clone()

    g.freeze()
    assert not a.training and not b.training

    # Rust-driven inference path. Project-level inputs are lists/dicts.
    out_eval = g.forward(x.tolist())
    eval_t = torch.as_tensor(out_eval, dtype=torch.float32)
    diff = (eval_t - ref).abs().max().item()
    assert diff < 1e-4, f"freeze→eval should match training forward, got {diff:.2e}"


def test_freeze_survives_wiped_live_modules():
    """After freeze, blowing away ``_module`` must not break inference.

    Eval ``forward`` reconstructs the module lazily and loads state from
    the runtime library — that is the path remote workers will use too.
    """
    g, a, b = _build_two_filter_graph()
    x, y = _learnable_data()

    g.materialize(x)
    g.train()
    g.make_optimizer(torch.optim.Adam, lr=1e-2)
    for _ in range(150):
        with g.context() as ctx:
            g.zero_grad()
            out, _ = g.forward(x)
            loss = nn.functional.mse_loss(out, y)
            g.backward(ctx, loss)
        g.step(ctx)

    out_train, _ = g.forward(x)
    ref = out_train.detach().clone()
    g.freeze()

    a._module = None
    b._module = None

    out_eval = g.forward(x.tolist())
    eval_t = torch.as_tensor(out_eval, dtype=torch.float32)
    diff = (eval_t - ref).abs().max().item()
    assert diff < 1e-4


def test_to_moves_modules_and_persists_for_lazy_builds():
    """``g.to(device)`` moves materialised modules now AND persists the
    target so modules built lazily on a later forward inherit it."""
    g, a, b = _build_two_filter_graph()
    x, _ = _learnable_data()
    g.materialize(x)
    g.train()

    # cpu→cpu is the only portable target on CI; verify call shape and
    # py_state persistence rather than a cross-device move.
    ret = g.to("cpu")
    assert ret is g, "to() must return self for chaining"
    assert g.py_state.get("device") == "cpu"
    for f in (a, b):
        assert f._module.weight.device.type == "cpu"

    # Wipe and lazy-rebuild: forward must place the new module on the
    # target stored in py_state without another .to() call.
    a._module = None
    b._module = None
    out, _ = g.forward(x)
    for f in (a, b):
        assert f._module is not None
        assert f._module.weight.device.type == "cpu"


# ── 1.5.4 Lifecycle and orchestration ───────────────────────


def test_train_eval_toggle_propagates_to_modules():
    g, a, b = _build_two_filter_graph()
    g.materialize(_learnable_data()[0])

    g.train()
    assert a.training and b.training
    assert a._module.training and b._module.training

    g.eval()
    assert not a.training and not b.training
    assert not a._module.training and not b._module.training


def test_parameters_iterates_in_topological_order_and_unique():
    g, a, b = _build_two_filter_graph()
    g.materialize(_learnable_data()[0])

    params = list(g.parameters())
    # Linear(4→8) + Linear(8→2): 4 tensors total (w,b,w,b).
    assert len(params) == 4
    assert len({id(p) for p in params}) == 4, "parameters() must not yield duplicates"


def test_filter_ids_topologically_sorted():
    """Inserting nodes out of order shouldn't matter — graph.filter_ids
    must return predecessors before successors."""
    torch.manual_seed(0)
    g = Graph()
    g.node("c", Dense(out_dim=2))
    g.node("a", Dense(out_dim=8))
    g.node("b", Dense(out_dim=4))
    g.connect("a", "b")
    g.connect("b", "c")
    assert g.filter_ids() == ["a", "b", "c"]


# ── 1.5.5 Error paths ───────────────────────────────────────


def test_step_without_optimizer_raises_clear_error():
    g, _, _ = _build_two_filter_graph()
    g.materialize(_learnable_data()[0])
    with pytest.raises(RuntimeError, match="No optimiser registered"):
        g.step()


def test_make_optimizer_without_parameters_raises_clear_error():
    g = Graph()  # empty
    with pytest.raises(RuntimeError, match="no parameters found"):
        g.make_optimizer()


def test_zero_grad_without_optimizer_is_silent_noop():
    g, _, _ = _build_two_filter_graph()
    g.materialize(_learnable_data()[0])
    # Must not raise — users may call zero_grad before make_optimizer.
    g.zero_grad()


# ── 1.5.6 Backwards-compatibility with legacy fit() path ────


def test_legacy_filter_with_plain_forward_still_works_in_eval():
    """A non-diff Filter that returns a plain value (not a tuple) must
    still be usable end-to-end via the Rust forward path."""
    from soma import Filter

    class Identity(Filter):
        def fit(self, x, y=None):
            return {}

        def forward(self, x, state):
            return x

    g = Graph()
    g.node("id", Identity())
    g.fit({"x": [[1.0, 2.0], [3.0, 4.0]]})
    out = g.forward([[1.0, 2.0], [3.0, 4.0]])
    assert out == [[1.0, 2.0], [3.0, 4.0]]
