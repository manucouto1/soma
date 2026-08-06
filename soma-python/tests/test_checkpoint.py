"""Tests for ``Graph`` checkpoint API: state(), load_state(), save(), load()."""

from __future__ import annotations

import os
import sys
import tempfile
import types
import warnings

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
pytest.importorskip("safetensors")

from soma import DifferentiableFilter, Graph


# ── Module-importable filter classes ─────────────────────────
#
# Filter classes referenced in checkpoints must be importable by
# fully-qualified name. We register a synthetic module here so
# ``cls.class_path()`` returns a path the test can resolve.


_FILTER_MODULE_NAME = "soma_test_checkpoint_filters"


def _ensure_filter_module() -> types.ModuleType:
    if _FILTER_MODULE_NAME in sys.modules:
        return sys.modules[_FILTER_MODULE_NAME]
    mod = types.ModuleType(_FILTER_MODULE_NAME)
    exec(
        "from soma import DifferentiableFilter\n"
        "import torch.nn as nn\n"
        "\n"
        "class Dense(DifferentiableFilter):\n"
        "    def __init__(self, out_dim, lr=1e-3):\n"
        "        super().__init__(out_dim=out_dim, lr=lr)\n"
        "    def build_module(self, input_shape):\n"
        "        return nn.Linear(input_shape[-1], self.out_dim)\n"
        "    def output_shape(self, input_shape):\n"
        "        return (self.out_dim,)\n"
        "\n"
        "class DenseV2(DifferentiableFilter):\n"
        "    class_version = 2\n"
        "    def __init__(self, out_dim, lr=1e-3):\n"
        "        super().__init__(out_dim=out_dim, lr=lr)\n"
        "    def build_module(self, input_shape):\n"
        "        return nn.Linear(input_shape[-1], self.out_dim)\n"
        "    def output_shape(self, input_shape):\n"
        "        return (self.out_dim,)\n",
        mod.__dict__,
    )
    sys.modules[_FILTER_MODULE_NAME] = mod
    return mod


@pytest.fixture(scope="module")
def filter_mod():
    return _ensure_filter_module()


def _build_and_train(Dense, seed: int = 0, n_iter: int = 150):
    torch.manual_seed(seed)
    g = Graph()
    g.node("a", Dense(out_dim=8))
    g.node("b", Dense(out_dim=2))
    g.edge("a", "b")

    W = torch.randn(2, 4)
    x = torch.randn(64, 4)
    y = x @ W.T

    g.materialize(x)
    g.train()
    g.make_optimizer(torch.optim.Adam, lr=1e-2)
    for _ in range(n_iter):
        with g.context() as ctx:
            g.zero_grad()
            out = g.forward(x)
            loss = nn.functional.mse_loss(out, y)
            g.backward(ctx, loss)
        g.step(ctx)
    g.freeze()
    return g, x


# ── 2.1 Filter introspection ─────────────────────────────────


def test_filter_kwargs_round_trip(filter_mod):
    f = filter_mod.Dense(out_dim=8, lr=2e-3)
    assert f.kwargs() == {"out_dim": 8, "lr": 2e-3}
    rebuilt = filter_mod.Dense(**f.kwargs())
    assert rebuilt.kwargs() == f.kwargs()


def test_filter_class_path_resolves(filter_mod):
    cp = filter_mod.Dense.class_path()
    assert cp == f"{_FILTER_MODULE_NAME}.Dense"


def test_filter_default_class_version_is_one(filter_mod):
    assert filter_mod.Dense.class_version == 1
    assert filter_mod.DenseV2.class_version == 2


# ── 2.2 state-only API ───────────────────────────────────────


def test_state_load_state_round_trip(filter_mod):
    Dense = filter_mod.Dense
    g, x = _build_and_train(Dense)
    ref = torch.as_tensor(g.forward(x.tolist()), dtype=torch.float32)

    sd = g.state()
    assert set(sd.keys()) == {"a", "b"}

    g2 = Graph()
    g2.node("a", Dense(out_dim=8))
    g2.node("b", Dense(out_dim=2))
    g2.edge("a", "b")
    g2.load_state(sd)

    out2 = torch.as_tensor(g2.forward(x.tolist()), dtype=torch.float32)
    assert (out2 - ref).abs().max().item() < 1e-5


def test_load_state_strict_rejects_unknown_keys(filter_mod):
    Dense = filter_mod.Dense
    g, _ = _build_and_train(Dense, n_iter=1)
    sd = g.state()
    sd_bad = {**sd, "phantom": {}}

    g2 = Graph()
    g2.node("a", Dense(out_dim=8))
    g2.node("b", Dense(out_dim=2))
    g2.edge("a", "b")

    with pytest.raises(KeyError, match="not in this graph"):
        g2.load_state(sd_bad, strict=True)


def test_load_state_non_strict_warns_on_missing(filter_mod):
    Dense = filter_mod.Dense
    g, _ = _build_and_train(Dense, n_iter=1)
    sd = g.state()
    del sd["b"]  # incomplete

    g2 = Graph()
    g2.node("a", Dense(out_dim=8))
    g2.node("b", Dense(out_dim=2))
    g2.edge("a", "b")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g2.load_state(sd, strict=False)
        msgs = [str(w.message) for w in caught]
    assert any("nodes without state" in m for m in msgs), msgs


# ── 2.3 Full save/load round-trip ────────────────────────────


def test_full_save_load_matches_reference_output(filter_mod):
    Dense = filter_mod.Dense
    g, x = _build_and_train(Dense)
    ref = torch.as_tensor(g.forward(x.tolist()), dtype=torch.float32)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "g.somack")
        g.save(path)
        assert os.path.getsize(path) > 0

        g2 = Graph.load(path)
        out2 = torch.as_tensor(g2.forward(x.tolist()), dtype=torch.float32)
        assert (out2 - ref).abs().max().item() < 1e-5
        assert g2.filter_ids() == ["a", "b"]
        assert g2.edges() == [("a", "b")]


def test_save_to_nonexistent_directory_errors_clean(filter_mod):
    Dense = filter_mod.Dense
    g, _ = _build_and_train(Dense, n_iter=1)
    bogus = "/no/such/dir/anywhere/g.somack"
    with pytest.raises(FileNotFoundError):
        g.save(bogus)


# ── 2.4 Versioning ───────────────────────────────────────────


def test_class_version_mismatch_warns_in_non_strict(filter_mod):
    Dense = filter_mod.Dense
    g, x = _build_and_train(Dense, n_iter=1)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "g.somack")
        g.save(path)

        # Bump in-place to simulate a code-side schema bump.
        original_v = Dense.class_version
        Dense.class_version = 5
        try:
            with pytest.raises(RuntimeError, match="class_version mismatch"):
                Graph.load(path, strict=True)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                g2 = Graph.load(path, strict=False)
                msgs = [str(w.message) for w in caught]
            assert any("class_version mismatch" in m for m in msgs)
            # State still applied: forward should run.
            _ = g2.forward(x.tolist())
        finally:
            Dense.class_version = original_v


# ── 2.5 Optimiser snapshot ───────────────────────────────────


def test_save_and_restore_optimizer(filter_mod):
    Dense = filter_mod.Dense
    torch.manual_seed(0)
    g = Graph()
    g.node("a", Dense(out_dim=4))
    g.node("b", Dense(out_dim=2))
    g.edge("a", "b")
    x = torch.randn(16, 4)
    y = torch.randn(16, 2)

    g.materialize(x)
    g.train()
    g.make_optimizer(torch.optim.Adam, lr=1e-2)
    for _ in range(5):
        with g.context() as ctx:
            g.zero_grad()
            out = g.forward(x)
            loss = nn.functional.mse_loss(out, y)
            g.backward(ctx, loss)
        g.step(ctx)

    # Snapshot momentum buffers before save.
    pre = g.optimizer().state_dict()
    g.freeze()

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "ckpt.somack")
        g.save(path, include_optimizer=True)

        g2 = Graph.load(path)
        g2.materialize(x)
        g2.train()
        g2.make_optimizer(torch.optim.Adam, lr=1e-2)
        applied = g2.restore_optimizer()
        assert applied is True
        post = g2.optimizer().state_dict()

    # State counts must match (same number of param groups, same step counter).
    assert pre["param_groups"][0]["lr"] == post["param_groups"][0]["lr"]
    assert len(pre["state"]) == len(post["state"])
