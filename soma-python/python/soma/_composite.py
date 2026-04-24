"""DifferentiableFilter: base class with a default ``composite_fit``.

Subclasses implement:
  - ``build_module(input_shape)`` → the ``nn.Module`` this filter wraps.
  - ``output_shape(input_shape)`` → the shape this filter emits.

When ``graph.fit(..., mode="differentiable")`` sees consecutive
:class:`DifferentiableFilter` nodes, the compiler collapses them into a
``Composite`` block. The runtime calls ``composite_fit`` on the first
filter of the block, which by default:

  1. Walks the peer filters in topological order.
  2. Calls ``build_module(shape)`` on each, threading the shape through.
  3. Chains them with :class:`torch.nn.Sequential`.
  4. Runs an Adam loop across the full ``x``/``y`` with mini-batch SGD.
  5. Returns ``(final_output, {node_id: {weights_b64}})`` to Soma.

Hyperparameters (``lr``, ``epochs``, ``batch_size``, optimiser, loss)
are taken from the *first* filter in the composite block. Subclasses
that need per-filter hyperparameters (e.g. different encoder_lr) should
override ``composite_fit`` explicitly.
"""

from __future__ import annotations

import base64
import io

import torch
import torch.nn as nn

from soma.filter import Filter


def _to_tensor(x, dtype=torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype) if x.dtype != dtype else x
    return torch.as_tensor(x, dtype=dtype)


def _serialize_state_dict(module: nn.Module) -> str:
    buf = io.BytesIO()
    torch.save(module.state_dict(), buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _deserialize_state_dict(b64: str) -> dict:
    buf = io.BytesIO(base64.b64decode(b64))
    return torch.load(buf, weights_only=True)


class DifferentiableFilter(Filter):
    """Base class for filters that wrap a trainable ``nn.Module``.

    Subclass contract::

        class MyLayer(DifferentiableFilter):
            def __init__(self, out_dim: int, lr=1e-3, epochs=10):
                super().__init__(out_dim=out_dim, lr=lr, epochs=epochs)

            def build_module(self, input_shape):
                # input_shape: tuple, e.g. (64,) for a feature vector
                return nn.Linear(input_shape[-1], self.out_dim)

            def output_shape(self, input_shape):
                return (self.out_dim,)

    ``forward``/``fit`` reload the serialised state and run the module
    without gradients, so the filter is usable outside a composite block
    too (non-differentiable inference path).
    """

    _kind = "trainable"
    _differentiable = True

    # Training hyperparameters. Composite block uses the *first* filter's values.
    lr: float = 1e-3
    epochs: int = 10
    batch_size: int = 64

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._module: nn.Module | None = None

    # ── Subclass hooks ───────────────────────────────────────────

    def build_module(self, input_shape: tuple[int, ...]) -> nn.Module:
        """Return the ``nn.Module`` that this filter wraps.

        ``input_shape`` excludes the batch dimension (e.g. ``(64,)`` for
        per-sample feature vectors of length 64).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement build_module(input_shape)"
        )

    def output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Return the output shape this filter emits given ``input_shape``.

        Needed so downstream peers in a composite block can size their
        modules without having to run the module once.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement output_shape(input_shape)"
        )

    def compute_loss(
        self, output: torch.Tensor, y: torch.Tensor, aux: dict | None = None,
    ) -> torch.Tensor:
        """Training loss. Default: MSE. Override for BCE/CE/custom."""
        return nn.functional.mse_loss(output, y)

    def make_optimizer(self, modules: list[nn.Module]) -> torch.optim.Optimizer:
        """Optimiser over all trainable params of the composite block.

        Default: single Adam at ``self.lr``. Override for per-filter lrs.
        """
        params = [p for m in modules for p in m.parameters() if p.requires_grad]
        return torch.optim.Adam(params, lr=self.lr)

    # ── Composite training (called by Soma runtime) ──────────────

    def composite_fit(self, peers: dict, x, y):
        """Build each peer's module, chain them, and run an Adam loop."""
        node_ids = list(peers.keys())

        x_t = _to_tensor(x)
        y_t = _to_tensor(y) if y is not None else None

        # 1. Build modules in topological order, propagating shape.
        modules: list[nn.Module] = []
        shape = tuple(x_t.shape[1:])
        for nid in node_ids:
            f = peers[nid]
            module = f.build_module(shape)
            f._module = module
            modules.append(module)
            shape = f.output_shape(shape)
        composite = nn.Sequential(*modules)

        # 2. Optimiser + mini-batch SGD over ``epochs``.
        opt = self.make_optimizer(modules)
        n_samples = x_t.shape[0]
        bs = min(self.batch_size, n_samples)
        for _ in range(self.epochs):
            perm = torch.randperm(n_samples)
            for i in range(0, n_samples, bs):
                idx = perm[i : i + bs]
                opt.zero_grad()
                out = composite(x_t[idx])
                yb = y_t[idx] if y_t is not None else None
                loss = self.compute_loss(out, yb)
                loss.backward()
                opt.step()

        # 3. Final forward for Soma to propagate downstream, plus states.
        with torch.no_grad():
            final_out = composite(x_t).tolist()
        states = {
            nid: {"weights_b64": _serialize_state_dict(m)}
            for nid, m in zip(node_ids, modules)
        }
        return final_out, states

    # ── Standalone (non-composite) paths ─────────────────────────

    def fit(self, x, y=None):
        """Train this filter's module alone.

        Used when the filter isn't inside a composite block — no
        gradient flow from downstream filters. Calls ``composite_fit``
        on a single-node "group" to share the training loop logic.
        """
        _, states = self.composite_fit({type(self).__name__: self}, x, y)
        return next(iter(states.values()))

    def forward(self, x, state):
        x_t = _to_tensor(x)
        if self._module is None:
            self._module = self.build_module(tuple(x_t.shape[1:]))
        if isinstance(state, dict) and "weights_b64" in state:
            self._module.load_state_dict(_deserialize_state_dict(state["weights_b64"]))
        self._module.eval()
        with torch.no_grad():
            return self._module(x_t).tolist()
