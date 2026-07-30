"""DifferentiableFilter: base class for filters wrapping a trainable ``nn.Module``.

Subclasses implement:
  - ``build_module(input_shape)`` → the ``nn.Module`` this filter wraps.
  - ``output_shape(input_shape)`` → the shape this filter emits.

The module is built once via :meth:`materialize` (or lazily on the first
forward) and is **persistent** for the lifetime of the filter instance.
The graph orchestrator drives training; this class only exposes building
blocks (``materialize``, ``forward``, ``compute_loss``, ``make_optimizer``).

Forward is polymorphic on ``self.training``:

  - ``training=True``  → autograd-live, returns ``(out, aux)`` with grads.
  - ``training=False`` → eval + no_grad, optionally loading ``state``.

Both branches always return ``(out, aux_dict)``; ``aux_dict`` is empty
unless the subclass produces auxiliary signals (gates, routing weights,
auxiliary losses).
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
    """Base class for filters that wrap a persistent trainable ``nn.Module``.

    Subclass contract::

        class MyLayer(DifferentiableFilter):
            def __init__(self, out_dim: int, lr=1e-3):
                super().__init__(out_dim=out_dim, lr=lr)

            def build_module(self, input_shape):
                return nn.Linear(input_shape[-1], self.out_dim)

            def output_shape(self, input_shape):
                return (self.out_dim,)

    The module is materialised once (lazy on first forward, or eager via
    :meth:`materialize`) and lives on the filter instance. ``forward(x,
    state=None)`` keeps a uniform signature and branches on
    ``self.training``:

      - **training=True**: returns ``(out_tensor, aux)`` with autograd
        live; the ``state`` argument is ignored — params live on
        ``self._module``.
      - **training=False**: optionally loads ``state["weights_b64"]``,
        runs ``no_grad``, returns ``(out_list, aux)``.

    Always returns ``(out, aux_dict)``. ``aux_dict`` is empty unless the
    subclass overrides ``forward`` to surface auxiliary signals (gates,
    routing weights, auxiliary losses) that the user combines with the
    main loss in the training loop.

    Driving training from a graph::

        g = Graph.somatize(MyLayer(8) >> MyLayer(2))
        g.materialize(sample_x)              # builds every _module once
        g.train()                            # set training=True everywhere
        g.make_optimizer(torch.optim.Adam, lr=1e-3)
        for x, y in batches:
            with g.context() as ctx:
                g.zero_grad()
                out, aux = g.forward(x)      # autograd-live, threaded
                loss = my_loss(out, y, aux)
                g.backward(ctx, loss)
            g.step(ctx)
        g.freeze()                           # weights → state library
        g.eval(); preds = g.forward(x_test)  # Rust inference path
    """

    _kind = "trainable"
    _differentiable = True

    lr: float = 1e-3

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._module: nn.Module | None = None
        # Default to inference mode. ``Graph.train()`` / ``Graph.eval()``
        # flip this for every live filter; a filter used outside a graph
        # still has a valid value for ``forward`` to branch on.
        self.training: bool = False

    # ── Subclass hooks ───────────────────────────────────────────

    def build_module(self, input_shape: tuple[int, ...]) -> nn.Module:
        """Return the ``nn.Module`` this filter wraps.

        ``input_shape`` excludes the batch dimension (e.g. ``(64,)`` for
        per-sample feature vectors of length 64).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement build_module(input_shape)"
        )

    def output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Return the shape this filter emits given ``input_shape``.

        Needed so downstream peers can size their modules without having
        to run the module first.
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
        """Optimiser over all trainable params. Default: single Adam at ``self.lr``."""
        params = [p for m in modules for p in m.parameters() if p.requires_grad]
        return torch.optim.Adam(params, lr=self.lr)

    # ── Lifecycle ────────────────────────────────────────────────

    def materialize(self, input_shape: tuple[int, ...]) -> nn.Module:
        """Build ``self._module`` once for the given input shape.

        Idempotent: subsequent calls return the existing module without
        rebuilding. Called eagerly by the graph orchestrator before the
        training loop, or lazily by :meth:`forward` on first use.
        """
        if self._module is None:
            self._module = self.build_module(tuple(input_shape))
        return self._module

    # ── Notebook display ─────────────────────────────────────────

    def _repr_html_(self) -> str:
        """Architecture diagram of the built model, inline in the
        notebook: submodules with parameters, chained in definition
        order, parameter counts as sublabels. Before ``materialize``
        there is nothing to draw yet."""
        name = type(self).__name__
        if self._module is None:
            return (
                f"<div style='font-family:system-ui;font-size:13px'>"
                f"<b>{name}</b> <span style='color:#898781'>(sin "
                f"materializar — llama a g.materialize(x) o haz un "
                f"forward para construir el módulo)</span></div>"
            )

        import json as _json
        import re as _re

        from soma import _soma

        children = [
            (path, mod)
            for path, mod in self._module.named_modules()
            if path and any(True for _ in mod.parameters(recurse=True))
            and "." not in path  # direct children only
        ]
        if not children:
            children = [("module", self._module)]

        def _params(n: int) -> str:
            if n >= 1_000_000:
                return f"{n / 1e6:.1f}M θ"
            if n >= 1_000:
                return f"{n / 1e3:.1f}k θ"
            return f"{n} θ"

        nodes, edges, overlay = [], [], {}
        prev = None
        for path, mod in children:
            nid = "m_" + _re.sub(r"[^0-9A-Za-z_]", "_", path)
            nodes.append(
                {
                    "id": nid,
                    "label": f"{path}: {type(mod).__name__}",
                    "kind": {"type": "Filter", "filter_name": type(mod).__name__},
                }
            )
            overlay[nid] = {
                "sublabel": _params(sum(p.numel() for p in mod.parameters()))
            }
            if prev is not None:
                edges.append(
                    {
                        "id": f"e_{len(edges)}",
                        "source": prev,
                        "target": nid,
                        "kind": "Data",
                        "label": None,
                    }
                )
            prev = nid
        svg = _soma.graph_json_to_svg(
            _json.dumps({"nodes": nodes, "edges": edges}),
            overlay={"nodes": overlay},
        )
        total = sum(p.numel() for p in self._module.parameters())
        return (
            f"<div style='font-family:system-ui;font-size:13px'>"
            f"<b>{name}</b> <span style='color:#898781'>"
            f"({_params(total)} en {len(children)} submódulos)</span>"
            f"<div style='margin-top:6px'>{svg}</div></div>"
        )

    # ── Forward ──────────────────────────────────────────────────

    def forward(self, x, state=None):
        """Polymorphic forward.

        Returns ``(out, aux)``:
          - training: ``out`` is a tensor with autograd live; ``state``
            is ignored (the persistent ``_module`` holds the params).
          - eval: ``out`` is a plain list; if ``state`` carries
            ``weights_b64``, it is loaded into ``_module`` first.

        Subclasses that produce auxiliary signals (gates, aux losses)
        should override and return them in ``aux``.
        """
        x_t = _to_tensor(x)
        self.materialize(tuple(x_t.shape[1:]))

        if self.training:
            return self._module(x_t), {}

        if isinstance(state, dict) and "weights_b64" in state:
            self._module.load_state_dict(
                _deserialize_state_dict(state["weights_b64"])
            )
        self._module.eval()
        with torch.no_grad():
            return self._module(x_t).tolist(), {}
