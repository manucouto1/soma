"""DifferentiableFilter: default composite_fit handles a tensor chain.

Three filters chained (``a >> b >> c``). Each one subclasses
:class:`DifferentiableFilter` and implements only ``build_module`` and
``output_shape``. The base class's ``composite_fit`` composes them with
``nn.Sequential`` and runs a shared Adam loop — gradients flow from the
loss at the tail all the way back to ``a``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from soma import DifferentiableFilter, Graph


class DenseFilter(DifferentiableFilter):
    """A Linear + optional activation. Pure tensor → tensor."""

    def __init__(
        self,
        out_dim: int,
        activation: str | None = "relu",
        lr: float = 1e-2,
        epochs: int = 200,
        batch_size: int = 16,
    ):
        super().__init__(
            out_dim=out_dim, activation=activation,
            lr=lr, epochs=epochs, batch_size=batch_size,
        )

    def build_module(self, input_shape):
        layers: list[nn.Module] = [nn.Linear(input_shape[-1], self.out_dim)]
        if self.activation == "relu":
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def output_shape(self, input_shape):
        return (*input_shape[:-1], self.out_dim)


def test_three_filter_chain_converges():
    torch.manual_seed(0)
    # Nonlinear target: y = tanh(2*x0 - x1 + 0.5*x0*x1) in [-1, 1]
    import math
    xs = [(i - 32) / 32.0 for i in range(64)]
    x = [[xi, xi + 0.3] for xi in xs]
    y = [[math.tanh(2.0 * p[0] - p[1] + 0.5 * p[0] * p[1])] for p in x]

    a = DenseFilter(out_dim=8,  activation="relu",   epochs=300)
    b = DenseFilter(out_dim=8,  activation="relu")
    c = DenseFilter(out_dim=1,  activation=None)

    g = Graph()
    g.node("a", a)
    g.node("b", b)
    g.node("c", c)
    g.edge("a", "b")
    g.edge("b", "c")

    assert a._module is None and b._module is None and c._module is None

    g.fit(x, y, mode="differentiable")

    # Every peer got a module built.
    assert a._module is not None
    assert b._module is not None
    assert c._module is not None

    # End-to-end residual loss drops well below the input variance.
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    with torch.no_grad():
        pred = c._module(b._module(a._module(x_t)))
        final_loss = nn.functional.mse_loss(pred, y_t).item()
    # Initial loss on random init is O(1); < 0.02 is decisive convergence.
    assert final_loss < 2e-2, f"chain did not converge: loss={final_loss:.4e}"


def test_shape_propagates_via_output_shape():
    """build_module receives the output_shape of the predecessor."""
    received_shapes: list[tuple] = []

    class ShapeRecorder(DenseFilter):
        def build_module(self, input_shape):
            received_shapes.append(input_shape)
            return super().build_module(input_shape)

    a = ShapeRecorder(out_dim=5, epochs=1)
    b = ShapeRecorder(out_dim=3, epochs=1)
    c = ShapeRecorder(out_dim=1, activation=None, epochs=1)

    g = Graph()
    g.node("a", a)
    g.node("b", b)
    g.node("c", c)
    g.edge("a", "b")
    g.edge("b", "c")

    x = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]]
    y = [[0.0]]
    g.fit(x, y, mode="differentiable")

    # a got input_shape (7,)  — raw feature dim
    # b got input_shape (5,)  — a.output_shape((7,))
    # c got input_shape (3,)
    assert received_shapes == [(7,), (5,), (3,)]
