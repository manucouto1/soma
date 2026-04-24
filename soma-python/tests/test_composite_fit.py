"""TDD: gradient flow through a Composite block.

Two Filters wrap ``nn.Linear`` modules. ``graph.fit(x, y, mode="differentiable")``
must trigger ``composite_fit`` on the first filter of the composite block,
which runs a single training loop back-propagating gradients through every
filter in the block.

Post-conditions verified by the test:
  * ``composite_fit`` was invoked (both filters' ``_module`` attribute is set).
  * The chained training loop reduces MSE below a low threshold — i.e.
    the gradient actually flowed from the last module to the first.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from soma import Filter, Graph


class _LinearModule(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class LinearFilter(Filter):
    """Differentiable Filter wrapping one ``nn.Linear``.

    ``composite_fit`` builds the module of every peer, chains them with
    ``nn.Sequential`` and runs a short Adam loop.
    """

    _kind = "trainable"
    _differentiable = True

    def __init__(self, in_dim: int, out_dim: int, lr: float = 1e-2, epochs: int = 80):
        super().__init__(in_dim=in_dim, out_dim=out_dim, lr=lr, epochs=epochs)
        self._module: nn.Module | None = None   # populated by composite_fit

    def fit(self, x, y=None):
        # Standalone fit — never invoked inside a composite block.
        return {}

    def forward(self, x, state):
        return x

    def composite_fit(self, peers: dict, x, y):
        """peers: {node_id: Filter instance}, ordered topologically (incl. self)."""
        node_ids = list(peers.keys())
        modules = []
        for nid in node_ids:
            f = peers[nid]
            m = _LinearModule(f.in_dim, f.out_dim)
            modules.append(m)
            f._module = m     # expose for external inspection
        composite = nn.Sequential(*modules)

        x_t = torch.tensor(x, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        opt = torch.optim.Adam(composite.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = loss_fn(composite(x_t), y_t)
            loss.backward()
            opt.step()

        with torch.no_grad():
            final_out = composite(x_t).tolist()
        # States keyed by node_id — one entry per filter in the block.
        states = {nid: {"fitted": True} for nid in node_ids}
        return final_out, states


def test_gradient_flows_through_composite_block():
    torch.manual_seed(0)
    # Normalised linear target in [-1, 1] so the test is scale-invariant.
    x = [[(i - 32) / 32.0, (i - 31) / 32.0] for i in range(64)]
    y = [[2.0 * xi[0] - xi[1]] for xi in x]

    a = LinearFilter(in_dim=2, out_dim=4, epochs=200)
    b = LinearFilter(in_dim=4, out_dim=1, epochs=200)
    assert a._module is None and b._module is None

    g = Graph()
    g.node("a", a)
    g.node("b", b)
    g.edge("a", "b")

    g.fit(x, y, mode="differentiable")

    # composite_fit must have been invoked on both filters.
    assert a._module is not None, "composite_fit didn't build module for filter a"
    assert b._module is not None, "composite_fit didn't build module for filter b"

    # Gradients flowed end-to-end → chained output fits the linear target.
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    with torch.no_grad():
        pred = b._module(a._module(x_t))
        final_loss = nn.functional.mse_loss(pred, y_t).item()
    # Random init loss is O(1); convergence below 1e-3 requires gradients
    # to have propagated all the way back to filter a.
    assert final_loss < 1e-3, f"expected training loss < 1e-3, got {final_loss:.4e}"
