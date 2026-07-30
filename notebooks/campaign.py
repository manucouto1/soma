"""Shared synthetic campaign for notebooks 10–12.

One 1-D sensor stream with three regimes, seen through two complementary
views. numpy and torch only: no downloads, no sklearn, and the whole
thing is reproducible from a seed.

Why this lives in a module rather than in a cell
------------------------------------------------

Three reasons, and each one is load-bearing:

1. ``Graph.load()`` resolves every node through ``importlib``, so a
   filter class that only exists in a kernel has no import path and a
   checkpoint cannot be reopened. Notebook 12 restores a saved graph.
2. The notebook runner gives each notebook a fresh temporary working
   directory, so all three would otherwise re-declare the generator.
3. The pathology switches have to be *identical* across notebooks 11 and
   12, or the derivation deltas between their runs are comparing two
   different things and quietly lying about it.

The data
--------

Three regimes over a 128-sample window:

``flat``   smoothed baseline noise only.
``drift``  baseline plus a slow rising ramp  → lives in the SHAPE view.
``burst``  baseline plus a short high-frequency packet → lives in the
           SPECTRAL view.

The ramp always rises. A sign-symmetric one would put the ``drift``
class centroid back on top of ``flat`` and the shape view would stop
carrying any linear signal at all.

The two views are each deliberately *insufficient*: the spectral view
drops the low bins where a linear drift puts its energy, and the shape
view mean-pools hard enough that a fast oscillation averages away.
Alone each reaches roughly 0.5–0.75 accuracy; together, ~0.95. That gap
is the entire reason the architecture has two branches — and it is what
makes the leakage pathology in notebook 11 cost something real rather
than being a decorative flag.
"""

from __future__ import annotations

import numpy as np

WINDOW = 128
N_SPECTRAL = 16
N_SHAPE = 16
CLASSES = ("flat", "drift", "burst")

#: Channel layout of ``features()``, and the exact thing notebook 11's
#: ``ChannelConfig.groups`` names when it asks for cross-branch CKA.
CHANNELS = {"spectral": range(0, N_SPECTRAL), "shape": range(N_SPECTRAL, 2 * N_SHAPE)}


# ── data ────────────────────────────────────────────────────────────


def make_windows(n: int = 512, window: int = WINDOW, seed: int = 0):
    """``(X, y)`` — ``X`` is ``(n, window)`` float32, ``y`` is 0..2."""
    rng = np.random.default_rng(seed)
    t = np.arange(window) / window
    X = np.empty((n, window))
    y = rng.integers(0, 3, size=n)
    for i in range(n):
        base = 0.45 * rng.standard_normal(window)
        base = np.convolve(base, np.ones(7) / 7, mode="same")  # noise floor
        if y[i] == 1:  # drift
            base = base + rng.uniform(0.40, 0.80) * (t - t.mean())
        elif y[i] == 2:  # burst
            freq = rng.uniform(9.0, 13.0)  # cycles/window → bins 9..13
            centre = rng.integers(window // 4, 3 * window // 4)
            env = np.exp(-0.5 * ((np.arange(window) - centre) / (window / 12)) ** 2)
            base = base + 0.55 * env * np.sin(2 * np.pi * freq * t)
        X[i] = base
    return X.astype(np.float32), y.astype(np.int64)


def spectral_view(X, lo: int = 3, hi: int = 19):
    """``log|rFFT|`` over bins ``[lo, hi)`` → 16 features.

    Bins 0–2 are dropped on purpose: that is where a linear drift puts
    its energy, so this view is close to blind to the drift class.
    """
    win = np.hanning(np.shape(X)[1])
    mag = np.abs(np.fft.rfft(np.asarray(X, dtype=float) * win, axis=1))
    return np.log1p(mag[:, lo:hi])


def shape_view(X, bins: int = N_SHAPE):
    """Mean-pooled time profile → 16 features.

    An oscillation faster than the 8-sample pool averages out, so this
    view is close to blind to the burst class.
    """
    a = np.asarray(X, dtype=float)
    return a.reshape(a.shape[0], bins, -1).mean(axis=2)


def features(X):
    """``[spectral | shape]`` — the layout :data:`CHANNELS` describes."""
    return np.concatenate([spectral_view(X), shape_view(X)], axis=1)


def split(X, y, n_train: int = 384):
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def standardize(reference, F):
    """Standardize ``F`` using ``reference``'s statistics."""
    reference = np.asarray(reference, dtype=float)
    return (np.asarray(F, dtype=float) - reference.mean(0)) / (reference.std(0) + 1e-9)


def accuracy(logits, y) -> float:
    return float((np.asarray(logits).argmax(1) == np.asarray(y)).mean())


# ── model ───────────────────────────────────────────────────────────
#
# Imported lazily so the data half above stays usable without torch.


def _torch():
    import torch
    import torch.nn as nn

    return torch, nn


try:  # pragma: no cover - exercised by the notebooks, not the test suite
    import torch
    import torch.nn as nn
    from soma import DifferentiableFilter
    from soma._composite import _deserialize_state_dict

    class DualViewEncoder(DifferentiableFilter):
        """Two view-specific branches, a fusion point, and a deep trunk.

        Four switches, each injecting one classic pathology. They are
        constructor arguments rather than edits so that reverting one is
        a `NodeReconfigured` derivation and nothing else — which is what
        lets notebook 12 attribute a result to a single change.

        ``leak_wiring``     the shape branch was copy-pasted from the
                            spectral one and never rewired: same weights
                            *and* the same input slice. Both groups at
                            ``mix`` then encode the same thing → LEAKAGE.
        ``dead_bias``       four post-fusion biases sit at −6, below the
                            ReLU hinge, so those units output exactly 0
                            forever → DEAD_CHANNELS.
        ``starve_context``  the ``ctx`` branch enters the sum multiplied
                            by zero: alive in the forward pass, no
                            gradient ever reaches it → IGNORED_CHANNELS
                            and VANISHING.
        ``trunk_gain``      every tanh layer's weights scaled by 0.30, so
                            the trunk contracts → the staircase in
                            ``plot_module_flow``.
        """

        _cache_version = "campaign-dualview-v1"

        def __init__(
            self,
            hidden: int = 32,
            depth: int = 5,
            lr: float = 1e-2,
            leak_wiring: bool = False,
            dead_bias: bool = False,
            starve_context: bool = False,
            trunk_gain: float = 1.0,
        ):
            super().__init__(
                hidden=hidden,
                depth=depth,
                lr=lr,
                leak_wiring=leak_wiring,
                dead_bias=dead_bias,
                starve_context=starve_context,
                trunk_gain=trunk_gain,
            )

        def build_module(self, input_shape):
            h = self.hidden
            branches = nn.ModuleDict(
                {
                    "spectral": nn.Sequential(nn.Linear(N_SPECTRAL, 16), nn.ReLU()),
                    "shape": nn.Sequential(nn.Linear(N_SHAPE, 16), nn.ReLU()),
                }
            )
            if self.leak_wiring:
                with torch.no_grad():
                    branches["shape"][0].weight.copy_(branches["spectral"][0].weight)
                    branches["shape"][0].bias.copy_(branches["spectral"][0].bias)

            post = nn.Sequential(nn.Linear(32, h), nn.ReLU())
            with torch.no_grad():
                if self.dead_bias:
                    post[0].bias[-4:] = -6.0
                else:
                    post[0].bias.fill_(0.05)

            layers = []
            for _ in range(self.depth):
                lin = nn.Linear(h, h)
                with torch.no_grad():
                    lin.weight.mul_(self.trunk_gain)
                layers += [lin, nn.Tanh()]

            return nn.ModuleDict(
                {
                    "branches": branches,
                    "mix": nn.Identity(),  # the observable fusion point
                    "post": post,
                    "trunk": nn.Sequential(*layers),
                    "ctx": nn.Linear(h, h),
                    "out": nn.Linear(h, 8),
                }
            )

        def output_shape(self, input_shape):
            return (8,)

        def forward(self, x, state=None):
            x_t = (
                x
                if isinstance(x, torch.Tensor)
                else torch.as_tensor(np.asarray(x), dtype=torch.float32)
            )
            self.materialize(tuple(x_t.shape[1:]))
            # Overriding forward means taking on the base class's eval
            # contract too: in eval, saved weights arrive through
            # `state` and have to be loaded, or a graph restored from a
            # checkpoint quietly predicts with a fresh random module.
            if not self.training and isinstance(state, dict) and "weights_b64" in state:
                self._module.load_state_dict(
                    _deserialize_state_dict(state["weights_b64"])
                )
            m = self._module
            spec = m["branches"]["spectral"](x_t[:, :N_SPECTRAL])
            # The bug is one character: `:N_SPECTRAL` twice.
            shape_in = x_t[:, :N_SPECTRAL] if self.leak_wiring else x_t[:, N_SPECTRAL:]
            shp = m["branches"]["shape"](shape_in)
            mixed = m["mix"](torch.cat([spec, shp], dim=1))
            deep = m["trunk"](m["post"](mixed))
            gate = 0.0 if self.starve_context else 1.0
            out = m["out"](deep + gate * m["ctx"](deep))
            if self.training:
                return out, {}
            return out.detach().tolist(), {}

    class Head(DifferentiableFilter):
        """8 → hidden → 3. The classifier on top of the encoder."""

        _cache_version = "campaign-head-v1"

        def __init__(self, hidden: int = 16):
            super().__init__(hidden=hidden)

        def build_module(self, input_shape):
            return nn.Sequential(
                nn.Linear(input_shape[-1], self.hidden),
                nn.ReLU(),
                nn.Linear(self.hidden, len(CLASSES)),
            )

        def output_shape(self, input_shape):
            return (len(CLASSES),)

except ImportError:  # torch not installed — the data half still works
    DualViewEncoder = None  # type: ignore[assignment]
    Head = None  # type: ignore[assignment]
