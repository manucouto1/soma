"""Graph-level training orchestration for ``DifferentiableFilter`` pipelines.

This module installs Python methods on the Rust ``Graph`` class so users can
drive a training loop natively without serialising parameters between filters:

    graph.materialize(sample_input)
    graph.train()
    graph.make_optimizer(torch.optim.Adam, lr=1e-3)
    for epoch in range(n_epochs):
        for x, y in batches:
            with graph.context() as ctx:
                graph.zero_grad()
                out, aux = graph.forward(x)
                loss = compute_loss(out, y, aux)
                graph.backward(ctx, loss)
            graph.step(ctx)
    graph.eval()
    out, aux = graph.forward(x)        # delegates to Rust inference path

``context()`` / ``backward(ctx, loss)`` / ``step(ctx)`` / ``zero_grad()`` are
RPC-ready wrappers: locally they reduce to ``loss.backward()``,
``opt.step()``, ``opt.zero_grad()`` with a no-op context. Once filters live
on remote workers, the same call sites swap in ``dist.autograd.context()``,
``dist.autograd.backward(ctx, [loss])``, and ``DistributedOptimizer.step(ctx)``
without changing user code.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterable

from soma._soma import Graph as _RustGraph

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None     # type: ignore[assignment]


# Sentinel for "no distributed autograd context" — used so users can pass
# ``ctx`` from ``with graph.context() as ctx`` uniformly to ``backward``
# and ``step`` regardless of local vs. RPC mode.
_LOCAL_CTX = object()


def _is_diff(filter_obj: Any) -> bool:
    """A filter participates in autograd if it exposes ``_module``."""
    return hasattr(filter_obj, "_module")


def _any_training(graph: _RustGraph) -> bool:
    return any(getattr(f, "training", False) for _, f in graph.filters())


# ── Lifecycle ─────────────────────────────────────────────────


def materialize(self: _RustGraph, sample_input: Any) -> None:
    """Walk topology, build each ``DifferentiableFilter._module`` once.

    ``sample_input`` is a representative batch; its trailing shape (without
    the batch dim) is threaded through ``output_shape`` to size each module.
    Non-differentiable filters are skipped — their output shape, if needed,
    is only known at runtime, so the chain stops propagating shapes there
    and downstream diff filters fall back to lazy materialization on first
    forward.
    """
    if torch is None:
        return
    x = sample_input
    if not isinstance(x, torch.Tensor):
        try:
            x = torch.as_tensor(x)
        except Exception:
            # Heterogeneous inputs (dict, list of texts) — skip eager
            # materialization; lazy path in DifferentiableFilter.forward
            # will build modules on first call.
            return
    shape: tuple[int, ...] | None = tuple(x.shape[1:])
    for _, f in self.filters():
        # Eager materialisation threads one shape forward, so it only
        # describes a chain. A filter that reads several predecessors
        # owns its own construction — it is the only thing that knows how
        # it combines them — and is left to build lazily on first
        # forward, which is the same fallback a non-diff filter gets.
        if getattr(f, "_multi_input", False):
            shape = None
            continue
        if shape is not None and hasattr(f, "materialize") and hasattr(f, "output_shape"):
            f.materialize(shape)
            try:
                shape = tuple(f.output_shape(shape))
            except Exception:
                shape = None
        else:
            # Non-diff filter or one that doesn't propagate static shapes.
            shape = None


def train(self: _RustGraph, mode: bool = True) -> _RustGraph:
    """Set ``training=mode`` on every live filter (and its ``_module``)."""
    for _, f in self.filters():
        f.training = mode
        mod = getattr(f, "_module", None)
        if mod is not None and hasattr(mod, "train"):
            mod.train(mode)
    return self


def eval(self: _RustGraph) -> _RustGraph:
    return train(self, mode=False)


def to(self: _RustGraph, device: Any, *, dtype: Any = None) -> _RustGraph:
    """Move every materialised filter ``_module`` to ``device`` (and ``dtype``).

    Stores the target on ``py_state`` so modules built lazily by a later
    ``forward`` (e.g. inside a warm-up call after ``g.to(device)``) are
    moved transparently — no need to call ``g.to`` again. Returns self
    for chaining (``g.train().to('cuda').make_optimizer(...)``).
    """
    self.py_state["device"] = device
    if dtype is not None:
        self.py_state["dtype"] = dtype
    for _, f in self.filters():
        mod = getattr(f, "_module", None)
        if mod is None or not hasattr(mod, "to"):
            continue
        if dtype is not None:
            mod.to(device, dtype)
        else:
            mod.to(device)
    return self


def parameters(self: _RustGraph) -> Iterable[Any]:
    """Yield ``nn.Parameter`` objects from every diff filter's ``_module``.

    Order follows the topological order of the graph. Filters whose
    ``_module`` is ``None`` (not yet materialized) contribute nothing.
    """
    seen: set[int] = set()
    for _, f in self.filters():
        mod = getattr(f, "_module", None)
        if mod is None or not hasattr(mod, "parameters"):
            continue
        for p in mod.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            yield p


# ── Forward (polymorphic on training state) ──────────────────


def _predecessors(graph: _RustGraph, node_ids: list[str]) -> dict[str, list[str]]:
    """Live predecessors of each node, in edge order.

    The walk used to thread one filter's output into the next and refuse
    anything that was not a chain, because a chain is all it could
    describe. Resolving each node's input from its own predecessors is
    what lets it execute the graph the user actually drew: a fan-out is
    two consumers reading one output, and a fan-in is a node that reads
    two.
    """
    live = set(node_ids)
    preds: dict[str, list[str]] = {}
    for source, target in graph.edges():
        if source in live and target in live:
            preds.setdefault(target, []).append(source)
    return preds


def _input_for(
    node_id: str,
    node: Any,
    preds: list[str],
    outputs: dict[str, Any],
    graph_input: Any,
) -> Any:
    """What this node receives: the graph's input, one output, or several.

    Several arrive as a dict keyed by predecessor node id — the same
    shape the Rust path uses for a fan-in, and independent of the order
    the edges happened to be declared in. A filter has to say it accepts
    that (`_multi_input = True`), because a dict handed to a `forward`
    written for one tensor is a confusing failure deep inside torch
    rather than a clear one here.
    """
    if not preds:
        return graph_input
    if len(preds) == 1:
        return outputs[preds[0]]
    if not getattr(node, "_multi_input", False):
        raise NotImplementedError(
            f"node `{node_id}` has {len(preds)} predecessors ({', '.join(sorted(preds))}) "
            f"but {type(node).__name__} does not declare `_multi_input = True`. "
            "A multi-input filter receives a dict keyed by predecessor node id "
            "and owns how it combines them (concatenate, add, attend); the "
            "framework cannot pick for you. Set `_multi_input = True` and "
            "accept a dict in `forward`."
        )
    return {pid: outputs[pid] for pid in preds}


def differentiable_forward(self: _RustGraph, x: Any):
    """Walk the graph in Python so autograd survives it.

    Called by ``Graph.forward`` when the graph holds torch modules —
    autograd does not survive the ``Value`` boundary, so the Rust path
    cannot be used for these. Each node reads its own predecessors: one
    predecessor hands its output straight over, several arrive as a dict
    keyed by node id (see :func:`_input_for`), and a root gets the
    graph's input. ``DifferentiableFilter`` forwards branch on
    ``self.training`` (autograd live in train, ``no_grad`` in eval);
    ordinary filters get their state from the runtime library so a
    non-trainable node can sit in the same graph.

    The value returned is the last node in topological order — the same
    rule the chain version used, and still deterministic.

    Returns ``(out, aux_by_node)`` while training and bare ``out`` in
    eval, matching what the Rust path produces.

    This used to be installed *over* ``Graph.forward`` at import time,
    which meant two implementations answered to one name and the choice
    between them depended on what had been imported. The dispatch lives
    in ``Graph.forward`` now; this is the branch it names.
    """
    pairs = list(self.filters())
    preds = _predecessors(self, [node_id for node_id, _ in pairs])

    target_device = self.py_state.get("device")
    target_dtype = self.py_state.get("dtype")

    outputs: dict[str, Any] = {}
    out = x
    aux_by_node: dict[str, dict] = {}
    for node_id, f in pairs:
        # `pairs` is topological, so every predecessor has already run.
        out = _input_for(node_id, f, preds.get(node_id, []), outputs, x)
        # Idempotent: ``Module.to(device)`` is a no-op when already on
        # that device. Catches both warm-up materialisation and
        # lazy-built modules created on first forward.
        mod = getattr(f, "_module", None)
        if mod is not None and target_device is not None and hasattr(mod, "to"):
            if target_dtype is not None:
                mod.to(target_device, target_dtype)
            else:
                mod.to(target_device)
        if hasattr(f, "build_module"):
            # DifferentiableFilter. In training, params live on the
            # module — state is irrelevant. In eval, pass state so that
            # a wiped/fresh module can reload weights_b64 from the
            # runtime library (e.g. after Graph.load).
            if getattr(f, "training", False):
                result = f.forward(out)
            else:
                try:
                    state = self.get_node_state(node_id)
                except Exception:
                    state = None
                result = f.forward(out, state)
        else:
            # Legacy filter: feed its stored runtime state.
            try:
                state = self.get_node_state(node_id)
            except Exception:
                state = None
            result = f.forward(out, state if state is not None else {})
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            out, aux = result
            if aux:
                aux_by_node[node_id] = aux
        else:
            out = result
        outputs[node_id] = out
    # Return shape matches the user's mode:
    #   training (any filter has training=True) → (out, aux_by_node) so
    #     callers can fold aux into the loss;
    #   eval (no training, only inference) → just ``out`` to match the
    #     legacy Rust-forward contract that pure-legacy graphs already
    #     produce, so users don't have to special-case the return shape.
    if _any_training(self):
        return out, aux_by_node
    return out


# ── Training-loop primitives (RPC-ready signatures) ──────────


def set_optimizer(self: _RustGraph, optimizer: Any) -> Any:
    """Register an externally-built optimiser on this graph.

    Stored under ``graph.py_state['optimizer']`` (PyGraph has no
    ``__dict__``). Returns the optimiser for chaining.
    """
    self.py_state["optimizer"] = optimizer
    return optimizer


def make_optimizer(self: _RustGraph, optim_cls: Any = None, **kwargs: Any) -> Any:
    """Build and register an optimiser over ``self.parameters()``.

    Default class is ``torch.optim.Adam``. Equivalent under RPC will swap
    in ``DistributedOptimizer`` over ``RRef`` parameters; users keep
    calling the same ``graph.step(ctx)`` / ``graph.zero_grad()``.
    """
    if torch is None:
        raise RuntimeError("torch is required to build an optimiser")
    if optim_cls is None:
        optim_cls = torch.optim.Adam
    params = list(self.parameters())
    if not params:
        raise RuntimeError(
            "graph.make_optimizer(): no parameters found. Did you call "
            "graph.materialize(sample_input) first?"
        )
    return set_optimizer(self, optim_cls(params, **kwargs))


def optimizer(self: _RustGraph) -> Any:
    """Return the registered optimiser, or raise if none has been set."""
    opt = self.py_state.get("optimizer")
    if opt is None:
        raise RuntimeError(
            "No optimiser registered. Call graph.make_optimizer(...) "
            "or graph.set_optimizer(opt) before step/zero_grad."
        )
    return opt


@contextlib.contextmanager
def context(self: _RustGraph):
    """Autograd context for the training step.

    Locally a no-op that yields a sentinel. In the RPC future this becomes
    ``with dist.autograd.context() as ctx`` so backward/step can refer to
    a specific autograd context id.
    """
    yield _LOCAL_CTX


def backward(self: _RustGraph, ctx: Any, loss: Any) -> None:
    """Run backward over ``loss``.

    Locally: ``loss.backward()`` and ignore ``ctx``. Under RPC: switches
    to ``dist.autograd.backward(ctx, [loss])`` (same call site).

    If a :class:`soma.Audit` is active (set by :meth:`Graph.gradient_audit`
    on context entry), its ``_snapshot_after_backward`` hook is invoked
    once gradients have fully accumulated. Reading ``p.grad`` inside a
    backward hook is timing-dependent; doing it here is reliable.
    """
    if loss is None:
        raise ValueError("graph.backward(ctx, loss): loss is None")
    loss.backward()
    audit = self.py_state.get("active_audit")
    if audit is not None:
        audit._snapshot_after_backward()
    # Inside graph.track_run(...): emit a coarse liveness marker so
    # trackers and live subscribers see training progress.
    run = self.py_state.get("active_run")
    if run is not None:
        step = self.py_state.get("train_step", 0)
        run.step_completed(step)
        self.py_state["train_step"] = step + 1


def step(self: _RustGraph, ctx: Any = None) -> None:
    """Take an optimiser step.

    Locally: ``opt.step()`` and ignore ``ctx``. Under RPC:
    ``DistributedOptimizer.step(ctx)`` uses ``ctx`` to gather grads from
    every worker before applying.
    """
    optimizer(self).step()


def zero_grad(self: _RustGraph, set_to_none: bool = True) -> None:
    """Zero the optimiser's gradient buffers."""
    opt = self.py_state.get("optimizer")
    if opt is None:
        # Nothing to zero yet (e.g. user is about to make_optimizer).
        return
    opt.zero_grad(set_to_none=set_to_none)


# ── Freeze (training → inference) ────────────────────────────


def freeze(self: _RustGraph) -> _RustGraph:
    """Snapshot live ``_module`` weights into each filter's runtime state.

    After ``freeze()``, ``graph.eval()`` followed by ``graph.forward(x)``
    delegates to the Rust inference path, which loads the serialised
    ``weights_b64`` blob into each filter's ``_module`` exactly like the
    pre-refactor pickle-based path. This is the bridge between the
    autograd-live training loop and the cached, distributable inference
    pipeline.

    Idempotent: filters without ``_module`` (non-diff or not yet
    materialised) are skipped.
    """
    if torch is None:
        return self
    from soma._composite import _serialize_state_dict
    for node_id, f in self.filters():
        mod = getattr(f, "_module", None)
        if mod is None:
            continue
        self.set_node_state(node_id, {"weights_b64": _serialize_state_dict(mod)})
    self.mark_fitted()
    self.eval()
    return self
