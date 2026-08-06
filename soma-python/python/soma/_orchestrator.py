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
                out = graph.forward(x)
                loss = compute_loss(out, y, graph.py_state["last_aux"])
                graph.backward(ctx, loss)
            graph.step(ctx)
    graph.eval()
    out = graph.forward(x)             # delegates to Rust inference path

``context()`` / ``backward(ctx, loss)`` / ``step(ctx)`` / ``zero_grad()`` are
RPC-ready wrappers: locally they reduce to ``loss.backward()``,
``opt.step()``, ``opt.zero_grad()`` with a no-op context. Once filters live
on remote workers, the same call sites swap in ``dist.autograd.context()``,
``dist.autograd.backward(ctx, [loss])``, and ``DistributedOptimizer.step(ctx)``
without changing user code.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, Iterable, Iterator

from soma._soma import Graph as _RustGraph

if TYPE_CHECKING:  # the runtime import would be circular; the annotation is not
    from soma._graph import Graph

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


def _any_training(graph: Graph) -> bool:
    return any(getattr(f, "training", False) for _, f in graph.filters())


# ── Lifecycle ─────────────────────────────────────────────────


def materialize(self: Graph, sample_input: Any) -> None:
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
    sample_shape: tuple[int, ...] = tuple(x.shape[1:])

    # Each node's input shape comes from its PREDECESSORS, not from
    # whatever was built last. Threading one shape down the iteration order
    # only describes a chain, and it is wrong in two ways the moment a
    # graph forks: a second root is handed the previous branch's output
    # instead of the graph's input, and a fan-in node is handed one
    # predecessor's shape as though it were the whole input. Both build a
    # module of the wrong width and fail at the first forward with a
    # matmul error naming shapes the caller never chose.
    preds: dict[str, list[str]] = {}
    for source, target in self.edges():
        preds.setdefault(target, []).append(source)

    # Topological order, so a predecessor's output shape is always known
    # before its consumer asks for it. Relying on `filters()` order would
    # make this work by luck on the graphs where it happens to agree.
    by_id = dict(self.filters())
    ready = [n for n in by_id if not preds.get(n)]
    seen: set[str] = set()
    order: list[str] = []
    while ready:
        node = ready.pop(0)
        if node in seen:
            continue
        seen.add(node)
        order.append(node)
        for candidate, parents in preds.items():
            if candidate not in seen and all(p in seen for p in parents):
                ready.append(candidate)
    # A cycle would leave nodes out; the compiler rejects those, but this
    # runs before it, so anything left over is appended and builds lazily.
    order += [n for n in by_id if n not in seen]

    out_shape: dict[str, tuple[int, ...] | None] = {}
    for node_id in order:
        f = by_id[node_id]
        parents = preds.get(node_id, [])
        if not parents:
            # A root receives the graph's input. Every root does.
            shape: tuple[int, ...] | None = sample_shape
        elif len(parents) == 1:
            shape = out_shape.get(parents[0])
        else:
            # A filter reading several predecessors owns its own
            # construction — it is the only thing that knows how it
            # combines them — so it builds lazily on first forward, which
            # is the same fallback a non-differentiable filter gets.
            shape = None

        if getattr(f, "_multi_input", False):
            shape = None

        if shape is not None and hasattr(f, "materialize") and hasattr(f, "output_shape"):
            f.materialize(shape)
            try:
                out_shape[node_id] = tuple(f.output_shape(shape))
            except Exception:
                out_shape[node_id] = None
        else:
            # Non-diff filter, a fan-in, or one that doesn't propagate
            # static shapes: downstream falls back to lazy materialization.
            out_shape[node_id] = None

    # Whatever is still unbuilt gets built here, by running the graph once.
    # Shapes alone cannot size a fan-in — only the node knows how it
    # combines its inputs — so the only way to find out is to ask it.
    self.py_state["materialize_sample"] = x
    _complete_materialization(self)


def _unbuilt(self: Graph) -> list[str]:
    """Differentiable filters that still have no module.

    `parameters()` skips these silently, which is how an optimiser ends up
    training a subset of the graph and nothing says so.
    """
    return [
        node_id
        for node_id, f in self.filters()
        if getattr(f, "_differentiable", False) and getattr(f, "_module", None) is None
    ]


def _complete_materialization(self: Graph) -> list[str]:
    """Build anything left lazy, by running one forward on the sample.

    A fan-in cannot be sized from shapes — it is the only thing that knows
    how it combines its predecessors — so `materialize` leaves it alone and
    it builds on first forward. That is fine until something needs its
    parameters *before* the first forward, which is exactly what
    `make_optimizer` does: the gate of an ensemble stays at its initial
    weights for the whole run, and nothing raises.

    So: probe. A few rows, no grad, and every filter forced out of training
    mode so dropout and batch-norm statistics are untouched. Returns
    whatever is still unbuilt afterwards.
    """
    if torch is None:
        return []
    pending = _unbuilt(self)
    if not pending:
        return []
    sample = self.py_state.get("materialize_sample")
    if sample is None:
        return pending

    probe = sample[:8] if hasattr(sample, "__getitem__") and len(sample) > 8 else sample
    was_training = [(f, getattr(f, "training", False)) for _, f in self.filters()]
    try:
        for f, _ in was_training:
            f.training = False
        with torch.no_grad():
            self.forward(probe)
    except Exception:
        # A graph that cannot be run on its input alone keeps its lazy
        # nodes. Not fatal here — `make_optimizer` is where it matters,
        # and it names them.
        pass
    finally:
        for f, mode in was_training:
            f.training = mode
    return _unbuilt(self)


def train(self: Graph, mode: bool = True) -> Graph:
    """Set ``training=mode`` on every live filter (and its ``_module``)."""
    for _, f in self.filters():
        f.training = mode
        mod = getattr(f, "_module", None)
        if mod is not None and hasattr(mod, "train"):
            mod.train(mode)
    return self


def eval(self: Graph) -> Graph:
    return train(self, mode=False)


def to(self: Graph, device: Any, *, dtype: Any = None) -> Graph:
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


def parameters(self: Graph) -> Iterable[Any]:
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


def _predecessors(graph: Graph, node_ids: list[str]) -> dict[str, list[str]]:
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


def differentiable_forward(self: Graph, x: Any):
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

    Returns the output, and only the output — the same shape the Rust
    path produces, in both modes. Auxiliary signals land in
    ``graph.py_state["last_aux"]`` as ``{node_id: aux_dict}``, so a loss
    that needs them reads them from the graph rather than from a return
    value whose arity depended on the mode.

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
        if getattr(f, "_differentiable", False):
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
    # One return shape, always the output.
    #
    # This used to return `(out, aux_by_node)` while training and bare
    # `out` in eval, so what `g.forward(x)` gave back depended on a mode
    # set elsewhere — and a training loop written against one shape broke
    # on the same graph after `g.eval()`. The auxiliaries are still
    # produced; they are read from the graph, beside the context and the
    # optimizer the same loop already reaches for.
    self.py_state["last_aux"] = aux_by_node
    return out


# ── Training-loop primitives (RPC-ready signatures) ──────────


def set_optimizer(self: Graph, optimizer: Any) -> Any:
    """Register an externally-built optimiser on this graph.

    Stored under ``graph.py_state['optimizer']`` (PyGraph has no
    ``__dict__``). Returns the optimiser for chaining.
    """
    self.py_state["optimizer"] = optimizer
    return optimizer


def make_optimizer(self: Graph, optim_cls: Any = None, **kwargs: Any) -> Any:
    """Build and register an optimiser over ``self.parameters()``.

    Default class is ``torch.optim.Adam``. Equivalent under RPC will swap
    in ``DistributedOptimizer`` over ``RRef`` parameters; users keep
    calling the same ``graph.step(ctx)`` / ``graph.zero_grad()``.
    """
    if torch is None:
        raise RuntimeError("torch is required to build an optimiser")
    if optim_cls is None:
        optim_cls = torch.optim.Adam

    # An optimiser is a trigger for whatever construction is still
    # pending, not something the caller warms up by hand with a throwaway
    # forward.
    still_lazy = _complete_materialization(self)
    if still_lazy:
        raise RuntimeError(
            "graph.make_optimizer(): "
            f"{', '.join(repr(n) for n in still_lazy)} "
            f"{'has' if len(still_lazy) == 1 else 'have'} no module yet, so "
            "their parameters would be missing from the optimiser and they "
            "would never train. Call graph.materialize(sample_input) with an "
            "input the whole graph can run on — a fan-in can only be sized by "
            "running it."
        )

    params = list(self.parameters())
    if not params:
        raise RuntimeError(
            "graph.make_optimizer(): no parameters found. Did you call "
            "graph.materialize(sample_input) first?"
        )
    return set_optimizer(self, optim_cls(params, **kwargs))


def optimizer(self: Graph) -> Any:
    """Return the registered optimiser, or raise if none has been set."""
    opt = self.py_state.get("optimizer")
    if opt is None:
        raise RuntimeError(
            "No optimiser registered. Call graph.make_optimizer(...) "
            "or graph.set_optimizer(opt) before step/zero_grad."
        )
    return opt


@contextlib.contextmanager
def context(self: Graph) -> Iterator[Any]:
    """Autograd context for the training step.

    Locally a no-op that yields a sentinel. In the RPC future this becomes
    ``with dist.autograd.context() as ctx`` so backward/step can refer to
    a specific autograd context id.
    """
    yield _LOCAL_CTX


def backward(self: Graph, ctx: Any, loss: Any) -> None:
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


def step(self: Graph, ctx: Any = None) -> None:
    """Take an optimiser step.

    Locally: ``opt.step()`` and ignore ``ctx``. Under RPC:
    ``DistributedOptimizer.step(ctx)`` uses ``ctx`` to gather grads from
    every worker before applying.
    """
    optimizer(self).step()


def zero_grad(self: Graph, set_to_none: bool = True) -> None:
    """Zero the optimiser's gradient buffers."""
    opt = self.py_state.get("optimizer")
    if opt is None:
        # Nothing to zero yet (e.g. user is about to make_optimizer).
        return
    opt.zero_grad(set_to_none=set_to_none)


# ── Architecture, as data and as an annotation ───────────────


def _theta(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M θ"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k θ"
    return f"{n} θ"


def architecture(self: Graph) -> dict:
    """What each node *is*, as numbers rather than a picture.

    Per node: the module class it built, total / trainable / frozen
    parameter counts, whether it is differentiable at all, and whether it
    has been built yet. Plus the totals.

    Data first, deliberately: a figure is one consumer of this, a printed
    table is another, and a future GUI is a third. A graph of plain
    Filters reports zero throughout, which is the honest answer rather
    than an empty diagram.
    """
    nodes: dict[str, dict] = {}
    for node_id, f in self.filters():
        module = getattr(f, "_module", None)
        params = list(module.parameters()) if module is not None else []
        trainable = sum(p.numel() for p in params if p.requires_grad)
        frozen = sum(p.numel() for p in params if not p.requires_grad)
        nodes[node_id] = {
            "filter": type(f).__name__,
            "module": type(module).__name__ if module is not None else None,
            "differentiable": bool(getattr(f, "_differentiable", False)),
            "built": module is not None,
            "parameters": trainable + frozen,
            "trainable": trainable,
            "frozen": frozen,
        }
    return {
        "nodes": nodes,
        "total_parameters": sum(n["parameters"] for n in nodes.values()),
        "total_trainable": sum(n["trainable"] for n in nodes.values()),
        "total_frozen": sum(n["frozen"] for n in nodes.values()),
    }


def architecture_overlay(self: Graph, flags: Any = None) -> dict:
    """The same information in the shape the renderers already take.

    Feed it straight to ``graph.to_svg(overlay=…)``,
    ``to_mermaid(overlay=…)`` or ``to_graphviz(overlay=…)`` — every one of
    them already accepts an overlay, and ``NodeOverlay.sublabel`` is a
    free-form label line, so no renderer needs to learn about parameters.

    ``flags`` merges health in: pass an ``Audit.report()``, a
    ``RunView.overlay()``, or a plain ``{node_id: [flag, …]}``. A node that
    is differentiable but not built says so, because "0 θ" and "not built
    yet" are very different things to see on a diagram.
    """
    per_node = _flag_map(flags)
    arch = architecture(self)
    out: dict[str, dict] = {}
    for node_id, info in arch["nodes"].items():
        parts: list[str] = []
        if info["differentiable"] and not info["built"]:
            parts.append("not built")
        elif info["trainable"]:
            parts.append(_theta(info["trainable"]))
        if info["frozen"]:
            parts.append(f"{_theta(info['frozen'])} frozen")
        entry: dict[str, Any] = {}
        if parts:
            entry["sublabel"] = " · ".join(parts)
        node_flags = per_node.get(node_id)
        if node_flags:
            entry["flags"] = list(node_flags)
        if entry:
            out[node_id] = entry
    return {"nodes": out}


def _flag_map(flags: Any) -> dict[str, list[str]]:
    """Accept the three shapes a caller is likely to already be holding."""
    if flags is None:
        return {}
    # An Audit.report(): filters carry `filter_id` and `flags`, and the
    # ids of submodules are "<node>/<path>" — roll those up to the node.
    if hasattr(flags, "filters"):
        rolled: dict[str, list[str]] = {}
        for entry in flags.filters:
            if not entry.flags:
                continue
            node = entry.filter_id.split("/")[0]
            for flag in entry.flags:
                name = flag.split("(")[0]
                if name not in rolled.setdefault(node, []):
                    rolled[node].append(name)
        return rolled
    if isinstance(flags, dict):
        # A RunView.overlay(): {"nodes": {id: {"flags": [...]}}}
        inner = flags.get("nodes") or {} if "nodes" in flags else flags
        return {
            node_id: list(value.get("flags", []) if isinstance(value, dict) else value)
            for node_id, value in inner.items()
        }
    raise TypeError(f"cannot read health flags from {type(flags).__name__}")


# ── Freeze (training → inference) ────────────────────────────


def freeze(self: Graph) -> Graph:
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
