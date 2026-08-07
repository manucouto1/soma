"""Graph checkpoint: save/load full or state-only.

Surfaces installed on ``Graph`` (via ``_install`` at import time):

  - ``graph.state() -> dict[node_id, state]``
  - ``graph.load_state(sd, strict=True)``
  - ``graph.save(path, include_optimizer=False)``
  - ``Graph.load(path, strict=True)`` (classmethod)

Format (zip bundle on disk):

  manifest.json     — soma_version, saved_at, topology, per-node class info
  states/{nid}.safetensors  — tensor state for ``DifferentiableFilter`` modules
  states/{nid}.json         — non-tensor state (any other Python state)
  optimizer.pt              — optional, only if include_optimizer=True

Rationale: safetensors + JSON is safer than ``torch.save`` (no code
execution on load), framework-agnostic-friendly, and supports memmap for
large models. ``torch.save`` is reserved for optimiser state, where the
zero-code-execution rule isn't critical (you only re-load the optimiser
in trusted contexts where you also re-load filter classes).
"""

from __future__ import annotations

import base64
import datetime as _dt
import importlib
import io
import json
import os
import warnings
import zipfile
from typing import TYPE_CHECKING, Any

from soma._soma import Graph as _RustGraph

if TYPE_CHECKING:  # the runtime import would be circular; the annotation is not
    from soma._graph import Graph

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    from safetensors.torch import load as st_load_bytes
    from safetensors.torch import save as st_save_bytes
    _HAS_SAFETENSORS = True
except ImportError:
    _HAS_SAFETENSORS = False


SOMA_CHECKPOINT_FORMAT_VERSION = 1


# ── State-only API ───────────────────────────────────────────


def state(self: Graph) -> dict[str, Any]:
    """Snapshot every node's runtime state into a plain dict.

    Returns ``{node_id: state_value}``. Filters with no stored state are
    omitted. Useful for transfer-learning, debugging, and swapping
    topology around fixed weights.
    """
    out: dict[str, Any] = {}
    for nid in self.filter_ids():
        st = self.get_node_state(nid)
        if st is not None:
            out[nid] = st
    return out


def load_state(self: Graph, sd: dict[str, Any], strict: bool = True) -> None:
    """Apply a state dict produced by :meth:`state`.

    With ``strict=True`` (default), every key in ``sd`` must correspond
    to a node in this graph. With ``strict=False``, missing or unknown
    nodes are reported via :mod:`warnings` and the rest is restored.
    """
    known = set(self.filter_ids())
    unknown = [k for k in sd if k not in known]
    missing = [k for k in known if k not in sd]

    if unknown:
        msg = f"load_state: keys not in this graph: {sorted(unknown)}"
        if strict:
            raise KeyError(msg)
        warnings.warn(msg, stacklevel=2)
    if missing and not strict:
        warnings.warn(
            f"load_state: nodes without state in checkpoint: {sorted(missing)}",
            stacklevel=2,
        )

    for nid, st in sd.items():
        if nid not in known:
            continue
        self.set_node_state(nid, st)
    self.mark_fitted()


# ── Full checkpoint (zip bundle) ─────────────────────────────


def _split_state(state: Any) -> tuple[dict[str, "torch.Tensor"], Any]:
    """Split a runtime state into (tensor_dict, non_tensor_remainder).

    ``DifferentiableFilter`` stores its weights as a base64-encoded
    ``torch.save`` blob under ``state["weights_b64"]``. We pull it apart
    here so weights live in safetensors (memmap-friendly, safe to load)
    and the non-tensor remainder lives in JSON.
    """
    tensors: dict[str, "torch.Tensor"] = {}
    if isinstance(state, dict) and "weights_b64" in state and torch is not None:
        buf = io.BytesIO(base64.b64decode(state["weights_b64"]))
        sd = torch.load(buf, weights_only=True)
        if isinstance(sd, dict):
            tensors = {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}
            non_t = {k: v for k, v in sd.items() if not isinstance(v, torch.Tensor)}
            non_t_state = {k: v for k, v in state.items() if k != "weights_b64"}
            if non_t:
                non_t_state["_extra_non_tensor"] = non_t
            return tensors, non_t_state if non_t_state else None
    return tensors, state


def _merge_state(tensors: dict[str, "torch.Tensor"], non_tensor: Any) -> Any:
    """Inverse of :func:`_split_state`."""
    if not tensors:
        return non_tensor
    extra = None
    if isinstance(non_tensor, dict) and "_extra_non_tensor" in non_tensor:
        extra = non_tensor.pop("_extra_non_tensor")
    sd = dict(tensors)
    if isinstance(extra, dict):
        sd.update(extra)
    buf = io.BytesIO()
    torch.save(sd, buf)
    out: dict[str, Any] = {"weights_b64": base64.b64encode(buf.getvalue()).decode("ascii")}
    if isinstance(non_tensor, dict):
        out.update(non_tensor)
    return out


def _build_manifest(graph: Graph) -> dict:
    nodes = []
    for nid, f in graph.filters():
        cls = type(f)
        nodes.append({
            "id": nid,
            "class_path": cls.class_path() if hasattr(cls, "class_path") else f"{cls.__module__}.{cls.__qualname__}",
            "class_version": getattr(cls, "class_version", 1),
            "kwargs": f.kwargs() if hasattr(f, "kwargs") else {},
        })

    edges = [{"source": s, "target": t} for s, t in graph.edges()]
    return {
        "soma_checkpoint_format": SOMA_CHECKPOINT_FORMAT_VERSION,
        "saved_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "topology": {"nodes": nodes, "edges": edges},
    }


def save(self: Graph, path: str, include_optimizer: bool = False) -> None:
    """Save topology + state to ``path`` (a zip bundle).

    The graph should be ``freeze()``-d first so every diff filter's
    ``_module`` weights are in the runtime state library.
    """
    if not _HAS_SAFETENSORS:
        raise RuntimeError(
            "graph.save requires the 'safetensors' package. "
            "Install with: pip install safetensors"
        )

    manifest = _build_manifest(self)
    states = self.state()

    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        raise FileNotFoundError(f"parent directory does not exist: {parent}")

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        for nid, st in states.items():
            tensors, non_t = _split_state(st)
            if tensors:
                zf.writestr(f"states/{nid}.safetensors", st_save_bytes(tensors))
            if non_t is not None:
                try:
                    zf.writestr(f"states/{nid}.json", json.dumps(non_t))
                except (TypeError, ValueError) as e:
                    raise RuntimeError(
                        f"Filter '{nid}' has non-JSON-serialisable state: {e}. "
                        "Override Filter.kwargs() to drop non-serialisable items, "
                        "or rebuild them inside build_module/forward."
                    ) from e

        if include_optimizer:
            opt = self.py_state.get("optimizer")
            if opt is None:
                warnings.warn("save(include_optimizer=True) called but no optimiser registered", stacklevel=2)
            elif torch is None:
                warnings.warn("save(include_optimizer=True) needs torch", stacklevel=2)
            else:
                buf = io.BytesIO()
                torch.save(opt.state_dict(), buf)
                zf.writestr("optimizer.pt", buf.getvalue())


def _import_class(class_path: str) -> type:
    mod_name, _, cls_name = class_path.rpartition(".")
    if not mod_name:
        raise ImportError(f"invalid class_path: {class_path!r}")
    mod = importlib.import_module(mod_name)
    obj = mod
    for part in cls_name.split("."):
        obj = getattr(obj, part)
    return obj  # type: ignore[return-value]


def load(cls: type, path: str, strict: bool = True) -> Graph:
    """Rebuild a graph from a checkpoint produced by :meth:`save`.

    Reconstructs every filter via ``filter_class(**kwargs)``, rewires the
    topology, then loads state. Edges come from the manifest, so a
    diamond comes back a diamond.

    Wiring nodes in manifest order is the *fallback*, for a checkpoint
    written before the manifest carried edges. This docstring used to
    present that fallback as the only behaviour and point at a TODO in
    :func:`_build_manifest` that is not there — the manifest has carried
    ``graph.edges()`` for a while, and the round-trip test was a two-node
    chain, for which the fallback and the real edges are
    indistinguishable.
    """
    if not _HAS_SAFETENSORS:
        raise RuntimeError("graph.load requires safetensors")

    with zipfile.ZipFile(path, "r") as zf:
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        fmt = manifest.get("soma_checkpoint_format", 0)
        if fmt > SOMA_CHECKPOINT_FORMAT_VERSION:
            raise RuntimeError(
                f"checkpoint format version {fmt} > supported {SOMA_CHECKPOINT_FORMAT_VERSION}"
            )

        graph = cls()
        nodes = manifest["topology"]["nodes"]
        for n in nodes:
            klass = _import_class(n["class_path"])
            saved_v = n.get("class_version", 1)
            current_v = getattr(klass, "class_version", 1)
            if saved_v != current_v:
                msg = (
                    f"class_version mismatch on node {n['id']!r} "
                    f"({klass.__name__}): checkpoint={saved_v}, code={current_v}. "
                    "Best-effort restore — provide a migration if needed."
                )
                if strict:
                    raise RuntimeError(msg)
                warnings.warn(msg, stacklevel=2)
            inst = klass(**n["kwargs"])
            graph.node(n["id"], inst)

        # Edges: explicit if present, else linear chain from node order.
        edges = manifest["topology"].get("edges") or []
        if edges:
            for e in edges:
                graph.edge(e["source"], e["target"])
        else:
            for prev, nxt in zip(nodes, nodes[1:]):
                graph.edge(prev["id"], nxt["id"])

        # Restore state.
        sd: dict[str, Any] = {}
        for n in nodes:
            nid = n["id"]
            tensors: dict[str, "torch.Tensor"] = {}
            non_t: Any = None
            tname = f"states/{nid}.safetensors"
            jname = f"states/{nid}.json"
            try:
                tensors = st_load_bytes(zf.read(tname))
            except KeyError:
                pass
            try:
                non_t = json.loads(zf.read(jname).decode("utf-8"))
            except KeyError:
                pass
            merged = _merge_state(tensors, non_t)
            if merged is not None:
                sd[nid] = merged
        graph.load_state(sd, strict=strict)

        # Optional optimiser restore (lazy: needs an optimiser registered first).
        try:
            opt_bytes = zf.read("optimizer.pt")
            graph.py_state["optimizer_state_dict_pending"] = opt_bytes
        except KeyError:
            pass

    return graph


def restore_optimizer(self: Graph) -> bool:
    """Apply a pending optimiser state dict captured by :meth:`load`.

    Call after ``g.materialize(sample)`` + ``g.make_optimizer(...)`` (or
    ``g.set_optimizer(...)``) to restore step counts, momentum buffers,
    etc. Returns ``True`` if a state was applied, ``False`` if no
    optimiser snapshot was bundled.
    """
    blob = self.py_state.get("optimizer_state_dict_pending")
    if blob is None:
        return False
    if torch is None:
        raise RuntimeError("restore_optimizer needs torch")
    opt = self.py_state.get("optimizer")
    if opt is None:
        raise RuntimeError(
            "restore_optimizer: no optimiser registered. Call "
            "graph.make_optimizer(...) before restore_optimizer()."
        )
    sd = torch.load(io.BytesIO(blob), weights_only=True)
    opt.load_state_dict(sd)
    del self.py_state["optimizer_state_dict_pending"]
    return True
