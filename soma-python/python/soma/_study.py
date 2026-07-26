"""Graph-level hyperparameter search: aggregate filter search spaces,
apply sampled params to live filters, and build a Study from a graph.

Installed onto the Rust ``Graph`` at import (same pattern as
``_orchestrator``):

- ``graph.search_space()`` — collect every filter's ``search()``
  descriptors, prefixed with the node id (``"encoder.lr"``), realizing
  the aggregation described in ``docs/design/optimization.md``.
- ``graph.apply_params(params)`` — write a sampled configuration back
  onto the live filter instances.
- ``graph.study(name, ...)`` — a ``soma.Study`` over that space.
"""

from __future__ import annotations

from typing import Any

from soma._soma import Graph as _RustGraph
from soma._soma import Study as _Study


def _graph_search_space(self: _RustGraph) -> list[dict]:
    """Aggregate ``_soma_search_space`` from every live filter, with
    dimension names prefixed by the node id (``"<node_id>.<param>"``)."""
    dims: list[dict] = []
    for node_id, f in self.filters():
        for dim in getattr(type(f), "_soma_search_space", []):
            prefixed = dict(dim)
            prefixed["name"] = f"{node_id}.{dim['name']}"
            dims.append(prefixed)
    return dims


def _apply_params(self: _RustGraph, params: dict[str, Any]) -> None:
    """Apply a sampled configuration to the live filter instances.

    Keys are ``"<node_id>.<attr>"`` (as produced by
    :func:`_graph_search_space`); a bare ``"<attr>"`` is accepted when
    exactly one filter declares that attribute. Unknown keys raise.
    """
    by_node = dict(self.filters())
    for key, value in params.items():
        if "." in key:
            node_id, attr = key.split(".", 1)
            f = by_node.get(node_id)
            if f is None:
                raise KeyError(f"apply_params: no node '{node_id}' (from '{key}')")
            setattr(f, attr, value)
            continue

        owners = [
            f
            for f in by_node.values()
            if any(d["name"] == key for d in getattr(type(f), "_soma_search_space", []))
        ]
        if not owners:
            raise KeyError(f"apply_params: no filter declares search param '{key}'")
        if len(owners) > 1:
            raise KeyError(
                f"apply_params: param '{key}' is ambiguous across {len(owners)} filters; "
                f"use the 'node_id.{key}' form"
            )
        setattr(owners[0], key, value)


def _graph_study(self: _RustGraph, name: str, **kwargs: Any) -> _Study:
    """Create a :class:`soma.Study` whose search space is aggregated
    from this graph's filters. Accepts every ``Study(...)`` keyword
    (strategy, n_trials, objective, objectives, direction, pruning,
    seed, tracking, root, tags)."""
    if "search_space" in kwargs:
        raise TypeError(
            "graph.study() builds the search space from the graph's filters; "
            "pass extra dimensions by declaring search() descriptors instead"
        )
    return _Study(name, search_space=self.search_space(), **kwargs)


_RustGraph.search_space = _graph_search_space
_RustGraph.apply_params = _apply_params
_RustGraph.study = _graph_study
