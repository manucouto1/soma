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


def graph_search_space(self: _RustGraph) -> list[dict]:
    """Aggregate the searchable dimensions of every node, with names
    prefixed by the node id (``"<node_id>.<param>"``).

    Filters declare theirs as class attributes (``lr = search(...)``);
    agents and judges declare theirs at the call site
    (``Agent(model=search(choices=[...]))``), because an agent's
    hyperparameters are its constructor arguments. Both end up as the same
    kind of dimension, so a study can search a graph that mixes them.
    """
    dims: list[dict] = []
    for node_id, f in self.filters():
        for dim in getattr(type(f), "_soma_search_space", []):
            prefixed = dict(dim)
            prefixed["name"] = f"{node_id}.{dim['name']}"
            dims.append(prefixed)
    for node_id, step in self.steps():
        for dim in step.search_space():
            prefixed = dict(dim)
            prefixed["name"] = f"{node_id}.{dim['name']}"
            dims.append(prefixed)
    for source, target in self.optional_edges():
        # Whether a connection should exist at all is a dimension like any
        # other — the edge optimization that agentic-graph search papers do
        # by hand, over the sampler Soma already has.
        dims.append(
            {
                "type": "categorical",
                "name": f"edge:{source}->{target}",
                "choices": [True, False],
            }
        )
    return dims


def _declared(node: Any) -> list[dict]:
    """The dimensions a node declares, wherever it keeps them."""
    space = getattr(node, "search_space", None)
    if callable(space):
        return space()
    return getattr(type(node), "_soma_search_space", [])


def apply_params(self: _RustGraph, params: dict[str, Any]) -> None:
    """Apply a sampled configuration to the live filter instances.

    Keys are ``"<node_id>.<attr>"`` (as produced by
    :func:`graph_search_space`); a bare ``"<attr>"`` is accepted when
    exactly one node declares that attribute. Unknown keys raise.

    Writing to a live agent is enough: the immutable step behind it is
    rebuilt from the agent before the next run.
    """
    by_node = dict(self.filters())
    by_node.update(dict(self.steps()))
    for key, value in params.items():
        if key.startswith("edge:"):
            source, _, target = key[len("edge:") :].partition("->")
            self.set_edge(source, target, bool(value))
            continue

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
            if any(d["name"] == key for d in _declared(f))
        ]
        if not owners:
            raise KeyError(f"apply_params: no node declares search param '{key}'")
        if len(owners) > 1:
            raise KeyError(
                f"apply_params: param '{key}' is ambiguous across {len(owners)} nodes; "
                f"use the 'node_id.{key}' form"
            )
        setattr(owners[0], key, value)


def graph_study(self: _RustGraph, name: str, **kwargs: Any) -> _Study:
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




# ── Study.run with an optional progress bar ──────────────────


_rust_study_run = _Study.run


def _study_run(self, executor, on_event=None, resume=False, progress=False):
    """Run the study; ``progress=True`` draws a tqdm bar (part of the
    ``somatize[viz]`` extra) fed by live ``StudyProgress`` events, with
    the current best as postfix. Chains with a user ``on_event``.

    Events reach the bar through the lossy broadcast subscriber, so the
    bar is finalized from ``study.n_trials`` once the run returns."""
    if not progress:
        return _rust_study_run(self, executor, on_event=on_event, resume=resume)

    try:
        # tqdm.auto warns when a notebook kernel lacks ipywidgets; fall
        # back to the text bar silently in that case.
        import ipywidgets  # noqa: F401

        from tqdm.auto import tqdm
    except ImportError:
        try:
            from tqdm import tqdm
        except ImportError as e:
            raise RuntimeError(
                "Study.run(progress=True) needs tqdm — "
                "install it with: pip install 'somatize[viz]'"
            ) from e

    state: dict[str, Any] = {"bar": None}

    def _on_event(event: dict) -> None:
        if event.get("event_type") == "StudyProgress":
            bar = state["bar"]
            if bar is None:
                bar = state["bar"] = tqdm(
                    total=event["total"], desc=self.name, unit="trial"
                )
            bar.n = event["completed"]
            bar.set_postfix_str(f"best={event['best_value']:.4g}", refresh=False)
            bar.refresh()
        if on_event is not None:
            on_event(event)

    try:
        return _rust_study_run(self, executor, on_event=_on_event, resume=resume)
    finally:
        import time

        time.sleep(0.2)  # drain the async subscriber's last events
        bar = state["bar"]
        if bar is None:
            bar = tqdm(total=self.n_trials, desc=self.name, unit="trial")
        bar.n = self.n_trials
        bar.refresh()
        bar.close()



# ── Study: the Rust class plus what is written in Python ─────


class Study(_Study):
    """A hyperparameter search.

    `run` wraps the Rust one to add the optional progress bar; the
    figures come from `soma.viz`. Both used to be assigned onto the Rust
    class at import time, so `help(Study)` showed neither.
    """

    run = _study_run

    def _viz(name):
        """Bind a `soma.viz` function as a method, resolved on first use."""

        def method(self, *args, **kwargs):
            from soma import viz

            return getattr(viz, name)(self, *args, **kwargs)

        method.__name__ = name
        return method

    plot_optimization_history = _viz("plot_optimization_history")
    plot_intermediate_values = _viz("plot_intermediate_values")
    plot_parallel_coordinate = _viz("plot_parallel_coordinate")
    plot_param_importances = _viz("plot_param_importances")
    plot_timeline = _viz("plot_timeline")
    plot_pareto_front = _viz("plot_pareto_front")
    trials_dataframe = _viz("trials_dataframe")
    to_html = _viz("study_to_html")

    del _viz
