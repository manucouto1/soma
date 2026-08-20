"""A graph's parameters: what an optimizer has to update."""

from __future__ import annotations

from soma_next._stage import learns


def parameters(graph):
    """The parameters of every node in the graph that has any.

    It asks for `.parameters()` and skips whoever lacks it, so a tokenizer does
    not stop being a node for having nothing to train. Without repeats **by
    identity** — two nodes can share a module — and in declaration order.

    And it skips whoever **trains itself**: those weights are updated by an
    optimizer of their own, wherever the node runs, so putting them in this one
    would be two optimizers over one tensor. It is also what keeps `NoGradient`
    meaning what it means — the parameters that legitimately get no gradient
    here are no longer the ones it is looking at.
    """
    seen, all_of_them = set(), []
    for node_id in graph.nodes():
        implementation = graph.implementation(node_id)
        collect = getattr(implementation, "parameters", None)
        if collect is None or learns(implementation):
            continue
        for parameter in collect():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                all_of_them.append(parameter)
    return all_of_them
