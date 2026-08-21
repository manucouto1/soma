"""A graph's parameters: what an optimizer has to update."""

from __future__ import annotations


def parameters(graph, without=()):
    """The parameters of every node in the graph that has any.

    It asks for `.parameters()` and skips whoever lacks it, so a tokenizer does
    not stop being a node for having nothing to train. Without repeats **by
    identity** — two nodes can share a module — and in declaration order.

    `without` names the nodes to leave out — the ones somebody else updates::

        trains = {"body": Split(SGD, lr=0.1)}
        Adam(parameters(g, without=trains), lr=1e-3)

    It takes anything you can ask `in` of, so the same dict you hand the
    `Trainer` does. Leaving them in is not a detail and is refused there: a node
    trained where it runs and also held by this optimizer is **updated twice**
    when where it runs is here.
    """
    seen, all_of_them = set(), []
    for node_id in graph.nodes():
        if node_id in without:
            continue
        implementation = graph.implementation(node_id)
        collect = getattr(implementation, "parameters", None)
        if collect is None:
            continue
        for parameter in collect():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                all_of_them.append(parameter)
    return all_of_them
