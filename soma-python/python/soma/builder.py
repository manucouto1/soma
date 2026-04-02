"""Graph materialization — turns Chain/Fork into executable Graphs.

``Graph.somatize()`` walks a lazy Chain/Fork tree and adds nodes + edges
to produce an executable Graph.
"""

from soma._soma import Graph as _RustGraph
from soma.chain import Chain, Fork


def somatize(topology):
    """Materialize a Chain, Fork, or Filter into an executable Graph.

    Usage::

        g = Graph.somatize(Scaler() >> PCA() >> Model())

        g = Graph.somatize(
            Scaler() >> (HeadA() | HeadB()) >> Ensemble()
        )

    Args:
        topology: A Chain, Fork, or single Filter.

    Returns:
        Graph: A materialized, executable graph.
    """
    g = _RustGraph()
    _walk(g, topology)
    return g


def _walk(graph, obj, sources=None):
    """Recursively materialize topology into graph nodes and edges.

    Args:
        graph: The Graph being built.
        obj: Chain, Fork, or Filter to materialize.
        sources: List of node IDs to connect FROM (upstream outputs).

    Returns:
        List of terminal node IDs (downstream outputs of this subtree).
    """
    if sources is None:
        sources = []

    if isinstance(obj, Chain):
        cursor = list(sources)
        for step in obj.steps:
            cursor = _walk(graph, step, cursor)
        return cursor

    elif isinstance(obj, Fork):
        all_terminals = []
        for branch in obj.branches:
            terminals = _walk(graph, branch, sources)
            all_terminals.extend(terminals)
        return all_terminals

    else:
        # Single filter — add node and connect from all sources
        node_id = graph.node(obj)
        for src in sources:
            graph.connect(src, node_id)
        return [node_id]
