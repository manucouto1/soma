"""The topology DSL: node, edge and the queries."""

import pytest

from conftest import Add, Identity


def test_a_freshly_created_graph_is_empty(g):
    assert len(g) == 0
    assert g.nodes() == []
    assert repr(g) == "Graph(0 nodes, 0 edges)"


def test_node_with_an_explicit_id_returns_the_id(g):
    assert g.node("clean", Identity()) == "clean"
    assert "clean" in g
    assert len(g) == 1


def test_node_without_an_id_derives_it_from_the_class(g):
    assert g.node(Identity()) == "identity"


def test_node_without_an_id_breaks_ties_by_suffixing(g):
    assert g.node(Identity()) == "identity"
    assert g.node(Identity()) == "identity_2"
    assert g.node(Identity()) == "identity_3"


def test_a_two_node_pipeline(g):
    g.node("clean", Identity())
    g.node("vectorize", Identity())
    g.edge("clean", "vectorize")

    assert g.edges() == [("clean", "vectorize")]
    assert g.roots() == ["clean"]
    assert g.leaves() == ["vectorize"]
    assert g.topological_sort() == ["clean", "vectorize"]
    assert repr(g) == "Graph(2 nodes, 1 edges)"


def test_the_registered_object_comes_back(g):
    node = Identity()
    g.node("clean", node)
    assert g.implementation("clean") is node
    assert g.implementation("does_not_exist") is None


def test_two_nodes_cannot_share_a_name(g):
    g.node("clean", Identity())
    with pytest.raises(ValueError, match="there is already a node called `clean`"):
        g.node("clean", Add(1))
    assert len(g) == 1


def test_an_edge_to_a_node_that_does_not_exist(g):
    g.node("clean", Identity())
    with pytest.raises(ValueError, match="names no node"):
        g.edge("clean", "ghost")
    assert g.edges() == []


def test_a_cycle_is_rejected_when_added(g):
    for name in ("a", "b", "c"):
        g.node(name, Identity())
    g.edge("a", "b")
    g.edge("b", "c")
    with pytest.raises(ValueError, match="would close a cycle"):
        g.edge("c", "a")
    assert len(g.edges()) == 2


def test_the_same_edge_is_not_added_twice(g):
    g.node("a", Identity())
    g.node("b", Identity())
    g.edge("a", "b")
    with pytest.raises(ValueError, match="already exists"):
        g.edge("a", "b")


def test_querying_a_node_that_does_not_exist(g):
    with pytest.raises(ValueError, match="names no node"):
        g.predecessors("ghost")


def test_node_with_absurd_arguments(g):
    with pytest.raises(ValueError, match="takes \\(object\\)"):
        g.node()


def test_parallel_branches_that_rejoin(g):
    for name in ("input", "left", "right", "join"):
        g.node(name, Identity())
    g.edge("input", "left")
    g.edge("input", "right")
    g.edge("left", "join")
    g.edge("right", "join")

    assert g.roots() == ["input"]
    assert g.leaves() == ["join"]
    assert g.predecessors("join") == ["left", "right"]
    assert g.successors("input") == ["left", "right"]

    order = g.topological_sort()
    assert order[0] == "input"
    assert order[-1] == "join"
