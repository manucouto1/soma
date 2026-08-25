"""The DSL: the graph as an expression."""

import pytest

from conftest import Add, Mean
from soma_next import Graph, Node


# ── Chaining ──


def test_a_single_node_is_already_a_graph():
    g = Graph.somatize(Add(1))
    assert g.nodes() == ["add"]
    assert g.forward(41) == 42.0


def test_a_chain():
    g = Graph.somatize(Add(1) >> Add(10) >> Add(100))
    assert g.nodes() == ["add", "add_2", "add_3"]
    assert g.forward(0) == 111.0


def test_named_sets_the_id():
    g = Graph.somatize(Add(1).named("first") >> Add(10).named("second"))
    assert g.nodes() == ["first", "second"]
    assert g.edges() == [("first", "second")]


# ── Opening and closing branches ──


def test_a_diamond_reads_at_a_glance():
    g = Graph.somatize(Add(1) >> (Add(10).named("left") | Add(100).named("right")) >> Mean())
    assert g.edges() == [
        ("add", "left"),
        ("add", "right"),
        ("left", "mean"),
        ("right", "mean"),
    ]
    assert g.forward(0) == 56.0


def test_open_branches_come_out_as_a_map():
    g = Graph.somatize(Add(1) >> (Add(10).named("left") | Add(100).named("right")))
    assert g.forward(0) == {"left": 11.0, "right": 101.0}


def test_one_branch_can_be_longer_than_the_other():
    g = Graph.somatize(
        Add(1).named("source")
        >> ((Add(1).named("left") >> Add(1).named("left2")) | Add(1).named("right"))
    )
    assert g.edges() == [
        ("source", "left"),
        ("left", "left2"),
        ("source", "right"),
    ]


def test_three_branches():
    g = Graph.somatize(Add(0) >> (Add(1).named("a") | Add(2).named("b") | Add(3).named("c")))
    assert g.forward(0) == {"a": 1.0, "b": 2.0, "c": 3.0}


# ── The class forces it, and it is the DSL's only door ──


def test_the_class_forces_forward_to_be_implemented():
    class WithoutForward(Node):
        pass

    with pytest.raises(TypeError, match="abstract method 'forward'"):
        WithoutForward()


def test_in_the_dsl_you_have_to_inherit_from_node():
    class Loose:
        def forward(self, x, ctx):
            return x

    with pytest.raises(TypeError, match="has to inherit from soma_next.Node"):
        Graph.somatize(Loose() >> Add(1))


def test_what_cannot_be_a_node_says_so():
    with pytest.raises(TypeError, match="has to inherit from soma_next.Node"):
        Graph.somatize(Add(1) >> "this is not a node")


def test_an_outside_object_still_comes_in_through_the_lower_door(g):
    class Foreign:  # inherits from nothing of ours
        def forward(self, x, ctx):
            return x * 2

    g.node("foreign", Foreign())
    assert g.forward(21) == 42.0


# ── The DSL and the calls build the same thing ──


def test_the_dsl_is_nothing_but_node_and_edge():
    dsl = Graph.somatize(Add(1).named("a") >> Add(10).named("b"))

    by_hand = Graph()
    by_hand.node("a", Add(1))
    by_hand.node("b", Add(10))
    by_hand.edge("a", "b")

    assert dsl.nodes() == by_hand.nodes()
    assert dsl.edges() == by_hand.edges()
    assert dsl.plan() == by_hand.plan()


def test_the_argument_count_is_still_checked_by_rust(g):
    with pytest.raises(ValueError, match="takes \\(object\\)"):
        g.node()
