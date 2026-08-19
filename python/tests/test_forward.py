"""Executing: the engine is in Rust, this only supplies the implementations."""

import pytest

from conftest import Add, Identity, Mean
from soma_next import Done, Node


class Upper(Node):
    def forward(self, x, ctx):
        return Done(x.upper())


class Fail(Node):
    def forward(self, x, ctx):
        raise RuntimeError("I broke")


# ── The happy path ──


def test_an_empty_graph_returns_its_input(g):
    assert g.forward("intact") == "intact"


def test_a_single_node(g):
    g.node("add", Add(1))
    assert g.forward(41) == 42.0


def test_a_chain_chains_the_outputs(g):
    g.node("a", Add(1))
    g.node("b", Add(10))
    g.node("c", Add(100))
    g.edge("a", "b")
    g.edge("b", "c")
    assert g.forward(0) == 111.0


def test_text_crosses_the_boundary(g):
    g.node("shout", Upper())
    assert g.forward("hello") == "HELLO"


def test_a_list_goes_and_comes_back_the_same(g):
    g.node("id", Identity())
    assert g.forward([1, 2, 3]) == [1.0, 2.0, 3.0]


def test_a_nested_list_too(g):
    g.node("id", Identity())
    assert g.forward([1, ["two", None], 3]) == [1.0, ["two", None], 3.0]


def test_a_dict_goes_and_comes_back_the_same(g):
    g.node("id", Identity())
    assert g.forward({"b": 1, "a": ["two", None]}) == {"b": 1.0, "a": ["two", None]}


def test_without_input_the_node_receives_none(g):
    class Receives(Node):
        def forward(self, x, ctx):
            assert x is None
            return Done("fine")

    g.node("receives", Receives())
    assert g.forward() == "fine"


# ── The failures ──


def test_an_object_without_forward_fails_when_registered(g):
    class NotOne:
        pass

    with pytest.raises(TypeError, match="missing forward"):
        g.node("bad", NotOne())
    assert len(g) == 0


def test_a_nodes_exception_says_which_one_it_was(g):
    g.node("bomb", Fail())
    with pytest.raises(ValueError, match="node `bomb` failed"):
        g.forward(1)


def test_a_type_that_does_not_cross_says_so(g):
    class Returns(Node):
        def forward(self, x, ctx):
            return Done({"I", "do", "not", "cross"})  # a set

    g.node("returns", Returns())
    with pytest.raises(ValueError, match="a `set` does not cross"):
        g.forward(1)


def test_a_bool_is_not_converted_silently(g):
    g.node("add", Add(1))
    with pytest.raises(TypeError, match="a bool does not cross"):
        g.forward(True)


def test_a_key_that_is_not_text_says_so(g):
    g.node("id", Identity())
    with pytest.raises(TypeError, match="keys of a dict"):
        g.forward({1: "one"})


def test_returning_something_that_is_not_a_transition_says_so(g):
    class Confused(Node):
        def forward(self, x, ctx):
            return "I forgot the Done"

    g.node("confused", Confused())
    with pytest.raises(ValueError, match="must return Done"):
        g.forward(1)


# ── Fans, in both directions ──


def test_several_leaves_come_out_as_a_map_keyed_by_name(g):
    g.node("source", Add(1))
    g.node("left", Add(10))
    g.node("right", Add(100))
    g.edge("source", "left")
    g.edge("source", "right")

    assert g.forward(0) == {"left": 11.0, "right": 101.0}


def test_a_node_with_two_inputs_receives_a_map(g):
    g.node("left", Add(10))
    g.node("right", Add(100))
    g.node("join", Mean())
    g.edge("left", "join")
    g.edge("right", "join")

    assert g.forward(0) == 55.0


def test_a_diamond_comes_back_round(g):
    g.node("source", Add(1))
    g.node("left", Add(10))
    g.node("right", Add(100))
    g.node("join", Mean())
    for a, b in (("source", "left"), ("source", "right"), ("left", "join"), ("right", "join")):
        g.edge(a, b)

    assert g.forward(0) == 56.0


def test_the_map_keeps_the_order_the_edges_were_declared_in(g):
    class Keys(Node):
        def forward(self, inputs, ctx):
            return Done(list(inputs.keys()))

    g.node("second", Add(1))
    g.node("first", Add(1))
    g.node("join", Keys())
    g.edge("second", "join")
    g.edge("first", "join")

    assert g.forward(0) == ["second", "first"]


def test_the_plan_can_be_looked_at(g):
    g.node("source", Add(1))
    g.node("left", Add(10))
    g.edge("source", "left")

    plan = g.plan()
    assert "Sequence" in plan
    assert "from" in plan
