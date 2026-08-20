"""Cutting a graph into stages.

Nothing here imports torch, and that is the point being checked as much as any
assertion: how many cuts a graph has is a fact of the **graph** — who runs where
and who trains itself —, and it is decided before anybody talks about a loss.

A cut is derived from `(host, learns)`, so a node that learns is written here as
what it is: any object with a `learn`, asked as a duck. Nothing calls it.
"""

import pytest

from soma_next import Done, Graph, Node
from soma_next._stage import Held, Tap, stages

from conftest import Add, Identity, Mean


class Learner(Add):
    """A node that trains itself. What says so is the `learn`, not a type."""

    def learn(self, signal, ctx):
        return signal


def cut(graph, source, target):
    """Whether that edge is a cut, said the way the module says it."""
    hosts = graph.hosts()

    def side(node_id):
        return hosts.get(node_id), hasattr(graph.implementation(node_id), "learn")

    return side(source) != side(target)


def walk(graph, input_):
    """Runs the whole graph stage by stage and returns everything produced."""
    produced = {}
    for stage in stages(graph):
        stage.fill(produced)
        out = stage.graph.forward(input_ if stage.level == 0 else None)
        produced.update(stage.read(out))
    return produced


def test_an_empty_graph_has_no_stages(g):
    assert stages(g) == []


def test_with_nothing_said_the_whole_graph_is_one_stage(g):
    g.node("a", Add(1))
    g.node("b", Add(2))
    g.edge("a", "b")

    [only] = stages(g)
    assert only.nodes == ("a", "b")
    assert only.holds == {}
    assert only.taps == {"b": "out:b"}


def test_another_host_cuts(g):
    g.node("a", Add(1))
    g.node("b", Add(2))
    g.edge("a", "b")
    g.place_at("b", "w1")

    assert [stage.nodes for stage in stages(g)] == [("a",), ("b",)]


def test_a_node_that_learns_cuts(g):
    g.node("a", Add(1))
    g.node("b", Learner(2))
    g.edge("a", "b")

    assert [stage.nodes for stage in stages(g)] == [("a",), ("b",)]


def test_the_same_host_on_both_ends_does_not_cut(g):
    g.node("a", Add(1))
    g.node("b", Add(2))
    g.edge("a", "b")
    g.place_at("a", "w1")
    g.place_at("b", "w1")

    assert [stage.nodes for stage in stages(g)] == [("a", "b")]


def test_coming_back_here_cuts_again(g):
    g.node("a", Add(1))
    g.node("b", Add(2))
    g.node("c", Add(3))
    g.edge("a", "b")
    g.edge("b", "c")
    g.place_at("b", "w1")

    assert [stage.nodes for stage in stages(g)] == [("a",), ("b",), ("c",)]


def test_two_hosts_side_by_side_are_one_stage(g):
    """A stage is not uniform in host: what cuts is an edge, and these two do not
    touch. They run in one `forward`, and the wave is rebuilt inside it."""
    g.node("a", Add(1))
    g.node("b", Add(2))
    g.place_at("a", "left")
    g.place_at("b", "right")

    assert [stage.nodes for stage in stages(g)] == [("a", "b")]


def test_the_wave_is_rebuilt_inside_the_stage():
    """What keeping the waves means: the two branches still leave at the same
    time, inside one `forward`, without the stage knowing what a wave is."""
    graph = Graph.somatize(Add(1) >> (Add(2).at("a") | Add(3).at("b")))

    _, second = stages(graph)
    plan = second.graph.plan()
    assert "Wave" in plan and 'Host("a")' in plan and 'Host("b")' in plan


def test_a_join_lands_after_the_deepest_branch(g):
    g.node("source", Add(1))
    g.node("far", Add(10))
    g.node("near", Add(100))
    g.node("join", Mean())
    for source, target in [
        ("source", "far"),
        ("source", "near"),
        ("far", "join"),
        ("near", "join"),
    ]:
        g.edge(source, target)
    g.place_at("far", "w1")

    assert [stage.nodes for stage in stages(g)] == [
        ("source", "near"),
        ("far",),
        ("join",),
    ]


# ── What a stage is made of ──


def test_a_hold_is_named_after_the_real_producer(g):
    g.node("a", Add(1))
    g.node("b", Add(2))
    g.edge("a", "b")
    g.place_at("b", "w1")

    _, second = stages(g)
    assert list(second.holds) == ["a"]
    assert second.graph.nodes() == ["a", "b", "out:b"]
    assert isinstance(second.graph.implementation("a"), Held)
    assert isinstance(second.graph.implementation("out:b"), Tap)


def test_the_fan_in_map_is_keyed_the_same_as_in_the_whole_graph(g):
    """What a hold is named for: `Mean` reads a map keyed by producer, and after
    a cut the producers are holds. Rename them and the node reads another map."""
    seen = {}

    class Watching(Node):
        def forward(self, inputs, ctx):
            seen.update(inputs)
            return Done(sum(inputs.values()))

    g.node("left", Learner(1))
    g.node("right", Learner(2))
    g.node("join", Watching())
    g.edge("left", "join")
    g.edge("right", "join")

    walk(g, 0)
    assert sorted(seen) == ["left", "right"]


def test_a_node_that_feeds_inside_and_outside_comes_back(g):
    """Without a tap `run` gives back only the terminals, and `shared` is not one
    of them: what the next stage holds would never arrive."""
    g.node("shared", Add(1))
    g.node("here", Add(10))
    g.node("there", Learner(100))
    g.edge("shared", "here")
    g.edge("shared", "there")

    first, second = stages(g)
    assert first.nodes == ("shared", "here")
    assert sorted(first.taps) == ["here", "shared"]
    assert sorted(second.holds) == ["shared"]

    produced = walk(g, 0)
    assert produced == {"shared": 1.0, "here": 11.0, "there": 101.0}


def test_a_hold_and_a_tap_are_never_placed(g):
    """So they do not show up in `hosts()`, do not go into `_share_out` and are
    in no artifact."""
    g.node("a", Add(1))
    g.node("b", Add(2))
    g.edge("a", "b")
    g.place_at("b", "w1")

    _, second = stages(g)
    assert second.graph.hosts() == {"b": "w1"}


def test_the_stage_keeps_everything_that_was_said_about_its_nodes(g):
    g.node("a", Add(1))
    g.node("b", Add(2))
    g.edge("a", "b")
    g.place("a", "cpu")
    g.freeze("a", "d41d8cd9")
    g.cache("a", "a100-fp16")
    g.written_as("a", "43b0bf6e")
    g.place_at("b", "w1")

    first, _ = stages(g)
    assert first.graph.devices() == {"a": "cpu"}
    assert first.graph.frozen() == {"a": "d41d8cd9"}
    assert first.graph.cached() == {"a": "a100-fp16"}
    assert first.graph.fingerprints() == {"a": "43b0bf6e"}


def test_the_same_objects_travel_and_not_copies(g):
    """The state trained in a stage is the state of the graph, or a step would
    update weights nobody executes."""
    node = Learner(1)
    g.node("a", node)

    [only] = stages(g)
    assert only.graph.implementation("a") is node


def test_a_hold_nobody_filled_says_so(g):
    g.node("a", Add(1))
    g.node("b", Learner(2))
    g.edge("a", "b")

    _, second = stages(g)
    with pytest.raises(ValueError, match="never handed in"):
        second.graph.forward(None)


def test_a_stage_takes_what_it_holds_and_ignores_the_rest(g):
    g.node("a", Add(1))
    g.node("b", Learner(2))
    g.edge("a", "b")

    _, second = stages(g)
    second.fill({"a": 7.0, "somebody_else": 9.0})
    assert second.graph.forward(None) == 9.0


# ── The three properties the backward pass hangs off ──


TOPOLOGIES = {
    "a chain through another host": lambda: Add(1) >> Add(2).at("w1") >> Add(3),
    "a fan with one branch away": (
        lambda: Add(1) >> (Add(2).at("w1") | Add(3)) >> Mean()
    ),
    "a learner in the middle": lambda: Add(1) >> Learner(2) >> Add(3),
    "two hosts side by side": (
        lambda: Add(1) >> (Add(2).at("a") | Add(3).at("b")) >> Mean()
    ),
    "everything here": lambda: Add(1) >> Add(2) >> Identity(),
}


@pytest.mark.parametrize("topology", TOPOLOGIES.values(), ids=list(TOPOLOGIES))
def test_every_cut_edge_crosses_a_boundary_and_no_edge_goes_backwards(topology):
    graph = Graph.somatize(topology())
    where = {
        node_id: stage.level for stage in stages(graph) for node_id in stage.nodes
    }

    for source, target in graph.edges():
        assert where[source] <= where[target], "an edge going backwards"
        if cut(graph, source, target):
            assert where[source] < where[target], "a cut edge inside a stage"


def test_the_stages_run_to_what_the_whole_graph_runs_to(g):
    g.node("source", Add(1))
    g.node("left", Learner(10))
    g.node("right", Add(100))
    g.node("join", Mean())
    for source, target in [
        ("source", "left"),
        ("source", "right"),
        ("left", "join"),
        ("right", "join"),
    ]:
        g.edge(source, target)

    assert walk(g, 0)["join"] == g.forward(0)
