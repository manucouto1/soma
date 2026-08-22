"""Nodes that ask for something before finishing, and whoever serves it.

They are not a separate type: the only thing that tells them apart is that they
sometimes return `Await` instead of `Done`.
"""

import pytest

from conftest import Add, Ask, Shout
from soma_next import Await, Done, Node


class Insatiable(Node):
    def forward(self, x, ctx):
        return Await(["again"])


class Stingy:
    """A driver that returns fewer results than it was asked for."""

    def perform(self, requests):
        return []


def test_whoever_finishes_on_the_first_turn_needs_no_driver(g):
    g.node("add", Add(1))
    assert g.forward(41) == 42.0


def test_it_asks_for_something_and_the_driver_gives_it(g):
    g.node("question", Ask("hello"))
    assert g.forward(driver=Shout()) == "HELLO"


def test_without_a_driver_the_one_that_asks_says_so(g):
    g.node("question", Ask("hello"))
    with pytest.raises(ValueError, match="no driver"):
        g.forward()


def test_a_driver_without_perform_fails_when_used(g):
    g.node("question", Ask("hello"))
    with pytest.raises(TypeError, match="missing perform"):
        g.forward(driver=object())


def test_a_driver_that_returns_too_few_says_so(g):
    g.node("question", Ask("a", "b"))
    with pytest.raises(ValueError, match="returned 0"):
        g.forward(driver=Stingy())


def test_the_one_that_cannot_stop_spends_its_turns(g):
    g.node("never", Insatiable())
    with pytest.raises(ValueError, match="cannot stop"):
        g.forward(driver=Shout())


def test_the_two_kinds_of_node_chain(g):
    g.node("add", Add(1))
    g.node("question", Ask("x"))
    g.edge("add", "question")
    assert g.forward(41, driver=Shout()) == "X"


def test_the_plan_does_not_tell_apart_who_asks_for_turns(g):
    g.node("add", Add(1))
    g.node("question", Ask("x"))
    g.edge("add", "question")
    assert g.plan().count("Execute") == 2
    assert "Step {" not in g.plan()


def test_a_node_can_evolve_without_changing_type(g):
    """It starts always finishing; a new branch adds a turn to it."""

    class Evolves(Node):
        def forward(self, x, ctx):
            if ctx.turn > 0:
                return Done(ctx.results[0])
            return Await(["negative"]) if x < 0 else Done(x)

    g.node("evolves", Evolves())
    assert g.forward(1) == 1.0                        # it does not even need a driver
    assert g.forward(-1, driver=Shout()) == "NEGATIVE"


def test_the_context_says_the_turn_and_what_the_driver_brought(g):
    seen = []

    class Watches(Node):
        def forward(self, x, ctx):
            seen.append((ctx.turn, list(ctx.results)))
            if ctx.turn < 2:
                return Await([f"t{ctx.turn}"])
            return Done("end")

    g.node("watches", Watches())
    assert g.forward("x", driver=Shout()) == "end"
    assert seen == [(0, []), (1, ["T0"]), (2, ["T1"])]


# ── A node that keeps something, which is the only kind `ctx` cannot serve ──


class Gathers(Node):
    """Keeps what every turn brought, because `ctx.results` does not."""

    def __init__(self, turns=3):
        self.turns, self.seen = turns, []

    def forward(self, x, ctx):
        self.seen.extend(ctx.results)
        if ctx.turn < self.turns:
            return Await([f"t{ctx.turn}"])
        return Done("|".join(self.seen))


def test_a_node_that_wants_more_than_the_last_turn_has_to_keep_it_itself(g):
    # `ctx.results` is what the driver brought for the **previous** turn and not
    # a history: three turns hand a node three separate answers and never all
    # three at once. Whatever wants them together holds them.
    node = Gathers()
    g.node("gathers", node)

    assert g.forward("x", driver=Shout()) == "T0|T1|T2"
    assert node.seen == ["T0", "T1", "T2"]


def test_and_what_it_kept_is_still_there_the_next_time_the_graph_runs(g):
    # The catalog holds **the node**, not a copy per run, so its state outlives
    # a `forward`. That is not incidental: a worker keeping its catalog is what
    # lets an activation stay alive on the far side of a cut, which is what
    # `Split` is built on.
    #
    # The other face of it is a trap, and this is where it is written down: a
    # node that accumulates answers the second run differently from the first,
    # and nothing warns. The engine promises the same **plan**, not that a node
    # without memory is the only kind there is.
    node = Gathers(turns=1)
    g.node("gathers", node)

    assert g.forward("x", driver=Shout()) == "T0"
    assert g.forward("x", driver=Shout()) == "T0|T0", "the second run started fresh"


def test_a_node_that_keeps_nothing_answers_the_same_thing_every_time(g):
    # The contrast, and the reason the one above is a choice rather than the
    # shape of the framework: a node written without state is asked the same
    # question twice and says the same thing.
    g.node("evolves", Ask("x"))

    assert g.forward(None, driver=Shout()) == "X"
    assert g.forward(None, driver=Shout()) == "X"
