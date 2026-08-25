"""Branches launched at the same time.

`|` opens branches in the DSL, and since CU9 that is not just topology: a wave's
branches **run at the same time**, each one whole on its own thread. What is
checked here is the three things that can go wrong crossing the boundary: that
the plan has the shape it says, that the threads really exist, and that the GIL
does not spoil it.
"""

import subprocess
import sys
import textwrap
import threading

import pytest

from soma_next import Graph, Node
from conftest import Add, Mean

DEADLINE = 10


# ── Nodes that watch how they get executed ──


class Noter(Node):
    """Notes itself in a shared list: who, when and on what thread."""

    def __init__(self, name, journal, lock):
        self.name = name
        self.journal = journal
        self.lock = lock

    def forward(self, x, ctx):
        with self.lock:
            self.journal.append((self.name, threading.get_ident()))
        return x


class Rendezvous(Node):
    """Does not finish until the other branch has arrived.

    If the branches were executed one after the other, the first would wait for
    a second that has not started yet and the barrier would blow up when the
    deadline ran out. `Barrier.wait()` releases the GIL while it waits, which is
    what makes this able to work with Python nodes.
    """

    def __init__(self, barrier, fails=None):
        self.barrier = barrier
        self.fails = fails

    def forward(self, x, ctx):
        self.barrier.wait()
        if self.fails:
            raise ValueError(self.fails)
        return x


def in_another_process(source):
    """Runs a program that uses waves, with a deadline, in a separate interpreter.

    It has to be another **process** and not another thread: if the engine did
    not release the GIL, the wave's threads would be waiting for it and with them
    the whole interpreter — even a `join(timeout=…)` on the main thread needs the
    GIL to return. A hang like that cannot be caught by anything inside.
    """
    try:
        return subprocess.run(
            [sys.executable, "-c", textwrap.dedent(source)],
            capture_output=True,
            text=True,
            timeout=DEADLINE,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"the program did not return in {DEADLINE}s. The engine did not release "
            "the GIL: the wave's branches are waiting for it and whoever holds it is "
            "blocked inside Rust. `py.allow_threads` is missing around `executor.run` "
            "in `python/src/lib.rs`."
        )


# ── The shape of the plan ──


def test_a_chain_has_no_waves():
    g = Graph.somatize(Add(1).named("a") >> Add(10).named("b"))
    assert "Wave" not in g.plan()


def test_two_branches_come_out_as_a_wave_in_the_plan():
    g = Graph.somatize(
        Add(1).named("s") >> (Add(10).named("l") | Add(100).named("r")) >> Mean().named("j")
    )
    plan = g.plan()
    assert "Wave" in plan
    assert plan.count("Execute") == 4


def test_a_long_branch_is_a_single_branch_of_the_wave():
    # `a >> (b >> b2 | c) >> d`: the wave carries a sequence inside.
    g = Graph.somatize(
        Add(1).named("a")
        >> ((Add(10).named("b") >> Add(20).named("b2")) | Add(100).named("c"))
        >> Mean().named("d")
    )
    plan = g.plan()
    assert "Wave" in plan
    # The long branch's sequence is inside the wave, not outside.
    assert plan.index("Wave") < plan.index('NodeId("b2")')


def test_the_dsl_with_branches_gives_the_same_plan_as_node_and_edge():
    # Decision 6 of CU5, now with waves in the way: the plan comes out of the
    # graph, not the expression, so both doors give the same thing.
    dsl = Graph.somatize(
        Add(1).named("s") >> (Add(10).named("l") | Add(100).named("r")) >> Mean().named("j")
    )

    by_hand = Graph()
    for name, node in [("s", Add(1)), ("l", Add(10)), ("r", Add(100)), ("j", Mean())]:
        by_hand.node(name, node)
    for source, target in [("s", "l"), ("s", "r"), ("l", "j"), ("r", "j")]:
        by_hand.edge(source, target)

    assert dsl.plan() == by_hand.plan()


# ── That the threads really exist ──


def test_the_branches_run_at_the_same_time():
    barrier = threading.Barrier(2, timeout=DEADLINE)
    g = Graph.somatize(Rendezvous(barrier).named("left") | Rendezvous(barrier).named("right"))

    output = g.forward("x")
    assert output == {"left": "x", "right": "x"}


def test_three_branches_too():
    barrier = threading.Barrier(3, timeout=DEADLINE)
    g = Graph.somatize(
        Add(1).named("s")
        >> (
            Rendezvous(barrier).named("x")
            | Rendezvous(barrier).named("y")
            | Rendezvous(barrier).named("z")
        )
    )
    g.forward(0)


def test_a_whole_branch_runs_on_the_same_thread():
    # What decomposing by branch buys: the day a node has a device, torch pins it
    # per thread and the branch does not hop from one to another.
    journal, lock = [], threading.Lock()

    def witness(name):
        return Noter(name, journal, lock).named(name)

    g = Graph.somatize(
        witness("a")
        >> ((witness("b") >> witness("b2") >> witness("b3")) | (witness("c") >> witness("c2")))
        >> witness("d")
    )
    g.forward("x")

    threads = dict(journal)
    assert threads["b"] == threads["b2"] == threads["b3"]
    assert threads["c"] == threads["c2"]
    assert threads["b"] != threads["c"], "the two branches share a thread: they are not concurrent"
    assert threads["a"] == threads["d"], "what is outside the wave runs on the executing thread"


def test_the_real_execution_order_respects_the_edges():
    # The plan states an order; this looks at the one that actually happened,
    # with threads.
    journal, lock = [], threading.Lock()

    def witness(name):
        return Noter(name, journal, lock).named(name)

    edges = [("a", "b"), ("b", "b2"), ("a", "c"), ("b2", "d"), ("c", "d")]
    g = Graph.somatize(
        witness("a") >> ((witness("b") >> witness("b2")) | witness("c")) >> witness("d")
    )
    g.forward("x")

    order = [name for name, _ in journal]
    assert sorted(order) == ["a", "b", "b2", "c", "d"], f"one is spare or missing: {order}"
    for source, target in edges:
        assert order.index(source) < order.index(target), (
            f"{target} executed before {source}: {order}"
        )


# ── That the result depends on none of the above ──


def test_the_diamond_gives_the_same_spread_out_as_in_a_row():
    g = Graph.somatize(
        Add(1).named("s") >> (Add(10).named("l") | Add(100).named("r")) >> Mean().named("j")
    )
    assert g.forward(0) == 56.0


def test_what_each_branch_produces_inside_reaches_the_end():
    g = Graph.somatize(
        Add(1).named("a")
        >> (
            (Add(10).named("b") >> Add(20).named("b2"))
            | (Add(100).named("c") >> Add(200).named("c2"))
        )
        >> Mean().named("d")
    )
    # 0 → 1 → branch b: 11, 31 · branch c: 101, 301 → mean 166
    assert g.forward(0) == 166.0


def test_executing_twice_gives_the_same_thing():
    g = Graph.somatize(
        Add(1).named("s") >> (Add(10).named("l") | Add(100).named("r")) >> Mean().named("j")
    )
    assert [g.forward(0) for _ in range(5)] == [56.0] * 5


# ── Failures ──


def test_if_two_branches_fail_the_first_is_always_the_one_reported():
    # Both genuinely fail at the same time — they agree to meet before breaking
    # — so which one arrives first is a race; the error reported is not.
    barrier = threading.Barrier(2, timeout=DEADLINE)
    g = Graph.somatize(
        Rendezvous(barrier, fails="the left one broke").named("left")
        | Rendezvous(barrier, fails="the right one broke").named("right")
    )

    with pytest.raises(ValueError, match="the left one broke"):
        g.forward("x")


def test_the_error_says_which_branch_it_was():
    class Fail(Node):
        def forward(self, x, ctx):
            raise ValueError("I broke")

    g = Graph.somatize(Add(1).named("healthy") | Fail().named("bad"))
    with pytest.raises(ValueError, match="bad"):
        g.forward(0)


# ── What the DSL cannot write ──


def test_a_graph_that_is_not_series_parallel_executes_even_without_being_spread():
    # The "N": `a→c, a→d, b→d`. It has no series-parallel tree — that is a
    # theorem — so there is no wave; and it cannot be written with `>>` and `|`,
    # it has to be built with node()/edge(). It still executes as always.
    g = Graph()
    for name, node in [("a", Add(1)), ("b", Add(2)), ("c", Add(100)), ("d", Mean())]:
        g.node(name, node)
    for source, target in [("a", "c"), ("a", "d"), ("b", "d")]:
        g.edge(source, target)

    assert "Wave" not in g.plan(), "the N has no tree to recover"
    assert g.forward(0) == {"c": 101.0, "d": 1.5}


# ── The GIL ──


def test_the_engine_releases_the_gil_while_it_runs():
    """The guard on everything above, and the only one that cannot live inside.

    A wave spawns threads that call Python objects' `forward`. If the thread that
    came in through `Graph.forward` kept the GIL, those threads would block on
    asking for it and the whole process would freeze — not an error, a hang. That
    is why this test lives in another process: there the deadline works.
    """
    done = in_another_process("""
        import threading
        from soma_next import Graph, Node

        barrier = threading.Barrier(2, timeout=5)

        class Rendezvous(Node):
            def forward(self, x, ctx):
                barrier.wait()
                return x

        g = Graph.somatize(Rendezvous().named("left") | Rendezvous().named("right"))
        assert g.forward("x") == {"left": "x", "right": "x"}
        print("it returned")
    """)

    assert done.returncode == 0, done.stderr
    assert "it returned" in done.stdout


def test_two_python_nodes_in_the_same_wave_give_the_right_result():
    """The GIL interleaves them and nothing fixes that.

    That both are in flight is proved by `test_the_branches_run_at_the_same_time`,
    with a rendezvous and no driver. What the GIL prevents is the other thing:
    that more than one runs at any instant, i.e. that spreading them out gains
    time.

    What does have to hold is the result: the interpreter handing out turns as it
    likes cannot change what comes out. It is the price said plainly and the
    limit of what a wave buys on this side of the boundary.
    """
    counter, lock = [], threading.Lock()

    class Counts(Node):
        def forward(self, x, ctx):
            for _ in range(2000):
                with lock:
                    counter.append(1)
            return len(counter)

    g = Graph.somatize(Counts().named("one") | Counts().named("other"))
    output = g.forward(0)

    assert len(counter) == 4000, "increments were lost"
    assert set(output) == {"one", "other"}
