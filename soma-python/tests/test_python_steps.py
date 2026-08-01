"""Steps written in Python, and the dynamic fan-out they unlock.

Until this existed every step was Rust, so `Transition::Spawn` — the map
half of map-reduce, where the width is only known at runtime — was
unreachable from the language everyone actually writes graphs in.

A step is any object with `poll(ctx)`. No base class: the same duck typing
filters get, for the same reason.
"""

import pytest

import soma
from soma.agentic import Await, Done, Fanout, Goto, Llm, Run, Spawn


class Square:
    """A spawned child."""

    _cache_version = "1"

    def poll(self, ctx):
        return Done(ctx.input * ctx.input)


class Fan:
    """Spawns one child per input item — width decided by the data."""

    _cache_version = "1"

    def __init__(self, runs="worker"):
        self.runs = runs

    def poll(self, ctx):
        if ctx.turn == 0:
            return Spawn([Run(self.runs, x, label=f"w{i}")
                          for i, x in enumerate(ctx.input)])
        return Done(sum(r["output"] for r in ctx.results))


def fan_graph(items_step=None):
    g = soma.Graph(cache="memory")
    g.node("fanout", items_step or Fan())
    g.register_step("worker", Square())
    return g


# ── The bridge itself ──


def test_a_python_object_with_poll_is_a_step():
    class Echo:
        _cache_version = "1"

        def poll(self, ctx):
            return Done(ctx.input)

    g = soma.Graph(cache="memory")
    g.node("echo", Echo())
    assert g.forward("hello") == "hello"


def test_step_sees_its_context():
    seen = {}

    class Inspect:
        _cache_version = "1"

        def poll(self, ctx):
            seen.update(node_id=ctx.node_id, turn=ctx.turn,
                        results=len(ctx.results), run_id=bool(ctx.run_id))
            return Done("ok")

    g = soma.Graph(cache="memory")
    g.node("inspect", Inspect())
    g.forward("x")

    assert seen["node_id"] == "inspect"
    assert seen["turn"] == 0
    assert seen["results"] == 0      # nothing has been asked for yet
    assert seen["run_id"] is True


def test_poll_must_return_a_transition():
    class Bad:
        _cache_version = "1"

        def poll(self, ctx):
            return "just a string"

    g = soma.Graph(cache="memory")
    g.node("bad", Bad())
    with pytest.raises(RuntimeError, match="soma.Done"):
        g.forward("x")


def test_a_python_exception_names_the_step():
    class Boom:
        _cache_version = "1"

        def poll(self, ctx):
            raise ZeroDivisionError("nope")

    g = soma.Graph(cache="memory")
    g.node("boom", Boom())
    with pytest.raises(RuntimeError, match="Boom"):
        g.forward("x")


# ── Dynamic fan-out: the point of the exercise ──


@pytest.mark.parametrize("items", [[2, 3, 4], [5], [1, 2, 3, 4, 5, 6]])
def test_spawn_width_comes_from_the_data(items):
    """Same graph, different widths — nothing in the topology says how many."""
    assert fan_graph().forward(items) == sum(x * x for x in items)


def test_spawned_children_are_labelled_apart():
    ids = []

    class Recorder:
        _cache_version = "1"

        def poll(self, ctx):
            ids.append(ctx.node_id)
            return Done(1)

    g = soma.Graph(cache="memory")
    g.node("fanout", Fan())
    g.register_step("worker", Recorder())
    g.forward([1, 2, 3])

    # `<parent>/<label>`, so two children of the same parent never collide.
    assert sorted(ids) == ["fanout/w0", "fanout/w1", "fanout/w2"]


def test_spawning_nothing_is_an_error_not_a_spin():
    class Empty:
        _cache_version = "1"

        def poll(self, ctx):
            return Spawn([])

    g = soma.Graph(cache="memory")
    g.node("empty", Empty())
    with pytest.raises(RuntimeError, match="spawned nothing"):
        g.forward([1])


def test_spawning_an_unregistered_step_says_so():
    g = soma.Graph(cache="memory")
    g.node("fanout", Fan(runs="nobody"))
    g.register_step("worker", Square())
    with pytest.raises(RuntimeError, match="nobody"):
        g.forward([1, 2])


def test_register_step_does_not_add_a_node():
    """A spawn target must not also be a root, or it runs on the graph's own
    input for no reason."""
    g = fan_graph()
    assert "worker" not in g.to_mermaid()
    assert "fanout" in g.to_mermaid()


# ── The rest of the protocol ──


def test_goto_hands_control_on():
    class Handoff:
        _cache_version = "1"

        def poll(self, ctx):
            return Goto("target", carry="passed along")

    class Target:
        _cache_version = "1"

        def poll(self, ctx):
            return Done(f"target got: {ctx.input}")

    g = soma.Graph(cache="memory")
    g.node("start", Handoff())
    g.node("target", Target())
    # A handoff is control, not data, and has to be declared.
    g.handoff("start", "target")
    assert "passed along" in str(g.forward("x"))


def test_goto_to_an_undeclared_target_is_refused():
    """A step jumping somewhere the graph never said it could is an error,
    not a silent hop."""

    class Wanderer:
        _cache_version = "1"

        def poll(self, ctx):
            return Goto("elsewhere", carry="x")

    g = soma.Graph(cache="memory")
    g.node("start", Wanderer())
    g.node("elsewhere", Square())
    g.connect("start", "elsewhere")      # data, not control
    with pytest.raises(RuntimeError, match="handoff"):
        g.forward("x")


def test_await_rejects_an_empty_effect_list():
    class Nothing:
        _cache_version = "1"

        def poll(self, ctx):
            return Await()

    g = soma.Graph(cache="memory")
    g.node("nothing", Nothing())
    with pytest.raises(RuntimeError, match="soma.Done"):
        g.forward("x")


def test_llm_effect_is_well_formed():
    """The helper builds what the bridge parses; the model call itself is
    covered by the live tests."""
    effect = Llm("ollama/qwen2.5", "hello", system="be brief")
    assert effect == {"effect": "llm", "model": "ollama/qwen2.5",
                      "prompt": "hello", "system": "be brief"}


# ── Fanout, and orchestrate on top of it ──


@pytest.mark.parametrize("value,expected", [
    (["a", "b"], ["a", "b"]),
    ({"tasks": ["a"]}, ["a"]),
    ("first\nsecond", ["first", "second"]),
    ("- bullet\n- points", ["bullet", "points"]),
    ("1. numbered\n2. list", ["numbered", "list"]),
    ("  \n\n", []),
])
def test_fanout_reads_a_plan_in_the_shapes_a_planner_writes(value, expected):
    assert Fanout.tasks(value) == expected


def test_fanout_caps_the_width():
    class Counter:
        _cache_version = "1"

        def poll(self, ctx):
            return Done(1)

    g = soma.Graph(cache="memory")
    g.node("fanout", Fanout(runs="worker", max_workers=2))
    g.register_step("worker", Counter())
    # Five tasks, a ceiling of two: a planner emitting hundreds of lines
    # must not open hundreds of conversations.
    assert len(g.forward(["a", "b", "c", "d", "e"])) == 2


def test_fanout_of_an_empty_plan_is_an_empty_answer():
    g = soma.Graph(cache="memory")
    g.node("fanout", Fanout(runs="worker"))
    g.register_step("worker", Square())
    assert g.forward([]) == []


# ── Effects a Python step can ask for ──


def test_a_python_step_can_sleep_and_the_journal_remembers():
    """`Sleep` and `Custom` were unreachable: `parse_effect` knew `llm`
    and `tool` and rejected everything else, so two of the five `Effect`
    variants existed only for Rust."""
    import soma
    from soma.agentic import Await, Custom, Done, Sleep

    class Naps:
        _cache_version = "v1"

        def poll(self, ctx):
            if not ctx.results:
                return Await([Sleep(0.001)])
            return Done("rested")

    g = soma.Graph()
    g.node("nap", Naps())
    assert g.forward("go") == "rested"

    class Asks:
        _cache_version = "v1"

        def poll(self, ctx):
            if not ctx.results:
                return Await([Custom("soma.test.echo", {"n": 1})])
            return Done("asked")

    g2 = soma.Graph()
    g2.node("ask", Asks())
    # No handler claims it, so it comes back as a failure the step sees —
    # which is the contract, and is reached only because the effect parses.
    with pytest.raises(RuntimeError, match="soma.test.echo"):
        g2.forward("go")
