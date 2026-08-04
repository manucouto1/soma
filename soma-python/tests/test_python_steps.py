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
    # The message must name a path that actually imports — it used to say
    # `soma.Done`, which does not exist at the top level.
    with pytest.raises(RuntimeError, match=r"soma\.agentic\.Done"):
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
    with pytest.raises(RuntimeError, match=r"soma\.agentic\.Done"):
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


# ── Fitting a graph that contains a step ──


class Centre:
    """A trainable filter: learns a mean, subtracts it."""

    _cache_version = "1"
    _kind = "trainable"

    def fit(self, x, y=None):
        return {"mean": sum(x) / len(x)}

    def forward(self, x, state):
        return [v - state["mean"] for v in x]


class Total:
    """A step that reduces whatever reaches it."""

    _cache_version = "1"

    def poll(self, ctx):
        return Done(sum(ctx.input))


def test_a_graph_mixing_a_filter_and_a_step_can_be_fitted():
    """Fit used to be a separate, filter-only walk.

    It resolved every node id against the filter half of the catalog, so a
    step id came back as `NodeNotFound` and the whole fit failed — a graph
    you could `run` was a graph you could not `fit`. Fit and forward go
    through the same execution site now, so a step is simply a node.
    """
    g = soma.Graph(cache="memory")
    g.node("centre", Centre())
    g.node("total", Total())
    g.edge("centre", "total")

    g.fit([1.0, 2.0, 3.0])

    # The filter learned, and the step ran on what it produced:
    # centre([1,2,3]) = [-1,0,1], summing to 0.
    assert g.get_node_state("centre") == {"mean": 2.0}
    assert g.forward([1.0, 2.0, 3.0]) == 0.0


# ── What JSON may hold, and what keeps its pickle ────────────


class Echo:
    """Returns whatever reached it, so a value's crossing is observable."""

    _cache_version = "1"

    def poll(self, ctx):
        return Done(ctx.input)


def _round_trip(value):
    """Send `value` through a step and read back what arrived."""
    g = soma.Graph(cache="memory")
    g.node("echo", Echo())
    return g.forward(value)


def test_a_plain_dict_crosses_as_json():
    """A dict JSON can hold becomes JSON, so a loop, a branch, a report
    and a worker in another language can all read it."""
    assert _round_trip({"a": 1, "b": [1, 2], "c": "x"}) == {
        "a": 1,
        "b": [1, 2],
        "c": "x",
    }


def test_nested_structures_survive():
    value = {"outer": {"inner": [1, {"deep": True}, None]}}
    assert _round_trip(value) == value


def test_a_tuple_does_not_pretend_to_be_a_list():
    """`json.dumps` would turn it into a list, so it is not JSON-holdable
    and keeps its pickle instead of arriving changed."""
    out = _round_trip({"pair": (1, 2)})
    assert out == {"pair": (1, 2)}, "a tuple must come back a tuple"


def test_an_integer_keyed_dict_keeps_its_keys():
    """`json.dumps` would stringify the key to `"1"`."""
    out = _round_trip({1: "one", 2: "two"})
    assert out == {1: "one", 2: "two"}


def test_a_non_finite_float_is_not_flattened_to_null():
    """`serde_json` writes a non-finite float as `null`.

    That is the same silent flattening that once gave two different
    tensors one cache key, so such a dict must not take the JSON path.
    """
    out = _round_trip({"x": float("nan")})
    assert isinstance(out["x"], float)
    assert out["x"] != out["x"], "NaN must come back NaN, not null"

    inf = _round_trip({"x": float("inf")})
    assert inf["x"] == float("inf")


def test_a_bool_is_not_an_int():
    """In Python `bool` is an `int`, so order of checks is load-bearing."""
    out = _round_trip({"flag": True, "count": 1})
    assert out["flag"] is True
    assert out["count"] == 1


# ── Declared schemas ─────────────────────────────────────────────


def test_compile_reports_step_schema_mismatch():
    """The compiler's edge check is reachable from Python.

    A tensor feeding a node that expects a conversation has no possible
    reading — this used to be silently unchecked because Python nodes
    carried no schemas at all, so ``SomaSchemaMismatch`` could never fire
    for a Python-built graph.
    """

    class Numbers(soma.Filter):
        _cache_version = "1"
        _kind = "stateless"
        _output_schema = {"dtype": "Float64", "shape": None}

        def forward(self, x, state):
            return [1.0, 2.0]

    class Chat:
        _cache_version = "1"
        _input_schema = "messages"

        def poll(self, ctx):
            return Done("never reached")

    g = soma.Graph(cache="memory")
    g.node("numbers", Numbers())
    g.node("chat", Chat())
    g.connect("numbers", "chat")
    with pytest.raises(Exception, match="no conversion"):
        g.compile("no_cache")


def test_filter_schema_attrs_flow():
    """The same declaration works on a filter→filter edge."""

    class Text(soma.Filter):
        _cache_version = "1"
        _kind = "stateless"
        _output_schema = "text"

        def forward(self, x, state):
            return "words"

    class WantsTensor(soma.Filter):
        _cache_version = "1"
        _kind = "stateless"
        _input_schema = {"dtype": "Float64", "shape": None}

        def forward(self, x, state):
            return x

    g = soma.Graph(cache="memory")
    g.node("text", Text())
    g.node("tensor", WantsTensor())
    g.connect("text", "tensor")
    with pytest.raises(Exception, match="no conversion"):
        g.compile("no_cache")


def test_a_schema_typo_fails_at_registration():
    """An invalid shorthand is an error when the node is added, not a
    silently unchecked edge."""

    class Typo(soma.Filter):
        _cache_version = "1"
        _input_schema = "mesages"

        def forward(self, x, state):
            return x

    g = soma.Graph(cache="memory")
    with pytest.raises(Exception, match="mesages"):
        g.node("typo", Typo())
