"""The two patterns nothing exercised: `react` and `orchestrate`.

`react` is one agent with tools in a loop; `orchestrate` is the
orchestrator-workers shape whose pool is sized by the plan at runtime.
Both return ordinary graphs, so these tests run them the way any user
would — `g.forward(...)` — with the model, where one is needed, a
scripted HTTP mock.
"""

from __future__ import annotations

import threading

import pytest

import soma
from conftest import MockProvider, Reply
from soma.agentic import Done, orchestrate, react


# ── react ──


def test_react_calls_a_tool_and_answers(providers_file):
    """The reason-act loop end to end: the model asks for a tool, the tool
    actually runs, its result goes back, and the model's final prose is
    what the graph returns."""
    calls = []

    @soma.tool
    def lookup(topic: str) -> str:
        """Look a topic up. Call this when the answer needs a fact."""
        calls.append(topic)
        return f"facts about {topic}"

    script = [
        Reply.wants_tool("lookup", {"topic": "soma"}),
        Reply.says("Soma is a graph runtime."),
    ]
    with MockProvider(script) as provider:
        providers_file(provider.base_url)

        g = react(
            "mock/any-model",
            tools=[lookup],
            system="Answer with facts.",
            max_turns=3,
        )
        out = g.forward("what is soma?")

        assert out == "Soma is a graph runtime."
        assert calls == ["soma"], "the Python tool should have run once"
        # The tool was advertised on the first call, and the second call
        # carried its result back to the model.
        assert provider.received[0]["tools"][0]["function"]["name"] == "lookup"
        tool_turn = provider.received[1]["messages"][-1]
        assert tool_turn["role"] == "tool"
        assert tool_turn["content"] == "facts about soma"


# ── orchestrate ──


class Plans:
    """A planner that needs no model: it Done's a fixed list of tasks."""

    _cache_version = "1"

    def __init__(self, tasks):
        self.tasks = list(tasks)

    def poll(self, ctx):
        return Done(self.tasks)


class Works:
    """A worker that records it ran, then echoes its task."""

    _cache_version = "1"

    _seen: list = []
    _lock = threading.Lock()

    def poll(self, ctx):
        with Works._lock:
            Works._seen.append(ctx.input)
        return Done(f"did {ctx.input}")


class Joins:
    """A synthesizer: joins whatever the pool produced."""

    _cache_version = "1"

    def poll(self, ctx):
        return Done(" | ".join(ctx.input))


@pytest.fixture(autouse=True)
def _fresh_worker_ledger():
    Works._seen = []
    yield


def test_orchestrate_sizes_the_pool_from_the_plan():
    """planner → fanout → synthesize, with the fan-out width coming from
    the plan rather than the topology: three tasks means three workers,
    decided while the graph is running."""
    g = orchestrate(
        Plans(["measure", "cut", "sand"]),
        Works(),
        Joins(),
        cache="memory",
    )
    out = g.forward("build a table")

    assert sorted(Works._seen) == ["cut", "measure", "sand"], (
        "one worker per planned task should have run"
    )
    assert out == "did measure | did cut | did sand"


def test_orchestrate_caps_the_width():
    """`max_workers` is a ceiling on how far a plan can fan out: a planner
    emitting five tasks against a cap of two opens two conversations, not
    five — the guard against a planner that writes four hundred lines."""
    g = orchestrate(
        Plans(["a", "b", "c", "d", "e"]),
        Works(),
        Joins(),
        max_workers=2,
        cache="memory",
    )
    out = g.forward("go")

    # Sorted, like the sibling test above and for the same reason: the
    # capped workers run in parallel, so `_seen` records whichever
    # finished first. Asserting arrival order made this flaky — it went
    # red on CI as `['b', 'a']` after passing everywhere for weeks.
    #
    # The claim survives the sort: {"a", "b"} is still two tasks and still
    # the *first* two, which is what a cap of 2 against a five-task plan
    # has to mean. Output order is pinned by the assertion below, and that
    # one is deterministic — the join reads the spawn order, not the
    # completion order.
    assert sorted(Works._seen) == ["a", "b"], "the cap keeps the first two tasks only"
    assert out == "did a | did b"
