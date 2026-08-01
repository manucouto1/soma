"""The named agentic patterns, end to end against a mock model.

Each test runs the pattern rather than inspecting its topology: what matters
is that a router hands the arm the *request* and not the label, and that a
refine loop shows the worker its own last attempt.
"""

import json

import pytest

import soma
from soma.agentic import debate, parallel_vote, refine, route

from conftest import MockProvider, says  # noqa: F401


class Echo(soma.Filter):
    """Records everything it was handed, and passes it along."""

    _kind = "stateless"
    _cache_version = "1"

    def __init__(self, label=""):
        self.label = label
        self.seen = []

    def forward(self, x, state):
        self.seen.append(x)
        return x


class Says(soma.Filter):
    """A classifier with a fixed answer — a stand-in for a real router."""

    _kind = "stateless"
    _cache_version = "1"

    def __init__(self, answer):
        self.answer = answer

    def forward(self, x, state):
        return self.answer


class Stops(soma.Filter):
    """A judge that passes on the nth call."""

    _kind = "stateless"
    _cache_version = "1"

    def __init__(self, pass_on):
        self.pass_on = pass_on
        self.calls = 0
        self.seen = []

    def forward(self, x, state):
        self.calls += 1
        self.seen.append(x)
        return {"done": self.calls >= self.pass_on, "value": x, "round": self.calls}


class Speaks(soma.Filter):
    """Adds its turn to the transcript, so every round differs."""

    _kind = "stateless"
    _cache_version = "1"

    def __init__(self, name):
        self.name = name
        self.seen = []

    def forward(self, x, state):
        self.seen.append(x)
        return f"{x}>{self.name}"


class Joins(soma.Filter):
    """Reconciles a fan-in: the input is a dict keyed by node id."""

    _kind = "stateless"
    _cache_version = "1"

    def forward(self, x, state):
        return "|".join(str(x[k]) for k in sorted(x))


# ── route ──


def test_route_sends_the_request_not_the_label():
    billing = Echo("billing")
    tech = Echo("tech")

    g = route(Says("billing"), {"billing": billing, "tech": tech})
    out = g.forward("my invoice is wrong")

    # The arm gets the customer's question. Handing it the string "billing"
    # instead is the context loss that breaks multi-agent handoffs.
    assert billing.seen == ["my invoice is wrong"]
    assert tech.seen == [], "the unselected arm must not run"
    assert out == "my invoice is wrong"


def test_route_falls_back_to_a_default_arm():
    fallback = Echo()
    g = route(Says("something-unexpected"), {"billing": Echo(), "default": fallback})
    g.forward("hello")
    assert fallback.seen == ["hello"]


def test_route_without_a_matching_arm_is_an_error():
    g = route(Says("nope"), {"billing": Echo()})
    with pytest.raises(RuntimeError, match="nope"):
        g.forward("hello")


def test_route_needs_arms():
    with pytest.raises(ValueError, match="arm"):
        route(Says("x"), {})


def test_an_agent_can_be_the_router(providers_file):  # noqa: F811
    with MockProvider([says("tech")]) as provider:
        providers_file(provider.base_url)

        tech = Echo()
        g = route(
            soma.Agent(model="mock/any", system="Answer with one word."),
            {"billing": Echo(), "tech": tech},
            cache="memory",
        )
        g.forward("my laptop will not boot")

        assert tech.seen == ["my laptop will not boot"]


# ── refine ──


def test_refine_stops_when_the_judge_passes():
    worker = Echo()
    judge = Stops(pass_on=2)

    g = refine(worker=worker, judge=judge, max_rounds=5)
    out = g.forward("draft this")

    assert judge.calls == 2, "should stop the round the judge passes"
    assert out["done"] is True
    assert out["round"] == 2


def test_refine_shows_the_worker_the_last_verdict():
    worker = Echo()
    judge = Stops(pass_on=3)

    refine(worker=worker, judge=judge, max_rounds=5).forward("first draft")

    # Round 1 sees the request; every later round sees its own last attempt
    # and the critique of it, which is the only way "refine" means anything.
    assert worker.seen[0] == "first draft"
    assert len(worker.seen) == 3
    assert "first draft" in worker.seen[1]
    assert "Revise it." in worker.seen[1]


def test_refine_hands_the_worker_something_it_can_read():
    """A verdict is a mapping; an agent's input is a conversation.

    Converting one to the other is a node in the loop, not leniency buried
    in the agent — so it is visible, and replaceable.
    """
    from soma.agentic import Revise

    worker = Echo()
    refine(
        worker=worker,
        judge=Stops(pass_on=2),
        max_rounds=3,
        revise=Revise(instruction="Try again, shorter."),
    ).forward("go")

    assert "Try again, shorter." in worker.seen[1]


def test_refine_gives_up_at_max_rounds():
    judge = Stops(pass_on=99)
    out = refine(worker=Echo(), judge=judge, max_rounds=3).forward("x")

    assert judge.calls == 3
    assert out["done"] is False, "an exhausted loop must not report success"


def test_refine_needs_a_round():
    with pytest.raises(ValueError, match="round"):
        refine(worker=Echo(), judge=Stops(1), max_rounds=0)


def test_a_judge_verdict_carries_what_it_judged(providers_file):  # noqa: F811
    verdict = says(json.dumps({"score": 0.95, "reason": "good"}))
    with MockProvider([verdict]) as provider:
        providers_file(provider.base_url)

        g = soma.Graph(cache="memory")
        g.node("critic", soma.Judge(model="mock/any", rubric="Be good"))
        out = g.forward("the artifact")

        # Without this the next refine round would be asked to improve
        # something it can no longer see.
        assert out["value"] == "the artifact"


# ── debate ──


def test_debate_runs_every_agent_each_round():
    a, b = Speaks("a"), Speaks("b")
    out = debate([a, b], rounds=3, cache="memory").forward("motion")

    # Three rounds of two speakers, each answering the one before — which is
    # only true if the loop hands each round what the last one ended with.
    assert out == "motion>a>b>a>b>a>b"
    assert len(a.seen) == 3
    assert len(b.seen) == 3
    assert b.seen[0] == "motion>a"


def test_debate_stops_early_when_the_judge_is_satisfied():
    judge = Stops(pass_on=2)
    debate([Echo(), Echo()], rounds=5, judge=judge).forward("go")
    assert judge.calls == 2


def test_debate_needs_two_agents():
    with pytest.raises(ValueError, match="two agents"):
        debate([Echo()])


# ── parallel_vote ──


def test_parallel_vote_asks_everyone_and_reconciles():
    g = parallel_vote([Says("yes"), Says("no"), Says("yes")], Joins())
    assert g.forward("should we ship?") == "yes|no|yes"


def test_parallel_vote_needs_two_agents():
    with pytest.raises(ValueError, match="two agents"):
        parallel_vote([Says("yes")], Joins())


# ── the structural primitives underneath ──


def test_loop_rejects_a_condition_node_that_does_not_exist():
    g = soma.Graph(cache="memory")
    g.node("w", Echo())
    with pytest.raises(ValueError, match="names no node"):
        g.loop("l", body="w", until="ghost")


def test_until_true_is_refused_as_a_typo():
    g = soma.Graph(cache="memory")
    g.node("w", Echo())
    with pytest.raises(ValueError, match="before it runs"):
        g.loop("l", body="w", until=True)


def test_branch_arms_may_name_existing_nodes():
    handler = Echo()
    g = soma.Graph(cache="memory")
    g.node("handler", handler)
    g.branch("router", Says("a"), {"a": "handler"})
    g.forward("payload")
    assert handler.seen == ["payload"]


def test_a_branch_arm_naming_a_missing_node_is_an_error():
    g = soma.Graph(cache="memory")
    with pytest.raises(ValueError, match="names no node"):
        g.branch("router", Says("a"), {"a": "nowhere"})


def test_an_agentic_graph_serializes_whole():
    """The JSON is the contract an editor or another language reads."""
    g = soma.Graph(cache="memory")
    g.node("draft", Echo())
    g.node("critic", Stops(1))
    g.connect("draft", "critic")
    g.loop("refine", body="draft", until="critic", max_iterations=3)
    g.branch("router", Says("a"), {"a": Echo(), "default": Echo()})

    graph = json.loads(g.graph_json())
    kinds = {n["id"]: n["kind"] for n in graph["nodes"]}

    # A resolved loop condition names its node. This is what a graph with
    # any loop in it could not serialize at all.
    assert kinds["refine"]["until"] == {"type": "WhenSignaled", "node": "critic"}
    assert kinds["refine"]["max_iterations"] == 3
    assert kinds["router"]["arms"] == ["a", "default"]

    # Control edges carry the arm label; data edges do not.
    control = {
        (e["source"], e["target"]): e["label"]
        for e in graph["edges"]
        if e["kind"] == "Control"
    }
    assert control[("router", "a")] == "a"
    assert control[("refine", "draft")] is None
