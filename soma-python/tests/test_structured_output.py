"""Asking a model for a shape, and checking it got one.

Two halves that are easy to confuse. `soma.Agent(schema=...)` asks the
*endpoint* to constrain its decoding, and checks the reply — one wrong
answer buys one correction. `Validate` checks a value that is already in the
graph, and returns a verdict instead of raising, so a branch can route on it
without spending another model call to find out what went wrong.
"""

import json

import pytest

import soma
from conftest import MockProvider, Reply
from soma.agentic import Validate, _structural_errors

SCHEMA = {
    "type": "object",
    "required": ["score", "reason"],
    "properties": {
        "score": {"type": "number"},
        "reason": {"type": "string"},
        "tags": {"type": "array"},
    },
}


def graded(score=0.9, reason="fine"):
    return Reply.says(json.dumps({"score": score, "reason": reason}))


# ── The request that goes out ──


def test_an_endpoint_that_can_constrain_decoding_is_asked_to(providers_file):
    with MockProvider([graded()]) as p:
        providers_file(p.base_url, quirks={"supports_json_schema": True})

        g = soma.Graph(cache="memory")
        g.node("extract", soma.Agent(model="mock/any", schema=SCHEMA))
        g.forward("go")

        sent = p.received[0]
        # Constrained decoding cannot produce prose. A prompt can.
        assert sent["response_format"]["type"] == "json_schema"
        assert sent["response_format"]["json_schema"]["schema"] == SCHEMA


def test_an_endpoint_that_cannot_is_asked_in_words(providers_file):
    with MockProvider([graded()]) as p:
        providers_file(p.base_url, quirks={"supports_json_schema": False})

        g = soma.Graph(cache="memory")
        g.node("extract", soma.Agent(model="mock/any", schema=SCHEMA))
        g.forward("go")

        sent = p.received[0]
        assert "response_format" not in sent, "must not send what it cannot honour"
        # Worse than enforcement, but better than dropping the requirement
        # and leaving the caller wondering why nothing validates.
        system = sent["messages"][0]
        assert system["role"] == "system"
        assert "JSON Schema" in system["content"]
        assert "score" in system["content"]


def test_no_schema_means_no_response_format(providers_file):
    with MockProvider([Reply.says("hello")]) as p:
        providers_file(p.base_url)

        g = soma.Graph(cache="memory")
        g.node("chat", soma.Agent(model="mock/any"))
        g.forward("go")

        assert "response_format" not in p.received[0]


# ── The reply that comes back ──


def test_a_reply_in_prose_buys_one_correction(providers_file):
    with MockProvider([Reply.says("I'd say about 0.9, pretty good"), graded()]) as p:
        providers_file(p.base_url)

        g = soma.Graph(cache="memory")
        g.node("extract", soma.Agent(model="mock/any", schema=SCHEMA))
        out = g.forward("go")

        assert p.hits == 2, "one correction, not more"
        assert json.loads(out)["score"] == 0.9

        # The correction says what was wrong, or the model is guessing.
        correction = p.received[1]["messages"][-1]
        assert correction["role"] == "user"
        assert "not JSON" in correction["content"]


def test_a_missing_field_is_named_in_the_correction(providers_file):
    incomplete = Reply.says(json.dumps({"score": 0.9}))
    with MockProvider([incomplete, graded()]) as p:
        providers_file(p.base_url)

        g = soma.Graph(cache="memory")
        g.node("extract", soma.Agent(model="mock/any", schema=SCHEMA))
        g.forward("go")

        correction = p.received[1]["messages"][-1]["content"]
        assert "reason" in correction, correction


def test_a_wrong_field_type_is_named(providers_file):
    wrong = Reply.says(json.dumps({"score": "high", "reason": "fine"}))
    with MockProvider([wrong, graded()]) as p:
        providers_file(p.base_url)

        g = soma.Graph(cache="memory")
        g.node("extract", soma.Agent(model="mock/any", schema=SCHEMA))
        g.forward("go")

        correction = p.received[1]["messages"][-1]["content"]
        assert "score" in correction and "number" in correction, correction


def test_a_model_that_cannot_produce_the_shape_gives_up(providers_file):
    """A repair loop without a ceiling is an open invoice.

    A model that cannot produce the shape will not learn to on the fifteenth
    try, and every attempt is billed.
    """
    with MockProvider([Reply.says("still prose, sorry")]) as p:
        providers_file(p.base_url)

        g = soma.Graph(cache="memory")
        g.node("extract", soma.Agent(model="mock/any", schema=SCHEMA))

        with pytest.raises(RuntimeError) as excinfo:
            g.forward("go")

        assert p.hits == 2, "the first answer plus one correction"
        message = str(excinfo.value)
        assert "required shape" in message, message
        assert "still prose" in message, "the last reply is evidence: " + message


def test_the_repair_budget_is_configurable(providers_file):
    with MockProvider([Reply.says("prose"), Reply.says("prose"), graded()]) as p:
        providers_file(p.base_url)

        g = soma.Graph(cache="memory")
        g.node("extract", soma.Agent(model="mock/any", schema=SCHEMA, max_repairs=2))
        g.forward("go")

        assert p.hits == 3


def test_a_schema_survives_a_rate_limit(providers_file):
    with MockProvider([Reply.error(429), graded()]) as p:
        providers_file(p.base_url)

        g = soma.Graph(cache="memory")
        g.node("extract", soma.Agent(model="mock/any", schema=SCHEMA))

        # A retried call is the same call: it must not eat the repair budget.
        assert json.loads(g.forward("go"))["score"] == 0.9
        assert p.hits == 2


# ── Validate, for a value already in the graph ──


def test_validate_reports_rather_than_raises():
    verdict = Validate(SCHEMA).forward({"score": 0.9, "reason": "fine"}, None)
    assert verdict["ok"] is True
    assert verdict["errors"] == []
    assert verdict["value"]["score"] == 0.9


def test_validate_names_every_problem():
    verdict = Validate(SCHEMA).forward({"score": "high"}, None)
    assert verdict["ok"] is False
    joined = " ".join(verdict["errors"])
    assert "reason" in joined
    assert "score" in joined


def test_validate_parses_a_json_string():
    verdict = Validate(SCHEMA).forward('{"score": 1, "reason": "ok"}', None)
    assert verdict["ok"] is True


def test_validate_says_so_when_it_is_not_json_at_all():
    verdict = Validate(SCHEMA).forward("I think it's fine", None)
    assert verdict["ok"] is False
    assert "not JSON" in verdict["errors"][0]


def test_validate_can_be_told_to_raise():
    with pytest.raises(ValueError, match="validation failed"):
        Validate(SCHEMA, strict=True).forward({}, None)


def test_a_branch_can_route_on_the_verdict():
    """The point of returning a verdict: the invalid case goes somewhere
    that handles it, without a second model call to discover the problem."""

    class Marks(soma.Filter):
        _kind = "stateless"
        _cache_version = "1"

        def __init__(self, mark):
            self.mark = mark
            self.seen = []

        def forward(self, x, state):
            self.seen.append(x)
            return self.mark

    good, bad = Marks("used"), Marks("repaired")
    g = soma.Graph(cache="memory")
    g.branch("check", Validate(SCHEMA), {"ok": good, "invalid": bad})

    assert g.forward({"score": 1, "reason": "ok"}) == "used"
    assert g.forward({"nope": True}) == "repaired"


# ── The fallback, for a machine without `jsonschema` ──


def test_the_structural_fallback_catches_what_matters():
    assert _structural_errors(SCHEMA, {"score": 1, "reason": "ok"}) == []

    missing = _structural_errors(SCHEMA, {"score": 1})
    assert missing == ["missing required field `reason`"]

    wrong_root = _structural_errors(SCHEMA, [1, 2, 3])
    assert len(wrong_root) == 1, "the root type buries everything else"
    assert "expected object" in wrong_root[0]

    wrong_field = _structural_errors(SCHEMA, {"score": "x", "reason": "ok"})
    assert "should be number" in wrong_field[0]


def test_the_structural_fallback_does_not_invent_violations():
    """Permissive on purpose: a violation missed costs an error the consumer
    would have hit anyway; one invented sends a correct answer back to be
    'fixed', and that costs a real model call."""
    # Constructs it does not model — `oneOf`, `pattern`, `minimum` — must
    # not become rejections.
    exotic = {
        "type": "object",
        "properties": {"n": {"type": "integer", "minimum": 10}},
    }
    assert _structural_errors(exotic, {"n": 3}) == []
    assert _structural_errors({"oneOf": [{"type": "string"}]}, 42) == []

    # An integer field accepting 2.0 is JSON Schema's own rule.
    assert _structural_errors(exotic, {"n": 2.0}) == []
    # But a bool is not a number, however Python feels about it.
    assert _structural_errors(exotic, {"n": True}) != []
