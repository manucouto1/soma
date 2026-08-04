"""What an agentic run does when the endpoint misbehaves.

Every other agentic test scripts a well-behaved model. These script the ways
a real one fails: rate limits, gateways, truncation, and replies that are not
what they claim to be. That gap is where production lives — a single 429 used
to end a whole run, because nothing retried and the step reads a failed model
call as final.

All of it against the mock on localhost. No key, no network, no flakiness.
"""

import json
import time

import pytest

import soma
from conftest import MockProvider, Reply


def one_agent(cache="memory"):
    g = soma.Graph(cache=cache)
    g.node("assistant", soma.Agent(model="mock/any"))
    return g


# ── Rate limits and outages ──


def test_a_rate_limit_is_ridden_out(providers_file):
    with MockProvider([Reply.error(429), Reply.error(429), Reply.says("done")]) as p:
        providers_file(p.base_url)

        assert one_agent().forward("hi") == "done"
        assert p.hits == 3, "should have knocked three times"


def test_a_gateway_outage_is_ridden_out(providers_file):
    with MockProvider([Reply.error(503), Reply.says("done")]) as p:
        providers_file(p.base_url)

        assert one_agent().forward("hi") == "done"
        assert p.hits == 2


def test_a_permanent_rate_limit_gives_up_and_says_why(providers_file):
    with MockProvider([Reply.error(429)]) as p:
        providers_file(p.base_url, retry={"max_attempts": 3})

        with pytest.raises(RuntimeError) as excinfo:
            one_agent().forward("hi")

        assert p.hits == 3
        message = str(excinfo.value)
        # Named endpoint, status, and how many times — anything less and a
        # misconfigured key looks like an outage.
        assert "mock" in message, message
        assert "429" in message, message
        assert "3 attempt" in message, message


def test_a_retry_after_header_is_obeyed(providers_file):
    with MockProvider([Reply.error(429, retry_after=1), Reply.says("done")]) as p:
        # A ceiling above what is being asked for, so this measures the
        # instruction being followed rather than the cap.
        providers_file(p.base_url, retry={"max_ms": 5000})

        started = time.monotonic()
        assert one_agent().forward("hi") == "done"
        assert time.monotonic() - started >= 0.9, "should have waited the second"


# ── Failures that are not worth retrying ──


def test_a_bad_request_is_asked_exactly_once(providers_file):
    with MockProvider([Reply.error(400)]) as p:
        providers_file(p.base_url)

        with pytest.raises(RuntimeError):
            one_agent().forward("hi")

        # Four round trips to reach the same error is four wasted, and on a
        # paid endpoint it is four charges.
        assert p.hits == 1


def test_a_rejected_key_is_asked_exactly_once(providers_file):
    with MockProvider([Reply.error(401)]) as p:
        providers_file(p.base_url)

        with pytest.raises(RuntimeError):
            one_agent().forward("hi")
        assert p.hits == 1


# ── Replies that are not what they claim ──


def test_a_gateway_page_is_retried_then_reported(providers_file):
    with MockProvider([Reply.garbage(), Reply.says("done")]) as p:
        providers_file(p.base_url)

        # A 200 carrying HTML is a proxy answering for the endpoint — not
        # something the caller fixes by changing the request.
        assert one_agent().forward("hi") == "done"
        assert p.hits == 2


def test_a_body_that_is_never_a_completion_fails_without_a_traceback(providers_file):
    with MockProvider([Reply.garbage()]) as p:
        providers_file(p.base_url, retry={"max_attempts": 2})

        with pytest.raises(RuntimeError) as excinfo:
            one_agent().forward("hi")
        assert "not a chat completion" in str(excinfo.value)


def test_an_endpoint_that_hangs_up_is_retried(providers_file):
    with MockProvider([Reply.hangs_up(), Reply.says("done")]) as p:
        providers_file(p.base_url)

        assert one_agent().forward("hi") == "done"
        assert p.hits == 2


# ── Turns that end badly ──


def test_a_truncated_turn_is_not_read_as_a_finished_answer(providers_file):
    """`finish_reason: length` means the model was cut off mid-thought.

    Reading it as a finished answer is how a half-written tool call becomes
    a confident wrong result.
    """
    truncated = Reply.wants_tool("search", {"q": "x"}, finish_reason="length")
    with MockProvider([truncated, Reply.says("recovered")]) as p:
        providers_file(p.base_url)

        @soma.tool
        def search(q: str) -> str:
            """Search. Call this for anything current."""
            return "result"

        g = soma.Graph(cache="memory")
        g.node("assistant", soma.Agent(model="mock/any", tools=[search]))

        with pytest.raises(RuntimeError) as excinfo:
            g.forward("hi")

        message = str(excinfo.value)
        assert "max_tokens" in message, message
        # The partial text is in the message rather than thrown away —
        # a cut-off thought is still evidence.
        assert "ran out of tokens" in message, message


def test_malformed_tool_arguments_reach_the_model_instead_of_crashing(providers_file):
    """Models emit broken JSON in tool arguments often enough that it must
    be information, not an exception."""
    broken = Reply.wants_tool("echo", "{not valid json", finish_reason="tool_calls")
    with MockProvider([broken, Reply.says("I will try again")]) as p:
        providers_file(p.base_url)

        seen = []

        @soma.tool
        def echo(text: str = "") -> str:
            """Echo the text back. Call this to repeat something."""
            seen.append(text)
            return "echoed"

        g = soma.Graph(cache="memory")
        g.node("assistant", soma.Agent(model="mock/any", tools=[echo]))

        assert g.forward("hi") == "I will try again"
        # The run survived, and the model got a turn to correct itself.
        assert p.hits == 2


def test_a_refusal_is_not_read_as_an_answer(providers_file):
    """A content-filtered turn is the model declining, not answering.

    Returning its empty content is how a refusal ends up written into a
    report as the agent's considered reply.
    """
    with MockProvider([Reply.stops("content_filter", "")]) as p:
        providers_file(p.base_url)

        with pytest.raises(RuntimeError) as excinfo:
            one_agent().forward("hi")

        message = str(excinfo.value)
        assert "declined" in message, message
        assert "content_filter" in message, message


# ── A judge is the place a bad reply does the most damage ──


def test_a_judge_given_garbage_does_not_pass_silently(providers_file):
    with MockProvider([Reply.says("I'm not sure, maybe 7/10?")]) as p:
        providers_file(p.base_url)

        g = soma.Graph(cache="memory")
        g.node("critic", soma.Judge(model="mock/any", rubric="Has a price column"))
        verdict = g.forward("the artifact")

        # A verdict that cannot be read is a failure to grade, and a refine
        # loop reading `done: true` off it would stop on nothing.
        assert verdict["passed"] is False
        assert verdict["done"] is False
        assert verdict["score"] == 0.0


def test_a_judge_survives_an_outage_and_still_grades(providers_file):
    good = Reply.says(json.dumps({"score": 0.9, "reason": "fine"}))
    with MockProvider([Reply.error(503), good]) as p:
        providers_file(p.base_url)

        g = soma.Graph(cache="memory")
        g.node("critic", soma.Judge(model="mock/any", rubric="Anything"))

        assert g.forward("the artifact")["score"] == 0.9
        assert p.hits == 2


# ── A loop is where a rate limit hurts most ──


def test_a_refine_loop_survives_a_rate_limit_mid_flight(providers_file):
    """The case that motivated all of this.

    A study running forty trials against a metered endpoint will meet a 429
    somewhere in the middle. Before retries, that one response ended the
    whole run.
    """
    from soma.agentic import refine

    verdict = Reply.says(json.dumps({"score": 0.95, "reason": "good"}))
    script = [
        Reply.says("first draft"),
        Reply.error(429),  # the judge gets rate limited
        verdict,
    ]
    with MockProvider(script) as p:
        providers_file(p.base_url)

        g = refine(
            worker=soma.Agent(model="mock/any"),
            judge=soma.Judge(model="mock/any", rubric="Anything"),
            max_rounds=3,
            cache="memory",
        )
        out = g.forward("write something")

        assert out["passed"] is True
        assert p.hits == 3, "the 429 cost one extra knock, not the run"
