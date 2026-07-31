"""The agentic stack against real endpoints.

Every other agentic test answers from a mock on localhost, which proves the
wiring but never proves the catalog resolves, the bearer header is accepted,
or that a real model's tool call survives the round trip. This file does,
and it is opt-in twice over — the `live` marker plus `$SOMA_LIVE` — because
it needs a network, credentials, and somebody's tokens.

    SOMA_LIVE=1 OLLAMA_HOST=http://box:11434 NVIDIA_API_KEY=nvapi-... \
        pytest tests/test_live_providers.py -m live -v

Each provider skips itself when its credentials are absent, so running with
only one configured is a supported way to use this.
"""

import os

import pytest

import soma

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("SOMA_LIVE"), reason="set SOMA_LIVE=1 to run"),
]


# ── Which endpoints to exercise ──
#
# Both are overridable: model availability is a property of somebody's
# server and their account, not of Soma.

OLLAMA_MODEL = os.environ.get("SOMA_LIVE_OLLAMA_MODEL", "ollama/qwen2.5:14b")
# The 8B rather than the 70B: on NVIDIA's free tier the large models are
# heavily contended, and a test that waits five minutes for a queue slot
# tells you nothing about Soma.
NVIDIA_MODEL = os.environ.get(
    "SOMA_LIVE_NVIDIA_MODEL", "nvidia/meta/llama-3.1-8b-instruct"
)


# ── Separating "Soma is wrong" from "the provider is busy" ──
#
# A free-tier endpoint returns 503 ResourceExhausted when its worker pool is
# full, and sometimes just never answers. Reporting either as a test failure
# would be a false negative: the assertion is about Soma's behaviour, not the
# provider's uptime. So those become skips, and everything else still fails.

_UNAVAILABLE = (
    "503",
    "resourceexhausted",
    "service unavailable",
    "429",
    "rate limit",
    "timed out",
    "timeout",
    "connection refused",
)


def run(graph, x):
    """Forward `x`, turning provider unavailability into a skip."""
    try:
        return graph.forward(x)
    except RuntimeError as e:
        message = str(e).lower()
        if any(marker in message for marker in _UNAVAILABLE):
            pytest.skip(f"provider unavailable: {str(e)[:200]}")
        raise

PROVIDERS = [
    pytest.param(
        OLLAMA_MODEL,
        marks=pytest.mark.skipif(
            not os.environ.get("OLLAMA_HOST"),
            reason="OLLAMA_HOST not set",
        ),
        id="ollama",
    ),
    pytest.param(
        NVIDIA_MODEL,
        marks=pytest.mark.skipif(
            not os.environ.get("NVIDIA_API_KEY"),
            reason="NVIDIA_API_KEY not set",
        ),
        id="nvidia",
    ),
]


@soma.tool
def population(city: str) -> str:
    """Return the population of a city. Call this for any question about how many people live in a place."""
    return {"Lugo": "98025", "Vigo": "296692"}.get(city, "unknown")


@pytest.mark.parametrize("model", PROVIDERS)
def test_agent_completes(model):
    """The plainest thing there is: catalog -> auth -> completion -> text."""
    g = soma.Graph(cache="memory")
    g.node("a", soma.Agent(model=model, system="Answer in one short sentence."))

    answer = run(g, "What is a compiler?")

    assert isinstance(answer, str)
    assert "compil" in answer.lower() or "translat" in answer.lower()


@pytest.mark.parametrize("model", PROVIDERS)
def test_agent_runs_a_tool(model):
    """A full react loop: the model asks for the tool, Soma runs it, the
    model answers from the result.

    Retried, and this is not flakiness we are papering over. Small
    open-weight models intermittently emit their tool-call *markup* as prose
    instead of using the `tool_calls` field — observed with qwen2.5:14b at
    roughly one call in five, where Ollama returns `finish_reason: "stop"`
    and a leaked `</tool_call>` in the content. By the OpenAI contract that
    response is a final answer, so Soma is right to return it; the failure
    is upstream and no amount of client code fixes it. What we can still
    assert is that the loop closes when the model does its part.
    """
    for attempt in range(3):
        g = soma.Graph(cache="memory")
        g.node(
            "a",
            soma.Agent(
                model=model,
                system="Use the tools you are given. Answer in one sentence.",
                tools=[population],
            ),
        )
        answer = run(g, "How many people live in Lugo?")

        if "98" in answer and "tool_call" not in answer:
            return
        last = answer

    pytest.fail(f"no clean tool round trip in 3 attempts; last answer: {last!r}")


@pytest.mark.parametrize("model", PROVIDERS)
def test_judge_returns_a_score(model):
    """Structured output: the judge has to come back as parsed JSON, not prose."""
    g = soma.Graph(cache="memory")
    g.node(
        "j",
        soma.Judge(
            model=model,
            rubric="The text names at least one concrete programming language.",
            threshold=0.5,
        ),
    )

    verdict = run(g, "Rust and Python are both widely used.")

    assert isinstance(verdict, dict)
    assert 0.0 <= verdict["score"] <= 1.0
    assert verdict["passed"] is True
    assert verdict["reason"]


@pytest.mark.parametrize("model", PROVIDERS)
def test_refine_loop_terminates(model):
    """A loop whose exit condition is a real model's verdict.

    The point is termination: a live judge decides when to stop, and the
    loop has to honour that rather than burn `max_rounds` every time.
    """
    flow = soma.agentic.refine(
        worker=soma.Agent(model=model, system="Write one vivid sentence about the sea."),
        judge=soma.Judge(
            model=model,
            rubric="The sentence contains a concrete, physical image.",
            threshold=0.5,
        ),
        max_rounds=2,
        cache="memory",
    )

    verdict = run(flow, "write about the sea")

    assert verdict["done"] is True
    assert 0.0 <= verdict["score"] <= 1.0
    assert verdict["value"]
