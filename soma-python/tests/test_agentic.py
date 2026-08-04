"""Agentic graphs from Python.

These are what a notebook actually does: declare a tool, put an agent in a
graph next to ordinary filters, run it. The model is a mock HTTP server on
localhost, so nothing here needs a key or a network.
"""

import json

import pytest

import soma
from conftest import MockProvider, Reply, says, wants_tool  # noqa: F401


# ── Tools ──


def test_tool_takes_its_description_from_the_docstring():
    @soma.tool
    def search(query: str) -> str:
        """Search the web. Call this when the answer needs current information."""
        return "result"

    assert search.name == "search"
    assert "Call this when" in search.description
    assert search.schema["properties"]["query"]["type"] == "string"
    assert search.schema["required"] == ["query"]


def test_a_tool_without_a_description_is_refused():
    def undocumented(x: str) -> str:
        return x

    # A model picks tools by their descriptions, so one without is a tool it
    # will never use — better to say so than to register something inert.
    with pytest.raises(ValueError, match="description"):
        soma.Tool(undocumented)


def test_optional_arguments_are_not_required():
    @soma.tool
    def fetch(url: str, timeout: int = 30) -> str:
        """Fetch a URL."""
        return url

    assert fetch.schema["required"] == ["url"]
    assert fetch.schema["properties"]["timeout"]["type"] == "integer"


def test_an_explicit_schema_wins_over_the_signature():
    @soma.tool(schema={"type": "object", "properties": {"anything": {}}})
    def odd(*args, **kwargs):
        """Takes whatever."""
        return "ok"

    assert "anything" in odd.schema["properties"]


# ── Graphs with agents ──


def test_an_agent_node_runs_and_returns_its_answer(providers_file):
    with MockProvider([says("a graph runtime")]) as provider:
        providers_file(provider.base_url)

        g = soma.Graph(cache="memory")
        g.node("assistant", soma.Agent(model="mock/any-model"))
        out = g.forward("what is soma?")

        assert out == "a graph runtime"
        assert provider.received[0]["messages"][-1]["content"] == "what is soma?"


def test_an_agent_calls_a_python_tool(providers_file):
    calls = []

    @soma.tool
    def weather(city: str) -> str:
        """Current weather for a city. Call this when asked about conditions."""
        calls.append(city)
        return f"sunny in {city}"

    script = [wants_tool("weather", {"city": "Vigo"}), says("It is sunny in Vigo.")]
    with MockProvider(script) as provider:
        providers_file(provider.base_url)

        g = soma.Graph(cache="memory")
        g.node("assistant", soma.Agent(model="mock/any-model", tools=[weather]))
        out = g.forward("weather in Vigo?")

        assert out == "It is sunny in Vigo."
        assert calls == ["Vigo"], "the Python tool should have run"

        # The tool was advertised, and its result went back paired to the
        # call that asked for it.
        assert provider.received[0]["tools"][0]["function"]["name"] == "weather"
        second = provider.received[1]["messages"]
        assert second[-1]["role"] == "tool"
        assert second[-1]["tool_call_id"] == "call_1"
        assert second[-1]["content"] == "sunny in Vigo"


def test_a_tool_that_raises_is_reported_to_the_model(providers_file):
    @soma.tool
    def broken(x: str) -> str:
        """Always fails. Call this to see what happens."""
        raise RuntimeError("kaboom")

    script = [wants_tool("broken", {"x": "1"}), says("I could not do that.")]
    with MockProvider(script) as provider:
        providers_file(provider.base_url)

        g = soma.Graph(cache="memory")
        g.node("assistant", soma.Agent(model="mock/any-model", tools=[broken]))
        # The run completes: a broken tool is information for the model, not
        # a reason to abandon the work.
        assert g.forward("go") == "I could not do that."

        tool_turn = provider.received[1]["messages"][-1]
        assert "kaboom" in tool_turn["content"]


def test_agents_and_filters_compose_in_one_graph(providers_file):
    class Shout(soma.Filter):
        _cache_version = "1"
        _kind = "stateless"

        def fit(self, x, y=None):
            return {}

        def forward(self, x, state):
            return str(x).upper()

    with MockProvider([says("quiet answer")]) as provider:
        providers_file(provider.base_url)

        g = soma.Graph(cache="memory")
        g.node("assistant", soma.Agent(model="mock/any-model"))
        g.node("shout", Shout())
        g.connect("assistant", "shout")

        # A step's output feeds an ordinary filter with no adapter between.
        assert g.forward("hello") == "QUIET ANSWER"


def test_a_default_provider_lets_models_go_unqualified(providers_file):
    with MockProvider([says("ok")]) as provider:
        providers_file(provider.base_url)

        g = soma.Graph(cache="memory")
        g.use_provider("mock")
        g.node("assistant", soma.Agent(model="any-model"))

        assert g.forward("hi") == "ok"
        assert provider.received[0]["model"] == "any-model"


def test_a_judge_scores_against_its_rubric(providers_file):
    verdict = says(json.dumps({"score": 0.9, "reason": "has a price column"}))
    with MockProvider([verdict]) as provider:
        providers_file(provider.base_url)

        g = soma.Graph(cache="memory")
        g.node("critic", soma.Judge(model="mock/any", rubric="Has a numeric price column"))
        out = g.forward("the artifact")

        assert out["score"] == 0.9
        assert out["passed"] is True
        # `done` is the key a refine loop reads as its stop signal.
        assert out["done"] is True
        assert "price column" in provider.received[0]["messages"][0]["content"]


def test_an_unknown_provider_names_what_is_available(providers_file):
    with MockProvider([says("x")]) as provider:
        providers_file(provider.base_url)

        g = soma.Graph(cache="memory")
        g.node("assistant", soma.Agent(model="nosuchprovider/model"))

        with pytest.raises(Exception) as excinfo:
            g.forward("hi")
        assert "nosuchprovider" in str(excinfo.value)


def test_node_rejects_something_that_is_neither_filter_nor_step():
    g = soma.Graph(cache="memory")
    with pytest.raises(Exception):
        g.node("x", "just a string")


def test_a_step_node_is_visible_in_the_rendered_graph(providers_file, tmp_path):
    providers_file("http://127.0.0.1:1/v1")

    g = soma.Graph(cache="memory")
    g.node("assistant", soma.Agent(model="mock/any"))
    g.node("shout", _Identity())
    g.connect("assistant", "shout")

    mermaid = g.to_mermaid()
    # Effectful nodes render as parallelograms — they reach outside the graph.
    assert "assistant[/assistant/]" in mermaid
    assert "shout[shout]" in mermaid


class _Identity(soma.Filter):
    _cache_version = "1"

    def fit(self, x, y=None):
        return {}

    def forward(self, x, state):
        return x


# ── Providers ──


def test_the_provider_catalog_lists_the_usual_endpoints():
    ids = {p["id"] for p in soma.providers()}
    for expected in ["ollama", "hf", "nvidia", "kimi", "glm", "deepseek", "vllm"]:
        assert expected in ids, f"missing `{expected}`"


def test_local_providers_need_no_key():
    by_id = {p["id"]: p for p in soma.providers()}
    assert by_id["ollama"]["configured"] is True
    assert by_id["ollama"]["env"] is None
    assert by_id["kimi"]["env"] == "MOONSHOT_API_KEY"
