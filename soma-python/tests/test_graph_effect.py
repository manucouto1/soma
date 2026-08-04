"""A pipeline as a tool for a step: soma.agentic.RunGraph.

The effect carries the sub-graph's *structure*; its implementations are
made runnable with ``g.register_graph(sub)``. These pin the seam from the
Python side — the Rust side has its own tests in
soma-runtime/src/effects/graph_handler.rs.
"""

import pytest

import soma
from soma.agentic import Await, Done, RunGraph


class Upper(soma.Filter):
    _cache_version = "1"
    _kind = "stateless"

    def forward(self, x, state):
        return x.upper()


class CallsPipeline:
    """Awaits the sub-pipeline once, then hands back what it produced.

    The graph is stored underscored: a live Graph cannot enter the step's
    JSON identity, and it does not need to — the journal keys the effect
    by the graph's own content.
    """

    _cache_version = "1"

    def __init__(self, sub):
        self._sub = sub

    def poll(self, ctx):
        if ctx.turn == 0:
            return Await(RunGraph(self._sub, input=ctx.input))
        result = ctx.result()
        if result["kind"] == "failed":
            return Done("failed: " + result["message"])
        return Done(result["output"])


def sub_pipeline():
    sub = soma.Graph(cache="memory")
    sub.node("upper", Upper())
    return sub


def test_step_awaits_a_graph_effect(tmp_path, monkeypatch):
    monkeypatch.setenv("SOMA_CACHE_DIR", str(tmp_path))
    sub = sub_pipeline()

    g = soma.Graph(cache="memory")
    g.node("planner", CallsPipeline(sub))
    g.register_graph(sub)

    assert g.forward("hola") == "HOLA"


def test_run_graph_without_registration_fails_readably(tmp_path, monkeypatch):
    """A sub-graph nobody registered is a result the step reads, not a crash."""
    monkeypatch.setenv("SOMA_CACHE_DIR", str(tmp_path))
    sub = sub_pipeline()

    g = soma.Graph(cache="memory")
    g.node("planner", CallsPipeline(sub))
    # no register_graph(sub)

    out = g.forward("hola")
    assert out.startswith("failed: ")
    assert "upper" in out, out


def test_a_sub_graph_may_contain_a_step(tmp_path, monkeypatch):
    """Agent → pipeline → agent, all in Python, no model involved."""
    monkeypatch.setenv("SOMA_CACHE_DIR", str(tmp_path))

    class Inner:
        _cache_version = "1"

        def poll(self, ctx):
            return Done("inner saw: " + ctx.input)

    sub = soma.Graph(cache="memory")
    sub.node("inner", Inner())

    g = soma.Graph(cache="memory")
    g.node("outer", CallsPipeline(sub))
    g.register_graph(sub)

    assert g.forward("x") == "inner saw: x"


def test_unknown_effect_message_lists_graph(tmp_path, monkeypatch):
    monkeypatch.setenv("SOMA_CACHE_DIR", str(tmp_path))

    class Bogus:
        _cache_version = "1"

        def poll(self, ctx):
            return Await({"effect": "bogus"})

    g = soma.Graph(cache="memory")
    g.node("bogus", Bogus())
    with pytest.raises(Exception, match="graph"):
        g.forward("x")


def test_registering_a_conflicting_node_id_is_refused(tmp_path, monkeypatch):
    """Two implementations under one id would answer for each other's cache."""
    monkeypatch.setenv("SOMA_CACHE_DIR", str(tmp_path))

    class Lower(soma.Filter):
        _cache_version = "1"
        _kind = "stateless"

        def forward(self, x, state):
            return x.lower()

    sub = soma.Graph(cache="memory")
    sub.node("upper", Lower())  # same id as the outer graph's, other impl

    g = soma.Graph(cache="memory")
    g.node("upper", Upper())
    with pytest.raises(Exception, match="upper"):
        g.register_graph(sub)
