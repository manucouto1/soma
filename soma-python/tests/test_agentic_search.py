"""Searching over an agentic graph.

An agent's prompt, model and turn budget are hyperparameters, and so is
whether two nodes should be connected at all. Both end up in the same
``search_space()`` a computational graph produces, which is what lets one
``Study`` tune a graph that mixes filters and agents.
"""

import pytest

import soma

from conftest import MockProvider, says  # noqa: F401


class Scale(soma.Filter):
    """An ordinary trainable filter, to prove the two spaces merge."""

    factor = soma.search(0.1, 10.0, scale="log")
    _cache_version = "1"

    def fit(self, x, y=None):
        return {}

    def forward(self, x, state):
        return x


class Sink(soma.Filter):
    """Marks what it produces, so two of them are not the same filter.

    Two identical stateless filters given identical input share a cache
    entry — correctly — so a test that wants both to run has to make them
    distinguishable.
    """

    _kind = "stateless"
    _cache_version = "1"

    def __init__(self, mark="."):
        self.mark = mark
        self.seen = []

    def forward(self, x, state):
        self.seen.append(x)
        return f"{x}{self.mark}"


class Merge(soma.Filter):
    """Records exactly which producers reached it."""

    _kind = "stateless"
    _cache_version = "1"

    def __init__(self):
        self.seen = []

    def forward(self, x, state):
        self.seen.append(x)
        if isinstance(x, dict):
            return " ".join(f"{k}={x[k]}" for k in sorted(x))
        return x


def _names(space):
    return [d["name"] for d in space]


# ── an agent's arguments are its hyperparameters ──


def test_a_searchable_argument_still_has_a_value():
    # The graph has to be runnable before any study samples it, so a
    # declared space resolves to its first choice / lower bound.
    agent = soma.Agent(
        model=soma.search(choices=["ollama/a", "ollama/b"]),
        max_turns=soma.search(4, 16),
    )
    assert agent.model == "ollama/a"
    assert agent.max_turns == 4


def test_a_default_beats_the_bounds():
    agent = soma.Agent(model=soma.search(choices=["a", "b"], default="b"))
    assert agent.model == "b"


def test_a_space_with_nothing_to_start_from_is_refused():
    with pytest.raises(ValueError, match="no value to start from"):
        soma.Agent(model=soma.search())


def test_agent_and_filter_spaces_merge_into_one():
    g = soma.Graph(cache="memory")
    g.node("scale", Scale())
    g.node(
        "writer",
        soma.Agent(model="mock/m", system=soma.search(choices=["terse", "florid"])),
    )
    g.node("critic", soma.Judge(model="mock/m", rubric="r", threshold=soma.search(0.5, 0.9)))

    assert _names(g.search_space()) == [
        "scale.factor",
        "critic.threshold",
        "writer.system",
    ]


def test_a_sampled_configuration_reaches_the_live_agent():
    agent = soma.Agent(model="mock/m", system=soma.search(choices=["terse", "florid"]))
    g = soma.Graph(cache="memory")
    g.node("writer", agent)

    g.apply_params({"writer.system": "florid", "writer.max_turns": 7})
    assert agent.system == "florid"
    assert agent.max_turns == 7


def test_a_sampled_prompt_reaches_the_model(providers_file):  # noqa: F811
    with MockProvider([says("one"), says("two")]) as provider:
        providers_file(provider.base_url)

        agent = soma.Agent(
            model="mock/any", system=soma.search(choices=["be terse", "be florid"])
        )
        g = soma.Graph(cache="memory")
        g.node("writer", agent)

        g.forward("hello")
        g.apply_params({"writer.system": "be florid"})
        g.forward("hello again")

        # A Step is immutable once built; this only works because the graph
        # rebuilds it from the live agent before each run.
        assert provider.received[0]["messages"][0]["content"] == "be terse"
        assert provider.received[1]["messages"][0]["content"] == "be florid"


def test_an_unknown_param_is_refused():
    g = soma.Graph(cache="memory")
    g.node("writer", soma.Agent(model="mock/m"))
    with pytest.raises(KeyError, match="no node 'ghost'"):
        g.apply_params({"ghost.model": "x"})


# ── topology as a dimension ──


def test_an_optional_edge_becomes_a_dimension():
    g = soma.Graph(cache="memory")
    g.node("a", Sink())
    g.node("b", Sink())
    g.edge("a", "b")
    g.optional("a", "b")

    assert _names(g.search_space()) == ["edge:a->b"]
    assert g.optional_edges() == [("a", "b")]


def test_cutting_an_edge_changes_what_the_consumer_sees():
    # Two producers feeding one consumer, one of the two connections up for
    # debate. This is the shape topology search is actually for: does the
    # critic need the retriever's output, or only the draft?
    merge = Merge()
    g = soma.Graph(cache="memory")
    g.node("a", Sink("[a]"))
    g.node("b", Sink("[b]"))
    g.node("merge", merge)
    g.edge("a", "merge")
    g.edge("b", "merge")
    g.optional("a", "merge")

    assert g.forward("x") == "a=x[a] b=x[b]"

    g.apply_params({"edge:a->merge": False})
    g.forward("x")
    # Cut, the merge has one producer left and reads it directly. (The
    # *graph's* output is ambiguous here — cutting leaves `a` a leaf of its
    # own — so the consumer's input is what this asserts.)
    assert merge.seen == [{"a": "x[a]", "b": "x[b]"}, "x[b]"]


def test_restoring_an_edge_restores_it_exactly():
    g = soma.Graph(cache="memory")
    g.node("a", Sink())
    g.node("b", Sink())
    g.node("c", Sink())
    # More than one edge, and the optional one is not the last: restoring it
    # on the end would leave a graph that renders and fingerprints
    # differently from the one this trial started with.
    g.edge("a", "b")
    g.edge("b", "c")
    g.optional("a", "b")

    before = g.to_mermaid()
    g.apply_params({"edge:a->b": False})
    assert g.to_mermaid() != before
    g.apply_params({"edge:a->b": True})
    # A trial that cuts an edge has to leave the graph the next trial starts
    # from byte-identical, or the search is comparing different things.
    assert g.to_mermaid() == before


def test_cutting_twice_is_harmless():
    g = soma.Graph(cache="memory")
    g.node("a", Sink())
    g.node("b", Sink())
    g.edge("a", "b")
    g.optional("a", "b")

    g.apply_params({"edge:a->b": False})
    cut = g.to_mermaid()
    g.apply_params({"edge:a->b": False})
    assert g.to_mermaid() == cut


def test_an_edge_that_does_not_exist_cannot_be_optional():
    g = soma.Graph(cache="memory")
    g.node("a", Sink())
    with pytest.raises(ValueError, match="no edge"):
        g.optional("a", "b")


def test_a_control_edge_cannot_be_optional():
    g = soma.Graph(cache="memory")
    g.node("w", Sink())
    g.loop("l", body="w", until=False, max_iterations=2)
    # Cutting it would change what the loop owns, not just what flows.
    with pytest.raises(ValueError, match="control edge"):
        g.optional("l", "w")


def test_an_undeclared_edge_cannot_be_toggled():
    g = soma.Graph(cache="memory")
    g.node("a", Sink())
    g.node("b", Sink())
    g.edge("a", "b")
    with pytest.raises(ValueError, match="never declared optional"):
        g.set_edge("a", "b", False)


# ── the two together ──


def test_a_study_over_an_agentic_graph_covers_both():
    g = soma.Graph(cache="memory")
    g.node("writer", soma.Agent(model=soma.search(choices=["mock/a", "mock/b"])))
    g.node("critic", soma.Judge(model="mock/m", rubric="r"))
    g.edge("writer", "critic")
    g.optional("writer", "critic")

    assert sorted(_names(g.search_space())) == [
        "edge:writer->critic",
        "writer.model",
    ]
    # And a Study accepts that space without knowing an agent from a filter.
    assert g.study("shape", n_trials=4).name == "shape"
