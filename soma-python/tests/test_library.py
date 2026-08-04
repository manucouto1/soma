"""The behaviour library.

Ordinary filters, so most of this is arithmetic and string handling — but
the degenerate cases are the point. A scorer that returns 0.0 for an empty
set, or a compactor that silently drops the question everything refers to,
does its damage quietly and months later.
"""

import json

import pytest

import soma
from conftest import MockProvider, Reply
from soma.agentic import MajorityVote, self_consistency
from soma.library import Accumulator, Compact, Eval, Retriever, normalize_answer


# ── Eval ──


def test_accuracy_counts_what_matched():
    out = Eval(["accuracy"]).forward(
        {"prediction": ["a", "b", "c"], "reference": ["a", "x", "c"]}, None
    )
    assert out["accuracy"] == pytest.approx(2 / 3)
    assert out["n"] == 3


def test_a_single_pair_needs_no_list():
    out = Eval(["accuracy"]).forward({"prediction": 42, "reference": 42}, None)
    assert out["accuracy"] == 1.0
    assert out["n"] == 1


def test_accuracy_crosses_the_string_number_line_but_not_the_prose_one():
    # `1` and `"1"` are the same label written twice; `"The 1"` is not.
    assert Eval(["accuracy"]).forward({"prediction": 1, "reference": "1"}, None)[
        "accuracy"
    ] == 1.0
    assert Eval(["accuracy"]).forward(
        {"prediction": "The 1", "reference": "1"}, None
    )["accuracy"] == 0.0


def test_exact_match_forgives_formatting():
    """Comparing raw strings makes a scorer measure formatting."""
    out = Eval(["exact_match"]).forward(
        {"prediction": "The answer is 42.", "reference": "answer is 42"}, None
    )
    assert out["exact_match"] == 1.0


def test_f1_rewards_partial_overlap():
    out = Eval(["f1"]).forward(
        {"prediction": "a red brick house", "reference": "the red house"}, None
    )
    # Normalization drops the articles first, so this is
    # ["red","brick","house"] against ["red","house"]: precision 2/3,
    # recall 2/2.
    assert out["f1"] == pytest.approx(0.8)


def test_f1_of_two_empty_answers_is_agreement():
    assert Eval(["f1"]).forward({"prediction": "", "reference": ""}, None)["f1"] == 1.0
    assert Eval(["f1"]).forward({"prediction": "", "reference": "x"}, None)["f1"] == 0.0


def test_top_k_looks_down_the_candidate_list():
    out = Eval(["top_k"], k=3).forward(
        {"prediction": [["x", "y", "right"], ["a", "b", "c"]],
         "reference": ["right", "z"]},
        None,
    )
    assert out["top_k"] == 0.5


def test_several_metrics_at_once():
    out = Eval(["accuracy", "exact_match", "f1"]).forward(
        {"prediction": "42", "reference": "42"}, None
    )
    assert set(out) == {"accuracy", "exact_match", "f1", "n"}


def test_it_can_read_from_a_fan_in():
    g = soma.Graph(cache="memory")

    class Const(soma.Filter):
        _kind = "stateless"
        _cache_version = "1"

        def __init__(self, value):
            self.value = value

        def forward(self, x, state):
            return self.value

    g.node("model", Const(["a", "b"]))
    g.node("truth", Const(["a", "c"]))
    g.node("score", Eval(["accuracy"], prediction="model", reference="truth"))
    g.connect("model", "score")
    g.connect("truth", "score")

    assert g.forward("go")["accuracy"] == 0.5


# ── Eval: the cases that do damage quietly ──


def test_scoring_nothing_is_an_error_not_a_zero():
    """An accuracy computed over nothing is undefined, not zero.

    Reporting 0.0 puts a fabricated number into a record someone compares
    against later.
    """
    with pytest.raises(ValueError, match="nothing to score"):
        Eval(["accuracy"]).forward({"prediction": [], "reference": []}, None)


def test_a_missing_value_is_an_error_by_default():
    with pytest.raises(ValueError, match="missing"):
        Eval(["accuracy"]).forward(
            {"prediction": ["a", None], "reference": ["a", "b"]}, None
        )


def test_missing_values_can_be_skipped_or_scored_wrong():
    pairs = {"prediction": ["a", None], "reference": ["a", "b"]}

    skipped = Eval(["accuracy"], missing="skip").forward(pairs, None)
    assert skipped["n"] == 1 and skipped["accuracy"] == 1.0

    zeroed = Eval(["accuracy"], missing="zero").forward(pairs, None)
    assert zeroed["n"] == 2 and zeroed["accuracy"] == 0.5


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="line up"):
        Eval(["accuracy"]).forward(
            {"prediction": ["a", "b"], "reference": ["a"]}, None
        )


def test_a_missing_side_names_what_was_there():
    with pytest.raises(ValueError, match="reference"):
        Eval(["accuracy"]).forward({"prediction": "a"}, None)


def test_an_unknown_metric_is_refused_at_construction():
    # Better now than after a study has spent an hour.
    with pytest.raises(ValueError, match="unknown metric"):
        Eval(["bleu"])


def test_normalization_is_the_squad_one():
    assert normalize_answer("The  Answer, is 42!") == "answer is 42"


# ── Accumulator ──


def test_it_keeps_everything_that_passed_through():
    acc = Accumulator()
    acc.forward("first", None)
    out = acc.forward("second", None)

    assert out["history"] == ["first", "second"]
    assert out["value"] == "second"


def test_a_mapping_keeps_its_own_keys():
    acc = Accumulator()
    out = acc.forward({"score": 0.5, "reason": "meh"}, None)
    assert out["score"] == 0.5
    assert out["history"] == [{"score": 0.5, "reason": "meh"}]


def test_the_limit_keeps_the_most_recent():
    """A worker improves against what it just did, not what it did first."""
    acc = Accumulator(limit=2)
    for value in ["a", "b", "c"]:
        out = acc.forward(value, None)
    assert out["history"] == ["b", "c"]


def test_it_can_be_reset():
    acc = Accumulator()
    acc.forward("a", None)
    acc.reset()
    assert acc.forward("b", None)["history"] == ["b"]


def test_it_survives_a_loop_and_remembers_every_round():
    class Judge(soma.Filter):
        _kind = "stateless"
        _cache_version = "1"

        def __init__(self):
            self.round = 0

        def forward(self, x, state):
            self.round += 1
            return {"done": self.round >= 3, "round": self.round}

    acc = Accumulator()
    g = soma.Graph(cache="memory")
    g.node("judge", Judge())
    g.node("remember", acc)
    g.connect("judge", "remember")
    g.loop("loop", body="judge", until="judge", max_iterations=5)
    g.forward("go")

    assert [h["round"] for h in acc.history] == [1, 2, 3]


# ── Compact ──


def test_short_text_is_left_alone():
    """Bit-identical, so a short run's journal is unaffected."""
    compact = Compact(max_chars=100)
    assert compact.forward("short", None) == "short"


def test_a_long_transcript_keeps_the_head_and_the_tail():
    text = "QUESTION: what is it?" + ("x" * 5000) + "LATEST ANSWER"
    out = Compact(max_chars=500, head_chars=100).forward(text, None)

    assert len(out) < len(text)
    # The head is where the question lives, and everything else refers to it.
    assert out.startswith("QUESTION: what is it?")
    assert out.endswith("LATEST ANSWER")
    assert "characters dropped" in out


def test_it_says_how_much_it_dropped():
    out = Compact(max_chars=200, head_chars=50).forward("y" * 1000, None)
    assert "800 characters dropped" in out


def test_a_summarizer_replaces_the_middle():
    class Summarizer:
        def forward(self, x, state):
            return "they argued about lunch"

    out = Compact(max_chars=200, head_chars=50, summarizer=Summarizer()).forward(
        "z" * 1000, None
    )
    assert "they argued about lunch" in out
    assert "characters dropped" not in out


def test_a_head_with_no_room_for_a_tail_is_refused():
    with pytest.raises(ValueError, match="room for a tail"):
        Compact(max_chars=100, head_chars=100)


def test_a_mapping_is_compacted_as_json():
    out = Compact(max_chars=200, head_chars=50).forward({"k": "v" * 1000}, None)
    assert isinstance(out, str)
    assert "characters dropped" in out


# ── Retriever ──


def test_an_empty_pool_is_not_an_error(tmp_path, monkeypatch):
    """A new project has no pool. Saying 'nothing found' beats ending a run."""
    monkeypatch.chdir(tmp_path)
    out = Retriever().forward("anything at all", None)
    assert out["retrieved"] == []
    assert out["query"] == "anything at all"


def test_it_finds_what_was_recorded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOMA_CACHE_DIR", str(tmp_path / "cache"))

    class Noop(soma.Filter):
        _kind = "stateless"
        _cache_version = "1"

        def forward(self, x, state):
            return x

    g = soma.Graph(cache="memory")
    g.node("n", Noop())
    with g.track_run("normalization-sweep", tags=["retrieval-test"]):
        g.forward("data")

    out = Retriever(k=3).forward("normalization", None)
    assert out["retrieved"], "a tracked run belongs in the pool"
    assert "score" in out["retrieved"][0]


# ── self_consistency ──


def test_it_needs_more_than_one_sample():
    with pytest.raises(ValueError, match="two samples"):
        self_consistency(soma.Agent(model="mock/any"), n=1)


def test_the_majority_answer_wins(providers_file):
    script = [Reply.says("blue"), Reply.says("green"), Reply.says("blue")]
    with MockProvider(script) as p:
        providers_file(p.base_url)

        g = self_consistency(soma.Agent(model="mock/any"), n=3, cache="memory")
        out = g.forward("what colour?")

        assert out["answer"] == "blue"
        assert out["votes"] == {"blue": 2, "green": 1}
        assert out["agreement"] == pytest.approx(2 / 3)
        assert p.hits == 3


def test_unanimity_is_reported():
    votes = MajorityVote(mode="text").forward(
        {"a": "yes", "b": "Yes.", "c": "yes"}, None
    )
    # The normalization is what makes "Yes." and "yes" one answer rather
    # than a tie between two.
    assert votes["answer"] == "yes"
    assert votes["done"] is True


def test_number_mode_reads_a_worked_solution():
    votes = MajorityVote(mode="number").forward(
        {"a": "so the total is 18", "b": "I get \\boxed{18}", "c": "42"}, None
    )
    assert votes["answer"] == "18"


def test_the_mode_is_checked():
    with pytest.raises(ValueError, match="number.*text"):
        MajorityVote(mode="vibes")


def test_a_custom_aggregator_is_used(providers_file):
    class Longest(soma.Filter):
        _kind = "stateless"
        _cache_version = "1"

        def forward(self, x, state):
            answers = [v for k, v in sorted(x.items())] if isinstance(x, dict) else [x]
            return max(answers, key=len)

    script = [Reply.says("no"), Reply.says("a longer answer")]
    with MockProvider(script) as p:
        providers_file(p.base_url)

        g = self_consistency(
            soma.Agent(model="mock/any"), n=2, aggregator=Longest(), cache="memory"
        )
        assert g.forward("q") == "a longer answer"
