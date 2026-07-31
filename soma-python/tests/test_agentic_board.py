"""The board pattern: a panel answers, a chair moderates, the panel answers again.

This is Du et al.'s multi-agent debate (ICML 2024) with the summarizer
variant that paper introduces for larger panels. The model is a mock HTTP
server, so nothing here needs a key or a network — what is asserted is the
machinery (does it loop, does it stop, does round two see round one), never
a model's cleverness.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import soma
from soma.agentic import PANEL_MARKER, Brief, MajorityVote, board


# ── A mock endpoint that answers per member and per round ──


class _Panel(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 — BaseHTTPRequestHandler's naming
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        system = body["messages"][0]["content"]
        user = body["messages"][-1]["content"]
        self.server.seen.append(body)

        answer = self.server.answer(system, PANEL_MARKER in user)
        payload = json.dumps({
            "choices": [{"message": {"content": f"I reason, therefore \\boxed{{{answer}}}"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args):
        pass


class MockPanel:
    """`answer(system, is_later_round) -> str` drives what each member says."""

    def __init__(self, answer):
        self.answer = answer

    def __enter__(self):
        self.server = HTTPServer(("127.0.0.1", 0), _Panel)
        self.server.seen = []
        self.server.answer = self.answer
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *_exc):
        self.server.shutdown()

    @property
    def seen(self):
        return self.server.seen

    def model(self, monkeypatch, tmp_path):
        catalog = tmp_path / "providers.toml"
        catalog.write_text(
            f'[providers.mock]\nbase_url = "http://127.0.0.1:{self.server.server_port}/v1"\n'
            f'auth = {{ type = "none" }}\n'
        )
        monkeypatch.setenv("SOMA_PROVIDERS", str(catalog))
        return "mock/panel"


def members(model, n):
    return [soma.Agent(model=model, system=f"solver-{i}") for i in range(n)]


# ── The chair, on its own ──


def test_majority_vote_counts_and_normalizes():
    chair = MajorityVote()
    out = chair.forward(
        {"brief": "Q?", "a": r"\boxed{18}", "b": "the answer is 18.0", "c": r"\boxed{5}"},
        None,
    )
    # 18 and 18.0 are the same answer and must not split the vote.
    assert out["votes"] == {"18": 2, "5": 1}
    assert out["answer"] == "18"
    assert out["agreement"] == pytest.approx(2 / 3)
    assert out["done"] is False


def test_majority_vote_is_done_only_when_unanimous():
    chair = MajorityVote()
    assert chair.forward({"brief": "Q", "a": r"\boxed{7}", "b": r"\boxed{7}"}, None)["done"]
    assert not chair.forward({"brief": "Q", "a": r"\boxed{7}", "b": r"\boxed{8}"}, None)["done"]


def test_unreadable_answers_are_not_votes():
    """The reference implementation scores an unparseable answer as correct.
    Dropping it is the honest reading, and a panel with nothing readable has
    no answer rather than a wrong one."""
    chair = MajorityVote()
    out = chair.forward({"brief": "Q", "a": "I have no idea", "b": "nor I"}, None)
    assert out["answer"] is None
    assert out["votes"] == {}
    assert out["done"] is False


def test_extract_prefers_boxed_over_stray_numbers():
    assert MajorityVote.extract(r"I tried 3 and 4, so \boxed{12}") == "12"
    assert MajorityVote.extract("no box here, just 1,234") == "1234"
    assert MajorityVote.extract("nothing numeric") is None


def test_brief_passes_the_question_through_on_round_one():
    assert Brief().forward("How many eggs?", None) == "How many eggs?"


def test_brief_does_not_nest_briefs():
    """A brief built from a chair's verdict must restate the question once.
    Feeding the composed text back in is what made it grow every round."""
    brief = Brief()
    first = brief.forward("Q?", None)
    verdict = MajorityVote().forward({"brief": first, "a": r"\boxed{1}", "b": r"\boxed{2}"}, None)
    second = brief.forward(verdict, None)
    assert second.count(PANEL_MARKER) == 1
    assert second.startswith("Q?")


# ── The shape ──


def test_board_topology():
    g = board(members("mock/x", 3), rounds=2)
    mermaid = g.to_mermaid()
    for i in range(3):
        assert f"brief --> member_{i}" in mermaid
        assert f"member_{i} --> chair" in mermaid
    # The chair reads the brief too, or a second round would not know the
    # question it is deciding.
    assert "brief --> chair" in mermaid


def test_board_rejects_a_panel_of_one():
    with pytest.raises(ValueError, match="at least two"):
        board(members("mock/x", 1))
    with pytest.raises(ValueError, match="at least one round"):
        board(members("mock/x", 2), rounds=0)


# ── The loop ──


def test_board_debates_then_converges(monkeypatch, tmp_path):
    """Round one disagrees, round two is unanimous, and the board stops
    there rather than spending the round it was still allowed."""

    def answer(system, later):
        return "18" if (later or system != "solver-2") else "5"

    with MockPanel(answer) as panel:
        model = panel.model(monkeypatch, tmp_path)
        g = board(members(model, 3), rounds=3, cache="memory")
        verdict = g.forward("Janet has 16 eggs, eats 3, bakes 4. How many left?")

        assert verdict["answer"] == "18"
        assert verdict["done"] is True
        assert verdict["agreement"] == 1.0
        # 3 members x 2 rounds. A third round would be 9.
        assert len(panel.seen) == 6


def test_round_two_sees_round_one(monkeypatch, tmp_path):
    """The members must actually be shown what the panel said. This is the
    regression for a merge that discarded every re-run output and left the
    chair reading iteration one forever."""

    def answer(system, later):
        return "18" if (later or system != "solver-2") else "5"

    with MockPanel(answer) as panel:
        model = panel.model(monkeypatch, tmp_path)
        question = "How many are left?"
        board(members(model, 3), rounds=2, cache="memory").forward(question)

        later = [b for b in panel.seen if PANEL_MARKER in b["messages"][-1]["content"]]
        assert later, "no second round happened"
        prompt = later[0]["messages"][-1]["content"]
        assert question in prompt          # the question survived the round
        assert "18" in prompt and "5" in prompt   # and so did the disagreement


def test_board_runs_every_round_when_it_never_converges(monkeypatch, tmp_path):
    """A panel that never agrees spends its whole budget and says so."""

    def answer(system, later):
        return {"solver-0": "1", "solver-1": "2"}.get(system, "3")

    with MockPanel(answer) as panel:
        model = panel.model(monkeypatch, tmp_path)
        verdict = board(members(model, 3), rounds=2, cache="memory").forward("Q?")

        assert verdict["done"] is False
        assert verdict["agreement"] == pytest.approx(1 / 3)
        assert len(panel.seen) == 6


def test_board_is_searchable(monkeypatch, tmp_path):
    """A panel is a graph, so its members' prompts are hyperparameters like
    any other node's."""
    g = board(
        [soma.Agent(model="mock/x", system=soma.search(choices=["terse", "verbose"]))
         for _ in range(2)],
        rounds=2,
    )
    names = [d["name"] for d in g.search_space()]
    assert "member_0.system" in names
    assert "member_1.system" in names
