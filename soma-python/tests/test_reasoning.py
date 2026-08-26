"""The reasoning read back and laid out.

Two halves, and they are tested differently on purpose. The **layout** is pure
over the rows the readers return, so it is given rows written here — which is
also what proves an app can hand it its own. The **reading** needs a store with
an investigation in it, and the only thing that writes one is the terminal, so
those tests drive `somatize-tree` and skip when it has not been built.

The derivations themselves — what stands where, what folds, what a scope
reaches — are Rust's and are tested there. What is defended here is that they
arrive whole and that a position comes out of the shape.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from somatize import Store, reasoning

BINARY = Path(__file__).resolve().parents[2] / "target" / "debug" / "somatize-tree"


def move(name, kind, prose, under=(), scope=(), standing=None, course=None, about=(), cites=()):
    """One row of `reasoning.moves`, written by hand. The columns are the
    documented ones, so a change to them breaks this rather than the drawing."""
    return {
        "name": name,
        "id": 0,
        "kind": kind,
        "prose": prose,
        "who": "me",
        "when": 0,
        "under": list(under),
        "about": list(about),
        "scope": list(scope),
        "cites": list(cites),
        "course": course,
        "standing": standing,
        "pruned": False,
    }


def written(*rows):
    """The rows in the order they were made, which is the order they are drawn
    in — so the ids have to say so."""
    for nth, row in enumerate(rows):
        row["id"] = nth
    return list(rows)


def a_fan():
    """One question with three variants under it, the shape an investigation
    has: one idea, several things tried from it."""
    return written(
        move("q", "question", "does more capacity help?", standing="open"),
        move("a", "attempt", "twice the width", under=["q"]),
        move("b", "attempt", "twice the depth", under=["q"]),
        move("c", "attempt", "both at once", under=["q"]),
    )


def test_a_parent_is_centred_over_its_children_span_and_not_their_average():
    # An uneven fan drawn on the average leans and looks like it is falling
    # over. `a` opens a branch of its own, so the span is not the average.
    rows = written(
        move("q", "question", "does more capacity help?"),
        move("a", "attempt", "twice the width", under=["q"]),
        move("a1", "finding", "recall moved", under=["a"]),
        move("a2", "finding", "and precision did not", under=["a"]),
        move("b", "attempt", "twice the depth", under=["q"]),
    )
    placed, _ = reasoning.cards(rows)
    where = {card.name: card for card in placed}

    assert where["a"].y == (where["a1"].y + where["a2"].y) / 2
    assert where["q"].y == (where["a"].y + where["b"].y) / 2
    # And that is not where their average is: `a` sits between two rows and `b`
    # takes a third, so an average would put `q` above the middle of the fan.
    assert where["q"].y != (where["a"].y + where["b"].y + where["a1"].y) / 3


def test_a_lane_is_never_handed_out_twice():
    # Freeing one when a branch ends looks thrifty and stacks three variants
    # into a single row pretending to be one history.
    placed, _ = reasoning.cards(a_fan())
    rows = [card.y for card in placed if card.kind == "attempt"]

    assert len(set(rows)) == 3


def test_siblings_come_out_in_the_order_they_were_made():
    placed, _ = reasoning.cards(a_fan())
    where = {card.name: card for card in placed}

    assert where["a"].y < where["b"].y < where["c"].y


def test_depth_grows_to_the_right_and_a_second_parent_pushes_a_move_further():
    # A move under two questions sits to the right of **both**, or an edge from
    # the deeper one would run backwards and read as an arrow the other way.
    rows = written(
        move("q", "question", "does more capacity help?"),
        move("deep", "question", "and does it read better?", under=["q"]),
        move("both", "attempt", "wider and deeper at once", under=["q", "deep"]),
    )
    placed, edges = reasoning.cards(rows)
    where = {card.name: card for card in placed}

    assert where["q"].x < where["deep"].x < where["both"].x
    # Drawn once and pointed at twice: the branch is not repeated.
    assert ("q", "both", "again") in edges or ("deep", "both", "again") in edges
    assert len([card for card in placed if card.name == "both"]) == 1


def test_a_move_nobody_hung_anywhere_is_drawn():
    # Work waiting for a place, not a move that hides.
    rows = written(
        move("q", "question", "does more capacity help?"),
        move("loose", "question", "and what about the checkpoint?"),
    )
    placed, _ = reasoning.cards(rows)

    assert {card.name for card in placed} == {"q", "loose"}


def test_a_decision_is_drawn_beside_what_it_abandons_and_not_floating():
    # Its scope is the only thing tying it to the line it ended.
    rows = written(
        move("a", "attempt", "cross-attention"),
        move("drop", "decision", "too slow", about=["a"], scope=["a"], course="abandon"),
    )
    placed, edges = reasoning.cards(rows)
    where = {card.name: card for card in placed}

    assert ("a", "drop", "under") in edges
    assert where["drop"].x > where["a"].x


def test_a_folded_line_says_how_many_it_hides_and_why_and_opens_when_it_is_not_given():
    rows = written(
        move("q", "question", "does cross-attention help?"),
        move("a", "attempt", "tried it", under=["q"]),
    )
    folds = [
        {
            "root": "q",
            "by": "drop",
            "course": "abandon",
            "why": "it costs more than it gives",
            "hides": ["q", "a"],
        }
    ]

    shut, _ = reasoning.cards(rows, (), folds)
    assert [card.name for card in shut] == ["q"]
    assert shut[0].hides == 2
    assert shut[0].why == "it costs more than it gives"

    # Folding is what you hand it, so handing it nothing opens everything —
    # which is what makes the reader's folding an app's and not this one's.
    open_, _ = reasoning.cards(rows)
    assert [card.name for card in open_] == ["q", "a"]
    assert open_[0].hides is None


def test_how_a_question_stands_is_written_on_it_and_not_only_coloured():
    # A colour is never the only place a finding lives.
    rows = written(move("q", "hypothesis", "width is the bottleneck", standing="depends"))
    placed, _ = reasoning.cards(rows)

    assert placed[0].said == "depends"


def test_combines_is_an_edge_of_its_own_and_is_not_under():
    # It says this attempt **is** the composition of those, which is what lets
    # *each worked alone, together they cancel* read as what it is.
    rows = written(
        move("a", "attempt", "wider"),
        move("b", "attempt", "deeper"),
        move("both", "attempt", "wider and deeper", under=["a"]),
    )
    says = [{"from": "both", "says": "combines", "to": "b", "scope": [], "partly": False}]
    _, edges = reasoning.cards(rows, says)

    assert ("b", "both", "combines") in edges
    assert ("b", "both", "under") not in edges


def test_asking_for_a_name_nobody_has_says_so():
    with pytest.raises(ValueError, match="nowhere"):
        reasoning.cards(a_fan(), under="nowhere")


def test_an_empty_investigation_draws_a_statement_and_not_an_exception(tmp_path):
    figure = reasoning.figure(Store(str(tmp_path)), tree="nothing-here")

    assert figure.layout.annotations[0].text.startswith("Nothing has been written")


# ── Against a store somebody wrote from the terminal ─────────────────────────


@pytest.fixture(scope="module")
def investigated(tmp_path_factory):
    """A small investigation, written the only way one is written: from the
    command line, one move at a time."""
    if not BINARY.exists():
        pytest.skip("`somatize-tree` is not built; `cargo build -p somatize-tree`")
    at = tmp_path_factory.mktemp("investigation")
    repo, store = at / "repo", at / "store"
    repo.mkdir()
    git = ["git", "-C", str(repo)]
    subprocess.run([*git, "init", "-q", "."], check=True)
    subprocess.run([*git, "config", "user.email", "you@example.com"], check=True)
    subprocess.run([*git, "config", "user.name", "You"], check=True)
    (repo / "note.txt").write_text("a repository with a history and no graph\n")
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-qm", "the base"], check=True)
    commit = subprocess.run(
        [*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()

    def wrote(*args):
        subprocess.run(
            [str(BINARY), *args, "--repo", str(repo), "--store", str(store), "--tree", "t"],
            check=True,
            capture_output=True,
        )

    wrote("ask", "capacity", "-m", "does more capacity help?")
    wrote("suppose", "wider", "-m", "width is the bottleneck", "--under", "capacity")
    wrote("tried", "wider-2x", "-m", "twice the width", "--under", "wider", "--cites", commit)
    wrote("found", "it-moved", "-m", "recall moved four points", "--under", "wider-2x")
    wrote("says", "it-moved", "validates", "wider")
    wrote("tried", "never-ran", "-m", "four times the width", "--under", "capacity")
    wrote("decide", "abandon", "drop-4x", "-m", "it will not fit", "--about", "never-ran")
    return Store(str(store))


def test_every_cross_reference_arrives_as_a_name(investigated):
    # The id stops identifying a move the moment nobody holds it in a variable,
    # and reading a store back is exactly that moment.
    rows = {one["name"]: one for one in reasoning.moves(investigated, tree="t")}

    assert rows["wider-2x"]["under"] == ["wider"]
    assert rows["it-moved"]["under"] == ["wider-2x"]
    assert rows["capacity"]["id"] == 0


def test_a_standing_is_derived_and_only_a_question_or_a_hypothesis_has_one(investigated):
    said = reasoning.standing(investigated, tree="t")

    assert said == {"capacity": "open", "wider": "validated"}
    rows = {one["name"]: one for one in reasoning.moves(investigated, tree="t")}
    assert rows["wider-2x"]["standing"] is None


def test_an_attempt_nobody_ran_is_folded_although_it_cites_no_commit(investigated):
    folded = reasoning.folds(investigated, tree="t")

    assert [one["root"] for one in folded] == ["never-ran"]
    assert folded[0]["hides"] == ["never-ran"]
    assert folded[0]["why"] == "it will not fit"


def test_a_commit_says_which_moves_cite_it_with_no_index_saying_so(investigated):
    cited = reasoning.cites(investigated, tree="t")

    assert list(cited) == ["commit"]
    assert list(cited["commit"].values()) == [["wider-2x"]]


def test_a_scope_reaches_down_the_dag_and_a_name_nobody_has_is_refused(investigated):
    assert reasoning.covered(investigated, tree="t", by=["wider"]) == [
        "wider",
        "wider-2x",
        "it-moved",
    ]
    with pytest.raises(ValueError):
        reasoning.covered(investigated, tree="t", by=["nowhere"])


def test_what_was_said_arrives_with_where_it_holds(investigated):
    said = reasoning.says(investigated, tree="t")

    assert said == [
        {"from": "it-moved", "says": "validates", "to": "wider", "scope": [], "partly": False}
    ]


def test_the_reasoning_is_drawn_from_what_is_stored_with_nothing_run_again(investigated):
    pytest.importorskip("plotly")
    figure = reasoning.figure(investigated, tree="t")

    # Six moves; `never-ran` comes folded, and the decision that abandoned it
    # is inside its own fold — which is right: its reason is on the fold line.
    assert len(figure.layout.shapes) == 5
    assert any("⋯1 folded" in note.text for note in figure.layout.annotations)
