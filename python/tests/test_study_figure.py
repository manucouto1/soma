"""A study, read and drawn: which knob mattered, and where the good trials are.

The tests that matter are about **what is counted**. A pruned score is real and
was measured after fewer epochs, so a figure that ranks it against a finished one
says a trial that was stopped early did badly, when all that is known is that it
was stopped.
"""

import math

import pytest

from soma_next import Store
from soma_next.study import (
    DONE,
    PRUNED,
    Space,
    coordinates,
    importance,
    influence,
    report,
    table,
    take,
)

pytest.importorskip("plotly")

STUDY = "widths"


@pytest.fixture
def space():
    return Space().real("lr", 1e-4, 1e-1, log=True).int("width", 8, 64).choice("opt", ["adam", "sgd"])


@pytest.fixture
def store(tmp_path):
    return Store(str(tmp_path))


def scored(store, space, points, states=None):
    """Writes trials with the scores given, so a figure has something to draw."""
    for trial, (point, score) in enumerate(points):
        p = space.read(point)
        take(store, p, study=STUDY, trial=trial, me="one")
        state = (states or {}).get(trial, DONE)
        report(store, p, [score * 2, score], study=STUDY, trial=trial, me="one", state=state)


def a_study(store, space, how_many=12, **rest):
    """One where `lr` decides the score and `width` does nothing at all."""
    points = []
    for i in range(how_many):
        lr = 10 ** (-4 + 3 * i / max(how_many - 1, 1))
        points.append((f"lr={lr},width={8 + i},opt={'adam' if i % 2 else 'sgd'}", math.log10(lr)))
    scored(store, space, points, **rest)


# ── Which knob mattered ──


def test_a_knob_that_decides_the_score_comes_out_near_one(store, space):
    a_study(store, space)

    said = dict(importance(store, space, study=STUDY))

    assert said["lr"] == pytest.approx(1.0), "the score is a function of it"
    assert said["width"] == pytest.approx(1.0), "and this one moved with it"


def test_a_knob_that_never_varied_is_zero_because_that_is_no_evidence(store, space):
    scored(
        store,
        space,
        [(f"lr={lr},width=16,opt=adam", lr) for lr in (1e-3, 1e-2, 1e-1)],
    )

    said = dict(importance(store, space, study=STUDY))

    assert said["width"] == 0.0
    assert said["opt"] == 0.0
    assert said["lr"] == pytest.approx(1.0)


def test_a_study_with_nothing_to_compare_says_nothing_rather_than_guessing(store, space):
    scored(store, space, [("lr=0.01,width=16,opt=adam", 1.0)])

    assert all(value == 0.0 for _, value in importance(store, space, study=STUDY))


def test_a_pruned_trial_does_not_vote(store, space):
    # It has a score and the score is real; it was measured after fewer epochs.
    # Counted, it would say `width` decides everything.
    scored(
        store,
        space,
        [
            ("lr=1e-3,width=8,opt=adam", 1.0),
            ("lr=1e-2,width=16,opt=adam", 2.0),
            ("lr=1e-1,width=64,opt=adam", 0.0),
        ],
        states={2: PRUNED},
    )

    said = dict(importance(store, space, study=STUDY))

    assert said["lr"] == pytest.approx(1.0), "only the two that finished"
    assert said["width"] == pytest.approx(1.0)


def test_the_biggest_comes_first(store, space):
    a_study(store, space)

    said = importance(store, space, study=STUDY)

    assert [value for _, value in said] == sorted((v for _, v in said), reverse=True)


# ── Drawn ──


def test_the_table_shows_the_pruned_ones_too_and_says_which(store, space):
    # They are what the study spent its time on: hiding them would make a run of
    # twelve look like a run of eight.
    a_study(store, space, states={1: PRUNED, 3: PRUNED})

    cells = table(store, space, study=STUDY).data[0].cells.values
    states = list(cells[1])

    assert states.count(PRUNED) == 2
    assert states.count(DONE) == 10
    assert "12 scored, 10 finished" in table(store, space, study=STUDY).layout.title.text


def test_the_table_has_a_column_per_knob_in_the_space(store, space):
    a_study(store, space)

    header = list(table(store, space, study=STUDY).data[0].header.values)

    assert header == ["<b>trial</b>", "<b>state</b>", "<b>lr</b>", "<b>width</b>",
                      "<b>opt</b>", "<b>score</b>"]


def test_the_influence_bars_are_the_numbers_importance_gives(store, space):
    a_study(store, space)

    said = dict(importance(store, space, study=STUDY))
    bars = influence(store, space, study=STUDY).data[0]

    assert dict(zip(bars.y, bars.x)) == pytest.approx(said)


def test_every_finished_trial_is_a_curve_and_a_pruned_one_is_not(store, space):
    a_study(store, space, how_many=6, states={0: PRUNED})

    figure = coordinates(store, space, study=STUDY)

    assert len(figure.data) == 5
    assert all(one.line.shape == "spline" for one in figure.data), "curves, not zigzags"


def test_a_knob_searched_over_orders_of_magnitude_gets_a_log_axis(store, space):
    # The original's rule, and measured rather than declared: a `Space` does not
    # say how a knob was searched.
    a_study(store, space)

    named = [one.text for one in coordinates(store, space, study=STUDY).layout.annotations]

    assert "lr (log)" in named
    assert "width" in named, "eight to twenty is not orders of magnitude"


def test_the_goal_decides_which_end_of_the_scale_is_good(store, space):
    # Getting it backwards is the quietest lie a figure can tell: everything is
    # drawn, nothing raises, and the region read as promising is the wrong one.
    a_study(store, space, how_many=4)

    best = min(range(4), key=lambda i: i)
    lower = coordinates(store, space, study=STUDY, goal="min").data[0].line.color
    higher = coordinates(store, space, study=STUDY, goal="max").data[0].line.color

    assert lower != higher, f"trial {best} is drawn the same either way"


def test_a_study_nobody_has_finished_is_a_statement_and_not_an_exception(store, space):
    a_study(store, space, how_many=3, states={0: PRUNED, 1: PRUNED, 2: PRUNED})

    figure = coordinates(store, space, study=STUDY)

    assert figure.data == ()
    assert "nothing has finished yet" in figure.layout.annotations[0].text
