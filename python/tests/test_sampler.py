"""Where to look next, and what is being searched over.

The three schemes differ in what they look at — the space's shape, nothing, or
what already happened — and two of the three answer from the **index** alone,
which is what lets a study spread over a shared folder without a coordinator.
"""

import pytest

from soma_next.study import Sampler, Space


def space():
    return (
        Space()
        .real("lr", 1e-5, 1e-1, log=True)
        .int("batch", 16, 128)
        .choice("opt", ["adam", "sgd"])
    )


# ── The space ──


def test_it_is_built_up_and_every_call_gives_back_a_new_one():
    # So the same base can be handed to two studies without one of them growing
    # a knob the other did not ask for.
    base = Space().real("lr", 0.0, 1.0)
    wider = base.int("batch", 16, 128)

    assert len(base) == 1
    assert len(wider) == 2
    assert base.names() == ["lr"]


def test_the_knobs_keep_the_order_they_were_declared_in():
    assert space().names() == ["lr", "batch", "opt"]


def test_a_knob_with_nothing_to_draw_from_is_refused_where_it_was_written():
    with pytest.raises(ValueError, match="nothing to draw from"):
        Space().real("lr", 1.0, 0.0)
    with pytest.raises(ValueError, match="nothing to draw from"):
        Space().choice("opt", [])
    with pytest.raises(ValueError, match="above zero"):
        Space().real("lr", 0.0, 0.1, log=True)


def test_two_knobs_by_the_same_name_are_refused():
    with pytest.raises(ValueError, match="already a dimension"):
        Space().real("lr", 0.0, 1.0).int("lr", 1, 2)


# ── A point is a mapping, and it is the trial's name ──


def test_a_point_is_handed_straight_to_a_factory():
    point = Sampler.random().ask(space(), 0)

    assert set(point.keys()) == {"lr", "batch", "opt"}
    assert isinstance(point["lr"], float)
    assert isinstance(point["batch"], int)
    assert point["opt"] in ("adam", "sgd")
    assert dict(**point) == dict(point.items())


def test_a_point_writes_itself_down_because_that_is_the_trial_name():
    point = Sampler.grid(2).ask(space(), 0)

    assert str(point).startswith("lr=")
    assert str(point) != str(Sampler.grid(2).ask(space(), 1))


# ── The three schemes ──


def test_a_grid_walks_every_combination_and_then_runs_out():
    # `None` is how a `for` stops without being told a number.
    grid, s = Sampler.grid(2), space()
    total = grid.total(s)

    assert total == 2 * 2 * 2
    walked = {str(grid.ask(s, trial)) for trial in range(total)}
    assert len(walked) == total
    assert grid.ask(s, total) is None


def test_only_the_grid_knows_how_many_there_are():
    assert Sampler.random().total(space()) is None
    assert Sampler.tpe().total(space()) is None


def test_the_same_seed_and_index_give_the_same_point_however_it_is_asked_for():
    # The property that lets a machine which claimed trial 7 out of a shared
    # folder derive it without replaying the first six.
    how, s = Sampler.random(seed=42), space()

    assert str(how.ask(s, 7)) == str(how.ask(s, 7))
    assert str(how.ask(s, 7)) != str(how.ask(s, 8))
    assert str(Sampler.random(seed=1).ask(s, 0)) != str(Sampler.random(seed=2).ask(s, 0))


def test_what_is_drawn_stays_inside_every_knob():
    how, s = Sampler.random(seed=7), space()

    for trial in range(100):
        point = how.ask(s, trial)
        assert 1e-5 <= point["lr"] <= 1e-1
        assert 16 <= point["batch"] <= 128
        assert point["opt"] in ("adam", "sgd")


def test_a_logarithmic_knob_spreads_over_the_decades_and_not_over_the_line():
    # Drawn linearly, four fifths of 1e-5..1e-1 sits above 0.02 and a search
    # never sees a small learning rate at all.
    how, s = Sampler.random(seed=3), space()
    below = sum(how.ask(s, trial)["lr"] < 1e-3 for trial in range(400))

    assert 150 <= below <= 250


def test_tpe_goes_where_the_good_trials_were():
    s = Sampler.tpe(goal="min", startup=4, seed=11)
    knobs = space()
    finished = [(Sampler.random(seed=i).ask(knobs, 0), 9.0) for i in range(8)]
    # Three that did well, all with a small learning rate.
    good = Space().real("lr", 1e-5, 1e-1, log=True).int("batch", 16, 128)
    good = good.choice("opt", ["adam", "sgd"])
    for trial in range(3):
        point = Sampler.grid(2).ask(good, trial)
        finished.append((point, 0.1))

    proposals = [s.ask(knobs, trial, finished)["lr"] for trial in range(40)]
    assert sum(lr < 1e-3 for lr in proposals) > 20


def test_before_it_has_anything_to_learn_from_tpe_is_the_random_one():
    s = space()

    assert str(Sampler.tpe(startup=5, seed=11).ask(s, 2, [])) == str(
        Sampler.random(seed=11).ask(s, 2)
    )


def test_two_of_the_three_ignore_what_finished_and_that_is_why_there_are_three():
    s = space()
    finished = [(Sampler.random(seed=i).ask(s, 0), float(i)) for i in range(8)]

    for how in (Sampler.grid(2), Sampler.random(seed=0)):
        assert str(how.ask(s, 2)) == str(how.ask(s, 2, finished))

    guided = Sampler.tpe(startup=4, seed=0)
    assert str(guided.ask(s, 2, [])) != str(guided.ask(s, 2, finished))


# ── What is refused, and what is written down ──


def test_a_direction_that_is_not_one_is_caught_where_it_was_typed():
    with pytest.raises(ValueError, match="`min`"):
        Sampler.tpe(goal="minimise")


def test_nothing_to_search_is_nowhere_to_look():
    for how in (Sampler.grid(2), Sampler.random(), Sampler.tpe()):
        assert how.ask(Space(), 0) is None


def test_a_sampler_writes_itself_down_for_the_record_of_a_run():
    assert str(Sampler.grid(4)) == "grid:4"
    assert str(Sampler.random(seed=7)) == "random:7"
    assert repr(Sampler.tpe(seed=7)).startswith("Sampler(tpe:min:startup:10")

    all_of_them = [Sampler.grid(4), Sampler.grid(5), Sampler.random(), Sampler.tpe()]
    assert len({str(how) for how in all_of_them}) == 4
    assert len({Sampler.grid(4), Sampler.grid(4), Sampler.random()}) == 2
