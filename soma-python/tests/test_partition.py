"""Cutting the samples into folds, from the side the loop is written on.

Everything decided about a cut lives in Rust; what is tested here is that it
arrives in the shape a `for` expects — sklearn's `(train, test)` pairs — and
that what cannot be honoured is a `ValueError` before a single index comes out.
"""

import pytest

from soma_next.study import Partition


def held_out(folds):
    """Every index that is held out, across all the folds."""
    return sorted(i for _, test in folds for i in test)


def test_k_folds_are_a_partition_of_the_samples():
    folds = Partition.kfold(5).folds(20)

    assert len(folds) == 5
    assert held_out(folds) == list(range(20))
    for train, test in folds:
        assert len(train) + len(test) == 20
        assert not set(train) & set(test)


def test_it_yields_what_a_loop_written_against_sklearn_already_expects():
    # The pair is `(train, test)` and in that order, so moving a study over is
    # changing the import and nothing else.
    for train, test in Partition.kfold(4, shuffle=0).folds(12):
        assert len(test) == 3
        assert len(train) == 9


def test_the_same_seed_gives_the_same_cut():
    # What makes a fold reproducible from the record of a run, on any machine.
    assert Partition.kfold(4, shuffle=1).folds(40) == Partition.kfold(4, shuffle=1).folds(40)
    assert Partition.kfold(4, shuffle=1).folds(40) != Partition.kfold(4, shuffle=2).folds(40)


def test_stratifying_keeps_the_share_of_every_class():
    # Clumped on purpose: cut plainly, the first fold would be all of one class.
    classes = [0] * 8 + [1] * 4

    plain = Partition.kfold(2).folds(12)
    assert all(classes[i] == 0 for i in plain[0][1])

    for _, test in Partition.stratified(2).folds(12, classes=classes):
        assert sum(classes[i] for i in test) == 2


def test_grouping_never_puts_a_group_on_both_sides():
    groups = [1, 1, 1, 2, 2, 3, 3, 4, 4, 4]

    for _, test in Partition.grouped(2).folds(10, groups=groups):
        whole = {groups[i] for i in test}
        assert {i for i, g in enumerate(groups) if g in whole} == set(test)


def test_both_at_once_keeps_the_groups_whole_and_the_classes_even():
    classes = [0, 0, 1, 1, 0, 0, 1, 1]
    groups = [1, 1, 1, 1, 2, 2, 2, 2]

    for _, test in Partition.stratified_grouped(2).folds(8, classes=classes, groups=groups):
        assert len({groups[i] for i in test}) == 1
        assert sum(classes[i] for i in test) == 2


def test_time_series_never_trains_on_its_own_future():
    folds = Partition.time_series(3).folds(12)

    for train, test in folds:
        assert max(train) < min(test)
    assert [len(train) for train, _ in folds] == [3, 6, 9]


def test_a_gap_drops_what_sits_between_the_two_sides():
    # Purged and embargoed cross-validation, which is a parameter here and not
    # a scheme of its own.
    folds = Partition.time_series(2, gap=2).folds(9)

    assert folds[0] == ([0], [3, 4, 5])
    assert folds[1] == ([0, 1, 2, 3], [6, 7, 8])


def test_leave_one_out_is_k_equal_to_n():
    folds = Partition.kfold(6).folds(6)

    assert len(folds) == 6
    assert all(len(test) == 1 for _, test in folds)


def test_what_a_cut_needs_and_was_not_given_says_which_argument_supplies_it():
    with pytest.raises(ValueError, match="by_class"):
        Partition.stratified(2).folds(10)
    with pytest.raises(ValueError, match="in_groups"):
        Partition.grouped(2).folds(10)


def test_a_cut_that_cannot_be_honoured_fails_before_a_single_index_comes_out():
    with pytest.raises(ValueError, match="not a cut"):
        Partition.kfold(1).folds(10)
    with pytest.raises(ValueError, match="nothing in them"):
        Partition.kfold(20).folds(10)
    with pytest.raises(ValueError, match="cannot be in"):
        Partition.stratified(3).folds(10, classes=[0] * 9 + [1])
    with pytest.raises(ValueError, match="does not split"):
        Partition.grouped(3).folds(6, groups=[1, 1, 1, 2, 2, 2])


def test_one_key_per_sample_or_it_says_which_one_is_short():
    with pytest.raises(ValueError, match="one per sample"):
        Partition.stratified(2).folds(10, classes=[0, 1, 0])


def test_keys_that_are_there_and_are_not_needed_change_nothing():
    # What is missing fails, what is spare is ignored: it is what lets the same
    # data be cut two ways to compare them, and what makes stratifying by
    # accident impossible — the scheme asks for it, never the presence of `y`.
    bare = Partition.kfold(3, shuffle=4).folds(6)
    laden = Partition.kfold(3, shuffle=4).folds(6, classes=[0, 1] * 3, groups=[1, 1, 2, 2, 3, 3])

    assert bare == laden


def test_a_cut_writes_itself_down_so_a_key_can_be_made_of_it():
    # The reason it is an enum and not a class you subclass: without a name,
    # fold 3 of one cut and fold 3 of another are the same cache entry.
    assert str(Partition.kfold(5)) == "kfold:5"
    assert str(Partition.kfold(5, shuffle=7)) == "kfold:5:shuffled:7"
    assert str(Partition.time_series(5, gap=2)) == "timeseries:5:gap:2"
    assert repr(Partition.stratified(5)) == "Partition(stratified:5)"

    cuts = [Partition.kfold(5), Partition.kfold(5, shuffle=0), Partition.stratified(5)]
    assert len({str(c) for c in cuts}) == 3


def test_two_cuts_that_say_the_same_thing_are_the_same_cut():
    assert Partition.kfold(5) == Partition.kfold(5)
    assert Partition.kfold(5) != Partition.kfold(4)
    assert len({Partition.kfold(5), Partition.kfold(5), Partition.stratified(5)}) == 2


def test_how_many_folds_without_producing_them():
    assert Partition.kfold(5).k == 5
    assert Partition.time_series(3, gap=1).k == 3
