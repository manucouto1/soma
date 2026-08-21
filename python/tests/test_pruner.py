"""Giving up on a trial that is going badly, from the side the loop is on.

The thing this file is really defending is at the bottom: **nothing is asked of
the `Trainer`**. A pruner answers and the loop stops calling it, which is why
this slice added zero lines to level 2.
"""

import pytest

from soma_next import Done, Graph, Node, Opaque
from soma_next.study import Pruner
from soma_next.torch import Trainer, parameters

torch = pytest.importorskip("torch")
nn = torch.nn

IN, CLASSES = 4, 2


class Layer(Node):
    """The CU10 pattern, with the placement left out: this is not about devices."""

    def __init__(self, in_, out):
        self.lin = nn.Linear(in_, out)

    def forward(self, x, ctx):
        return Done(Opaque(self.lin(x)))

    def parameters(self):
        return list(self.lin.parameters())


def batches(n=4):
    torch.manual_seed(0)
    return [(torch.randn(6, IN), torch.randint(0, CLASSES, (6,))) for _ in range(n)]


def trainer():
    g = Graph.somatize(Layer(IN, CLASSES).named("layer"))
    return Trainer(
        g,
        objective=nn.functional.cross_entropy,
        optimizer=torch.optim.Adam(parameters(g), lr=0.05),
    )


# ── The three schemes, and what each judges against ──


def test_the_median_drops_what_is_behind_the_trials_that_finished():
    finished = [[3.0], [1.0], [2.0], [4.0]]
    median = Pruner.median(startup=1)

    assert median.verdict([5.0], finished) is not None
    assert median.verdict([2.0], finished) is None


def test_a_verdict_is_a_reason_or_nothing_which_is_what_an_if_wants():
    why = Pruner.median(startup=1).verdict([9.0], [[1.0], [2.0]])

    assert isinstance(why, str)
    assert "behind the bar" in why
    assert Pruner.median(startup=1).verdict([0.5], [[1.0], [2.0]]) is None


def test_percentile_is_the_share_that_is_kept_so_smaller_prunes_more():
    finished = [[1.0], [2.0], [3.0], [4.0]]

    assert Pruner.percentile(0, startup=1).verdict([2.0], finished) is not None
    assert Pruner.percentile(50, startup=1).verdict([2.0], finished) is None


def test_a_threshold_needs_no_other_trial_so_it_works_on_the_first_one():
    assert Pruner.threshold(upper=10.0).verdict([11.0]) is not None
    assert Pruner.threshold(upper=10.0).verdict([5.0]) is None


def test_what_blew_up_goes_under_every_scheme_even_during_the_warmup():
    nan = [float("nan")]

    assert "diverged" in Pruner.diverged().verdict(nan)
    assert "diverged" in Pruner.median(warmup=99, startup=99).verdict(nan, [])
    assert "diverged" in Pruner.patience(99).verdict(nan)


def test_patience_judges_the_trial_against_itself_and_nobody_else():
    # Doing far better than a field of 100s, and still going nowhere.
    field = [[100.0] * 4] * 5
    plateau = [1.0, 1.0, 1.0]

    assert Pruner.median(startup=1).verdict(plateau, field) is None
    assert Pruner.patience(2).verdict(plateau, field) is not None
    assert Pruner.patience(2).verdict([3.0, 2.0, 1.0]) is None


def test_a_delta_stops_noise_from_looking_like_progress():
    creeping = [5.0, 4.999, 4.998, 4.997]

    assert Pruner.patience(2).verdict(creeping) is None
    assert Pruner.patience(2, min_delta=0.01).verdict(creeping) is not None


def test_maximizing_is_said_and_not_guessed_from_the_numbers():
    accuracy = [0.9]
    finished = [[0.95], [0.97]]

    assert Pruner.median(goal="max", startup=1).verdict(accuracy, finished) is not None
    assert Pruner.median(goal="min", startup=1).verdict(accuracy, finished) is None


# ── What is refused, and where ──


def test_a_direction_that_is_not_one_is_caught_where_it_was_typed():
    # Not as a search that quietly optimised backwards for an afternoon.
    with pytest.raises(ValueError, match="`min`"):
        Pruner.median(goal="minimise")


def test_zero_patience_cannot_be_written_at_all():
    # It would prune every trial at its first report, improvement or not.
    with pytest.raises(ValueError):
        Pruner.patience(0)


def test_a_pruner_writes_itself_down_for_the_record_of_a_run():
    assert str(Pruner.median(warmup=2, startup=5)) == "percentile:50:min:warmup:2:startup:5"
    assert str(Pruner.diverged()) == "threshold:lower:none:upper:none"
    assert repr(Pruner.patience(3, goal="max")) == "Pruner(patience:3:delta:0:max)"

    rules = [Pruner.median(), Pruner.median(goal="max"), Pruner.diverged()]
    assert len({str(r) for r in rules}) == 3
    assert len({Pruner.median(), Pruner.median(), Pruner.diverged()}) == 2


# ── And the whole point: nothing is asked of the trainer ──


def test_a_pruned_trial_simply_stops_being_stepped():
    # No callback, no flag, no `trainer.stop()`. The loop breaks, and the
    # `Trainer` never finds out there was a pruner in the room.
    finished = [[0.0] * 10] * 5  # a field nothing is going to beat
    pruner = Pruner.median(warmup=1, startup=1)

    t, data, reported, why = trainer(), batches(), [], None
    for _ in range(10):
        reported.append(t.fit(data, epochs=1).loss)
        if why := pruner.verdict(reported, finished):
            break

    assert len(reported) == 2, "warmup covers the first report, the second is judged"
    assert "behind the bar" in why


def test_a_trial_that_is_holding_its_own_runs_to_the_end():
    # The same loop, the same trainer, a field it beats: nothing is cut short.
    finished = [[100.0] * 10] * 5
    pruner = Pruner.median(warmup=1, startup=1)

    t, data, reported = trainer(), batches(), []
    for _ in range(10):
        reported.append(t.fit(data, epochs=1).loss)
        if pruner.verdict(reported, finished):
            break

    assert len(reported) == 10
