"""N machines that never speak to each other, searching one space together.

There is no server here, no port and no protocol: a folder they all mounted, and
`claim`. A trial is a number, `ask` is a function of that number, so a machine
that claims trial 7 works out where to look on its own — without replaying the
first six and without asking anybody.

The test that matters starts **real processes**, because two loops in one
interpreter never race the way two Slurm tasks do.
"""

import subprocess
import sys

import pytest

from soma_next import Store
from soma_next.study import (
    DONE,
    PRUNED,
    RUNNING,
    Sampler,
    Space,
    curves,
    finished,
    report,
    take,
    trials,
)

HOW_MANY = 24


def space():
    return (
        Space()
        .real("lr", 1e-5, 1e-1, log=True)
        .int("batch", 16, 128)
        .choice("opt", ["adam", "sgd"])
    )


@pytest.fixture
def store(tmp_path):
    return Store(str(tmp_path / "shared"))


def ran(store, how, space, trial, me="m", state=DONE, score=None, reports=None):
    """One whole trial, start to finish, for a machine called `me`."""
    point = how.ask(space, trial, [])
    if not take(store, point, study="s", trial=trial, me=me):
        return None
    report(
        store,
        point,
        reports if reports is not None else [1.0, 0.5, score if score else 0.25],
        study="s",
        trial=trial,
        me=me,
        state=state,
    )
    return point


# ── Handing the work out, which costs no message ──


def test_a_trial_somebody_claimed_is_not_claimable_twice(store):
    # The whole of how work is handed out: the loser simply goes to the next
    # number. Nothing is sent anywhere, so nothing can be lost in flight.
    how, knobs = Sampler.sobol(seed=0), space()
    point = how.ask(knobs, 7, [])

    assert take(store, point, study="s", trial=7, me="one") is True
    assert take(store, point, study="s", trial=7, me="other") is False


def test_two_studies_sharing_a_directory_are_two_studies(store):
    how, knobs = Sampler.sobol(seed=0), space()
    point = how.ask(knobs, 0, [])

    assert take(store, point, study="one", trial=0, me="m") is True
    assert take(store, point, study="other", trial=0, me="m") is True


def test_whatever_else_is_in_the_store_is_not_a_trial(store):
    # A store holds a cache, an artifact, another run. The scan has to ask
    # whether a name is one of ours, not assume it.
    how, knobs = Sampler.sobol(seed=0), space()
    store.bind("s/trial/not-a-number/0", store.put(b"x"))
    store.bind("s/trial/0", store.put(b"x"))
    store.bind("somebody/elses/cache", store.put(b"x"))
    ran(store, how, knobs, 0)

    assert len(trials(store, knobs, study="s")) == 1


# ── What a scan costs, which is the reason for the split ──


class Counting:
    """A store that says how many blobs were fetched through it."""

    def __init__(self, store):
        self.store, self.fetches = store, 0

    def get(self, digest):
        self.fetches += 1
        return self.store.get(digest)

    def __getattr__(self, name):
        return getattr(self.store, name)


def test_the_history_a_sampler_wants_comes_back_with_no_fetches_at_all(store):
    # Why the configuration and the score live in the record and the curve does
    # not. A hundred trials is one scan, not a hundred round trips.
    how, knobs = Sampler.sobol(seed=0), space()
    for trial in range(5):
        ran(store, how, knobs, trial)

    counting = Counting(store)
    history = finished(counting, knobs, study="s")

    assert len(history) == 5
    assert counting.fetches == 0


def test_and_the_curves_a_pruner_wants_are_what_costs_one_fetch_each(store):
    how, knobs = Sampler.sobol(seed=0), space()
    for trial in range(5):
        ran(store, how, knobs, trial)

    counting = Counting(store)
    drawn = curves(counting, study="s")

    assert [len(one) for one in drawn] == [3] * 5
    assert counting.fetches == 5


def test_a_machine_that_ran_none_of_them_rebuilds_the_whole_history(store):
    # The point of writing the configuration down instead of keeping it: whoever
    # reads the folder next did not have to be there.
    how, knobs = Sampler.sobol(seed=0), space()
    mine = [ran(store, how, knobs, trial) for trial in range(6)]

    somebody_else = Store(str(store).removeprefix("Store(").removesuffix(")"))
    history = finished(somebody_else, knobs, study="s")

    assert [str(point) for point, _ in history] == [str(point) for point in mine]


# ── What each state means, and that they are not the same ──


def test_a_pruned_trial_is_not_a_configuration_that_scored_badly(store):
    # Its score is real and it is **not** comparable: it was measured after
    # fewer epochs. A sampler told otherwise learns something untrue.
    how, knobs = Sampler.sobol(seed=0), space()
    ran(store, how, knobs, 0, state=DONE)
    ran(store, how, knobs, 1, state=PRUNED, reports=[9.0])

    assert len(finished(store, knobs, study="s")) == 1
    assert [one["state"] for one in trials(store, knobs, study="s")] == [DONE, PRUNED]


def test_the_curve_is_watchable_while_it_is_still_being_drawn(store):
    # Rewritten on every report, which is what makes a notebook on another
    # machine able to draw a trial that has not finished.
    how, knobs = Sampler.sobol(seed=0), space()
    point = how.ask(knobs, 0, [])
    take(store, point, study="s", trial=0, me="m")

    drawing = []
    for epoch in range(3):
        drawing.append(1.0 / (epoch + 1))
        report(store, point, drawing, study="s", trial=0, me="m")
        seen = trials(store, knobs, study="s")[0]
        assert seen["state"] == RUNNING
        assert seen["score"] is None

    assert curves(store, study="s") == []  # nothing has finished yet
    report(store, point, drawing, study="s", trial=0, me="m", state=DONE)
    assert curves(store, study="s") == [[1.0, 0.5, 1 / 3]]


def test_a_retry_is_the_next_attempt_and_whoever_reads_keeps_the_higher(store):
    # A trial whose machine died stays claimed for ever, because that is what a
    # claim is. Rescuing it is a claim of the next attempt, not a write over the
    # old one — which would be a race.
    how, knobs = Sampler.sobol(seed=0), space()
    point = how.ask(knobs, 0, [])
    take(store, point, study="s", trial=0, me="the one that died")

    assert take(store, point, study="s", trial=0, me="the next one", attempt=1) is True
    report(
        store, point, [0.1], study="s", trial=0, me="the next one",
        attempt=1, state=DONE,
    )

    assert [one["who"] for one in trials(store, knobs, study="s")] == ["the next one"]
    assert len(finished(store, knobs, study="s")) == 1


# ── And the one it is all for: four processes, one folder ──


MACHINE = """
import sys, time
from soma_next import Store
from soma_next.study import DONE, Sampler, Space, report, take

where, me, how_many = sys.argv[1], sys.argv[2], int(sys.argv[3])
store = Store(where)
space = (Space().real("lr", 1e-5, 1e-1, log=True)
                .int("batch", 16, 128)
                .choice("opt", ["adam", "sgd"]))
sampler = Sampler.sobol(seed=0)

mine = []
for trial in range(how_many):
    # Derived from the index and nothing else: this machine has not seen a
    # single one of the trials the others are running.
    point = sampler.ask(space, trial, [])
    if not take(store, point, study="s", trial=trial, me=me):
        continue
    time.sleep(0.01)
    report(store, point, [1.0 / (trial + 1)], study="s", trial=trial, me=me,
           state=DONE)
    mine.append(trial)
print(" ".join(str(one) for one in mine))
"""


def run_them(where, machines, how_many):
    """The same script on `machines` processes, all of them at once."""
    running = [
        subprocess.Popen(
            [sys.executable, "-c", MACHINE, where, f"m{which}", str(how_many)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for which in range(machines)
    ]
    took = []
    for which, machine in enumerate(running):
        out, err = machine.communicate(timeout=180)
        assert machine.returncode == 0, f"machine {which}: {err}"
        took.append([int(one) for one in out.split()])
    return took


def test_four_processes_over_one_folder_run_every_trial_exactly_once(tmp_path):
    # The use case. Nobody is coordinating and nobody is being told anything:
    # every machine walks the same numbers, and a claim settles who gets which.
    where = str(tmp_path / "shared")
    Store(where)

    took = run_them(where, machines=4, how_many=HOW_MANY)

    everything = sorted(one for machine in took for one in machine)
    assert everything == list(range(HOW_MANY)), "a trial ran twice or not at all"
    assert sum(1 for machine in took if machine) >= 2, "one machine took it all"


def test_and_what_they_searched_is_what_one_machine_alone_would_have(tmp_path):
    # The claim underneath: `ask` is a function of the index, so spreading the
    # study over four machines does not change **which** points get tried. If it
    # did, a distributed run and a local one would be two different searches.
    where = str(tmp_path / "shared")
    store, knobs = Store(where), space()

    run_them(where, machines=4, how_many=HOW_MANY)

    alone = Sampler.sobol(seed=0)
    assert [str(point) for point, _ in finished(store, knobs, study="s")] == [
        str(alone.ask(knobs, trial, [])) for trial in range(HOW_MANY)
    ]
