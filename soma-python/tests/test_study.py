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

from somatize import Store
from somatize.study import (
    DONE,
    MAX,
    MIN,
    PRUNED,
    RUNNING,
    Sampler,
    Space,
    abandoned,
    curves,
    direction,
    finished,
    in_flight,
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


def ran(store, how, space, trial, me="m", state=DONE, score=None, reports=None,
        goal=None):
    """One whole trial, start to finish, for a machine called `me`."""
    point = how.ask(space, trial, [])
    if not take(store, point, study="s", trial=trial, me=me, goal=goal):
        return None
    report(
        store,
        point,
        reports if reports is not None else [1.0, 0.5, score if score else 0.25],
        study="s",
        trial=trial,
        me=me,
        state=state,
        goal=goal,
    )
    return point


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


class Said:
    """A record, made by hand, so a test can say when it was written."""

    def __init__(self, name, meta, when):
        self.name, self.meta, self.when, self.digest = name, list(meta.items()), when, "x"


class Folder:
    """A store that holds exactly the records a test hands it."""

    def __init__(self, *records):
        self.records = list(records)

    def bound(self):
        return self.records


def running_at(trial, point, when=0, who="other"):
    return Said(f"s/trial/{trial}/0", {"state": RUNNING, "point": str(point), "who": who}, when)


def done_at(trial, point, score, when=0):
    return Said(
        f"s/trial/{trial}/0",
        {"state": DONE, "point": str(point), "score": repr(score), "who": "m"},
        when,
    )


def test_a_trial_another_machine_is_holding_comes_back_with_no_score(store):
    # Which is what says *running*. Nothing is made up: `ask` puts it in the pile
    # to keep away from without letting it vote on how big the other pile is.
    how, knobs = Sampler.sobol(seed=0), space()
    ran, holding = how.ask(knobs, 0, []), how.ask(knobs, 1, [])
    folder = Folder(done_at(0, ran, 0.4), running_at(1, holding))

    assert in_flight(folder, knobs, study="s") == [(holding, None)]


def test_it_costs_a_scan_and_no_fetches_like_the_history_does(store):
    how, knobs = Sampler.sobol(seed=0), space()
    for trial in range(4):
        ran(store, how, knobs, trial)
    take(store, how.ask(knobs, 4, []), study="s", trial=4, me="other")

    counting = Counting(store)
    lied = in_flight(counting, knobs, study="s")

    assert len(lied) == 1
    assert counting.fetches == 0


def test_it_sends_a_guided_sampler_away_from_what_is_being_tried(store):
    # The point of all of it, end to end: what comes off the folder reaches
    # `ask` and moves it. Two promising regions and eight scored trials, which
    # is where the quota tips — the Rust test of the same name says why that is
    # the number that matters.
    import math

    knobs = space()
    scored = [
        (knobs.read("lr=0.09,batch=64,opt=adam"), 0.10),
        (knobs.read("lr=0.00011,batch=32,opt=sgd"), 0.11),
        (knobs.read("lr=0.08,batch=60,opt=adam"), 0.12),
        (knobs.read("lr=0.00013,batch=30,opt=sgd"), 0.13),
        (knobs.read("lr=0.005,batch=100,opt=adam"), 5.0),
        (knobs.read("lr=0.008,batch=90,opt=sgd"), 6.0),
        (knobs.read("lr=0.003,batch=110,opt=adam"), 7.0),
        (knobs.read("lr=0.002,batch=120,opt=sgd"), 8.0),
    ]
    folder = Folder(
        *[done_at(i, point, at) for i, (point, at) in enumerate(scored)],
        running_at(8, knobs.read("lr=0.085,batch=64,opt=adam")),
    )
    told = in_flight(folder, knobs, study="s")
    assert told == [(knobs.read("lr=0.085,batch=64,opt=adam"), None)]

    def busy(seen):
        return sum(
            abs(
                math.log(
                    float(
                        Sampler.tpe(goal="min", startup=4, seed=t)
                        .ask(knobs, t, seen)["lr"]
                    )
                )
                - math.log(0.085)
            )
            < 0.7
            for t in range(200)
        )

    assert busy(scored + told) < busy(scored), "being told did not move it away"


def test_the_schemes_that_look_at_nothing_are_unmoved_by_it(store):
    # Which is why handing it to every sampler is safe: four of the five ignore
    # what they are given, so the loop does not have to know which it has.
    knobs = space()
    told = [(knobs.read("lr=0.05,batch=64,opt=adam"), None)]

    for how in (Sampler.sobol(seed=0), Sampler.halton(seed=0), Sampler.random(seed=0),
                Sampler.grid(steps=3)):
        assert how.ask(knobs, 2, []) == how.ask(knobs, 2, told), str(how)


def test_a_trial_nobody_has_touched_for_a_while_stops_being_lied_about(store):
    how, knobs = Sampler.sobol(seed=0), space()
    folder = Folder(
        done_at(0, how.ask(knobs, 0, []), 0.4, when=10_000),
        running_at(1, how.ask(knobs, 1, []), when=100),
    )

    assert in_flight(folder, knobs, study="s", stale=100_000) != []
    assert in_flight(folder, knobs, study="s", stale=10) == []


def test_it_is_measured_against_the_others_and_not_against_this_clock(store):
    # Two machines sharing a folder are two clocks, and on a cluster they
    # disagree by minutes as a matter of course. Everything here was written long
    # before now, and none of it is stale: what is compared is writers with
    # writers.
    how, knobs = Sampler.sobol(seed=0), space()
    folder = Folder(
        done_at(0, how.ask(knobs, 0, []), 0.4, when=1),
        running_at(1, how.ask(knobs, 1, []), when=1),
    )

    assert in_flight(folder, knobs, study="s", stale=60) != []


def test_abandoned_says_which_ones_stopped_and_decides_nothing(store):
    how, knobs = Sampler.sobol(seed=0), space()
    folder = Folder(
        done_at(0, how.ask(knobs, 0, []), 0.4, when=10_000),
        running_at(1, how.ask(knobs, 1, []), when=100),
        running_at(2, how.ask(knobs, 2, []), when=10_000),
    )

    assert abandoned(folder, study="s", stale=10) == [(1, 0)]
    assert abandoned(folder, study="s", stale=100_000) == []


def test_and_a_study_where_everything_stopped_does_not_look_abandoned(store):
    # The honest hole, written down: staleness is relative, so if nothing is
    # writing there is nothing to be behind. It costs nothing — if nobody is
    # writing, nobody is asking this either.
    how, knobs = Sampler.sobol(seed=0), space()
    folder = Folder(
        running_at(1, how.ask(knobs, 1, []), when=100),
        running_at(2, how.ask(knobs, 2, []), when=100),
    )

    assert abandoned(folder, study="s", stale=10) == []


MACHINE = """
import sys, time
from somatize import Store
from somatize.study import DONE, Sampler, Space, report, take

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


def test_a_score_carries_the_direction_it_was_searched_in(store):
    # The reason this is in the record and not only in the script: `0.0837` is
    # a good score or a bad one and nothing in the number says which. Whoever
    # reads the study without the loop that wrote it has only the record.
    how, knobs = Sampler.sobol(seed=0), space()
    ran(store, how, knobs, 0, goal=MAX)

    assert direction(store, study="s") == MAX
    assert trials(store, knobs, study="s")[0]["goal"] == MAX


def test_a_trial_claimed_and_never_reported_still_says_which_way_it_looked(store):
    # `take` writes it too. A machine that claimed a trial and died leaves a
    # record, and that record is as readable as any other.
    how, knobs = Sampler.sobol(seed=0), space()

    take(store, how.ask(knobs, 0, []), study="s", trial=0, me="m", goal=MIN)

    assert direction(store, study="s") == MIN


def test_a_study_nobody_told_answers_none_rather_than_guessing_min(store):
    # The whole point. A default here would be a guess written into the store
    # that reads exactly like something somebody said.
    how, knobs = Sampler.sobol(seed=0), space()
    ran(store, how, knobs, 0)

    assert direction(store, study="s") is None
    assert "goal" not in dict(next(iter(store.bound())).meta)


def test_changing_your_mind_halfway_is_the_newest_record_and_not_a_vote(store):
    # The direction is not a fact about a trial: it is what the person running
    # the study currently means. So the newest one wins — and the old trials go
    # on saying what they were actually run for, which is worth having.
    how, knobs = Sampler.sobol(seed=0), space()
    ran(store, how, knobs, 0, goal=MIN)
    ran(store, how, knobs, 1, goal=MAX)

    assert direction(store, study="s") == MAX
    assert [one["goal"] for one in trials(store, knobs, study="s")] == [MIN, MAX]


def test_a_typo_is_caught_where_it_was_typed(store):
    # A study that wrote `minimize` into two thousand records is not a figure
    # that draws badly, it is a directory to migrate.
    how, knobs = Sampler.sobol(seed=0), space()
    point = how.ask(knobs, 0, [])

    with pytest.raises(ValueError, match="which way is better"):
        take(store, point, study="s", trial=0, me="m", goal="minimize")
    assert direction(store, study="s") is None, "and no record was written"
