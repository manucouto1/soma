"""What is asserted about the study in `searching.py`.

**The code being tested is not here.** It is in two files next to this one, and
they are the point — this file only starts them and checks what came out:

* `spam.py` — the domain: the messages, and the three nodes that classify them.
  What a user's own project would contain.
* `searching.py` — the study: the space, the sampler, the pruner, how a
  configuration becomes a graph, and the loop. **Read this one.** It is the whole
  use case in one screen and there is nothing test-shaped in it.

Everything else in this directory tests one property against containers. This
tests the use case they add up to, and it is the first test of level 3 with a
real pipeline under it rather than a tensor of noise.

Each machine gets **workers of its own**, and that is not a detail: a worker
holds one catalog, so two machines running different graphs against the same one
is the second of them being told to reconnect. What they share is the directory,
which is the only thing they should share. Both machines still say `.at("a")` and
`.at("gpu")` — what those names resolve to is each machine's business, which is
the whole reason a host is a name.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from somatize import Store
from somatize.study import DONE, PRUNED, finished, trials

from . import spam
from .searching import EPOCHS, STUDY, TRIALS, sampler, space

HERE = Path(__file__).resolve().parent

#: Which pair of workers each machine of the study gets. Two, because there are
#: two workers with torch in them and a machine searching needs one to itself.
PAIRS = [("a", "gpu"), ("b", "gpu-b")]

MESSAGES = 600


@pytest.fixture(scope="session")
def data():
    """The messages, fetched once. Skips if the hub cannot be reached — a suite
    that goes red because somebody's wifi did says nothing about this library."""
    pytest.importorskip("datasets")
    try:
        return spam.messages(MESSAGES)
    except Exception as why:  # noqa: BLE001 - a hub is a hub
        pytest.skip(f"`{spam.NAME}` could not be fetched: {why}")


@pytest.fixture(scope="session")
def searched(tmp_path_factory, cluster, two_with_torch, data):
    """Every machine, one directory, one study. Run once for the whole file."""
    where = str(tmp_path_factory.mktemp("study") / "shared")
    # The dataset goes in **once**, here, and every machine reads spans of it.
    spam.stored(Store(where), *data)

    running = [
        subprocess.Popen(
            [
                sys.executable, str(HERE / "searching.py"), where, f"m{which}",
                str(cluster[cheap]), str(cluster[with_torch]),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for which, (cheap, with_torch) in enumerate(PAIRS)
    ]

    took = []
    for which, machine in enumerate(running):
        out, err = machine.communicate(timeout=900)
        assert machine.returncode == 0, f"machine {which}:\n{err[-3000:]}"
        took.append(json.loads(out.splitlines()[-1]))
    return Store(where), took


def test_every_trial_ran_exactly_once_across_all_of_them(searched):
    # No queue, no messages, no coordinator. A claim is a claim, and that is the
    # whole of how eight trials get divided between the machines.
    _, took = searched

    everything = sorted(one["trial"] for machine in took for one in machine)

    assert everything == list(range(TRIALS)), "a trial ran twice or not at all"
    assert sum(1 for machine in took if machine) >= 2, "one machine took it all"


def test_and_they_searched_what_one_machine_alone_would_have(searched):
    # The claim underneath the whole design: `ask` is a function of the index, so
    # spreading a study over several machines does not change **which**
    # configurations get tried. If it did, a distributed run and a local one
    # would be two different searches and neither would mean anything.
    store, _ = searched

    tried = [str(one["point"]) for one in trials(store, space, study=STUDY)]

    assert tried == [str(sampler.ask(space, trial, [])) for trial in range(TRIALS)]


def test_the_record_says_which_machine_ran_which(searched):
    store, _ = searched

    who = {one["who"] for one in trials(store, space, study=STUDY)}

    assert who <= {f"m{which}" for which in range(len(PAIRS))}
    assert len(who) >= 2


def test_something_it_tried_actually_learnt_to_tell_spam_from_ham(searched):
    # Thirteen per cent of these messages are spam, so "always ham" is already
    # right 87% of the time and its loss sits around 0.39. A configuration that
    # gets well under that found something in the words.
    _, took = searched

    best = min(
        one["reports"][-1] for machine in took for one in machine if one["reports"]
    )

    assert best < 0.2, f"nothing the study tried learnt anything: best was {best}"


def test_the_configurations_are_not_all_the_same_one(searched):
    # A sampler that returns the same point every time would pass every test
    # above. Eight trials, eight configurations, spread over the space.
    store, _ = searched

    tried = [str(one["point"]) for one in trials(store, space, study=STUDY)]

    assert len(set(tried)) == TRIALS


def test_what_was_given_up_on_stopped_early_and_what_was_not_ran_to_the_end(searched):
    # A pruner stops nothing: it answers, and the loop stops calling `step`. So
    # the evidence it worked is the shape of the curves, not a flag.
    _, took = searched
    every = [one for machine in took for one in machine]

    for one in every:
        if one["state"] == PRUNED:
            assert len(one["reports"]) < EPOCHS, one
        else:
            assert one["state"] == DONE
            assert len(one["reports"]) == EPOCHS, one


def test_a_trial_that_was_given_up_on_is_not_a_configuration_that_scored_badly(searched):
    # Its score is real and it is **not** comparable: measured after fewer
    # epochs. What a sampler learns from is what ran to the end.
    store, _ = searched

    ran_out = [one for one in trials(store, space, study=STUDY) if one["state"] == DONE]

    assert len(finished(store, space, study=STUDY)) == len(ran_out)


def test_the_preprocessing_ran_where_there_is_no_torch_at_all(in_container, searched):
    # Which is the point of cutting the graph rather than sending the whole thing
    # to the big machine. `worker-a` is 193 MB and could not import torch if it
    # wanted to, and it is where every one of these messages was tokenised.
    said = in_container(
        "worker-a",
        "python",
        "-c",
        "import importlib.util; print(importlib.util.find_spec('torch'))",
    )

    assert said.strip() == "None", f"worker-a has torch after all: {said}"


def test_and_the_embedding_was_trained_on_the_machine_that_has_it(searched):
    # `embed` is a `Split`: it runs on the worker with torch and is trained
    # there, so its weights never came home. The loss coming down at all is the
    # proof the gradient reached it — nothing else could have moved it.
    _, took = searched
    every = [one["reports"] for machine in took for one in machine]

    assert any(
        len(reports) > 1 and reports[-1] < reports[0] * 0.9 for reports in every
    ), f"no trial's loss moved: the far half never learnt anything: {every}"
