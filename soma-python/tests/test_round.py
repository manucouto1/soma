"""A round that N processes finish together, with nobody in charge.

There is no server here, no port and no protocol: a folder they all mounted, and
`claim`. Everybody writes what they learnt, whoever finds the round complete
claims the averaging, and exactly one of them can win that — which is the whole
reason `claim` exists.

The tests that matter start **real processes**, because two clients in one
interpreter never race the way two Slurm tasks do.
"""

import subprocess
import sys
import time

import pytest

torch = pytest.importorskip("torch")

from somatize import Store  # noqa: E402
from somatize.torch import gather  # noqa: E402


def weights(value, n=4):
    """An export-shaped thing, as flat as it can be while still being one."""
    return {"body": {"0": torch.full((n,), float(value))}}


@pytest.fixture
def store(tmp_path):
    return Store(str(tmp_path / "shared"))


# ── One client at a time, where nothing races ──


def test_the_only_client_of_a_round_averages_it_itself(store):
    average = gather(store, weights(4), run="r", round=0, clients=1, mine=0)

    assert torch.equal(average["body"]["0"], torch.full((4,), 4.0))


def test_what_it_gives_back_is_published_for_everybody_else_to_find(store):
    gather(store, weights(4), run="r", round=0, clients=1, mine=0)

    assert store.recall("r/round/0/average") is not None


def test_a_client_that_arrives_after_the_average_finds_it_and_does_not_wait(store):
    # The straggler: it puts its round in, sees the average already there and
    # leaves with it. Its own contribution is lost for that round, which is what
    # being late means.
    store.keep("r/round/0/average", weights(99))

    average = gather(store, weights(1), run="r", round=0, clients=8, mine=3, within=0.0)

    assert torch.equal(average["body"]["0"], torch.full((4,), 99.0))


def test_two_runs_sharing_a_directory_are_two_runs(store):
    one = gather(store, weights(1), run="one", round=0, clients=1, mine=0)
    other = gather(store, weights(7), run="other", round=0, clients=1, mine=0)

    assert torch.equal(one["body"]["0"], torch.full((4,), 1.0))
    assert torch.equal(other["body"]["0"], torch.full((4,), 7.0))


def test_and_so_are_two_rounds_of_one(store):
    gather(store, weights(1), run="r", round=0, clients=1, mine=0)
    second = gather(store, weights(7), run="r", round=1, clients=1, mine=0)

    assert torch.equal(second["body"]["0"], torch.full((4,), 7.0))


# ── When somebody does not turn up ──


def test_it_gives_up_after_the_deadline_and_says_who_is_missing(store):
    started = time.monotonic()

    with pytest.raises(TimeoutError) as e:
        gather(
            store, weights(1), run="r", round=0, clients=4, mine=0,
            within=0.3, asking=0.05,
        )

    assert "client 1, 2, 3" in str(e.value)
    assert time.monotonic() - started < 5, "it waited far longer than it was told"


def test_and_everybody_in_with_no_average_is_a_different_thing_to_say(store):
    # Rarer and worse: the round is complete, so whoever claimed the averaging
    # died holding it. Nobody else will try, because a claim is a claim.
    for which in range(3):
        store.keep(f"r/round/0/client/{which}", weights(which))
    store.claim("r/round/0/averaging", store.put(b"the one that died"))

    with pytest.raises(TimeoutError) as e:
        gather(
            store, weights(0), run="r", round=0, clients=3, mine=0,
            within=0.2, asking=0.05,
        )

    assert "did not finish it" in str(e.value)
    assert "started again" in str(e.value)


# ── How much data each of them saw, which nobody has to be asked ──


def test_the_size_travels_in_the_record_so_the_averaging_can_weigh_by_it(store):
    store.keep("r/round/0/client/1", weights(0), {"size": "100"})

    average = gather(
        store, weights(10), run="r", round=0, clients=2, mine=0, size=900
    )

    assert torch.equal(average["body"]["0"], torch.full((4,), 9.0))


def test_and_with_nobody_saying_theirs_they_weigh_the_same(store):
    store.keep("r/round/0/client/1", weights(0))

    average = gather(store, weights(10), run="r", round=0, clients=2, mine=0)

    assert torch.equal(average["body"]["0"], torch.full((4,), 5.0))


def test_one_client_saying_its_size_and_another_not_is_not_half_a_weighting(store):
    # All or none: weighing three of four and guessing the fourth would be worse
    # than not weighing at all, and quieter.
    store.keep("r/round/0/client/1", weights(0))

    average = gather(
        store, weights(10), run="r", round=0, clients=2, mine=0, size=900
    )

    assert torch.equal(average["body"]["0"], torch.full((4,), 5.0))


# ── And the one it is all for: four processes, one folder ──


CLIENT = """
import sys, torch
from somatize import Store
from somatize.torch import gather

where, mine, clients, rounds, within = (
    sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
)
store = Store(where)
value = float(mine)
did = 0
for r in range(rounds):
    average = gather(store, {"body": {"0": torch.full((4,), value)}},
                     run="fed", round=r, clients=clients, mine=mine,
                     within=within, asking=0.05)
    value = float(average["body"]["0"][0]) + 1.0
    who = store.resolve(f"fed/round/{r}/averaging")
    if store.get(who.digest) == f"fed/{r}/{mine}".encode():
        did += 1
print(value, did)
"""


def them(where, mine, clients, rounds, within):
    return subprocess.Popen(
        [
            sys.executable, "-c", CLIENT, where,
            str(mine), str(clients), str(rounds), str(within),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def run_them(where, clients, rounds, within=60.0):
    """The same script on `clients` processes, all of them at once."""
    running = [them(where, mine, clients, rounds, within) for mine in range(clients)]
    said = []
    for mine, client in enumerate(running):
        out, err = client.communicate(timeout=180)
        assert client.returncode == 0, f"client {mine}: {err}"
        value, did = out.split()
        said.append((float(value), int(did)))
    return said


def test_four_processes_leave_every_round_with_the_same_number(tmp_path):
    # The use case. Four clients that never speak to each other: each starts at
    # its own number, and every round they all leave with the mean of the four.
    # Two rounds, so that the second one starting from the first's average is
    # part of what is checked.
    #
    # Round 0: mean(0,1,2,3) = 1.5, and everybody goes on from 2.5.
    # Round 1: mean(2.5, 2.5, 2.5, 2.5) = 2.5, and everybody ends at 3.5.
    where = str(tmp_path / "shared")
    Store(where)

    said = run_them(where, clients=4, rounds=2)

    assert [value for value, _ in said] == [3.5] * 4


def test_and_exactly_one_of_them_did_the_averaging_each_round(tmp_path):
    # What `claim` is for, seen from above: over two rounds there are two
    # averagings, and the four clients did two between them. Nobody arranged it.
    where = str(tmp_path / "shared")
    store = Store(where)

    said = run_them(where, clients=4, rounds=2)

    assert sum(did for _, did in said) == 2, said
    for r in range(2):
        assert store.resolve(f"fed/round/{r}/averaging") is not None


def test_and_a_client_that_never_starts_stops_the_others_by_name(tmp_path):
    # Three of the four turn up. The others say which one did not, rather than
    # waiting for a weekend.
    where = str(tmp_path / "shared")
    Store(where)

    running = [them(where, mine, clients=4, rounds=1, within=1.0) for mine in range(3)]
    said = [client.communicate(timeout=180) for client in running]

    assert all(client.returncode != 0 for client in running)
    for mine, (_, err) in enumerate(said):
        assert "client 3 never turned up" in err, f"client {mine}: {err}"
