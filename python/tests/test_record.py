"""What happened, read back.

A `Recorder` writes; these read. The tests that matter are about **what each
call costs**, because that is the whole design: the summary is in the record and
the detail is in the blob, so a progress view scans and only the per-node
breakdown fetches.
"""

import pytest

from soma_next import Graph, Node, Recorder, Store
from soma_next.record import curve, curve_costs, facts, forwards, nodes, runs


class Add(Node):
    def __init__(self, how_much):
        self.how_much = how_much

    def forward(self, x, ctx):
        return x + self.how_much


class Boom(Node):
    def forward(self, x, ctx):
        raise ValueError("I broke")


@pytest.fixture
def g():
    return Graph.somatize(Add(1).named("a") >> Add(10).named("b"))


@pytest.fixture
def store(tmp_path):
    return Store(str(tmp_path))


def trained(g, store, steps=3, **how):
    """A run of `steps` forwards with a loss said after each, like a trainer's."""
    recorder = Recorder(store, run="tuesday", **how)
    for step in range(steps):
        g.forward(0.0, watching=recorder)
        recorder({"fact": "loss", "value": 1.0 / (step + 1)})
    return recorder


# ── What is in here at all ──


def test_a_store_says_which_runs_it_holds(g, store):
    trained(g, store)

    (run,) = runs(store)
    assert run["run"] == "tuesday"
    assert run["forwards"] == 3
    assert run["broke"] == 0
    assert run["took_us"] > 0


def test_what_is_not_a_run_is_not_read_as_one(g, store):
    # A store holds whatever anybody put in it — a cache, a study, artifacts —
    # so belonging to a run is a question and not an assumption.
    store.bind("run/not-a-forward", store.put(b"x"))
    store.bind("spam/trial/0/0", store.put(b"y"))
    trained(g, store, steps=1)

    assert [run["run"] for run in runs(store)] == ["tuesday"]


def test_a_store_nobody_recorded_into_says_so_rather_than_failing(store):
    assert runs(store) == []
    assert forwards(store, run="never") == []


# ── Step by step, which is the free one ──


def test_every_forward_comes_back_in_order_with_its_numbers_as_numbers(g, store):
    trained(g, store)

    rows = forwards(store, run="tuesday")
    assert [row["forward"] for row in rows] == [0, 1, 2]
    assert all(isinstance(row["took_us"], int) for row in rows)
    assert all(row["nodes"] == 2 for row in rows)


def test_a_forward_that_broke_is_visible_without_reading_its_blob(store):
    g = Graph.somatize(Add(1).named("a") >> Boom().named("boom"))
    recorder = Recorder(store, run="tuesday")
    with pytest.raises(Exception):
        g.forward(0.0, watching=recorder)

    (row,) = forwards(store, run="tuesday")
    assert row["state"] == "broke"
    assert row["nodes"] == 1, "only the one that got to run"


# ── The curve, which is the one everybody draws ──


def test_a_summarised_loss_is_read_with_one_scan(g, store):
    trained(g, store, summarising=["loss"])

    assert curve_costs(store, run="tuesday") == "scan"
    assert curve(store, run="tuesday") == [(0, 1.0), (1, 0.5), (2, 1 / 3)]


def test_and_without_summarising_it_is_still_read_and_says_it_cost_more(g, store):
    # It does not refuse — the numbers are there. It says which of the two it
    # did, because a reader that is quietly a thousand times slower is worse
    # than one that says so.
    trained(g, store)

    assert curve_costs(store, run="tuesday") == "fetch"
    assert curve(store, run="tuesday") == [(0, 1.0), (1, 0.5), (2, 1 / 3)]


def test_anything_a_fact_carries_can_be_a_curve(g, store):
    # `took_us` is in the record already, so how long each step took is free.
    trained(g, store)

    drawn = curve(store, run="tuesday", of="took_us")
    assert [forward for forward, _ in drawn] == [0, 1, 2]
    assert all(took > 0 for _, took in drawn)


# ── The detail, and the aggregate ──


def test_the_facts_of_one_forward_are_what_was_seen_live(g, store):
    # The same dicts a `watching=` callable was handed. One shape, so what you
    # looked at live is what you read back.
    seen = []
    recorder = Recorder(store, run="tuesday")
    g.forward(0.0, watching=[recorder, seen.append])

    assert facts(store, run="tuesday", forward=0) == seen


def test_a_forward_that_is_not_there_is_nothing_and_not_a_failure(g, store):
    # A run that is still going has not written the next one yet.
    trained(g, store, steps=1)

    assert facts(store, run="tuesday", forward=9) is None


def test_who_spent_the_time_is_added_up_across_the_forwards(g, store):
    trained(g, store)

    (first, second) = nodes(store, run="tuesday")
    assert {first["node"], second["node"]} == {"a", "b"}
    assert first["ran"] == 3
    assert first["took_us"] >= second["took_us"], "the slowest first"
    assert first["mean_us"] == first["took_us"] / 3


def test_only_the_last_forwards_can_be_asked_for_because_each_costs_a_fetch(g, store):
    trained(g, store, steps=5)

    assert nodes(store, run="tuesday", last=2)[0]["ran"] == 2


def test_a_node_that_was_read_back_is_not_averaged_as_a_fast_one(tmp_path):
    # A hit took no time at all. Averaging over it would say a cache makes a
    # node fast rather than saying it did not happen.
    g = Graph.somatize(Add(1).named("a").frozen().cached())
    store, kept = Store(str(tmp_path / "runs")), str(tmp_path / "cache")
    recorder = Recorder(store, run="tuesday")

    for _ in range(2):
        g.forward(0.0, store=kept, watching=recorder)

    (a,) = nodes(store, run="tuesday")
    assert a["ran"] == 1, "the second one was read back"
    assert a["recalled"] == 1
    assert a["mean_us"] == a["took_us"], "over the one time it really ran"
