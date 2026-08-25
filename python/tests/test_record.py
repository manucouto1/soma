"""What happened, read back.

A `Recorder` writes; these read. The tests that matter are about **what each
call costs**, because that is the whole design: the summary is in the record and
the detail is in the blob, so a progress view scans and only the per-node
breakdown fetches.
"""

import sys

import pytest

from soma_next import Broker, Graph, Node, Recorder, Store
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


# ── The fleet, which is the record turned the other way up ──


def test_a_run_says_what_each_machine_did(tmp_path):
    # The record is written run -> forward -> node and *where* is an attribute
    # of a fact. `fleet` inverts it, because *what is this machine doing* is a
    # question nobody could ask of it.
    from soma_next import Broker, Worker
    from soma_next.record import fleet

    g = Graph.somatize(
        Add(1).named("a") >> Add(10).named("b").at("w1") >> Add(100).named("c").at("w2")
    )
    workers = Broker.embedded(
        {
            name: Worker.spawn(
                [sys.executable, "-m", "soma_next.worker"], mode="network", send=["test_record"]
            )
            for name in ("w1", "w2")
        }
    )
    store = Store(str(tmp_path))
    recorder = Recorder(store, run="fleet")
    for _ in range(3):
        g.forward(0.0, broker=workers, watching=recorder)

    said = {one["host"]: one for one in fleet(store, run="fleet")}

    assert set(said) == {"here", "w1", "w2"}
    assert said["here"]["nodes"] == ["a"], "a fact with no host ran here"
    assert said["here"]["slices"] == 0, "nothing was sent to this machine"
    assert said["w1"]["nodes"] == ["b"]
    assert said["w1"]["slices"] == 3, "one slice per forward"
    assert said["w2"]["ran"] == 3


def test_and_what_it_was_waited_on_for(tmp_path):
    # The column that only exists up here: the round trip minus what actually
    # ran over there — the wire, the queue and the codec. Neither half of the
    # subtraction belongs to a node, which is why no per-node view can say it.
    from soma_next import Broker, Worker
    from soma_next.record import fleet

    g = Graph.somatize(Add(1).named("a") >> Add(10).named("b").at("w1"))
    worker = Broker.embedded(
        {
            "w1": Worker.spawn(
            [sys.executable, "-m", "soma_next.worker"], mode="network", send=["test_record"]
            ),
        }
    )
    store = Store(str(tmp_path))
    for _ in range(2):
        g.forward(0.0, broker=worker, watching=Recorder(store, run="fleet"))

    away = next(one for one in fleet(store, run="fleet") if one["host"] == "w1")

    assert away["trip_us"] > away["took_us"], "a round trip is more than the work"
    assert away["waiting_us"] == away["trip_us"] - away["took_us"]


def test_a_machine_nobody_sent_anything_to_is_not_in_it(tmp_path):
    # There is no registry: a machine is in a fleet because it **did**
    # something. Standing one up and never using it leaves nothing to say, and
    # inventing a row for it would be inventing the coordinator CU15 removed.
    from soma_next.record import fleet

    g = Graph.somatize(Add(1).named("a"))
    store = Store(str(tmp_path))
    g.forward(0.0, watching=Recorder(store, run="alone"))

    said = fleet(store, run="alone")

    assert [one["host"] for one in said] == ["here"]


def test_the_fleet_is_drawn_working_against_waited_on(tmp_path):
    pytest.importorskip("plotly")
    from soma_next import Broker, Worker
    from soma_next.record import machines

    g = Graph.somatize(Add(1).named("a") >> Add(10).named("b").at("w1"))
    worker = Broker.embedded(
        {
            "w1": Worker.spawn(
            [sys.executable, "-m", "soma_next.worker"], mode="network", send=["test_record"]
            ),
        }
    )
    store = Store(str(tmp_path))
    g.forward(0.0, broker=worker, watching=Recorder(store, run="fleet"))

    figure = machines(store, run="fleet")

    assert [one.name for one in figure.data if one.name] == ["working", "waited on"]
    assert set(figure.data[0].y) == {"here", "w1"}


def test_a_machine_says_what_only_it_can_say(tmp_path):
    # Everything else about a worker can be worked out from what the client
    # wrote down. How loaded it is cannot: nobody on this end can see it, and it
    # is the half of *the health of the workers* that no scan answers.
    from soma_next import Broker, Worker
    from soma_next.record import fleet

    g = Graph.somatize(Add(1).named("a") >> Add(10).named("b").at("w1"))
    worker = Broker.embedded(
        {
            "w1": Worker.spawn(
            [sys.executable, "-m", "soma_next.worker"], mode="network", send=["test_record"]
            ),
        }
    )
    store = Store(str(tmp_path))
    recorder = Recorder(store, run="m")
    for _ in range(2):
        g.forward(0.0, broker=worker, watching=recorder)

    away = next(one for one in fleet(store, run="m") if one["host"] == "w1")

    assert away["served"] == 2, "it counts what it ran, and says so itself"
    assert away["up_us"] is not None
    assert away["cores"] is None or away["cores"] >= 1


def test_and_it_arrives_saying_which_host_without_anybody_attributing_it(tmp_path):
    # The reason it cost no message: `Answer::Saw` already carries a `Fact`, the
    # client already relays one to its watcher, and the engine already wraps
    # whatever comes back in `Elsewhere`. A flat carrier rides all of that.
    from soma_next import Broker, Worker

    g = Graph.somatize(Add(1).named("a") >> Add(10).named("b").at("w1"))
    worker = Broker.embedded(
        {
            "w1": Worker.spawn(
            [sys.executable, "-m", "soma_next.worker"], mode="network", send=["test_record"]
            ),
        }
    )
    seen = []

    g.forward(0.0, broker=worker, watching=seen.append)

    said = next(one for one in seen if one["fact"] == "machine")
    assert said["host"] == "w1", "a worker does not know its own name; we do"
    assert "node" not in said, "a machine is not a node"


def test_a_run_with_nobody_else_in_it_says_nothing_about_machines(tmp_path):
    # This machine does not measure itself. It is the one you can look at with
    # `top`, and inventing a reading for it would be the only row in the table
    # nobody had to send.
    from soma_next.record import fleet

    g = Graph.somatize(Add(1).named("a"))
    store = Store(str(tmp_path))
    g.forward(0.0, watching=Recorder(store, run="alone"))

    here = fleet(store, run="alone")[0]

    assert here["host"] == "here"
    assert here["busy"] is None and here["served"] is None


# ── The idle half, which is the one no connection can carry ──


def _standing(store, port, seconds="0.3"):
    """A worker on a port, writing readings on a clock, that nobody is using."""
    import subprocess
    import time

    said = subprocess.Popen(
        [
            sys.executable, "-m", "soma_next.worker",
            "--listen", f"127.0.0.1:{port}", "--store", str(store), "--reporting", seconds,
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    said.stdout.readline()
    time.sleep(float(seconds) * 3)
    return said


def test_a_worker_nobody_is_using_still_says_it_is_there(tmp_path):
    # The reason the clock exists. A worker only speaks down a wire when
    # somebody gives it work, so the machine sitting idle — the one you most
    # want to see in a fleet — would otherwise not be in the picture at all.
    from soma_next.record import standing

    store = Store(str(tmp_path))
    worker = _standing(tmp_path, 7741)
    try:
        said = standing(store)
    finally:
        worker.terminate()

    assert len(said) == 1, said
    one = next(iter(said.values()))
    assert one["served"] == "0", "it has done nothing, and says so"
    assert one["fact"] == "machine"


def test_and_the_name_the_graph_gave_it_is_joined_on(tmp_path):
    # A worker does not know it is `w1`, so it files under what it calls itself.
    # The two names are only ever in the same row on a reading that came down a
    # wire — where the client attributed it — and that is the join.
    from soma_next import Broker, Worker
    from soma_next.record import fleet

    store = Store(str(tmp_path))
    worker = _standing(tmp_path, 7742)
    try:
        g = Graph.somatize(Add(1).named("a") >> Add(10).named("b").at("w1"))
        away = Broker.embedded(
            {"w1": Worker.at("127.0.0.1:7742", mode="network", send=["test_record"])}
        )
        g.forward(0.0, broker=away, watching=Recorder(store, run="m"))

        said = {one["host"]: one for one in fleet(store, run="m")}
    finally:
        worker.terminate()

    assert set(said) == {"here", "w1"}, "and not one row per name for one machine"
    assert said["w1"]["quiet_s"] == 0, "it wrote, and it is the newest writer"
    assert said["w1"]["busy"] is not None


def test_a_machine_that_wrote_and_was_never_asked_is_there_under_its_own_name(tmp_path):
    # Which is the honest answer: there is a machine here writing, and the graph
    # never gave it a name because it never placed anything on it.
    from soma_next.record import fleet

    store = Store(str(tmp_path))
    worker = _standing(tmp_path, 7743)
    try:
        g = Graph.somatize(Add(1).named("a"))
        g.forward(0.0, watching=Recorder(store, run="m"))
        said = {one["host"]: one for one in fleet(store, run="m")}
    finally:
        worker.terminate()

    idle = next(one for host, one in said.items() if host != "here")
    assert idle["slices"] == 0 and idle["ran"] == 0
    assert idle["busy"] is not None, "it said what it is even though nobody asked"


def test_how_quiet_a_machine_is_is_measured_against_the_other_writers(tmp_path):
    # CU18's rule, and it is not a preference: those are two clocks on two
    # machines sharing a folder, and on a cluster they disagree by minutes as a
    # matter of course. Comparing writers with writers makes the drift cancel.
    from soma_next.record import standing

    store = Store(str(tmp_path))
    digest = store.put(b"")
    store.bind("machine/old", digest, {"fact": "machine", "served": "1"})
    store.bind("machine/new", digest, {"fact": "machine", "served": "2"})

    said = standing(store)

    assert said["new"]["quiet_s"] == 0, "the newest writer is the reference"
    assert said["old"]["quiet_s"] >= 0
    assert said["old"]["quiet_s"] >= said["new"]["quiet_s"]
