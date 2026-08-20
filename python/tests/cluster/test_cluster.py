"""A cluster that is not a metaphor: four containers, each its own machine.

Everything else in this suite runs workers as subprocesses of the test, which
proves the protocol and cannot prove the rest: those workers share this
filesystem, this interpreter and these installed packages. Here they do not.
Each one is an image with **the wheel and nothing else** — no clone of the
project, no `PYTHONPATH` of yours, its own hostname and its own network — and
the client is pytest, outside, which is the real shape of it.

What only becomes provable here:

| | with subprocesses | with containers |
|---|---|---|
| "the worker cannot import your module" | a trick with `sys.path` | it really cannot |
| "another version of the code" | the file rewritten underneath | another image, another mount |
| "two hosts" | two processes on the loopback | two network namespaces |
| "a store shared between workers" | a common `/tmp` | a volume between machines |
| "a worker with a GPU" | — | a device, and torch, in one of them |

Opt-in: `SOMA_CLUSTER=1 python -m pytest tests/cluster -q`.
"""

from __future__ import annotations

import os
import time

import pytest

from soma_next import Graph

from . import nodes  # the module the containers cannot see  # noqa: E402

pipeline = pytest.importorskip("pipeline")  # the one they have mounted


def hostnames(*answers):
    """Which machine each of these came from."""
    return {answer["host"] for answer in answers}


# ── The code goes over the wire ──


def test_a_worker_with_none_of_your_code_runs_your_nodes(sends_the_code):
    # `Shout` lives in a file no image has and no mount reaches. If it ran over
    # there, it travelled.
    g = Graph.somatize(nodes.Shout().at("a"))

    out = g.forward("hello", workers={"a": sends_the_code("a")})

    assert out["text"] == "HELLO"
    assert out["host"] != os.uname().nodename, "it ran here"


def test_a_worker_that_has_not_got_your_code_says_so(has_the_code):
    # The other half, and the reason `project` is the default: it does not guess.
    # Sending names to a worker that cannot resolve them has to fail **saying
    # which name**, not somewhere inside a `loads`.
    g = Graph.somatize(nodes.Shout().at("a"))

    with pytest.raises(ValueError) as raised:
        g.forward("hello", workers={"a": has_the_code("a")})

    assert "nodes" in str(raised.value), raised.value


def test_two_hosts_are_two_machines(sends_the_code):
    g = Graph.somatize(nodes.Shout().at("a") >> nodes.Wrap().at("b"))

    out = g.forward("hello", workers={"a": sends_the_code("a"), "b": sends_the_code("b")})

    assert out["text"] == "[HELLO]"
    assert out["before"] != out["host"], "both slices ran in the same container"


def test_what_one_produces_reaches_the_other(sends_the_code):
    # The values cross two wires: up to the client and down to the next worker,
    # because only what a slice reads and does not produce travels with it.
    g = Graph.somatize(nodes.Shout().at("a") >> nodes.Wrap().at("b"))

    out = g.forward("crossing", workers={"a": sends_the_code("a"), "b": sends_the_code("b")})

    assert out["text"] == "[CROSSING]"


def test_two_branches_on_two_machines_really_overlap(sends_the_code):
    # Two nodes of a wave, one per container, each taking `Slow.SECONDS`. In one
    # process they would take twice that — the GIL is what `test_waves.py`
    # already documents. In two, they do not.
    g = Graph.somatize(nodes.Slow().named("left").at("a") | nodes.Slow().named("right").at("b"))

    started = time.monotonic()
    out = g.forward(None, workers={"a": sends_the_code("a"), "b": sends_the_code("b")})
    took = time.monotonic() - started

    assert len(hostnames(out["left"], out["right"])) == 2
    assert took < nodes.Slow.SECONDS * 1.9, f"they queued up: {took:.2f}s"


def test_the_driver_travels_with_the_nodes_and_serves_where_they_run(sends_the_code):
    # A node that answers `Await` finishes over there or not at all, so the
    # driver rides in the same artifact. There is no driver in that container
    # other than the one that arrived.
    g = Graph.somatize(nodes.Asks().at("a"))

    out = g.forward(None, workers={"a": sends_the_code("a")}, driver=nodes.Answers("indeed"))

    assert out["heard"] == "indeed: are you there?"


# ── The worker that already has the project ──


def test_names_and_state_are_enough_when_the_worker_has_the_code(has_the_code):
    # Forty bytes instead of a pickle, and no coupling between interpreters:
    # `pipeline` is mounted in that container, so only its **name** travels.
    g = Graph.somatize(pipeline.Scale().at("a"))

    assert g.forward(21.0, workers={"a": has_the_code("a")}) == 42.0


def test_another_version_of_the_code_stops_the_run(has_the_code):
    # `worker-old` has the same module one version behind. Nothing is staged:
    # another file, another mount, another fingerprint.
    g = Graph.somatize(pipeline.Scale().at("old"))

    with pytest.raises(ValueError) as raised:
        g.forward(21.0, workers={"old": has_the_code("old")})

    said = str(raised.value)
    assert "Scale" in said, said


def test_with_lucky_it_runs_its_own_version_and_says_so(has_the_code, worker_logs):
    # The other half of the knob. Running a different version **in silence** is
    # what gets discovered three days later, so it is reported — and what comes
    # back is that worker's answer, not the one the graph was written against.
    g = Graph.somatize(pipeline.Scale().at("lucky"))

    assert g.forward(21.0, workers={"lucky": has_the_code("lucky")}) == 63.0
    assert "Scale" in worker_logs("worker-lucky")


# ── A store shared between machines ──


def test_what_one_worker_kept_another_one_reads(sends_the_code):
    # The two slices meeting: CU12 carries the work, CU13 keeps what it produced,
    # and the store is a volume both containers mount.
    #
    # `Stamp` answers a clock reading and the name of its machine. If the run on
    # `b` comes back with `a`'s reading and `a`'s hostname, `b` did not compute
    # anything: it read what `a` left there. And note what the client keeps —
    # nothing at all. It has no store; what travels is **what is remembered**.
    def stamped(where):
        graph = Graph.somatize(nodes.Stamp().frozen().cached().at(where))
        return graph.forward("one input", workers={where: sends_the_code(where)})

    first = stamped("a")
    second = stamped("b")

    assert second == first, "b answered with what a produced, down to the clock"


def test_a_different_input_is_not_the_same_answer(sends_the_code):
    # The other half, or the test above would pass with a broken cache that
    # answers the same for everything.
    def stamped(text):
        graph = Graph.somatize(nodes.Stamp().frozen().cached().at("a"))
        return graph.forward(text, workers={"a": sends_the_code("a")})

    assert stamped("one thing") != stamped("another thing")


def test_the_artifact_is_kept_where_both_of_them_can_see_it(sends_the_code, in_container):
    # The `have`/`want` with a `have` at last, and across machines: what `a` was
    # sent is in the volume `b` mounts. That a second worker is then **not sent
    # it** is what `transport`'s own tests pin down, with bytes that are not a
    # spec; from here what is worth checking is that the shelf is really shared.
    Graph.somatize(nodes.Shout().at("a")).forward(
        "provisioning", workers={"a": sends_the_code("a")}
    )

    seen_from_b = in_container(
        "worker-b", "sh", "-c", "cat /store/names/*/* | grep -o 'artifact:[^\"]*'"
    )

    assert "artifact:pickle:" in seen_from_b, seen_from_b


# ── The one with a GPU ──


def test_a_node_sent_to_the_gpu_lands_on_the_gpu(gpu, sends_the_code):
    # The placement crosses the wire and is **obeyed over there**: the core says
    # where, the node moves itself. Here the two halves are on different
    # machines, which is the only way to see that the device is not a local
    # notion that leaked into the plan.
    g = Graph.somatize(nodes.OnTheDevice().at("gpu").on("cuda:0"))

    out = g.forward(None, workers={"gpu": sends_the_code("gpu")})

    assert out["cuda"] == 1.0, "that container has no GPU"
    assert out["said"] == "cuda:0"
    assert out["landed"].startswith("cuda"), out["landed"]


def test_the_same_node_on_a_machine_without_one_is_told_nothing(gpu, sends_the_code):
    # And a node with no device is not "on the cpu": it is **unplaced**, which is
    # a different thing and is what `ctx.device is None` says.
    g = Graph.somatize(nodes.OnTheDevice().at("gpu"))

    out = g.forward(None, workers={"gpu": sends_the_code("gpu")})

    assert out["said"] == ""
    assert out["landed"] == "cpu"
