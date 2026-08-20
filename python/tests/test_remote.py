"""Sending a slice of the graph to another process, from Python.

Two halves tested apart: **declaring** that a node runs away — which is `.at()`
and `place_at`, and starts no process — and **executing** it there, which does
stand up a real worker.

About the `register_pickle_by_value` below: cloudpickle serializes **by
reference** classes coming from an importable module, and **by value** those
defined in `__main__` or in an interactive session. That is correct and not a
whim: a notebook works without touching anything, and a class living in an
installed package has no business travelling. A test module is neither of those
things — it is importable, but not from the worker — so it has to be said.
"""

import os
import sys

import pytest

from soma_next import Await, Done, Graph, Node, Opaque, Worker

cloudpickle = pytest.importorskip("cloudpickle")
cloudpickle.register_pickle_by_value(sys.modules[__name__])


# ── Nodes that are going to travel ──


class Add(Node):
    def __init__(self, how_much):
        self.how_much = how_much

    def forward(self, x, ctx):
        return Done(x + self.how_much)


class Mean(Node):
    def forward(self, inputs, ctx):
        return Done(sum(inputs.values()) / len(inputs))


class WhereIRan(Node):
    def forward(self, x, ctx):
        return Done(float(os.getpid()))


class HowManyTimes(Node):
    """How many times **this object** has been called.

    It is what makes the `have`/`want` observable from here: if the second run
    resent the artifact, the worker would unpack a new object and the count
    would start over.
    """

    def __init__(self):
        self.times = 0

    def forward(self, x, ctx):
        self.times += 1
        return Done(float(self.times))


class Fail(Node):
    def forward(self, x, ctx):
        raise ValueError("I broke in the worker")


class Chatterbox(Node):
    """Prints to `stdout`, which is where the wire runs.

    With `flush=True` on purpose: without it, Python keeps what is printed in
    its buffer and releases it on process exit — once the answer has already
    travelled — so the test would pass just the same with the redirection
    removed. Verified.
    """

    def forward(self, x, ctx):
        print("hello from the worker", flush=True)
        return Done(x)


class Opaquely(Node):
    def forward(self, x, ctx):
        return Done(Opaque(object()))


class WhichDevice(Node):
    def forward(self, x, ctx):
        return Done(ctx.device)


class Ask(Node):
    """Asks for something before finishing. Needs a driver where it runs."""

    def forward(self, x, ctx):
        if ctx.turn == 0:
            return Await(["hello"])
        return Done(ctx.results[0])


class Shout:
    """A driver. Declared here, and not in `conftest`, so it travels by value."""

    def __init__(self, suffix=""):
        self.suffix = suffix

    def perform(self, requests):
        return [r.upper() + self.suffix for r in requests]


class WhereIServed:
    """A driver that answers with the pid of whoever served the request."""

    def perform(self, requests):
        return [float(os.getpid()) for _ in requests]


def generic(**how):
    """An empty worker that gets sent **the code**.

    `mode="network"` explicitly: the classes in this file exist in no worker's
    clone, so the `project` strategy — the default — could not resolve them. It
    is exactly the case `network` is for.

    And with no nodes: the graph sends them at run time, since it is the one
    that knows which go to each host.
    """
    return Worker.generic(mode="network", **how)


# ── Declaring, which starts nothing ──


def test_at_sends_a_node_to_a_host():
    g = Graph.somatize(Add(1).named("a") >> Add(2).named("b").at("w1"))
    assert g.hosts() == {"b": "w1"}


def test_at_sends_the_whole_piece():
    g = Graph.somatize((Add(1).named("a") >> Add(2).named("b")).at("w1"))
    assert g.hosts() == {"a": "w1", "b": "w1"}


def test_with_hosts_the_innermost_one_wins_too():
    g = Graph.somatize((Add(1).named("a").at("w1") >> Add(2).named("b")).at("w2"))
    assert g.hosts() == {"a": "w1", "b": "w2"}


def test_an_inner_device_does_not_stop_the_outer_host_from_arriving():
    # The counterexample that forced `.on` and `.at` to look at different
    # fields. With a single one, `a` would have ended up without a host for
    # having asked for a GPU.
    g = Graph.somatize((Add(1).named("a").on("meta") >> Add(2).named("b")).at("w1"))

    assert g.hosts() == {"a": "w1", "b": "w1"}
    assert g.devices() == {"a": "meta"}


def test_the_order_of_on_and_at_does_not_matter():
    one = Graph.somatize((Add(1).named("a") >> Add(2).named("b")).on("meta").at("w1"))
    other = Graph.somatize((Add(1).named("a") >> Add(2).named("b")).at("w1").on("meta"))

    assert one.hosts() == other.hosts()
    assert one.devices() == other.devices()


def test_place_at_by_hand_does_the_same():
    g = Graph()
    g.node("a", Add(1))
    g.place_at("a", "w1")
    assert g.hosts() == {"a": "w1"}


def test_place_at_on_a_node_that_does_not_exist_fails():
    g = Graph()
    with pytest.raises(ValueError, match="ghost"):
        g.place_at("ghost", "w1")


def test_the_plan_shows_the_trip():
    g = Graph.somatize(Add(1).named("a") >> Add(2).named("b").at("w1"))
    plan = g.plan()

    assert "Remote" in plan
    assert 'Host("w1")' in plan


def test_a_whole_chain_on_one_host_is_a_single_trip():
    g = Graph.somatize((Add(1).named("a") >> Add(2).named("b") >> Add(4).named("c")).at("w1"))
    assert g.plan().count("Remote") == 1


def test_without_hosts_the_plan_is_the_usual_one():
    g = Graph.somatize(Add(1).named("a") >> Add(2).named("b"))
    assert "Remote" not in g.plan()


def test_placing_devices_only_distributes_nothing():
    # The CU10 invariant: a device is inert as far as the traversal goes.
    g = Graph.somatize((Add(1).named("a") >> Add(2).named("b")).on("meta"))
    assert "Remote" not in g.plan()


# ── Executing there, with a real worker ──


def test_a_node_sent_away_runs_in_another_process():
    where = WhereIRan()
    g = Graph.somatize(where.named("where").at("w1"))
    w = generic()

    output = g.forward(None, workers={"w1": w})

    assert output != float(os.getpid()), "it ran here: the distribution gets nowhere"


def test_the_same_graph_undistributed_runs_here():
    where = WhereIRan()
    g = Graph.somatize(where.named("where"))
    assert g.forward(None) == float(os.getpid())


def test_what_is_produced_here_reaches_the_worker_and_what_is_there_comes_back():
    # The real seam: `a` runs here, `b` there and reads from `a`, `c` here again
    # and reads from `b`.
    a, b, c = Add(1), Add(10), Add(100)
    g = Graph.somatize(a.named("a") >> b.named("b").at("w1") >> c.named("c"))
    w = generic()

    assert g.forward(0, workers={"w1": w}) == 111


def test_a_fan_in_with_one_branch_away_gives_the_same_as_all_of_it_here():
    def net():
        return Add(1), Add(10), Add(100), Mean()

    s, l, r, j = net()
    here = Graph.somatize(s.named("s") >> (l.named("l") | r.named("r")) >> j.named("j"))
    expected = here.forward(0)

    s, l, r, j = net()
    away = Graph.somatize(s.named("s") >> (l.named("l").at("w1") | r.named("r")) >> j.named("j"))
    w = generic()

    assert away.forward(0, workers={"w1": w}) == expected


def test_two_workers_are_two_processes():
    one, other = WhereIRan(), WhereIRan()
    g = Graph.somatize(one.named("one").at("w1") | other.named("other").at("w2"))
    a, b = generic(), generic()

    output = g.forward(None, workers={"w1": a, "w2": b})

    assert output["one"] != output["other"]
    assert float(os.getpid()) not in output.values()


def test_the_artifact_is_sent_only_once():
    # `HowManyTimes` counts calls on **its** object. If the second `forward`
    # resent the artifact, the worker would unpack a new one and the count would
    # go back to 1.
    how_many = HowManyTimes()
    g = Graph.somatize(how_many.named("how_many").at("w1"))
    w = generic()

    assert [g.forward(None, workers={"w1": w}) for _ in range(3)] == [1.0, 2.0, 3.0]


def test_the_placement_travels_with_the_slice():
    # What `placement.rs` promised back in CU10, now crossing two seams:
    # Python → Rust → the wire → Rust → Python.
    q = WhichDevice()
    g = Graph.somatize(q.named("q").at("w1").on("meta"))
    w = generic()

    assert g.forward(None, workers={"w1": w}) == "meta"


def test_a_print_in_a_node_does_not_break_the_wire():
    # The protocol's messages go over the worker's `stdout`. The generic worker
    # redirects Python's `sys.stdout` to `stderr` for exactly this.
    chatterbox = Chatterbox()
    g = Graph.somatize(chatterbox.named("chatterbox").at("w1"))
    w = generic()

    assert g.forward("intact", workers={"w1": w}) == "intact"


# ── What goes wrong ──


def test_a_mode_that_is_not_one_of_the_two_is_rejected():
    with pytest.raises(ValueError, match="'project' or 'network'"):
        Worker.generic(mode="carrier-pigeon")


def test_a_node_that_asks_for_something_is_served_where_it_runs():
    # The driver travels in the artifact, like the nodes: the graph packs the
    # one this run was given and sends it to every worker it uses.
    g = Graph.somatize(Ask().named("ask").at("w1"))
    w = generic()

    assert g.forward(None, driver=Shout(), workers={"w1": w}) == "HELLO"


def test_without_a_driver_it_still_fails_where_it_runs():
    # The boundary that remains, and it is the same one as at home: a node that
    # asks for something needs someone to serve it. Not having packed one is
    # not "it cannot be sent away".
    g = Graph.somatize(Ask().named("ask").at("w1"))
    w = generic()

    with pytest.raises(ValueError) as e:
        g.forward(None, workers={"w1": w})

    said = str(e.value)
    assert "w1" in said, said
    assert "no driver" in said, said


def test_the_same_node_runs_here_because_here_there_is_a_driver():
    # The other half: nothing is wrong with the node.
    g = Graph.somatize(Ask().named("ask"))
    assert g.forward(None, driver=Shout()) == "HELLO"


def test_one_driver_serves_both_sides_of_the_same_run():
    # A node here and a node there, both asking. One `driver=` covers both,
    # which is what makes the seam invisible from where it is written.
    g = Graph.somatize(Ask().named("here") >> Ask().named("there").at("w1"))
    w = generic()

    assert g.forward(None, driver=Shout(), workers={"w1": w}) == "HELLO"


def test_the_drivers_state_travels_with_it():
    # It is packed exactly like a node: what is in its `__dict__` arrives.
    g = Graph.somatize(Ask().named("ask").at("w1"))
    w = generic()

    assert g.forward(None, driver=Shout("!!"), workers={"w1": w}) == "HELLO!!"


def test_the_driver_runs_over_there_and_not_here():
    # Which process serves it is the question, and the answer has to be the one
    # the node runs in: a driver that reports its pid proves it.
    g = Graph.somatize(Ask().named("ask").at("w1"))
    w = generic()

    where = g.forward(None, driver=WhereIServed(), workers={"w1": w})
    assert where != os.getpid(), "the driver was served in the client"


def test_a_host_without_a_worker_is_not_executed_here_just_in_case():
    where = WhereIRan()
    g = Graph.somatize(where.named("where").at("the_one_that_is_not_there"))

    with pytest.raises(ValueError, match="the_one_that_is_not_there"):
        g.forward(None)


def test_a_failure_over_there_comes_back_with_the_host_and_the_reason():
    broken = Fail()
    g = Graph.somatize(broken.named("broken").at("w1"))
    w = generic()

    with pytest.raises(ValueError) as e:
        g.forward(None, workers={"w1": w})

    said = str(e.value)
    assert "w1" in said, said
    assert "broken" in said, said
    assert "I broke in the worker" in said, said


def test_an_opaque_does_not_leave_this_process():
    opaquely, add = Opaquely(), Add(1)
    g = Graph.somatize(opaquely.named("opaquely") >> add.named("a").at("w1"))
    w = generic()

    with pytest.raises(ValueError, match="does not cross"):
        g.forward(None, workers={"w1": w})


def test_an_opaque_produced_over_there_does_not_come_back_either():
    opaquely = Opaquely()
    g = Graph.somatize(opaquely.named("opaquely").at("w1"))
    w = generic()

    with pytest.raises(ValueError, match="does not cross"):
        g.forward(None, workers={"w1": w})


def test_a_runtime_the_worker_does_not_accept_is_rejected_on_connect():
    # The original soma's scar, turned into a refusal with both versions in
    # front of you. A client from another interpreter is faked.
    a = Add(1)
    g = Graph.somatize(a.named("a").at("w1"))
    w = Worker(
        [sys.executable, "-m", "soma_next.worker"],
        kind="pickle",
        id="sha256:whatever",
        blob=cloudpickle.dumps({"a": a}),
        runtime="cpython-2.7/cloudpickle-0.1",
    )

    with pytest.raises(ValueError) as e:
        # Without going through `Graph.forward`, which would set the right
        # artifact on it.
        super(Graph, g).forward(0, workers={"w1": w})

    said = str(e.value)
    assert "cpython-2.7" in said, said
    assert "cpython-3" in said, said


def test_a_kind_of_artifact_it_does_not_know_is_rejected_by_name():
    a = Add(1)
    g = Graph.somatize(a.named("a").at("w1"))
    w = Worker(
        [sys.executable, "-m", "soma_next.worker"],
        kind="package",
        id="whatever",
        blob=b"does not matter",
        runtime=__import__("soma_next.worker", fromlist=["runtime"]).runtime(),
    )

    with pytest.raises(ValueError, match="package"):
        super(Graph, g).forward(0, workers={"w1": w})


def test_provisioning_asks_for_all_three_things_or_none():
    with pytest.raises(ValueError, match="all three or none"):
        Worker([sys.executable, "-c", "pass"], kind="pickle")


def test_workers_takes_a_dict_from_host_to_worker():
    a = Add(1)
    g = Graph.somatize(a.named("a").at("w1"))

    with pytest.raises(ValueError, match="w1"):
        g.forward(0, workers={"w1": "I am not a worker"})


# ── That the code really arrives ──
#
# The hole the first version of this had: cloudpickle serializes by
# **reference** what comes from an importable module, so a node living in
# `my_package/net.py` travels as a pointer to a module the worker does not have.
# Both cases, side by side.


def test_a_node_from_a_module_the_worker_does_not_have_says_what_to_do():
    from sample_net import Greet

    greet = Greet()
    g = Graph.somatize(greet.named("greet").at("w1"))
    w = generic()

    with pytest.raises(ValueError) as e:
        g.forward("world", workers={"w1": w})

    said = str(e.value)
    assert "sample_net" in said, said
    assert "send=" in said, f"the message has to say what to do: {said}"


def test_with_send_the_module_travels_inside_the_artifact():
    from sample_net import Greet

    greet = Greet()
    g = Graph.somatize(greet.named("greet").at("w1"))
    w = generic(send=["sample_net"])

    assert g.forward("world", workers={"w1": w}) == "hello, world"


def test_send_does_not_leave_cloudpickles_global_registry_touched():
    # `register_pickle_by_value` is global to the process: leaving it set would
    # change how everything else is serialized in this interpreter from there on.
    import sample_net

    before = cloudpickle.dumps(sample_net.Greet)
    generic(send=["sample_net"])

    assert cloudpickle.dumps(sample_net.Greet) == before


# ── What an artifact is called, which decides what a worker keeps ──


def test_the_artifact_does_not_depend_on_the_order_the_nodes_came_in():
    # The id is the digest of these bytes and a dict pickles in insertion order,
    # so the same nodes handed over in another order **were another artifact**.
    # Reaching for a private here on purpose: what is being pinned is the bytes,
    # and from outside the only symptom is a worker that quietly starts over.
    from soma_next._remote import _pack

    one, other = Add(1), Add(2)
    for mode in ("network", "project"):
        assert _pack({"a": one, "b": other}, None, mode, ()) == _pack(
            {"b": other, "a": one}, None, mode, ()
        ), f"`{mode}` still depends on the order"


def test_two_graphs_over_the_same_nodes_are_one_artifact():
    # Which is what lets a **second** graph — the transpose of the first, which
    # is how a backward pass crosses a wire — reach a worker without provisioning
    # it again and swapping the catalog it has live.
    from soma_next._remote import _pack

    # The same two objects in both, which is what the real case does: the
    # transpose is the forward graph with its edges the other way round.
    one, other = Add(1), Add(2)
    forward = Graph.somatize(one.named("n") >> other.named("m"))
    backward = Graph.somatize(other.named("m") >> one.named("n"))

    def artifact(graph):
        nodes = {i: graph.implementation(i) for i in graph.nodes()}
        return _pack(nodes, None, "network", ())

    assert artifact(forward) == artifact(backward)
