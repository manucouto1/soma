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

from soma_next import Broker, Graph, Node, Opaque, Worker

cloudpickle = pytest.importorskip("cloudpickle")
cloudpickle.register_pickle_by_value(sys.modules[__name__])


# ── Nodes that are going to travel ──


class Add(Node):
    def __init__(self, how_much):
        self.how_much = how_much

    def forward(self, x, ctx):
        return x + self.how_much


class Mean(Node):
    def forward(self, inputs, ctx):
        return sum(inputs.values()) / len(inputs)


class WhereIRan(Node):
    def forward(self, x, ctx):
        return float(os.getpid())


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
        return float(self.times)


class Fail(Node):
    def forward(self, x, ctx):
        raise ValueError("I broke in the worker")


class Doubles(Node):
    """A tensor in, the same tensor out, still wrapped so it crosses."""

    def forward(self, x, ctx):
        return Opaque(x * 2)


class Shape(Node):
    """Answers something only a tensor could answer."""

    def forward(self, x, ctx):
        return [float(n) for n in x.shape]


class MakesATensor(Node):
    def forward(self, x, ctx):
        import torch

        return Opaque(torch.tensor([1.0, 2.0, 3.0]))


class WhatAmIGiven(Node):
    def forward(self, x, ctx):
        return type(x).__name__


class CountsATensor(Node):
    """Says in a tensor how many times it was really asked.

    A hit is invisible from outside unless somebody counts, and it has to be a
    **tensor** for this to be about anything: a number is kept by a worker with
    no codecs in front of its store just as well.
    """

    def __init__(self):
        import torch

        self.calls = 0
        self.settled = [torch.tensor([1.0])]

    def parameters(self):
        return self.settled

    def forward(self, x, ctx):
        import torch

        self.calls += 1
        return Opaque(torch.tensor([float(self.calls)]))


class Chatterbox(Node):
    """Prints to `stdout`, which is where the wire runs.

    With `flush=True` on purpose: without it, Python keeps what is printed in
    its buffer and releases it on process exit — once the answer has already
    travelled — so the test would pass just the same with the redirection
    removed. Verified.
    """

    def forward(self, x, ctx):
        print("hello from the worker", flush=True)
        return x


class Opaquely(Node):
    def forward(self, x, ctx):
        return Opaque(object())


class WhichDevice(Node):
    def forward(self, x, ctx):
        return ctx.device


def generic(host="w1", **how):
    """A broker that knows where an empty worker is, one that gets sent **the
    code**.

    A broker and not a worker, because that is where the wire now lives: two
    runs that have to reach the same process share **this**, not a `Worker`.

    `mode="network"` explicitly: the classes in this file exist in no worker's
    clone, so the `project` strategy — the default — could not resolve them. It
    is exactly the case `network` is for.

    And with no nodes: the graph sends them at run time, since it is the one
    that knows which go to each host.
    """
    return Broker.embedded({host: Worker.generic(mode="network", **how)})


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

    output = g.forward(None, broker=w)

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

    assert g.forward(0, broker=w) == 111


def test_a_fan_in_with_one_branch_away_gives_the_same_as_all_of_it_here():
    def net():
        return Add(1), Add(10), Add(100), Mean()

    s, l, r, j = net()
    here = Graph.somatize(s.named("s") >> (l.named("l") | r.named("r")) >> j.named("j"))
    expected = here.forward(0)

    s, l, r, j = net()
    away = Graph.somatize(s.named("s") >> (l.named("l").at("w1") | r.named("r")) >> j.named("j"))
    w = generic()

    assert away.forward(0, broker=w) == expected


def test_two_workers_are_two_processes():
    one, other = WhereIRan(), WhereIRan()
    g = Graph.somatize(one.named("one").at("w1") | other.named("other").at("w2"))
    both = Broker.embedded(
        {
            "w1": Worker.generic(mode="network"),
            "w2": Worker.generic(mode="network"),
        }
    )

    output = g.forward(None, broker=both)

    assert output["one"] != output["other"]
    assert float(os.getpid()) not in output.values()


def test_the_artifact_is_sent_only_once():
    # `HowManyTimes` counts calls on **its** object. If the second `forward`
    # resent the artifact, the worker would unpack a new one and the count would
    # go back to 1.
    how_many = HowManyTimes()
    g = Graph.somatize(how_many.named("how_many").at("w1"))
    w = generic()

    assert [g.forward(None, broker=w) for _ in range(3)] == [1.0, 2.0, 3.0]


def test_the_placement_travels_with_the_slice():
    # What `placement.rs` promised back in CU10, now crossing two seams:
    # Python → Rust → the wire → Rust → Python.
    q = WhichDevice()
    g = Graph.somatize(q.named("q").at("w1").on("meta"))
    w = generic()

    assert g.forward(None, broker=w) == "meta"


def test_a_print_in_a_node_does_not_break_the_wire():
    # The protocol's messages go over the worker's `stdout`. The generic worker
    # redirects Python's `sys.stdout` to `stderr` for exactly this.
    chatterbox = Chatterbox()
    g = Graph.somatize(chatterbox.named("chatterbox").at("w1"))
    w = generic()

    assert g.forward("intact", broker=w) == "intact"


# ── What goes wrong ──


def test_a_mode_that_is_not_one_of_the_two_is_rejected():
    with pytest.raises(ValueError, match="'project' or 'network'"):
        Worker.generic(mode="carrier-pigeon")


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
        g.forward(None, broker=w)

    said = str(e.value)
    assert "w1" in said, said
    assert "broken" in said, said
    assert "I broke in the worker" in said, said


def test_an_opaque_nobody_can_write_down_does_not_leave_this_process():
    """The frontier did not disappear when a codec appeared, it moved: from "an
    opaque" to "an opaque nobody registered a codec for", which is the more
    precise of the two. An `object()` is on the far side of it forever."""
    opaquely, add = Opaquely(), Add(1)
    g = Graph.somatize(opaquely.named("opaquely") >> add.named("a").at("w1"))
    w = generic()

    with pytest.raises(ValueError, match="nothing says how to write one down"):
        g.forward(None, broker=w)


def test_and_it_says_which_type_it_was_and_how_to_say_so():
    opaquely, add = Opaquely(), Add(1)
    g = Graph.somatize(opaquely.named("opaquely") >> add.named("a").at("w1"))
    w = generic()

    with pytest.raises(ValueError) as e:
        g.forward(None, broker=w)

    said = str(e.value)
    assert "`object`" in said, said
    assert "codec(" in said, said


def test_an_opaque_produced_over_there_that_nobody_can_write_down_stays_there():
    """It is the slice's own value, so somebody here is waiting for it: refusing
    is the honest answer, and it comes back with the codec's words rather than
    the wire's."""
    opaquely = Opaquely()
    g = Graph.somatize(opaquely.named("opaquely").at("w1"))
    w = generic()

    with pytest.raises(ValueError, match="nothing says how to write one down"):
        g.forward(None, broker=w)


# ── And the half that is new: with a codec, it crosses ──


def test_a_tensor_crosses_whole_and_is_the_same_tensor_over_there():
    torch = pytest.importorskip("torch")
    import soma_next.torch  # noqa: F401  — registers the codec for a tensor

    doubles, shape = Doubles(), Shape()
    g = Graph.somatize(doubles.named("doubles") >> shape.named("shape").at("w1"))
    w = generic()

    # `shape` runs over there and answers about the object it was handed: a
    # list of floats has no `shape`, so this only passes if a tensor arrived.
    out = g.forward(Opaque(torch.ones(3, 4)), broker=w)

    assert out == [3.0, 4.0], out


def test_a_tensor_produced_over_there_comes_back_a_tensor():
    torch = pytest.importorskip("torch")
    import soma_next.torch  # noqa: F401

    makes = MakesATensor()
    g = Graph.somatize(makes.named("makes").at("w1"))
    w = generic()

    out = g.forward(None, broker=w)

    assert torch.is_tensor(out), type(out)
    assert torch.equal(out, torch.tensor([1.0, 2.0, 3.0])), out


def test_what_crosses_are_the_bytes_of_the_tensor_and_not_its_floats():
    """The point of the whole thing, made observable: what a node is handed on
    the other side is a tensor and not a list, so it is the same node here and
    there — which is the entire argument of `.at()`."""
    torch = pytest.importorskip("torch")
    import soma_next.torch  # noqa: F401

    what, w = WhatAmIGiven(), generic()
    g = Graph.somatize(what.named("what").at("w1"))

    assert g.forward(Opaque(torch.ones(2)), broker=w) == "Tensor"


def test_a_runtime_the_worker_does_not_accept_is_rejected_on_connect():
    # The original soma's scar, turned into a refusal with both versions in
    # front of you. A client from another interpreter is faked.
    a = Add(1)
    g = Graph.somatize(a.named("a").at("w1"))
    w = generic()
    # Staged by hand and not through `Graph.forward`, which would put the right
    # artifact on it.
    w.provision(
        "w1",
        "pickle",
        "sha256:whatever",
        cloudpickle.dumps({"a": a}),
        "cpython-2.7/cloudpickle-0.1",
    )

    with pytest.raises(ValueError) as e:
        super(Graph, g).forward(0, broker=w)

    said = str(e.value)
    assert "cpython-2.7" in said, said
    assert "cpython-3" in said, said


def test_a_kind_of_artifact_it_does_not_know_is_rejected_by_name():
    a = Add(1)
    g = Graph.somatize(a.named("a").at("w1"))
    w = generic()
    w.provision(
        "w1",
        "package",
        "whatever",
        b"does not matter",
        __import__("soma_next.worker", fromlist=["runtime"]).runtime(),
    )

    with pytest.raises(ValueError, match="package"):
        super(Graph, g).forward(0, broker=w)


def test_a_worker_is_declared_with_an_address_or_a_command():
    # What is left of the old constructor's checking now that a `Worker` carries
    # no artifact: the artifact is packed by the graph and handed over by the
    # broker, where all three parts are required by the signature itself.
    with pytest.raises(ValueError, match="argv"):
        Worker(7)
    with pytest.raises(ValueError, match="at least a program"):
        Worker([])


def test_a_broker_takes_a_dict_from_host_to_worker():
    a = Add(1)
    g = Graph.somatize(a.named("a").at("w1"))

    with pytest.raises(ValueError, match="w1"):
        g.forward(0, broker=Broker.embedded({"w1": "I am not a worker"}))


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
        g.forward("world", broker=w)

    said = str(e.value)
    assert "sample_net" in said, said
    assert "send=" in said, f"the message has to say what to do: {said}"


def test_with_send_the_module_travels_inside_the_artifact():
    from sample_net import Greet

    greet = Greet()
    g = Graph.somatize(greet.named("greet").at("w1"))
    w = generic(send=["sample_net"])

    assert g.forward("world", broker=w) == "hello, world"


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
        assert _pack({"a": one, "b": other}, mode, ()) == _pack({"b": other, "a": one}, mode, ()), f"`{mode}` still depends on the order"


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
        return _pack(nodes, "network", ())

    assert artifact(forward) == artifact(backward)


# ── What a worker is told it is going to need ──


def test_provision_says_out_loud_what_forward_says_on_its_own():
    # The same handover, so saying it first does not make the artifact travel
    # twice: `HowManyTimes` would go back to 1 if it did.
    how_many = HowManyTimes()
    g = Graph.somatize(how_many.named("how_many").at("w1"))
    w = generic()

    g.provision(w)
    assert [g.forward(None, broker=w) for _ in range(2)] == [1.0, 2.0]


def test_a_worker_that_gets_nothing_is_told_nothing():
    # An artifact with no nodes in it is a catalog too, and offering it to a
    # worker already serving one is refused. It is the case of a stage with
    # nothing on that host, in the middle of a graph run in pieces.
    w = generic()
    Graph.somatize(Add(1).named("there").at("w1")).forward(0, broker=w)

    nothing_of_its_own = Graph.somatize(Add(2).named("here"))
    assert nothing_of_its_own.forward(0, broker=w) == 2.0


def test_a_graph_run_in_pieces_keeps_the_worker_it_had():
    # Three stages over one live worker and two of them with nodes on it. Each
    # provisions the **whole** graph, so the artifact is one: the objects over
    # there survive from stage to stage and from pass to pass, which is what a
    # backward pass against a live optimizer is going to need.
    from soma_next._stage import stages

    how_many = HowManyTimes()
    g = Graph.somatize(
        how_many.named("how_many").at("w1")
        >> Add(1).named("here")
        >> Add(10).named("back_there").at("w1")
    )
    w = generic()

    def pass_over():
        produced = {}
        for stage in stages(g):
            stage.fill(produced)
            out = stage.graph.forward(
                None if stage.level else 0.0, broker=w
            )
            produced.update(stage.read(out))
        return produced["back_there"]

    assert [pass_over(), pass_over()] == [12.0, 13.0]


def test_a_cached_tensor_is_kept_by_the_worker_too(tmp_path):
    """CU13's hole, which only a codec could close: a worker keeps what is made
    of numbers and, with nothing in front of its store, quietly keeps nothing
    else — so the same `.cached()` node hit here and missed there for no reason
    anybody could see."""
    torch = pytest.importorskip("torch")
    import soma_next.torch

    counts = CountsATensor()
    g = Graph.somatize(counts.named("counts").frozen().cached().at("w1"))
    # Declaring it is the graph's half; making it true is torch's, and without
    # the digest of its weights two checkpoints would share one name.
    soma_next.torch.freeze(g)
    w = Broker.embedded(
        {
            "w1": Worker.spawn(
                [sys.executable, "-m", "soma_next.worker", "--store", str(tmp_path)],
                mode="network",
            )
        }
    )

    first = g.forward(None, broker=w)
    second = g.forward(None, broker=w)

    assert torch.equal(first, torch.tensor([1.0])), first
    assert torch.equal(second, torch.tensor([1.0])), "the worker ran it again"


# ── Two names for one place, which is one catalog ──
#
# The rule is proved next door in `soma-fabric`, against a wire: two hosts at
# one address share it, two with the same `argv` do not. What is proved here is
# the half that lives on this side — that the grouping ends in **one artifact
# holding both halves**. A worker has one catalog, and provisioning the same
# process twice, once per host name, replaces what it had live and takes every
# activation over there with it. That failure is silent, which is why it is
# worth a test that does not need a machine.


class Recording(Broker):
    """A broker that writes down what it was told to hand over.

    A subclass and not a fake, so the grouping and the packing under test are
    the real ones — only the last step, the handover, is watched.
    """

    def provision(self, host, kind, ident, blob, runtime):
        self.told.append((host, ident))


def recording(workers):
    broker = Recording.embedded(workers)
    broker.told = []
    return broker


def two_nodes_apart():
    """One node on `w1`, one on `w2` — whether those are one machine is what
    each of these tests changes."""
    return Graph.somatize(Add(1).named("a").at("w1") >> Add(2).named("b").at("w2"))


def artifact_of(graph, *ids, mode="network"):
    """The id the artifact of exactly these nodes would have."""
    nodes = {one: graph.implementation(one) for one in ids}
    _, ident, _ = Worker.at("wherever:7000", mode=mode).packed(nodes)
    return ident


def test_two_names_for_one_place_are_told_once_each_about_one_catalog():
    g = two_nodes_apart()
    one_box = recording(
        {
            "w1": Worker.at("box:7000", mode="network"),
            "w2": Worker.at("box:7000", mode="network"),
        }
    )

    g.provision(one_box)

    assert [host for host, _ in one_box.told] == ["w1", "w2"], "both names are told"
    assert {ident for _, ident in one_box.told} == {artifact_of(g, "a", "b")}, (
        "and what they are told about is the artifact holding **both** nodes"
    )


def test_while_two_addresses_are_two_catalogs_with_half_each():
    # The contrast, without which the one above would also pass if the grouping
    # collapsed everything into one.
    g = two_nodes_apart()
    apart = recording(
        {
            "w1": Worker.at("box:7000", mode="network"),
            "w2": Worker.at("other:7000", mode="network"),
        }
    )

    g.provision(apart)

    assert apart.told == [("w1", artifact_of(g, "a")), ("w2", artifact_of(g, "b"))]


def test_two_names_for_one_place_packed_differently_is_refused_by_name():
    # There is no honest answer: one catalog cannot be both `network` and
    # `project`, and picking either quietly would send the wrong one. So it says
    # which two hosts, and what each of them asked for.
    g = two_nodes_apart()
    both_ways = Broker.embedded(
        {
            "w1": Worker.at("box:7000", mode="network"),
            "w2": Worker.at("box:7000", mode="project"),
        }
    )

    with pytest.raises(ValueError, match="same place") as why:
        g.provision(both_ways)

    assert "w1" in str(why.value) and "w2" in str(why.value)


def test_a_host_the_broker_never_heard_of_is_left_out_and_not_raised_over():
    # Naming a host nobody listed is not this step's failure to report: the run
    # either reaches that slice or it does not, and whichever happens says so
    # with the slice in front of it.
    g = two_nodes_apart()
    half_known = recording({"w1": Worker.at("box:7000", mode="network")})

    g.provision(half_known)

    assert half_known.told == [("w1", artifact_of(g, "a"))]
