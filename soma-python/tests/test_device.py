"""Where a node runs: `.on("cuda:0")`, `place()` and what reaches the `forward`.

Two things worth being clear on before reading:

**Placing is not executing.** The core does not know how to move anything to a
GPU; what it does is carry the declaration all the way to the node's `ctx`. The
one that obeys is the node, with `.to(ctx.device)`. What keeps disobeying from
being free is the postcondition: if a placed node returns something that is
somewhere else, that is a named error.

**Placing changes nothing else.** Not the plan, not the order, not the result.
That is why the device does not live in the `Plan`: the plan says when each node
runs, not where.
"""

import pytest

from somatize import Graph, Node, Opaque

from conftest import Add, Identity


class Watch(Node):
    """Notes the device that reached it and returns its input."""

    def __init__(self):
        self.seen = "has not run"

    def forward(self, x, ctx):
        self.seen = ctx.device
        return x


# ── Declaring where ──


def test_on_places_a_node():
    g = Graph.somatize(Add(1).named("a").on("cuda:0"))
    assert g.devices() == {"a": "cuda:0"}


def test_without_on_there_is_no_device():
    g = Graph.somatize(Add(1) >> Add(2))
    assert g.devices() == {}


def test_on_spreads_over_the_whole_piece():
    g = Graph.somatize((Add(1).named("a") >> Add(2).named("b")).on("cpu"))
    assert g.devices() == {"a": "cpu", "b": "cpu"}


def test_the_innermost_one_wins():
    g = Graph.somatize((Add(1).named("a").on("cuda:0") >> Add(2).named("b")).on("meta"))
    assert g.devices() == {"a": "cuda:0", "b": "meta"}


def test_each_branch_in_its_own_place():
    g = Graph.somatize(
        Add(1).named("source")
        >> (Add(2).named("left").on("cuda:0") | Add(3).named("right").on("cpu"))
    )
    assert g.devices() == {"left": "cuda:0", "right": "cpu"}


def test_named_and_on_commute():
    one = Graph.somatize(Add(1).named("a").on("cpu"))
    other = Graph.somatize(Add(1).on("cpu").named("a"))
    assert one.devices() == other.devices() == {"a": "cpu"}


def test_place_does_the_same_as_on():
    # `.on()` needs the object inside an expression; `place()` only the id,
    # which is all that is left when the graph was built by hand or when the
    # placement is decided afterwards.
    dsl = Graph.somatize(Add(1).named("a").on("cuda:0"))

    by_hand = Graph()
    by_hand.node("a", Add(1))
    by_hand.place("a", "cuda:0")

    assert dsl.devices() == by_hand.devices()
    assert dsl.plan() == by_hand.plan()


def test_placing_afterwards_in_a_loop():
    # The case `.on()` cannot cover: the placement comes from what is on the
    # machine, not from what was written in the expression.
    g = Graph.somatize(Add(1).named("a") >> Add(2).named("b") >> Add(3).named("c"))
    for i, nid in enumerate(g.nodes()):
        g.place(nid, f"cuda:{i % 2}")

    assert g.devices() == {"a": "cuda:0", "b": "cuda:1", "c": "cuda:0"}


def test_replacing_overwrites_the_previous_one():
    g = Graph.somatize(Add(1).named("a").on("cpu"))
    g.place("a", "meta")
    assert g.devices() == {"a": "meta"}


# ── What gets rejected, and where ──


@pytest.mark.parametrize(
    "bad, warning",
    [
        ("cude:0", "unknown device"),
        ("gpu:0", "unknown device"),
        ("cuda", "does not say which one"),
        ("cuda:", "not shaped like a device"),
        ("cuda:x", "not shaped like a device"),
        ("cpu:0", "not shaped like a device"),
        ("", "not shaped like a device"),
    ],
)
def test_a_name_that_names_no_place_fails_at_declaration(bad, warning):
    # The reason `Device` is an enum: the typo is caught here, not inside torch
    # halfway through a run.
    with pytest.raises(ValueError, match=warning):
        Graph.somatize(Add(1).on(bad))


def test_placing_a_node_that_does_not_exist_fails():
    g = Graph.somatize(Add(1).named("a"))
    with pytest.raises(ValueError, match="encodr"):
        g.place("encodr", "cpu")


# ── What reaches the node ──


def test_the_node_sees_where_it_was_told_to_run():
    watch = Watch()
    Graph.somatize(watch.named("m").on("cuda:1")).forward(1.0)
    assert watch.seen == "cuda:1"


def test_unplaced_it_sees_none():
    watch = Watch()
    Graph.somatize(watch.named("m")).forward(1.0)
    assert watch.seen is None


def test_nobody_catches_the_neighbours():
    first, second = Watch(), Watch()
    Graph.somatize(first.named("a").on("meta") >> second.named("b")).forward(1.0)
    assert first.seen == "meta"
    assert second.seen is None


def test_each_branch_of_a_wave_sees_its_own():
    left, right = Watch(), Watch()
    g = Graph.somatize(
        Identity().named("source")
        >> (left.named("left").on("cuda:0") | right.named("right").on("cuda:1"))
    )
    assert "Wave" in g.plan()
    g.forward(1.0)

    assert left.seen == "cuda:0"
    assert right.seen == "cuda:1"


def test_the_device_shows_up_in_the_ctxs_repr():
    class Repr(Node):
        def __init__(self):
            self.seen = None

        def forward(self, x, ctx):
            self.seen = repr(ctx)
            return x

    node = Repr()
    Graph.somatize(node.named("n").on("cpu")).forward(1.0)
    assert node.seen == "Ctx(device=cpu)"


# ── What placing does NOT change ──


def test_placing_does_not_change_the_plan():
    without = Graph.somatize(Add(1).named("a") >> (Add(2).named("b") | Add(3).named("c")))
    with_ = Graph.somatize(
        (Add(1).named("a") >> (Add(2).named("b") | Add(3).named("c"))).on("meta")
    )
    assert without.plan() == with_.plan()
    assert "meta" not in with_.plan(), "the plan says when, not where"


def test_placing_does_not_change_the_result():
    without = Graph.somatize(Add(1).named("a") >> Add(10).named("b"))
    with_ = Graph.somatize((Add(1).named("a") >> Add(10).named("b")).on("meta"))
    assert without.forward(0.0) == with_.forward(0.0) == 11.0


# ── The postcondition: disobeying is not free ──


class Fake:
    """Anything that knows how to say where it is. No torch needed."""

    def __init__(self, device):
        self.device = device


class Returns(Node):
    def __init__(self, value):
        self.value = value

    def forward(self, x, ctx):
        return Opaque(self.value)


def test_a_node_that_returns_something_from_elsewhere_is_caught():
    g = Graph.somatize(Returns(Fake("cpu")).named("n").on("cuda:0"))
    with pytest.raises(
        ValueError, match="declared `cuda:0` but returned a value on `cpu`"
    ):
        g.forward(1.0)


def test_the_error_says_which_node_it_is():
    g = Graph.somatize(Returns(Fake("cpu")).named("encoder").on("meta"))
    with pytest.raises(ValueError, match="node `encoder` failed"):
        g.forward(1.0)


def test_obeying_passes_without_noise():
    g = Graph.somatize(Returns(Fake("cuda:0")).named("n").on("cuda:0"))
    assert g.forward(1.0).device == "cuda:0"


def test_what_does_not_know_where_it_is_is_not_checked():
    # A placed node that returns text cannot be checked from outside. Placing it
    # did not make much sense anyway.
    g = Graph.somatize(Identity().named("n").on("cuda:0"))
    assert g.forward("hello") == "hello"


def test_unplaced_nothing_is_checked():
    g = Graph.somatize(Returns(Fake("cpu")).named("n"))
    assert g.forward(1.0).device == "cpu"


# ── With real torch ──

torch = pytest.importorskip("torch")
nn = torch.nn


class Layer(Node):
    """The pattern: the parameters once, the input every time.

    It is everything you have to write in order to obey a placement, and it goes
    here by hand on purpose: until this same body repeats three times, there is
    nothing to pull out into a base class.
    """

    def __init__(self, in_, out):
        self.lin = nn.Linear(in_, out)
        self.placed = None

    def forward(self, x, ctx):
        if ctx.device:
            if self.placed != ctx.device:
                self.lin.to(ctx.device)  # the parameters, once
                self.placed = ctx.device
            x = x.to(ctx.device)  # the input, every time
        return Opaque(self.lin(x))

    def parameters(self):
        return list(self.lin.parameters())


def test_meta_tests_placement_without_hardware():
    # `meta` is the reason the variant exists: it checks end to end that the
    # placement arrives and is obeyed on any machine.
    g = Graph.somatize(Layer(4, 3).named("layer").on("meta"))
    output = g.forward(Opaque(torch.zeros(2, 4)))

    assert str(output.device) == "meta"
    assert output.shape == (2, 3)


def test_a_node_that_ignores_its_device_is_caught():
    class Deaf(Node):
        def forward(self, x, ctx):
            return Opaque(torch.zeros(2, 3))  # born on cpu, whatever happens

    g = Graph.somatize(Deaf().named("deaf").on("meta"))
    with pytest.raises(ValueError, match="declared `meta` but returned a value on `cpu`"):
        g.forward(Opaque(torch.zeros(2, 4)))


# ── And with a GPU, if there is one ──

no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")


@no_cuda
def test_one_node_runs_on_the_gpu_and_the_next_on_the_cpu():
    # With a single GPU on the machine this is what can really be tested:
    # spreading across two GPUs stays declared but not executed here.
    g = Graph.somatize(Layer(4, 3).named("gpu").on("cuda:0") >> Layer(3, 2).named("cpu").on("cpu"))
    output = g.forward(Opaque(torch.zeros(2, 4)))

    assert str(output.device) == "cpu"


@no_cuda
def test_the_backward_pass_crosses_the_device_hop():
    # It is what makes the whole of CU10 cheap: `.to()` between devices is
    # differentiable, so `Opaque` has not had to change a thing.
    torch.manual_seed(0)
    on_gpu, on_cpu = Layer(4, 3), Layer(3, 2)
    g = Graph.somatize(on_gpu.named("gpu").on("cuda:0") >> on_cpu.named("cpu").on("cpu"))
    target = torch.tensor([0, 1])

    opt = torch.optim.Adam(on_gpu.parameters() + on_cpu.parameters(), lr=0.05)
    input_ = torch.randn(2, 4)
    first = last = None
    for step in range(20):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(g.forward(Opaque(input_)), target)
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()

    assert all(p.grad is not None for p in on_gpu.parameters()), (
        "the gradients crossed back to the GPU"
    )
    assert last < first, f"the loss went down from {first:.4f} to {last:.4f}"
