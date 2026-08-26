"""What happened, as it happens.

The other answer a run gives. `forward` returns a value when it is over;
`watching=` is told things while it is going, which is the difference between a
progress bar and a report.

A fact arrives as a `dict` with a `fact` key naming it and text beside it — and
that is **the same shape it is written down as**, so what you print is what you
would find in the store afterwards.
"""

import sys

import pytest

from somatize import Graph, Node, Recorder, Store


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


def kinds(seen):
    return [fact["fact"] for fact in seen]


def test_nobody_watching_is_the_run_it_always_was(g):
    assert g.forward(0.0) == 11.0


def test_every_node_that_ran_is_said_so_in_order(g):
    seen = []

    g.forward(0.0, watching=seen.append)

    assert kinds(seen) == ["ran", "ran", "finished"]
    assert [f["node"] for f in seen if f["fact"] == "ran"] == ["a", "b"]


def test_a_fact_is_text_to_text_because_that_is_what_a_record_is(g):
    seen = []

    g.forward(0.0, watching=seen.append)

    assert all(isinstance(what, str) for fact in seen for what in fact.values())
    assert int(seen[-1]["took_us"]) > 0


def test_a_node_that_failed_says_which_one_before_the_error_arrives():
    # The whole reason it is a fact: by the time the exception reaches you the
    # run is over, and you wanted to know which node while it was happening.
    g = Graph.somatize(Add(1).named("a") >> Boom().named("boom"))
    seen = []

    with pytest.raises(Exception):
        g.forward(0.0, watching=seen.append)

    assert kinds(seen) == ["ran", "failed", "broke"]
    assert seen[1]["node"] == "boom"
    assert "I broke" in seen[1]["why"]


def test_several_are_told_and_something_that_is_not_callable_is_refused(g):
    one, other = [], []

    g.forward(0.0, watching=[one.append, other.append])

    assert kinds(one) == kinds(other) == ["ran", "ran", "finished"]
    with pytest.raises(ValueError, match="Recorder"):
        g.forward(0.0, watching=7)


def test_a_recorder_writes_one_record_per_forward(g, tmp_path):
    store = Store(str(tmp_path))
    recorder = Recorder(store, run="tuesday")

    for _ in range(3):
        g.forward(0.0, watching=recorder)

    assert recorder.run == "tuesday"
    assert sorted(b.name for b in store.bound()) == [
        f"run/tuesday/{n}" for n in range(3)
    ]


def test_a_scan_says_how_it_went_without_reading_a_blob(g, tmp_path):
    store = Store(str(tmp_path))
    recorder = Recorder(store, run="tuesday")

    g.forward(0.0, watching=recorder)

    said = dict(store.resolve("run/tuesday/0").meta)
    assert said["state"] == "ok"
    assert said["nodes"] == "2"
    assert int(said["took_us"]) > 0


def test_a_recorder_nobody_named_is_still_findable(g, tmp_path):
    # A forward in a notebook has no reason to invent a name, and still has to
    # be findable afterwards.
    store = Store(str(tmp_path))
    recorder = Recorder(store)

    g.forward(0.0, watching=recorder)

    assert store.resolve(f"run/{recorder.run}/0") is not None


def test_what_is_printed_is_what_is_written(g, tmp_path):
    # One shape and not two, which is the property the whole seam is built on:
    # the dict a watcher sees **is** `Fact::flattened`, and so is the blob.
    import json

    store = Store(str(tmp_path))
    recorder = Recorder(store, run="tuesday")
    seen = []

    g.forward(0.0, watching=[recorder, seen.append])

    bound = store.resolve("run/tuesday/0")
    written = json.loads(store.get(bound.digest))
    assert [fact["fact"] for fact in written] == kinds(seen)
    assert written[0]["node"] == seen[0]["node"]


def test_what_ran_on_a_worker_comes_back_saying_which_host(tmp_path):
    from somatize import Broker, Worker

    g = Graph.somatize(Add(1).named("a") >> Add(10).named("b").at("w1"))
    # The nodes above live in this file and no worker has it, so it travels
    # inside the artifact — which is what `network` is for.
    broker = Broker.embedded(
        {
            "w1": Worker.spawn(
                [sys.executable, "-m", "somatize.worker"],
                mode="network",
                send=["test_watching"],
            )
        }
    )
    seen = []

    out = g.forward(0.0, broker=broker, watching=seen.append)

    assert out == 11.0
    # `machine` sits between them: the worker says what it looks like before it
    # starts the slice, down the connection that is already open.
    assert kinds(seen) == ["ran", "machine", "ran", "left", "finished"]
    assert seen[0].get("host") is None, "`a` ran here"
    assert seen[1]["fact"] == "machine"
    assert seen[1]["host"] == "w1", "and it is attributed like everything else"
    assert seen[2]["node"] == "b"
    assert seen[2]["host"] == "w1", "a worker does not know its own name; we do"
    assert seen[3]["host"] == "w1", "and the round trip is its own fact"


def test_a_training_step_says_the_loss_and_when_it_moved(tmp_path):
    torch = pytest.importorskip("torch")
    import somatize.torch  # noqa: F401
    from somatize.torch import Trainer, parameters

    class Layer(Node):
        def __init__(self):
            self.lin = torch.nn.Linear(4, 2)

        def forward(self, x, ctx):
            from somatize import Opaque

            return Opaque(self.lin(x))

        def parameters(self):
            return list(self.lin.parameters())

    g = Graph.somatize(Layer().named("body"))
    seen = []
    t = Trainer(
        g,
        objective=torch.nn.functional.mse_loss,
        optimizer=torch.optim.SGD(parameters(g), lr=0.1),
        watching=seen.append,
    )

    t.step((torch.randn(8, 4), torch.randn(8, 2)))

    # The engine's vocabulary and this level's, in one stream and in order.
    assert kinds(seen) == ["ran", "finished", "updated", "loss"]
    assert float(seen[-1]["value"]) > 0


def test_a_loss_lands_in_the_forward_it_belongs_to(tmp_path):
    # A loss is computed **after** the forward that produced it has ended. A
    # record that only knew how to open a new one would file every loss a step
    # late, and every curve would be off by one.
    import json

    torch = pytest.importorskip("torch")
    import somatize.torch  # noqa: F401
    from somatize.torch import Trainer, parameters

    class Layer(Node):
        def __init__(self):
            self.lin = torch.nn.Linear(4, 2)

        def forward(self, x, ctx):
            from somatize import Opaque

            return Opaque(self.lin(x))

        def parameters(self):
            return list(self.lin.parameters())

    store = Store(str(tmp_path))
    recorder = Recorder(store, run="tuesday")
    g = Graph.somatize(Layer().named("body"))
    t = Trainer(
        g,
        objective=torch.nn.functional.mse_loss,
        optimizer=torch.optim.SGD(parameters(g), lr=0.1),
        watching=recorder,
    )

    for _ in range(2):
        t.step((torch.randn(8, 4), torch.randn(8, 2)))

    assert sorted(b.name for b in store.bound()) == ["run/tuesday/0", "run/tuesday/1"]
    for which in (0, 1):
        bound = store.resolve(f"run/tuesday/{which}")
        written = json.loads(store.get(bound.digest))
        assert [f["fact"] for f in written] == ["ran", "finished", "updated", "loss"], (
            f"step {which} did not get its own loss"
        )


def test_a_trainer_takes_the_same_watching_a_forward_does(tmp_path):
    # Found by writing an example. `Graph.forward` hands `watching=` to the
    # engine, which resolves a list itself; a `Trainer` calls it from Python, and
    # for a while a list worked through one door and raised through the other.
    # `watching=` meaning two things depending on the door is the kind of trap
    # this project exists not to build.
    torch = pytest.importorskip("torch")
    import somatize.torch  # noqa: F401
    from somatize.torch import Trainer, parameters

    class Layer(Node):
        def __init__(self):
            self.lin = torch.nn.Linear(4, 2)

        def forward(self, x, ctx):
            from somatize import Opaque

            return Opaque(self.lin(x))

        def parameters(self):
            return list(self.lin.parameters())

    g = Graph.somatize(Layer().named("body"))
    one, other = [], []
    t = Trainer(
        g,
        objective=torch.nn.functional.mse_loss,
        optimizer=torch.optim.SGD(parameters(g), lr=0.1),
        watching=[one.append, other.append],
    )

    t.step((torch.randn(8, 4), torch.randn(8, 2)))

    assert kinds(one) == kinds(other) == ["ran", "finished", "updated", "loss"]


def test_a_trainer_refuses_a_watching_it_cannot_call(tmp_path):
    torch = pytest.importorskip("torch")
    import somatize.torch  # noqa: F401
    from somatize.torch import Trainer, parameters

    class Layer(Node):
        def __init__(self):
            self.lin = torch.nn.Linear(4, 2)

        def forward(self, x, ctx):
            from somatize import Opaque

            return Opaque(self.lin(x))

        def parameters(self):
            return list(self.lin.parameters())

    g = Graph.somatize(Layer().named("body"))
    with pytest.raises(ValueError, match="Recorder"):
        Trainer(
            g,
            objective=torch.nn.functional.mse_loss,
            optimizer=torch.optim.SGD(parameters(g), lr=0.1),
            watching=7,
        )


def test_a_group_of_steps_moves_once_and_says_so_once(tmp_path):
    # `Trainer(every=N)` makes a group of steps into one update. What is said
    # has to be the same fact: one `updated`, at the end of the group.
    torch = pytest.importorskip("torch")
    import somatize.torch  # noqa: F401
    from somatize.torch import Trainer, parameters

    class Layer(Node):
        def __init__(self):
            self.lin = torch.nn.Linear(4, 2)

        def forward(self, x, ctx):
            from somatize import Opaque

            return Opaque(self.lin(x))

        def parameters(self):
            return list(self.lin.parameters())

    g = Graph.somatize(Layer().named("body"))
    seen = []
    t = Trainer(
        g,
        objective=torch.nn.functional.mse_loss,
        optimizer=torch.optim.SGD(parameters(g), lr=0.1),
        every=3,
        watching=seen.append,
    )

    for _ in range(3):
        t.step((torch.randn(8, 4), torch.randn(8, 2)))

    assert kinds(seen).count("loss") == 3
    assert kinds(seen).count("updated") == 1
