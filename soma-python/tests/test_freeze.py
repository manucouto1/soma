"""Settling a node, which is the only part of all this that torch has to do.

The core **declares** that a settled node's state does not change, and reasons
over it; here it is made true. Two things at once, and they are the same thing
seen twice: the gradient stops, and the weights get a digest that goes into the
key. Without the first, a value read back from a store is a leaf and the net
above it stops training in silence; without the second, two checkpoints of the
same class share a name, which is the one kind of cache hit that is a bug.
"""

import pytest

from somatize import Graph, Node, Opaque

torch = pytest.importorskip("torch")
nn = torch.nn

from somatize.torch import Trainer, freeze, parameters  # noqa: E402
from somatize.torch._codec import KIND, dump, load  # noqa: E402

IN, MID, CLASSES = 4, 3, 2


class Layer(Node):
    """A node with weights, and the one that obeys its placement."""

    def __init__(self, in_, out):
        self.lin = nn.Linear(in_, out)
        self.calls = 0

    def forward(self, x, ctx):
        self.calls += 1
        return Opaque(self.lin(x))

    def parameters(self):
        return list(self.lin.parameters())

    def state_dict(self):
        return self.lin.state_dict()


class Label(Node):
    """No weights and no state: not every node has any, and it does not stop
    being a node."""

    def forward(self, x, ctx):
        return x


def batch():
    torch.manual_seed(0)
    return torch.randn(6, IN)


# ── The gradient really does stop ──


def test_freezing_turns_the_gradient_off():
    layer = Layer(IN, MID)
    g = Graph.somatize(layer.named("encoder"))

    assert all(p.requires_grad for p in layer.parameters())
    freeze(g, "encoder")
    assert not any(p.requires_grad for p in layer.parameters())


def test_declaring_it_is_not_obeying_it():
    # The division the whole design rests on: the core says it, torch does it.
    # Until somebody obeys, the weights still ask for a gradient.
    layer = Layer(IN, MID)
    g = Graph.somatize(layer.named("encoder").frozen())

    assert g.frozen() == {"encoder": None}
    assert all(p.requires_grad for p in layer.parameters())

    freeze(g)
    assert not any(p.requires_grad for p in layer.parameters())


def test_obeying_hashes_the_weights_and_that_is_the_state():
    layer = Layer(IN, MID)
    g = Graph.somatize(layer.named("encoder").frozen())
    freeze(g)

    assert g.frozen()["encoder"].startswith("sha256:")


def test_other_weights_are_another_state():
    digests = []
    for _ in range(2):
        layer = Layer(IN, MID)
        g = Graph.somatize(layer.named("encoder").frozen())
        freeze(g)
        digests.append(g.frozen()["encoder"])

    assert digests[0] != digests[1], "two initializations are two states"


def test_the_same_weights_are_the_same_state():
    # It has to hold across processes and across machines, or a shared store is
    # not shared: the digest is of the tensors, not of a `torch.save`.
    layer = Layer(IN, MID)
    first = Graph.somatize(layer.named("encoder").frozen())
    second = Graph.somatize(layer.named("encoder").frozen())
    freeze(first)
    freeze(second)

    assert first.frozen()["encoder"] == second.frozen()["encoder"]


def test_a_node_with_no_state_has_no_digest_and_is_still_settled():
    g = Graph.somatize(Label().named("label").frozen())
    freeze(g)

    assert g.frozen() == {"label": None}


def test_training_obeys_whatever_was_declared():
    # `Trainer` calls it, so a `.frozen()` in the expression is true before the
    # first step and not after somebody notices the loss going flat.
    encoder, head = Layer(IN, MID), Layer(MID, CLASSES)
    g = Graph.somatize(encoder.named("encoder").frozen() >> head.named("head"))
    Trainer(
        g,
        objective=nn.functional.cross_entropy,
        optimizer=torch.optim.Adam(parameters(g), lr=0.05),
    )

    assert not any(p.requires_grad for p in encoder.parameters())
    assert all(p.requires_grad for p in head.parameters())


def test_the_optimizer_still_points_at_the_same_weights():
    # Settling does not replace a tensor, it flips a flag on it: whoever was
    # already holding the parameter is holding the same object afterwards.
    encoder = Layer(IN, MID)
    g = Graph.somatize(encoder.named("encoder").frozen() >> Layer(MID, CLASSES).named("head"))
    before = [id(p) for p in parameters(g)]

    freeze(g)

    assert [id(p) for p in parameters(g)] == before


# ── A tensor, written down and read back ──


def test_a_tensor_goes_to_bytes_and_comes_back():
    x = torch.randn(3, 4)
    assert torch.equal(load(dump(x)), x)


def test_it_comes_back_on_the_cpu():
    # A store is shared between machines; one that only reads back where it was
    # written is not shared at all. Whoever receives it moves it, which is what
    # a placed node already does with its input.
    assert load(dump(torch.randn(2, 2))).device.type == "cpu"


def test_the_codec_is_registered_by_importing():
    from somatize._somatize import codecs_registered

    assert KIND in codecs_registered()


# ── The whole thing, which is what it was for ──


def test_a_settled_encoder_is_kept_and_read_back(tmp_path):
    encoder = Layer(IN, MID)
    g = Graph.somatize(encoder.named("encoder").frozen().cached())
    freeze(g)
    x = batch()

    first = g.forward(Opaque(x), store=str(tmp_path))
    second = g.forward(Opaque(x), store=str(tmp_path))

    assert encoder.calls == 1, "the second run read it instead of running it"
    assert torch.equal(first, second), "and what came back is the same, bit for bit"


def test_changing_the_head_does_not_recompute_the_embedding(tmp_path):
    # The labchain case with real tensors: the head changes twenty times in an
    # afternoon and the encoder runs once.
    encoder = Layer(IN, MID)
    x = batch()

    for _ in range(3):
        head = Layer(MID, CLASSES)
        g = Graph.somatize(
            encoder.named("encoder").frozen().cached() >> head.named("head")
        )
        freeze(g, "encoder")
        g.forward(Opaque(x), store=str(tmp_path))

    assert encoder.calls == 1


def test_what_comes_back_is_a_leaf(tmp_path):
    # Said out loud because it will look like a bug the first time somebody sees
    # it: no gradient flows through a cached prefix. What is restored has no
    # history — it is why the prefix has to be settled in the first place.
    encoder = Layer(IN, MID)
    g = Graph.somatize(encoder.named("encoder").frozen().cached())
    freeze(g)
    x = batch()

    g.forward(Opaque(x), store=str(tmp_path))
    read_back = g.forward(Opaque(x), store=str(tmp_path))

    assert read_back.grad_fn is None
    assert not read_back.requires_grad


def test_a_node_that_only_answers_parameters_is_settled_too(tmp_path):
    # Two ducks, because the project's own nodes use both. If `freeze` only knew
    # `state_dict`, a node like this would be told to settle by the check before
    # a run and have nothing to settle with.
    class OnlyParameters(Node):
        def __init__(self):
            self.lin = nn.Linear(IN, MID)

        def forward(self, x, ctx):
            return Opaque(self.lin(x))

        def parameters(self):
            return list(self.lin.parameters())

    g = Graph.somatize(OnlyParameters().named("encoder").frozen().cached())
    freeze(g)

    assert g.frozen()["encoder"].startswith("sha256:")
    # And the check before a run lets it through, which is the half that would
    # have been a dead end.
    g.forward(Opaque(batch()), store=str(tmp_path))
