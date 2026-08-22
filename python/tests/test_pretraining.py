"""Pretraining a body, settling it, and keeping what it produces.

The case the whole cache was written for, end to end and with real gradients:

1. a **body** worth training once — here two blocks, pretrained against a
   decoder that is then thrown away;
2. a **head** that changes twenty times in an afternoon;
3. and a **tokenizer** that has no gradients at all, no weights and no state,
   and still costs something to run every time.

The three are nodes and nothing tells them apart but what they return, which is
the point of a single contract. What tells them apart *here* is what is said
about them: the tokenizer and the body are settled and kept, the head is not.

What this file is defending, beyond the numbers:

- **a node with no gradients is not a special case.** It is settled like any
  other, it is kept like any other, and what it produces does not have to be a
  tensor — a list of ints crosses and is kept without anybody registering a
  codec for it;
- **`freeze` settles what was declared, not the graph.** The head keeps its
  gradient in the same call that takes the body's away;
- **what is kept is per batch, not per step.** Six epochs over four batches is
  twenty-four forwards of the head and four of the body.
"""

import pytest

from soma_next import Graph, Node, Opaque

torch = pytest.importorskip("torch")
nn = torch.nn

from soma_next.torch import NoGradient, Trainer, freeze, parameters  # noqa: E402

VOCAB, DIM, HID, CLASSES, LENGTH = 32, 8, 6, 3, 4
TEXTS = ["the dog runs", "a cat sleeps", "birds fly high", "the fish swims"]


# ── A node with no gradients, no weights and no state ──


class Tokenize(Node):
    """Text to ids. Deterministic on purpose: `hash()` of a str is salted per
    process, and a cache keyed by content cannot live with that."""

    def __init__(self):
        self.calls = 0

    def forward(self, texts, ctx):
        self.calls += 1
        ids = []
        for text in texts:
            words = [sum(map(ord, w)) % VOCAB for w in text.split()][:LENGTH]
            ids.append(words + [0] * (LENGTH - len(words)))
        # A plain list: it crosses, and it is kept, without a codec.
        return ids


# ── The body: trainable first, settled afterwards ──


class Embed(Node):
    def __init__(self):
        self.table = nn.Embedding(VOCAB, DIM)
        self.calls = 0

    def forward(self, ids, ctx):
        self.calls += 1
        return Opaque(self.table(torch.tensor(ids).long()).mean(dim=1))

    def parameters(self):
        return list(self.table.parameters())

    def state_dict(self):
        return self.table.state_dict()


class Block(Node):
    def __init__(self, wide, tall):
        self.layers = nn.Sequential(nn.Linear(wide, tall), nn.ReLU())
        self.calls = 0

    def forward(self, x, ctx):
        self.calls += 1
        return Opaque(self.layers(x))

    def parameters(self):
        return list(self.layers.parameters())

    def state_dict(self):
        return self.layers.state_dict()


class Decoder(Block):
    """Only for pretraining. It never reaches the graph that gets trained."""


class Head(Node):
    """The classifier: logits, and no activation on them."""

    def __init__(self, wide=HID, tall=CLASSES):
        self.layers = nn.Linear(wide, tall)
        self.calls = 0

    def forward(self, x, ctx):
        self.calls += 1
        return Opaque(self.layers(x))

    def parameters(self):
        return list(self.layers.parameters())

    def state_dict(self):
        return self.layers.state_dict()


@pytest.fixture
def body():
    """The tokenizer and the two trainable pieces, declared once. The same
    objects go into every graph below, which is what makes them the same
    weights."""
    torch.manual_seed(0)
    tokenize, embed, block = Tokenize(), Embed(), Block(DIM, HID)
    # Nobody is named by hand: an id nobody gives comes from the class,
    # `CleanText` -> `clean_text`, and it is suffixed if it is already taken.
    return tokenize >> embed >> block, (tokenize, embed, block)


@pytest.fixture
def batches():
    return [([TEXTS[i], TEXTS[(i + 1) % 4]], torch.tensor([i % CLASSES, 0])) for i in range(4)]


def pretrained(expression, batches):
    """Phase one: train the body against a decoder, and throw the decoder away."""
    graph = Graph.somatize(expression >> Decoder(HID, DIM))
    return Trainer(
        graph,
        objective=torch.nn.functional.mse_loss,
        optimizer=torch.optim.Adam(parameters(graph), lr=1e-2),
    ).fit([(texts, torch.zeros(2, DIM)) for texts, _ in batches], epochs=3)


def settled(expression, head, store, lr=1e-2):
    """Phase two: the same pieces, settled and kept, under a head that trains."""
    graph = Graph.somatize(expression.frozen().cached() >> head)
    return graph, Trainer(
        graph,
        objective=torch.nn.functional.cross_entropy,
        optimizer=torch.optim.Adam(
            [p for p in parameters(graph) if p.requires_grad], lr=lr
        ),
        store=store,
    )


# ── The whole thing ──


def test_the_body_runs_once_per_batch_and_the_head_once_per_step(body, batches, tmp_path):
    expression, (tokenize, embed, block) = body
    pretrained(expression, batches)
    ran_before = (tokenize.calls, embed.calls, block.calls)

    graph, trainer = settled(expression, Head(HID, CLASSES), str(tmp_path))
    report = trainer.fit(batches, epochs=6)

    head = graph.implementation("head")
    assert head.calls == 24, "six epochs over four batches"
    assert tokenize.calls - ran_before[0] == 4, "one per batch, and never again"
    assert embed.calls - ran_before[1] == 4
    assert block.calls - ran_before[2] == 4
    assert report.loss < report.history[0], "and the head did train"


def test_another_head_over_the_same_body_runs_none_of_it(body, batches, tmp_path):
    # The labchain case: what changes is the head, and what is underneath it was
    # named before it existed.
    expression, (tokenize, embed, block) = body
    pretrained(expression, batches)
    _, first = settled(expression, Head(HID, CLASSES), str(tmp_path))
    first.fit(batches, epochs=2)
    ran = (tokenize.calls, embed.calls, block.calls)

    _, second = settled(expression, Head(HID, CLASSES), str(tmp_path), lr=5e-3)
    report = second.fit(batches, epochs=6)

    assert (tokenize.calls, embed.calls, block.calls) == ran, "not one of them ran"
    assert report.loss < report.history[0]


def test_training_the_head_does_not_move_the_body(body, batches, tmp_path):
    expression, (_, embed, block) = body
    pretrained(expression, batches)
    before = [p.clone() for p in embed.parameters() + block.parameters()]

    graph, trainer = settled(expression, Head(HID, CLASSES), str(tmp_path))
    trainer.fit(batches, epochs=3)

    assert all(
        torch.equal(was, now)
        for was, now in zip(before, embed.parameters() + block.parameters())
    ), "a settled body that moves would make every key kept under it a lie"


def test_the_gradient_reaches_the_head_and_stops_there(body, batches, tmp_path):
    expression, (_, embed, block) = body
    pretrained(expression, batches)
    graph, trainer = settled(expression, Head(HID, CLASSES), str(tmp_path))

    trainer.step(batches[0])

    head = graph.implementation("head")
    assert all(p.grad is not None for p in head.parameters())
    assert all(not p.requires_grad for p in embed.parameters() + block.parameters())


def test_what_is_settled_and_what_is_kept_is_what_was_said(body, batches, tmp_path):
    expression, _ = body
    pretrained(expression, batches)
    graph, _ = settled(expression, Head(HID, CLASSES), str(tmp_path))

    assert list(graph.frozen()) == ["tokenize", "embed", "block"]
    assert list(graph.cached()) == ["tokenize", "embed", "block"]
    assert graph.frozen()["tokenize"] is None, "nothing to hash, and it is settled"
    assert graph.frozen()["embed"].startswith("sha256:")
    assert graph.identities() == {
        "tokenize": "Tokenize",
        "embed": "Embed",
        "block": "Block",
        "head": "Head",
    }


def test_there_is_one_kept_value_per_node_and_batch(body, batches, tmp_path):
    expression, _ = body
    pretrained(expression, batches)
    _, trainer = settled(expression, Head(HID, CLASSES), str(tmp_path))
    trainer.fit(batches, epochs=3)

    names = list((tmp_path / "names").rglob("sha256*"))
    assert len(names) == 12, "three kept nodes over four batches"


# ── The node with no gradients, which is not a special case ──


def test_a_tokenizer_is_kept_without_anybody_registering_a_codec(body, batches, tmp_path):
    # What it returns is a list of ints, not a tensor: it crosses as data and is
    # kept as data. `Opaque` is for what cannot.
    expression, (tokenize, _, _) = body
    graph = Graph.somatize(tokenize.frozen().cached())

    first = graph.forward(TEXTS[:2], store=str(tmp_path))
    second = graph.forward(TEXTS[:2], store=str(tmp_path))

    assert tokenize.calls == 1
    assert first == second


def test_it_needs_no_freeze_call_because_it_has_no_state(body, tmp_path):
    # `soma_next.torch.freeze` is for turning gradients off and hashing weights.
    # Something with neither is settled by saying so and nothing else — and the
    # check before a run knows the difference.
    expression, (tokenize, _, _) = body
    graph = Graph.somatize(tokenize.frozen().cached())

    assert graph.forward(TEXTS[:2], store=str(tmp_path)) is not None


def test_it_still_has_to_be_declared_settled(body, batches, tmp_path):
    # And this is not a formality. The core cannot tell a tokenizer from a net:
    # what it knows is that everything upstream of something kept has to be
    # unable to change, and a tokenizer with an adaptive vocabulary would be
    # exactly the thing that breaks it.
    expression, (tokenize, embed, block) = body
    graph = Graph.somatize(tokenize >> (embed >> block).frozen().cached())
    # Obedience is checked first, and it is a different complaint: settle the
    # body so that what is left is the one being tested.
    freeze(graph)

    with pytest.raises(ValueError, match="tokenize"):
        graph.forward(TEXTS[:2], store=str(tmp_path))


def test_a_head_that_still_trains_cannot_be_kept(body, batches, tmp_path):
    expression, (tokenize, embed, block) = body
    graph = Graph.somatize(
        expression.frozen() >> Head(HID, CLASSES).frozen().cached()
    )
    freeze(graph, "tokenize", "embed", "block")

    with pytest.raises(ValueError, match="head"):
        graph.forward(TEXTS[:2], store=str(tmp_path))


def test_a_body_declared_settled_and_never_settled_refuses_to_run(body, tmp_path):
    # Without the digest of the weights the key does not depend on them, and two
    # checkpoints of the same class would be kept under one name.
    expression, _ = body
    graph = Graph.somatize(expression.frozen().cached() >> Head(HID, CLASSES))

    with pytest.raises(ValueError, match="checkpoints"):
        graph.forward(TEXTS[:2], store=str(tmp_path))


def test_the_trainer_settles_what_was_declared_on_its_own(body, batches, tmp_path):
    # Which is why the tests above never call `freeze` by hand: declaring it is
    # the graph's half, and `Trainer` does the other one before the first step.
    expression, (_, embed, _) = body
    graph, _ = settled(expression, Head(HID, CLASSES), str(tmp_path))

    assert graph.frozen()["embed"].startswith("sha256:")
    assert not any(p.requires_grad for p in embed.parameters())


# ── That nothing is trained by halves in silence ──


class Cut(Node):
    """Has weights and hands out something the backward pass cannot cross.

    It is what a node on **another host** looks like from here — what crosses a
    wire is the value, not the graph that made it — without needing another host
    to see it. A cache hit looks the same, and so does a branch the loss never
    reads: one symptom, one check.
    """

    def __init__(self):
        self.layers = nn.Linear(DIM, HID)
        self.calls = 0

    def forward(self, x, ctx):
        self.calls += 1
        return Opaque(self.layers(x).detach())

    def parameters(self):
        return list(self.layers.parameters())


@pytest.fixture
def tensors():
    """A batch that goes straight into a `Linear`, without the tokenizer."""
    torch.manual_seed(0)
    return Opaque(torch.randn(2, DIM)), torch.tensor([0, 1])


def test_training_something_the_gradient_never_reaches_stops(tensors):
    # Without this the run does not fail: it trains the head, the loss comes
    # down because the head is learning, and the body never moves. Silently
    # wrong numbers, which is the one failure worth a type of its own.
    cut = Cut()
    graph = Graph.somatize(cut >> Head())
    trainer = Trainer(
        graph,
        objective=torch.nn.functional.cross_entropy,
        optimizer=torch.optim.Adam(parameters(graph), lr=1e-2),
    )

    with pytest.raises(NoGradient) as raised:
        trainer.step(tensors)

    said = str(raised.value)
    assert "`cut`" in said, said
    assert "another host" in said, said


def test_leaving_them_out_of_the_optimizer_is_how_you_say_it_is_deliberate(tensors):
    # Split learning is exactly this on purpose: the far side runs its own
    # backward with its own optimizer, and this one holds only what it trains.
    cut = Cut()
    graph = Graph.somatize(cut >> Head())
    head = graph.implementation("head")
    trainer = Trainer(
        graph,
        objective=torch.nn.functional.cross_entropy,
        optimizer=torch.optim.Adam(head.parameters(), lr=1e-2),
    )

    assert trainer.step(tensors) > 0.0


def test_a_settled_body_is_not_an_orphan(body, batches, tmp_path):
    # And the everyday case does not trip it: `freeze` leaves `requires_grad`
    # off, so those parameters are not waiting for a gradient — they are done.
    expression, _ = body
    _, trainer = settled(expression, Head(HID, CLASSES), str(tmp_path))

    assert trainer.step(batches[0]) > 0.0


def test_it_is_asked_once_and_not_on_every_step(body, batches, tmp_path):
    expression, _ = body
    _, trainer = settled(expression, Head(HID, CLASSES), str(tmp_path))
    trainer.step(batches[0])

    assert trainer._checked, "the second step onwards pays nothing"
