"""A real end-to-end pipeline: text → encoder → bottleneck → LSTM.

It is the case that motivated `Opaque`, and it is here in full for two reasons:
it checks that the gradients cross several nodes, and it **documents the
pattern** — how a node with parameters is written, where the training loop goes
and what happens when a node in the pipeline has no gradients.

The "LLM" is an `Embedding` + `TransformerEncoderLayer`, not a real model: what
is tested is the seam, not the model. With a HuggingFace `AutoModel` the pattern
is identical.
"""

import pytest

from soma_next import Graph, Node, Opaque

torch = pytest.importorskip("torch")
nn = torch.nn

VOCAB, DIM, BOTTLENECK, CLASSES, LENGTH = 64, 32, 8, 3, 6


def _ids(texts):
    """A toy tokenizer, deliberately deterministic.

    `hash()` of a str is salted per process in Python, so a test using it would
    give a different tokenization on every run.
    """
    rows = []
    for t in texts:
        words = [sum(map(ord, w)) % VOCAB for w in t.split()][:LENGTH]
        rows.append(words + [0] * (LENGTH - len(words)))
    return torch.tensor(rows)


# ── A node without gradients: text → text, and it is NOT wrapped ──


class Lemmatizer(Node):
    def forward(self, texts, ctx):
        return [t.strip().lower().replace("running", "run") for t in texts]


# ── The three with parameters. Note: they hold the modules, they do not
# inherit from nn.Module ──
#
# Inheriting from `nn.Module` registers the parameters on its own, but breaks
# calling the node as a module: our `forward` carries `ctx` and torch calls it
# without one.


class Encoder(Node):
    """Where the gradient graph begins."""

    def __init__(self):
        self.emb = nn.Embedding(VOCAB, DIM)
        self.enc = nn.TransformerEncoderLayer(DIM, 4, dim_feedforward=64, batch_first=True)
        self.last_output = None

    def forward(self, texts, ctx):
        self.last_output = self.enc(self.emb(_ids(texts)))
        return Opaque(self.last_output)

    def parameters(self):
        return list(self.emb.parameters()) + list(self.enc.parameters())


class Bottleneck(Node):
    def __init__(self):
        self.proj = nn.Linear(DIM, BOTTLENECK)
        self.last_input = None

    def forward(self, h, ctx):
        self.last_input = h
        return Opaque(self.proj(h))

    def parameters(self):
        return list(self.proj.parameters())


class Classifier(Node):
    def __init__(self):
        self.lstm = nn.LSTM(BOTTLENECK, 16, batch_first=True)
        self.head = nn.Linear(16, CLASSES)

    def forward(self, h, ctx):
        output, _ = self.lstm(h)
        return Opaque(self.head(output[:, -1, :]))

    def parameters(self):
        return list(self.lstm.parameters()) + list(self.head.parameters())


TEXTS = ["  The dog Running fast  ", "cat sleeps a lot", "  Bird flies high  "]
LABELS = torch.tensor([0, 1, 2])


@pytest.fixture
def pipeline():
    torch.manual_seed(0)
    nodes = (Lemmatizer(), Encoder(), Bottleneck(), Classifier())
    return Graph.somatize(nodes[0] >> nodes[1] >> nodes[2] >> nodes[3]), nodes


def _parameters(g):
    """What a `soma_next.torch.parameters(g)` would do, if it did not exist."""
    return [p for nid in g.nodes() for p in getattr(g.implementation(nid), "parameters", list)()]


# ── The topology ──


def test_each_part_is_a_node(pipeline):
    g, _ = pipeline
    assert g.nodes() == ["lemmatizer", "encoder", "bottleneck", "classifier"]
    assert g.plan().count("Execute") == 4


# ── The gradients ──


def test_the_backward_pass_crosses_the_three_nodes_with_parameters(pipeline):
    g, (_, encoder, bottleneck, classifier) = pipeline

    logits = g.forward(TEXTS)
    assert logits.shape == (len(TEXTS), CLASSES)
    assert logits.grad_fn is not None, "the output is still attached to the graph"

    torch.nn.functional.cross_entropy(logits, LABELS).backward()

    for name, node in (("encoder", encoder), ("bottleneck", bottleneck), ("clf", classifier)):
        missing = [p for p in node.parameters() if p.grad is None]
        assert not missing, f"{name} has {len(missing)} parameters without a gradient"


def test_the_tensor_crosses_the_edge_as_the_same_object(pipeline):
    g, (_, encoder, bottleneck, _) = pipeline
    g.forward(TEXTS)
    assert bottleneck.last_input is encoder.last_output


def test_the_node_without_gradients_coexists_without_breaking_anything(pipeline):
    g, (lemmatizer, _, _, _) = pipeline
    # It has no parameters, and its output crosses converted (text), not opaque.
    assert not hasattr(lemmatizer, "parameters")
    assert lemmatizer.forward(["  Running  "], None) == ["run"]

    g.forward(TEXTS)  # and the whole pipeline still works


# ── The training loop, which goes OUTSIDE the graph ──


def test_the_pipeline_trains(pipeline):
    g, _ = pipeline
    opt = torch.optim.Adam(_parameters(g), lr=0.01)

    first = last = None
    for step in range(30):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(g.forward(TEXTS), LABELS)
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()

    assert last < first / 2, f"the loss went down from {first:.4f} to {last:.4f}"


def test_the_weights_the_optimizer_updates_are_the_ones_the_graph_uses(pipeline):
    g, (_, _, bottleneck, _) = pipeline
    before = bottleneck.proj.weight.detach().clone()

    opt = torch.optim.Adam(_parameters(g), lr=0.1)
    opt.zero_grad()
    torch.nn.functional.cross_entropy(g.forward(TEXTS), LABELS).backward()
    opt.step()

    assert not torch.allclose(before, bottleneck.proj.weight), "the step changed nothing"
    assert g.implementation("bottleneck") is bottleneck, "the graph holds the same object"
