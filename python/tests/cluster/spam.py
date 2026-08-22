"""The domain: real messages, and the three nodes that classify them.

This is the half a user's own project would contain — the data they have and the
pipeline they wrote. Nothing about searching or about testing is in here.

The dataset is the SMS Spam Collection: 5,574 real text messages, 747 of them
spam. Real rather than generated because what a study is asked to find is a
learning rate **that works on this data**, and noise has no learning rate that
works on it. The imbalance is useful too: 13% spam means "always ham" is already
right 87% of the time, so a loss that comes down is a loss that learnt something.

The three nodes are a pipeline and not a shape, and what makes them worth having
here is **where each one can run**:

    Clean          plain Python — runs on a 193 MB worker with no torch in it
      >> Embed     torch — runs on the machine that has it, and trains there
      >> Classify  torch — runs where the loop is

Only this module travels to the workers. The dataset does not: data reaches a
worker as the input of a graph, like everything else.
"""

from __future__ import annotations

from soma_next import Node

NAME = "ucirvine/sms_spam"

VOCAB = 4096
"""How many buckets a word can hash into, `0` reserved for padding. Small on
purpose: an embedding table is `VOCAB x dim` and this one crosses a wire on every
configuration a study tries."""

WIDTH = 32
"""How many words of a message are kept. Cutting to a fixed width is what makes a
batch a **rectangle**, and a batch that is not one cannot be the seam of a
`Split`: what comes back the other way is `dL/da`, and a ragged thing has no
shape to have a gradient of."""


# ── The pipeline ──


class Clean(Node):
    """Text in, a rectangle of token ids out — and **not one line of torch**.

    Which is the point: this runs on a worker whose image is 193 MB and has no
    tensors in it at all. The expensive machine is for the part that needs it,
    and preprocessing is not that part. A cluster is heterogeneous or it is just
    several of the same computer.
    """

    def __init__(self, width=WIDTH):
        self.width = width

    def forward(self, texts, ctx):
        return [_padded(_ids(one), self.width) for one in texts]


class Embed(Node):
    """Ids in, one vector per message out.

    Averaged over the words the message really has — the padding is left out of
    the mean rather than dragging it towards whatever `0` learnt. The cheapest
    embedding that is really one.

    It is also the only node here with something to learn that does **not** run
    where the loop is, which is what a `Split` is for.
    """

    def __init__(self, dim=16, vocab=VOCAB):
        import torch

        self.table = torch.nn.Embedding(vocab, dim, padding_idx=0)

    def forward(self, x, ctx):
        import torch

        from soma_next import Opaque

        landed = x if torch.is_tensor(x) else torch.tensor(x)
        ids = landed.long()
        said = self.table(ids)
        # The mean over the real words. `clamp(min=1)` is for a message that
        # cleaned down to nothing at all — dividing by zero would put a `NaN`
        # into the batch, and one `NaN` is the whole batch.
        kept = (ids != 0).unsqueeze(-1).to(said.dtype)
        return Opaque((said * kept).sum(1) / kept.sum(1).clamp(min=1.0))

    def parameters(self):
        return list(self.table.parameters())


class Classify(Node):
    """A vector in, a class out. Plain torch, and it runs where the loop is."""

    def __init__(self, dim=16, classes=2):
        import torch

        self.lin = torch.nn.Linear(dim, classes)

    def forward(self, x, ctx):
        import torch

        from soma_next import Opaque

        landed = x if torch.is_tensor(x) else torch.tensor(x, dtype=torch.float32)
        return Opaque(self.lin(landed))

    def parameters(self):
        return list(self.lin.parameters())


def _ids(text):
    """A message as bucket numbers, none of them zero.

    `crc32` and not `hash`, and it is not a preference: Python salts `hash` per
    **process**, so the same word would become a different id on every worker and
    an embedding trained on one machine would be nonsense on the next. What two
    machines must agree on cannot be left to something that is random per
    process — the same reason the samplers carry their own `splitmix`.
    """
    import re
    import zlib

    words = re.findall(r"[a-z0-9']+", text.lower())
    return [1 + zlib.crc32(word.encode()) % (VOCAB - 1) for word in words]


def _padded(ids, width):
    """Cut to `width`, and filled with the padding id when it is short."""
    return (ids + [0] * width)[:width]


# ── The data ──


def messages(how_many=1600):
    """`(texts, labels)` — the first `how_many`, shuffled, spam as 1.

    It raises rather than skipping. Whether an unreachable hub is a skip or a
    failure is the caller's to decide, and this module is imported by processes
    where `pytest.skip` would come out as a crash.
    """
    from datasets import load_dataset

    whole = load_dataset(NAME, split="train")
    # Seeded, so every machine in a study trains on the same messages in the same
    # order — otherwise two trials would differ by their data as well as by their
    # configuration, and the search would be comparing nothing.
    shuffled = whole.shuffle(seed=0).select(range(min(how_many, len(whole))))
    return list(shuffled["sms"]), list(shuffled["label"])


def batches(texts, labels, size=64):
    """The messages in batches, as a `Trainer` takes them."""
    import torch

    return [
        (texts[at : at + size], torch.tensor(labels[at : at + size], dtype=torch.long))
        for at in range(0, len(labels), size)
    ]
