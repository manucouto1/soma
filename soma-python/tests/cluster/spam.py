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

from somatize import Node

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
    """Rows in, a rectangle of token ids out — and **not one line of torch**.

    Which is the point: this runs on a worker whose image is 193 MB and has no
    tensors in it at all. The expensive machine is for the part that needs it,
    and preprocessing is not that part. A cluster is heterogeneous or it is just
    several of the same computer.

    What arrives is a `Frame`, because what feeds it is a dataset. `column`
    hands over plain Python values, so this worker needs no dataframe library
    either — the image stays what it is.
    """

    def __init__(self, width=WIDTH):
        self.width = width

    def forward(self, frame, ctx):
        return [_padded(_ids(one), self.width) for one in frame.column("sms")]


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

        from somatize import Opaque

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

        from somatize import Opaque

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


IN_STORE = "sms/train"
"""The name the dataset is bound under, in the directory every machine shares."""


def stored(store, texts, labels, name=IN_STORE):
    """The messages, written once as parquet into the store everybody shares.

    Fetched by whoever sets the study up and **not** by each machine: the hub is
    a hub, and a study whose every machine downloads the same file is one that
    goes red when somebody's wifi does. From here on the dataset is in the store
    like everything else, and a machine reads the spans it needs.
    """
    import pyarrow
    import pyarrow.parquet

    sink = pyarrow.BufferOutputStream()
    pyarrow.parquet.write_table(pyarrow.table({"sms": texts, "label": labels}), sink)
    store.bind(name, store.put(sink.getvalue().to_pybytes()))


def batches(store, size=64, name=IN_STORE):
    """`(span, target)` pairs, as a `Trainer` takes them.

    The input is a **coordinate** and not the messages: the graph reads the rows
    it names, and what a cache has to weigh is two numbers instead of a batch.

    The labels are the caller's, as they always were — a target is not part of
    the graph — so they come out of the same file, read here once.
    """
    import torch

    from somatize.data import Parquet

    source, at, out = Parquet(store, name), 0, []
    while True:
        frame = source.forward({"at": at, "take": size}, None)
        if not frame.rows:
            return out
        out.append(
            (
                {"at": at, "take": size},
                torch.tensor(frame.column("label"), dtype=torch.long),
            )
        )
        at += size
