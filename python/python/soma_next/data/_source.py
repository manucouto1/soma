"""Where the rows come from, and why a graph is handed a coordinate.

    from soma_next import Graph, Store
    from soma_next.data import Parquet, settle

    sms = Parquet(Store("/data"), "sms/train")
    g = Graph.somatize(sms.named("sms").frozen() >> Clean().named("clean").cached())
    settle(g)

    g.forward({"at": 0, "take": 64}, store="/data")

## What a source is, and what it is not

It is **a node**, and deliberately not a kind of its own: the DSL, `.on()`,
`.at()`, `.cached()`, the record and the figure all reach it because it has a
`forward` and for no other reason. There is no `Source` trait anywhere in this
library, and a second one with a method that does what `forward` does would have
been a hole with one tenant.

What it has that other nodes do not is a **version** — what this dataset *is* —
and that is what makes handing over a coordinate rather than a batch honest.

## Why a coordinate

The graph's input is the one value a cache hashes by looking at all of it. A
batch of images is 19 MB of looking, on every step, hit or miss: measured at
121 ms a step against 0,027 ms for a span. The rows are named instead by the
span they are and the version they came from, and both are known before anything
is read.

Which is also why a stream fits with nothing added: a span is a **position**, and
a position can be asked for twice. Rows 400..500 are the same rows tomorrow,
however much has arrived since — so a source read by span is settled, and what
moves is not its state but which spans exist.
"""

from __future__ import annotations

from soma_next import Node
from soma_next._soma_next import Source as _Source

__all__ = ["Parquet", "settle"]


class Parquet(Node):
    """A parquet file in a store, answering spans of rows.

    `Parquet(store, name)` where `name` is what the file is bound under. It
    resolves the name and reads **nothing**: a graph that names a dataset has
    not opened it.
    """

    def __init__(self, store, name):
        self._inner = _Source(store, name)

    @property
    def version(self):
        """What this dataset is: the digest of its content, which the store had
        already worked out. It is the duck `settle` and `soma_next.torch.freeze`
        both look for."""
        return self._inner.version

    @property
    def name(self):
        """The name it was declared under, which is the graph's word."""
        return self._inner.name

    def forward(self, input, ctx):
        """The rows that span names, as a `Frame`."""
        return self._inner.forward(input)

    def __repr__(self):
        return repr(self._inner)


def settle(graph, *node_ids):
    """Says what each declared-frozen source is settled at.

    The same shape as `soma_next.torch.freeze`, and for the same reason: the
    graph **declares** that a node's state does not change, and whoever knows
    what is inside makes it true. For weights that means hashing them; here it
    means repeating what the store already knew, so this costs nothing.

    Without it a dataset is not in the key of anything computed from it, and two
    datasets share a name — which the graph refuses before running rather than
    letting you find out from the wrong rows.
    """
    for node_id in node_ids or list(graph.frozen()):
        version = getattr(graph.implementation(node_id), "version", None)
        if version is not None:
            graph.freeze(node_id, version)
