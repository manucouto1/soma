"""Where the rows come from, and why a graph is handed a coordinate.

    from somatize.data import Parquet, settle

    sms = Parquet(Store("/data"), "sms/train")
    g = Graph.somatize(sms.named("sms").frozen() >> Clean().named("clean").cached())
    settle(g)
    g.forward({"at": 0, "take": 64}, store="/data")

A source is **a node**, deliberately not a kind of its own: the DSL, `.on()`,
`.at()`, `.cached()`, the record and the figure all reach it because it has a
`forward`. What it has that other nodes do not is a **version** — what this
dataset *is* — and that is what makes handing over a coordinate honest.

The graph's input is the one value a cache hashes by looking at all of it: a
batch of images is 19 MB of looking on every step, hit or miss — 121 ms a step
against 0,027 ms for a span. The rows are named instead by the span they are and
the version they came from, both known before anything is read.

Which is also why a stream fits with nothing added: a span is a **position**, so
what moves is not the source's state but which spans exist.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from somatize._graph import Graph
    from somatize._somatize import Ctx, Frame, Store

from somatize import Node
from somatize._somatize import Source as _Source

__all__ = ["Parquet", "settle"]


class Parquet(Node):
    """A parquet file in a store, answering spans of rows.

    `Parquet(store, name)` where `name` is what the file is bound under. It
    resolves the name and reads **nothing**: a graph that names a dataset has not
    opened it.
    """

    def __init__(self, store: "Store", name: str) -> None:
        self._inner = _Source(store, name)

    @property
    def version(self) -> str:
        """What this dataset is: the digest of its content, which the store had
        already worked out. It is the duck `settle` and `somatize.torch.freeze`
        both look for."""
        return self._inner.version

    @property
    def name(self) -> str:
        """The name it was declared under, which is the graph's word."""
        return self._inner.name

    def forward(self, input: Any, ctx: "Ctx") -> "Frame":
        """The rows that span names, as a `Frame`."""
        return self._inner.forward(input)

    def __repr__(self) -> str:
        return repr(self._inner)


def settle(graph: "Graph", *node_ids: str) -> None:
    """Says what each declared-frozen source is settled at. The same shape as
    `somatize.torch.freeze`: the graph **declares** that a node's state does not
    change and whoever knows what is inside makes it true — here by repeating
    what the store already knew, so it costs nothing.

    Without it a dataset is not in the key of anything computed from it and two
    datasets share a name, which the graph refuses before running.
    """
    for node_id in node_ids or list(graph.frozen()):
        version = getattr(graph.implementation(node_id), "version", None)
        if version is not None:
            graph.freeze(node_id, version)
