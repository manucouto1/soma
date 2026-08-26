"""soma: the re-derived twin of Soma, one use case at a time.

A node is anything with a `forward`. It takes what arrived along the edges and
returns what it produced — there is no wrapper around either::

    from somatize import Graph, Node

    class Clean(Node):
        def forward(self, x, ctx):
            return x.strip()

    class Shout(Node):
        def forward(self, x, ctx):
            return x.upper()

    g = Graph.somatize(Clean() >> Shout())
    g.forward("  hello  ")

Whatever a node takes to answer — a retry, a model, three rounds of something —
happens **inside it**. `Graph()` with `node()` and `edge()` is still there for
when the topology is built in a loop or comes from outside.

`codec(kind, type, dump=..., load=...)` says how something wrapped in `Opaque` is
written down, which is what lets a graph **keep** what it produces.

`Store(directory)` is that same place, opened by hand. A directory two machines
can both see is how a training run written down on one is read back on another::

    store.keep("round/3", trainer.export())
    trainer.load(store.recall("round/3"))
"""

from somatize._dsl import Node
from somatize._graph import Graph
from somatize._remote import Broker, Worker
from somatize._somatize import (
    Bound,
    Ctx,
    Opaque,
    Recorder,
    Store,
    codec,
    __version__,
)

__all__ = [
    "Bound",
    "Ctx",
    "Graph",
    "Node",
    "Opaque",
    "Recorder",
    "Store",
    "Broker",
    "Worker",
    "codec",
    "__version__",
]
