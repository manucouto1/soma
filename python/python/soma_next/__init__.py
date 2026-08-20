"""soma-next: the re-derived twin of Soma, one use case at a time.

A node is anything that knows how to advance one turn::

    from soma_next import Await, Done, Graph, Node

    class Clean(Node):
        def forward(self, x, ctx):
            return Done(x.strip())

    class Ask(Node):
        def forward(self, x, ctx):
            if ctx.turn == 0:
                return Await([f"and {x}?"])
            return Done(ctx.results[0])

    g = Graph.somatize(Clean() >> Ask())
    g.forward("  hello  ", driver=MyDriver())

`Graph()` with `node()` and `edge()` is still there for when the topology is
built in a loop or comes from outside.

`codec(kind, type, dump=..., load=...)` says how something wrapped in `Opaque`
is written down, which is what lets a graph **keep** what it produces:
`soma_next.torch` registers the one for a tensor on being imported.
"""

from soma_next._dsl import Node
from soma_next._graph import Graph
from soma_next._remote import Worker
from soma_next._soma_next import Await, Ctx, Done, Opaque, codec, __version__

__all__ = [
    "Await",
    "Ctx",
    "Done",
    "Graph",
    "Node",
    "Opaque",
    "Worker",
    "codec",
    "__version__",
]
