"""soma-next: el gemelo re-derivado de Soma, un caso de uso cada vez.

Un nodo es cualquier cosa que sabe avanzar un turno::

    from soma_next import Await, Done, Graph, Node

    class Limpiar(Node):
        def forward(self, x, ctx):
            return Done(x.strip())

    class Preguntar(Node):
        def forward(self, x, ctx):
            if ctx.turn == 0:
                return Await([f"¿y {x}?"])
            return Done(ctx.results[0])

    g = Graph.somatize(Limpiar() >> Preguntar())
    g.forward("  hola  ", driver=MiDriver())

`Graph()` con `node()` y `edge()` sigue estando para cuando la topología se
construye en un bucle o viene de fuera.
"""

from soma_next._dsl import Node
from soma_next._graph import Graph
from soma_next._soma_next import Await, Ctx, Done, Opaque, __version__

__all__ = ["Await", "Ctx", "Done", "Graph", "Node", "Opaque", "__version__"]
