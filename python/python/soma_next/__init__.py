"""soma-next: el gemelo re-derivado de Soma, un caso de uso cada vez.

La forma normal de escribir un grafo es la expresión::

    from soma_next import Filter, build

    class Limpiar(Filter):
        def forward(self, x): ...

    g = build(Limpiar() >> (Izq() | Der()) >> Media())
    g.forward(datos)

`Graph()` con `node()` y `edge()` sigue estando para cuando la topología se
construye en un bucle o viene de fuera.
"""

from soma_next._dsl import Filter, Step, build
from soma_next._soma_next import Graph, __version__

__all__ = ["Filter", "Graph", "Step", "build", "__version__"]
