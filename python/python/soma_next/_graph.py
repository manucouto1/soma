"""`Graph` — el objeto de Rust más lo que solo puede vivir en Python.

Casi todo el `Graph` está en Rust. Lo que no puede estarlo es `somatize`: recibe
una expresión hecha de objetos Python y la recorre, así que vive aquí.

Está escrito como un método en el cuerpo de la clase, y no asignado sobre la
clase de Rust al importar, porque un `#[pyclass]` es un tipo inmutable —no se le
pueden colgar atributos— y porque aunque se pudiera, lo asignado no lo ve ni
`help()`, ni un IDE, ni un comprobador de tipos.
"""

from __future__ import annotations

from soma_next import _dsl
from soma_next._soma_next import Graph as _RustGraph


class Graph(_RustGraph):
    """Un grafo de cómputo: nodos, aristas y lo que ejecuta cada uno.

    Todo lo demás —`node`, `step`, `edge`, `forward`, `plan` y las consultas de
    topología— se hereda de la clase de Rust.
    """

    @classmethod
    def somatize(cls, topology):
        """Materializa una expresión en un grafo ejecutable.

        Lo piensas, Soma lo somatiza::

            Graph.somatize(Fuente() >> (Izq() | Der()) >> Media())
        """
        return _dsl.somatize(cls, topology)
