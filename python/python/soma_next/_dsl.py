"""Declarar un grafo como una expresión, en vez de a base de llamadas.

    Graph.somatize(Fuente() >> (Izq() | Der()) >> Media())

`>>` encadena y `|` abre en ramas. Un `>>` entre dos ramas abiertas las une:
lo que salga de todas entra en lo que venga detrás, que es exactamente el
fan-in — el nodo de la derecha recibe un mapa con la clave de cada rama.

Hay **una sola clase de nodo**, igual que en el núcleo hay un solo trait. Un
nodo devuelve `Done(valor)` si ya está o `Await([peticiones])` si necesita algo
del mundo antes de seguir; lo que en otros sitios se llama un filtro es
simplemente un nodo que siempre contesta `Done`.

Ojo con la precedencia, que es la de Python (y la misma en Rust): `>>` aprieta
más que `|`, así que las ramas van entre paréntesis.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Topology:
    """Un trozo de grafo a medio declarar."""

    def __rshift__(self, other):
        return Chain(_steps(self) + _steps(_wrap(other)))

    def __rrshift__(self, other):
        return Chain(_steps(_wrap(other)) + _steps(self))

    def __or__(self, other):
        return Fork(_branches(self) + _branches(_wrap(other)))

    def __ror__(self, other):
        return Fork(_branches(_wrap(other)) + _branches(self))


class Chain(Topology):
    """Uno detrás de otro."""

    def __init__(self, steps):
        self.steps = steps


class Fork(Topology):
    """Ramas que no se tocan."""

    def __init__(self, branches):
        self.branches = branches


class Declared(Topology):
    """Un objeto declarado como nodo, con su id si se lo pusiste."""

    def __init__(self, obj, node_id=None):
        self.obj = obj
        self.node_id = node_id


class Node(Topology, ABC):
    """Lo que ejecuta un nodo del grafo.

    Obliga a escribir `forward`: sin él, la clase no se puede instanciar.
    """

    @abstractmethod
    def forward(self, input, ctx):
        """Avanza un turno.

        `ctx` trae `turn` y `results`. Devuelve `Done(valor)` o
        `Await([peticiones])`.
        """

    def named(self, node_id):
        """El mismo nodo, con el id que digas."""
        return Declared(self, node_id)


def _wrap(obj):
    """Cualquier cosa, vista como topología.

    Un `Node` del usuario es un `Topology` —de ahí le vienen los operadores—
    pero no es un nodo declarado todavía: hay que envolverlo.
    """
    if isinstance(obj, (Chain, Fork, Declared)):
        return obj
    if isinstance(obj, Node):
        return Declared(obj)
    raise TypeError(
        f"`{type(obj).__name__}` no puede ir en una expresión de grafo: tiene "
        "que heredar de soma_next.Node"
    )


def _steps(topology):
    return topology.steps if isinstance(topology, Chain) else [topology]


def _branches(topology):
    return topology.branches if isinstance(topology, Fork) else [topology]


def somatize(graph_cls, topology):
    """Materializa la expresión en un grafo de la clase que te den.

    La clase entra por parámetro y no se importa aquí: `soma_next._graph`
    importa este módulo, así que importarlo de vuelta sería un ciclo.
    """
    g = graph_cls()
    _walk(g, _wrap(topology), [])
    return g


def _walk(g, topology, sources):
    """Añade lo declarado y devuelve por dónde sale este trozo."""
    if isinstance(topology, Chain):
        cursor = sources
        for step in topology.steps:
            cursor = _walk(g, _wrap(step), cursor)
        return cursor

    if isinstance(topology, Fork):
        return [
            terminal
            for branch in topology.branches
            for terminal in _walk(g, _wrap(branch), sources)
        ]

    node_id = (
        g.node(topology.node_id, topology.obj) if topology.node_id else g.node(topology.obj)
    )
    for source in sources:
        g.edge(source, node_id)
    return [node_id]
