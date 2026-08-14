"""Declarar un grafo como una expresión, en vez de a base de llamadas.

    build(Fuente() >> (Izq() | Der()) >> Media())

`>>` encadena y `|` abre en ramas. Un `>>` entre dos ramas abiertas las une:
lo que salga de todas entra en lo que venga detrás, que es exactamente el
fan-in — el nodo de la derecha recibe un mapa con la clave de cada rama.

Los operadores viven en `Filter` y `Step`, que son mixins vacíos: no obligan a
nada, solo dan el azúcar. Un objeto sin heredar de ellos sigue valiendo para
`g.node(obj)`, y también dentro de una expresión mientras algo a su lado sí
herede — Python prueba `__rrshift__` cuando el operando izquierdo no sabe.

Ojo con la precedencia, que es la de Python (y la misma en Rust): `>>` aprieta
más que `|`, así que las ramas van entre paréntesis.
"""

from __future__ import annotations


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


class Node(Topology):
    """Un objeto declarado como nodo, con su id si se lo pusiste."""

    def __init__(self, obj, node_id=None):
        self.obj = obj
        self.node_id = node_id

    def named(self, node_id):
        """El mismo nodo, con el id que digas."""
        return Node(self.obj, node_id)


class Filter(Topology):
    """Mixin: da `>>`, `|` y `named` a tus filtros. Heredar es opcional."""

    def named(self, node_id):
        return Node(self, node_id)


class Step(Topology):
    """Mixin: lo mismo para tus steps. Heredar es opcional."""

    def named(self, node_id):
        return Node(self, node_id)


def _wrap(obj):
    """Cualquier cosa, vista como topología.

    Un `Filter` del usuario es un `Topology` —de ahí le vienen los
    operadores— pero no es un nodo declarado todavía: hay que envolverlo.
    """
    if isinstance(obj, (Chain, Fork, Node)):
        return obj
    if hasattr(obj, "forward") or hasattr(obj, "poll"):
        return Node(obj)
    raise TypeError(
        f"un `{type(obj).__name__}` no puede ir en una expresión de grafo: "
        "le falta forward() o poll()"
    )


def _steps(topology):
    return topology.steps if isinstance(topology, Chain) else [topology]


def _branches(topology):
    return topology.branches if isinstance(topology, Fork) else [topology]


def build(topology):
    """Materializa la expresión en un `Graph` ejecutable."""
    from soma_next._soma_next import Graph

    g = Graph()
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

    add = g.step if hasattr(topology.obj, "poll") else g.node
    node_id = add(topology.node_id, topology.obj) if topology.node_id else add(topology.obj)
    for source in sources:
        g.edge(source, node_id)
    return [node_id]
