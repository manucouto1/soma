"""Declarar un grafo como una expresión, en vez de a base de llamadas.

    Graph.somatize(Fuente() >> (Izq() | Der()) >> Media())

`>>` encadena y `|` abre en ramas. Un `>>` entre dos ramas abiertas las une:
lo que salga de todas entra en lo que venga detrás, que es exactamente el
fan-in — el nodo de la derecha recibe un mapa con la clave de cada rama.

`Filter` y `Step` son clases abstractas, y **la herencia es lo que decide** de
qué tipo es un nodo. Cada una exige su método —un `Filter` sin `forward` no se
puede ni instanciar— y `isinstance` es la única pregunta que se hace el DSL.

No es duck typing: mirar si el objeto *tiene* `poll` dejaba pasar que un
`class X(Step)` con solo `forward` acabara registrado como filtro, sin un
aviso. Y con un objeto que tuviera los dos métodos, `node()`, `step()` y el DSL
daban tres respuestas distintas.

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


class Node(Topology):
    """Un objeto declarado como nodo, con su id si se lo pusiste."""

    def __init__(self, obj, node_id=None):
        self.obj = obj
        self.node_id = node_id

    def named(self, node_id):
        """El mismo nodo, con el id que digas."""
        return Node(self.obj, node_id)


class Filter(Topology, ABC):
    """Algo que transforma un valor en otro y termina siempre.

    Heredar de aquí es lo que hace que un nodo sea un filtro, y obliga a
    escribir `forward`: sin él, la clase no se puede instanciar.
    """

    @abstractmethod
    def forward(self, input):
        """Transforma la entrada."""

    def named(self, node_id):
        """El mismo nodo, con el id que digas."""
        return Node(self, node_id)


class Step(Topology, ABC):
    """Algo que avanza por turnos y puede pedir cosas antes de terminar.

    Heredar de aquí es lo que hace que un nodo sea un step. El método se llama
    igual que en un filtro —`forward`— porque debajo el contrato es el mismo;
    lo que cambia es que recibe el contexto y devuelve una transición.
    """

    @abstractmethod
    def forward(self, input, ctx):
        """Avanza un turno.

        `ctx` trae `turn` y `results`. Devuelve `{"done": valor}` o
        `{"await": [peticiones]}`.
        """

    def named(self, node_id):
        """El mismo nodo, con el id que digas."""
        return Node(self, node_id)


def kind_of(obj):
    """`"filter"` o `"step"`, según de qué herede. La única pregunta que vale.

    Raises:
        TypeError: si no hereda de ninguna, o si hereda de las dos.
    """
    es_filtro = isinstance(obj, Filter)
    es_step = isinstance(obj, Step)
    if es_filtro and es_step:
        raise TypeError(
            f"`{type(obj).__name__}` hereda de Filter y de Step a la vez, así que "
            "no hay forma de saber qué es. Un nodo termina siempre o puede no "
            "terminar; las dos cosas no"
        )
    if es_filtro:
        return "filter"
    if es_step:
        return "step"
    raise TypeError(
        f"`{type(obj).__name__}` no puede ser un nodo: tiene que heredar de "
        "soma_next.Filter (termina siempre, escribe forward) o de soma_next.Step "
        "(puede pedir cosas antes de terminar, escribe poll)"
    )


def ensure_kind(obj, expected):
    """Comprueba que el objeto es del tipo que la llamada dice.

    Un objeto que no hereda de ninguna de las dos pasa: `node()` y `step()` son
    la puerta de abajo, y ahí el tipo lo elige quien llama. Lo que no pasa es la
    contradicción — `node()` con algo que hereda de `Step`.
    """
    if not isinstance(obj, (Filter, Step)):
        return
    actual = kind_of(obj)
    if actual != expected:
        raise TypeError(
            f"`{type(obj).__name__}` hereda de {actual.capitalize()}, así que se "
            f"añade con {'step' if actual == 'step' else 'node'}(), no con "
            f"{'node' if actual == 'step' else 'step'}()"
        )


def _wrap(obj):
    """Cualquier cosa, vista como topología.

    Un `Filter` del usuario es un `Topology` —de ahí le vienen los
    operadores— pero no es un nodo declarado todavía: hay que envolverlo.
    """
    if isinstance(obj, (Chain, Fork, Node)):
        return obj
    kind_of(obj)  # decide, o explica por qué no puede
    return Node(obj)


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

    add = g.step if kind_of(topology.obj) == "step" else g.node
    node_id = add(topology.node_id, topology.obj) if topology.node_id else add(topology.obj)
    for source in sources:
        g.edge(source, node_id)
    return [node_id]
