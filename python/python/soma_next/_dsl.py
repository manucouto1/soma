"""Declarar un grafo como una expresión, en vez de a base de llamadas.

    Graph.somatize(Fuente() >> (Izq() | Der()) >> Media())

`>>` encadena, `|` abre en ramas y `.on()` coloca. Un `>>` entre dos ramas
abiertas las une:
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

    def on(self, device):
        """El mismo trozo, colocado. **Gana el de dentro.**

        ``(A().on("cuda:0") >> B()).on("cuda:1")`` deja `A` en la 0 y `B` en la
        1: el de fuera rellena a los que no tienen sitio, no pisa a los que sí.
        Así se coloca una rama entera y luego se afina un nodo suelto, que es
        como se lee de dentro afuera.

        El nombre del dispositivo se valida al materializar el grafo, en Rust:
        `on("cude:0")` falla ahí, no dentro de torch a mitad de un run.
        """
        return _placed(self, device)

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
    """Un objeto declarado como nodo, con su id y su sitio si se los pusiste."""

    def __init__(self, obj, node_id=None, device=None):
        self.obj = obj
        self.node_id = node_id
        self.device = device

    def named(self, node_id):
        """El mismo nodo, con el id que digas. `.named` y `.on` conmutan."""
        return Declared(self.obj, node_id, self.device)


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


def _placed(topology, device):
    """Reparte un dispositivo a las hojas que no lo tengan ya puesto.

    Se reparte al declarar, y no se guarda como «el dispositivo de este trozo»,
    porque un trozo deja de existir al materializarse: lo que queda son nodos.
    Colocar es un hecho por nodo.
    """
    topology = _wrap(topology)
    if isinstance(topology, Chain):
        return Chain([_placed(step, device) for step in topology.steps])
    if isinstance(topology, Fork):
        return Fork([_placed(branch, device) for branch in topology.branches])
    # `is not None` y no un `or`: `.on("")` tiene que llegar a `place()` y
    # fallar allí, no desaparecer por ser una cadena vacía.
    puesto = topology.device if topology.device is not None else device
    return Declared(topology.obj, topology.node_id, puesto)


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
    if topology.device is not None:
        # `.on()` no es otro camino: termina en el mismo `place` que usa quien
        # construye el grafo a mano, y hereda su validación.
        g.place(node_id, topology.device)
    for source in sources:
        g.edge(source, node_id)
    return [node_id]
