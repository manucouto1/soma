"""Declaring a graph as an expression, instead of by calls.

    Graph.somatize(Source() >> (Left() | Right()) >> Mean())

`>>` chains, `|` opens branches, `.on()` places on a device and `.at()` sends to
another host. A `>>` between two open branches joins them: whatever leaves them
all enters whatever comes next, which is exactly fan-in — the node on the right
receives a map keyed by each branch.

There is **a single kind of node**, just as the core has a single trait. A node
returns `Done(value)` if it is finished or `Await([requests])` if it needs
something from the world first; what elsewhere is called a filter is simply a
node that always answers `Done`.

Mind the precedence, which is Python's (and the same in Rust): `>>` binds
tighter than `|`, so the branches go in parentheses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Topology:
    """A half-declared piece of graph."""

    def on(self, device):
        """The same piece, placed on a device. **The innermost one wins**, so
        ``(A().on("cuda:0") >> B()).on("cuda:1")`` leaves `A` on 0 and `B` on 1.

        The name is validated when the graph is materialized, in Rust.
        """
        return _placed(self, "device", device)

    def at(self, host):
        """The same piece, in another process. The innermost one wins alike, and
        **independently** of `.on()`, so the two can be written in any order.

        A host is a **name**: what it resolves to is said by whoever executes,
        with `forward(..., workers={...})`.
        """
        return _placed(self, "host", host)

    def __rshift__(self, other):
        return Chain(_steps(self) + _steps(_wrap(other)))

    def __rrshift__(self, other):
        return Chain(_steps(_wrap(other)) + _steps(self))

    def __or__(self, other):
        return Fork(_branches(self) + _branches(_wrap(other)))

    def __ror__(self, other):
        return Fork(_branches(_wrap(other)) + _branches(self))


class Chain(Topology):
    """One after another."""

    def __init__(self, steps):
        self.steps = steps


class Fork(Topology):
    """Branches that do not touch."""

    def __init__(self, branches):
        self.branches = branches


class Declared(Topology):
    """An object declared as a node, with its id and its place if you gave it one."""

    def __init__(self, obj, node_id=None, device=None, host=None):
        self.obj = obj
        self.node_id = node_id
        self.device = device
        self.host = host

    def named(self, node_id):
        """The same node, with the id you say. `.named`, `.on` and `.at` commute."""
        return Declared(self.obj, node_id, self.device, self.host)


class Node(Topology, ABC):
    """What a graph node executes. `forward` has to be written or the class
    cannot be instantiated."""

    @abstractmethod
    def forward(self, input, ctx):
        """Advances one turn: returns `Done(value)` or `Await([requests])`.

        `ctx` carries `turn`, `results` and `device`.
        """

    def named(self, node_id):
        """The same node, with the id you say."""
        return Declared(self, node_id)


def _placed(topology, field, value):
    """Hands a place out to the leaves that do not already have one.

    It is handed out at declaration time because a piece stops existing once
    materialized: placing is a per-node fact. `field` is `"device"` or `"host"`,
    and each looks at only its own, which is what makes them independent.
    """
    topology = _wrap(topology)
    if isinstance(topology, Chain):
        return Chain([_placed(step, field, value) for step in topology.steps])
    if isinstance(topology, Fork):
        return Fork([_placed(branch, field, value) for branch in topology.branches])
    # `is not None` and not an `or`: `.on("")` has to reach `place()` and fail
    # there, not vanish for being an empty string.
    place = {"device": topology.device, "host": topology.host}
    if place[field] is None:
        place[field] = value
    return Declared(topology.obj, topology.node_id, **place)


def _wrap(obj):
    """Anything, seen as a topology. A user's `Node` is one, but it is not a
    declared node yet."""
    if isinstance(obj, (Chain, Fork, Declared)):
        return obj
    if isinstance(obj, Node):
        return Declared(obj)
    raise TypeError(
        f"`{type(obj).__name__}` cannot go in a graph expression: it has to "
        "inherit from soma_next.Node"
    )


def _steps(topology):
    return topology.steps if isinstance(topology, Chain) else [topology]


def _branches(topology):
    return topology.branches if isinstance(topology, Fork) else [topology]


def somatize(graph_cls, topology):
    """Materializes the expression into a graph of the class you are given.

    The class comes in as a parameter and is not imported here: `soma_next._graph`
    imports this module, so importing it back would be a cycle.
    """
    g = graph_cls()
    _walk(g, _wrap(topology), [])
    return g


def _walk(g, topology, sources):
    """Adds what was declared and returns where this piece leaves from."""
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
        # `.on()` is not another path: it ends at the same `place` used by
        # whoever builds the graph by hand, and inherits its validation.
        g.place(node_id, topology.device)
    if topology.host is not None:
        g.place_at(node_id, topology.host)
    for source in sources:
        g.edge(source, node_id)
    return [node_id]
