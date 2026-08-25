"""Declaring a graph as an expression, instead of by calls.

    Graph.somatize(Source() >> (Left() | Right()) >> Mean())

`>>` chains, `|` opens branches, `.on()` places on a device, `.at()` sends to
another host, `.frozen()` says the state does not change and `.cached()` says
the output is worth keeping. A `>>` between two open branches joins them: whatever leaves them
all enters whatever comes next, which is exactly fan-in — the node on the right
receives a map keyed by each branch.

There is **a single kind of node**, just as the core has a single trait: a node
takes what arrived along the edges and returns what it produced. What elsewhere
is called a filter and what is called a step are the same thing here, and there
is no return type that could tell them apart.

Mind the precedence, which is Python's (and the same in Rust): `>>` binds
tighter than `|`, so the branches go in parentheses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar, Union

from soma_next._soma_next import Ctx

if TYPE_CHECKING:
    from soma_next._graph import Graph

#: The three shapes a piece actually takes once `_wrap` has had a look at it.
#: `Topology` is the base every one of them shares and is what a user's
#: expression is typed as; this is the closed set the walk matches on, and
#: naming it is what lets a checker see that "not a `Chain`, not a `Fork`"
#: leaves exactly a `Declared`.
Piece = Union["Chain", "Fork", "Declared"]

#: Whatever `somatize` was handed a class of, so the graph that comes back is
#: that class and not its base.
G = TypeVar("G", bound="Graph")


class Topology:
    """A half-declared piece of graph."""

    def on(self, device: str) -> Piece:
        """The same piece, placed on a device. **The innermost one wins**, so
        ``(A().on("cuda:0") >> B()).on("cuda:1")`` leaves `A` on 0 and `B` on 1.

        The name is validated when the graph is materialized, in Rust.
        """
        return _fill(self, "device", device)

    def at(self, host: str) -> Piece:
        """The same piece, in another process. The innermost one wins alike, and
        **independently** of `.on()`, so the two can be written in any order.

        A host is a **name**: what it resolves to is said by whoever executes,
        with `forward(..., broker=Broker.embedded({...}))`.
        """
        return _fill(self, "host", host)

    def frozen(self) -> Piece:
        """The same piece, settled: its state does not change while the graph
        runs. The innermost one wins alike.

        Here it is **declared**; making it true is `soma_next.torch.freeze`,
        which turns the gradient off and hashes the weights. The same division as
        `.on()`, where the core says where and the node moves itself.
        """
        return _fill(self, "frozen", True)

    def cached(self, salt: str | None = None) -> Piece:
        """The same piece, worth keeping: what each of its nodes produces is
        looked up before being computed, and kept after.

        It costs to keep, so it is opt-in — and a node without it **does not
        break the chain**: its key is still computed and passed on, it is just
        not stored.

        `salt` tells apart two runs the key cannot tell apart on its own:
        `.cached(salt="a100-fp16")`. What is **not** in the key is the device,
        nor the fingerprint of the code.
        """
        return _fill(self, "cached", _Kept(salt))

    def mapped(self) -> Piece:
        """The same piece, mapping over the items of its input: hand it a list
        and it answers with a list as long, item for item.

        It is what gives a cache the grain of an **item**. Without it, adding one
        document to a list of a thousand changes the name of the list and all
        thousand miss; with it, the thousand are read back and the one runs.

        The node is handed **only the items that are missing**, so it still
        batches: what arrives is a list and what goes back is a list of the same
        length, in the same order. An item is named after **itself** — the same
        document in another list is the same item — which is why this is worth
        anything and why its position would not do.
        """
        return _fill(self, "mapped", True)

    def __rshift__(self, other: object) -> Chain:
        return Chain(_steps(self) + _steps(_wrap(other)))

    def __rrshift__(self, other: object) -> Chain:
        return Chain(_steps(_wrap(other)) + _steps(self))

    def __or__(self, other: object) -> Fork:
        return Fork(_branches(self) + _branches(_wrap(other)))

    def __ror__(self, other: object) -> Fork:
        return Fork(_branches(_wrap(other)) + _branches(self))


class Chain(Topology):
    """One after another."""

    def __init__(self, steps: list[Topology]) -> None:
        self.steps = steps


class Fork(Topology):
    """Branches that do not touch."""

    def __init__(self, branches: list[Topology]) -> None:
        self.branches = branches


class _Kept:
    """`.cached()` was said, with the salt it was said with.

    A class of one field and not the salt itself, because `None` is a salt that
    was not given and that is **not** the same as never having said `.cached()`.
    """

    def __init__(self, salt: str | None) -> None:
        self.salt = salt


class Declared(Topology):
    """An object declared as a node, with its id and whatever was said about it.

    What was said lives in a **dict** and not in four attributes, because
    `.frozen()` and `.cached()` are already methods of `Topology` and an
    attribute of the same name would shadow them: you would declare one node and
    then find the second `.frozen()` was not callable. A key that is not there
    means nothing was said, which is not the same as having been said `None` —
    `.cached(salt=None)` is a cache without salt.
    """

    def __init__(self, obj: Node, node_id: str | None = None, **said: Any) -> None:
        self.obj = obj
        self.node_id = node_id
        self.said: dict[str, Any] = dict(said)

    def named(self, node_id: str) -> Declared:
        """The same node, with the id you say. `.named` and the rest commute."""
        return Declared(self.obj, node_id, **self.said)


class Node(Topology, ABC):
    """What a graph node executes. `forward` has to be written or the class
    cannot be instantiated."""

    def __init_subclass__(cls, **said: Any) -> None:
        """Remembers what each instance is **built with**, which is half of its
        key — `Embed(512)` and `Embed(64)` are one class and two answers.

        Captured here and not read off the object afterwards, and that is the
        whole point: what a node holds is not what it was built with. A node
        that counts its calls, caches a client or moves a tensor onto a device
        has attributes that **move while it runs**, and a key built from those
        would change under a graph that never changed. What was passed to
        `__init__` cannot move.

        Bound against the signature, so `Layer(64, 32)`, `Layer(64, out=32)` and
        `Layer(in_=64, out=32)` are one declaration and not three: a key that
        depended on how the call was typed would miss for a rename.
        """
        super().__init_subclass__(**said)
        from soma_next import _declaration

        _declaration.remember_what_built_it(cls)

    @abstractmethod
    def forward(self, input: Any, ctx: Ctx) -> Any:
        """Runs it: takes what arrived along the edges, returns what it made.

        `ctx` carries `device`, which is where this node was told to run.
        """

    def named(self, node_id: str) -> Declared:
        """The same node, with the id you say."""
        return Declared(self, node_id)


def _fill(topology: Topology, field: str, value: Any) -> Piece:
    """Hands one thing out to the leaves that were not told it already.

    It is handed out at declaration time because a piece stops existing once
    materialized: all of this is a per-node fact. Each field looks at only its
    own, which is what makes them independent — a node can be settled without
    being kept, placed without being settled, and any combination of the rest.
    """
    # A new name and not a reassignment, so that "neither a `Chain` nor a
    # `Fork`" narrows to `Declared` and the two attribute reads below are
    # checked rather than taken on trust.
    piece = _wrap(topology)
    if isinstance(piece, Chain):
        return Chain([_fill(step, field, value) for step in piece.steps])
    if isinstance(piece, Fork):
        return Fork([_fill(branch, field, value) for branch in piece.branches])
    # `setdefault` and not a check against `None`: `.on("")` has to reach
    # `place()` and fail there, not vanish for being an empty string.
    said = dict(piece.said)
    said.setdefault(field, value)
    return Declared(piece.obj, piece.node_id, **said)


def _wrap(obj: object) -> Piece:
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


def _steps(topology: Topology) -> list[Topology]:
    return topology.steps if isinstance(topology, Chain) else [topology]


def _branches(topology: Topology) -> list[Topology]:
    return topology.branches if isinstance(topology, Fork) else [topology]


def _note_the_code(g: "Graph", node_id: str, obj: object) -> None:
    """Which version of the class this graph was written against.

    **Metadata**, never part of a key: a cosmetic refactor must not invalidate
    half a store in silence. It gets compared on a hit and said on `stderr`.

    Only for what is kept, because computing it means parsing an AST. And a
    class with no source to read — a notebook, an `exec` — simply has none: it
    is a comparison that cannot be made, not a reason to fail.
    """
    from soma_next import _fingerprint

    try:
        g.written_as(node_id, _fingerprint.digest(type(obj)))
    except _fingerprint.CannotVersion:
        pass


def somatize(graph_cls: type[G], topology: Topology) -> G:
    """Materializes the expression into a graph of the class you are given.

    The class comes in as a parameter and is not imported here: `soma_next._graph`
    imports this module, so importing it back would be a cycle.
    """
    g = graph_cls()
    _walk(g, _wrap(topology), [])
    return g


def _walk(g: "Graph", topology: Piece, sources: list[str]) -> list[str]:
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
    said = topology.said
    if "device" in said:
        # `.on()` is not another path: it ends at the same `place` used by
        # whoever builds the graph by hand, and inherits its validation.
        g.place(node_id, said["device"])
    if "host" in said:
        g.place_at(node_id, said["host"])
    if "frozen" in said:
        g.freeze(node_id)
    if "cached" in said:
        g.cache(node_id, said["cached"].salt)
        _note_the_code(g, node_id, topology.obj)
    if "mapped" in said:
        g.mapped(node_id)
    for source in sources:
        g.edge(source, node_id)
    return [node_id]
