"""Cutting a graph into stages: what runs in one `forward` and what cannot.

A pass stops being a single `forward` where the chain joining the output to the
input breaks, and that happens for two reasons that are the same one: the value
**crossed a cable**, or **something trained the node that produced it** and let
go of the activation.

Hence the pair `(host, trained)`: an edge whose two ends do not share it is a
**cut**, and a cut is a stage boundary. Grouping by the pair is what makes local
greedy and split learning one path through the code::

    trained(n)  said by whoever trains, and not asked of the node
    where(n)    `hosts().get(n)`, `None` meaning here
    cut(p, c)   `(where(p), trained(p)) != (where(c), trained(c))`
    level(n)    0 with no predecessors, else `max(level(p) + cut(p, n))`
    stage k     the nodes with `level(n) == k`

Three properties fall out, and they make the backward pass demonstrable: every
cut edge crosses a boundary, no cut edge stays inside a stage, and no edge goes
backwards — so the stages in reverse are a valid order to walk back through.

**A stage is not uniform in host on purpose**: `A.at("a") | B.at("b")` is one
stage and a single `forward`, with the wave rebuilt inside it.

A stage is remade as a `Graph` of its own: a `Held` for each value from outside —
named with **the id of the real producer**, so a fan-in map is what it was in the
whole graph —, the same node objects, and a `Tap` for each value somebody outside
reads. Neither is ever placed, and a stage knows whose piece it is, because half
a catalog is another catalog and a worker has only one.

A `Held` gives back what it was handed **verbatim**: whoever fills it says how it
crosses. A `Tap` can wrap, being the last one.

`around` puts somebody **beside** a node, in the two positions training needs.
Nothing here knows what either does; what it knows is that a node with company
still has to be a node in a graph. There is no torch in this module on purpose.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Sequence

from somatize._dsl import Node
from somatize._somatize import Ctx, Opaque

if TYPE_CHECKING:
    from somatize._graph import Graph

__all__ = ["Held", "Stage", "Tap", "around", "stages", "takes_a_gradient"]

_TAP = "out:{}"
"""How a tap is named after the node it reads."""

_IN = "{}:in"
"""The first of the two positions a trainer takes around the node it trains."""

_COMPUTES = "{}:computes"
"""Where the node itself ends up when a trainer is put around it. The id it was
called by stays with the **last** position: what the rest of the graph knows as
`x` is what `x` gives out, and with a trainer around it that is what the trainer
let go of."""


class _Nothing:
    """That a hold was never filled, which `None` cannot say: `None` is a value
    a node can legitimately produce."""


_NOTHING = _Nothing()


class Held(Node):
    """A value an earlier stage produced, waiting to be handed on."""

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self.value: Any = _NOTHING

    def forward(self, x: Any, ctx: Ctx) -> Any:
        """What it was handed, verbatim. Fails if nobody handed it anything."""
        if self.value is _NOTHING:
            raise ValueError(
                f"`{self.node_id}` was never handed in: this stage takes it from "
                f"an earlier one, and no earlier one produced it"
            )
        return self.value


class Tap(Node):
    """A terminal, so what a stage produces comes back even when somebody inside
    reads it too."""

    def forward(self, x: Any, ctx: Ctx) -> Any:
        """The same value, wrapped: a tap is always the last one and always here,
        so nothing it wraps has a cable or a store ahead of it."""
        return Opaque(x)


class Stage:
    """One `forward` of a cut graph: its own graph, what it waits for and what it
    gives back."""

    def __init__(
        self,
        level: int,
        graph: "Graph",
        nodes: Iterable[str],
        holds: dict[str, Held],
        taps: dict[str, str],
    ) -> None:
        self.level = level
        self.graph = graph
        self.nodes = tuple(nodes)
        self.holds = holds
        self.taps = taps

    def fill(self, produced: dict[str, Any]) -> None:
        """Hands in what earlier stages produced: it takes what it holds and
        ignores the rest. **Every** hold is written and one that is not there is
        emptied — a value left over from the last step is worse than the error
        saying nobody handed it in.
        """
        for node_id, held in self.holds.items():
            held.value = produced.get(node_id, _NOTHING)

    def read(self, out: Any) -> dict[str, Any]:
        """What `forward` gave back, keyed by the node that produced it instead of
        by the tap that carried it. Off the leaves and not the taps: a transposed
        stage can end in a node nobody is waiting for.
        """
        if not self.taps:
            return {}
        if len(self.graph.leaves()) == 1:
            [node_id] = self.taps
            return {node_id: out}
        return {node_id: out[tap] for node_id, tap in self.taps.items()}

    def transposed(self) -> "Stage":
        """The same stage with its edges the other way round, which is what a
        backward pass is: another forward, of the transpose. The same objects,
        ids and `.at()`, because the gradient of a node is worked out where the
        node ran; what swap places are the two ends.

        **Only whatever takes a gradient** is transposed — the trainer beside a
        node and never the node, which would read an envelope as an input. What
        sits between them is walked **through**.
        """
        mine = [
            node_id
            for node_id in self.nodes
            if takes_a_gradient(self.graph.implementation(node_id))
        ]
        owes = {node_id: self._fed_by(node_id, mine) for node_id in mine}
        hosts, devices = self.graph.hosts(), self.graph.devices()
        back = type(self.graph)()
        holds: dict[str, Held] = {}
        taps: dict[str, str] = {}
        back._slice_of = self.graph._slice_of or self.graph

        for node_id in mine:
            if node_id in self.taps:
                holds[node_id] = Held(node_id)
                back.node(self.taps[node_id], holds[node_id])
        for node_id in mine:
            back.node(node_id, self.graph.implementation(node_id))
            if node_id in hosts:
                back.place_at(node_id, hosts[node_id])
            if node_id in devices:
                back.place(node_id, devices[node_id])
        for node_id in mine:
            for owed in owes[node_id]:
                if owed not in mine and owed not in taps:
                    taps[owed] = owed
                    back.node(owed, Tap())
        for node_id in mine:
            if node_id in self.taps:
                back.edge(self.taps[node_id], node_id)
            for owed in owes[node_id]:
                back.edge(node_id, owed)
        return Stage(self.level, back, mine, holds, taps)

    def _fed_by(self, node_id: str, mine: list[str]) -> list[str]:
        """Who fed this one, walking **up through** whatever is not transposed
        until it reaches something that is — or a hold, which is where this stage
        was fed by the one before it."""
        found: list[str] = []
        seen: set[str] = set()
        pending = list(self.graph.predecessors(node_id))
        while pending:
            other = pending.pop(0)
            if other in seen:
                continue
            seen.add(other)
            if other in mine or other in self.holds:
                found.append(other)
            else:
                pending.extend(self.graph.predecessors(other))
        return found


def stages(graph: "Graph", learns: Iterable[str] = ()) -> list[Stage]:
    """The graph cut into stages, in the order they run. One stage means there is
    no cut. `learns` is the ids of whatever breaks the chain where it stands,
    said by whoever trains and not asked of the nodes.
    """
    level = _levels(graph, set(learns))
    if not level:
        return []
    order = graph.topological_sort()
    return [
        _stage(graph, k, [n for n in order if level[n] == k])
        for k in range(max(level.values()) + 1)
    ]


def _levels(graph: "Graph", learns: set[str]) -> dict[str, int]:
    """How many cuts each node is behind, which is the stage it falls in."""
    hosts = graph.hosts()
    side: dict[str, tuple[str | None, bool]] = {}
    level: dict[str, int] = {}
    for node_id in graph.topological_sort():
        side[node_id] = (hosts.get(node_id), node_id in learns)
        level[node_id] = max(
            (
                level[before] + int(side[before] != side[node_id])
                for before in graph.predecessors(node_id)
            ),
            default=0,
        )
    return level


def takes_a_gradient(implementation: object) -> bool:
    """Whether this is something that takes a gradient rather than an input. A
    duck that only ever meets **the framework's own** objects: a user's node is
    never asked this.
    """
    return getattr(implementation, "learn", None) is not None


def _stage(graph: "Graph", level: int, mine: Sequence[str]) -> Stage:
    """One stage remade as a graph: the holds it waits for, the same node objects
    with everything that was said about them, and a tap per output."""
    inside = set(mine)
    hosts, devices = graph.hosts(), graph.devices()
    frozen, cached, fingerprints = graph.frozen(), graph.cached(), graph.fingerprints()
    stage = type(graph)()
    holds: dict[str, Held] = {}
    taps: dict[str, str] = {}
    # Whose catalog a worker of this stage is holding: its own half is another
    # catalog, and one host's worker keeps only one.
    stage._slice_of = graph._slice_of or graph

    for node_id in mine:
        for producer in graph.predecessors(node_id):
            if producer not in inside and producer not in holds:
                holds[producer] = Held(producer)
                stage.node(producer, holds[producer])
    for node_id in mine:
        stage.node(node_id, graph.implementation(node_id))
        if node_id in hosts:
            stage.place_at(node_id, hosts[node_id])
        if node_id in devices:
            stage.place(node_id, devices[node_id])
        if node_id in frozen:
            stage.freeze(node_id, frozen[node_id])
        if node_id in cached:
            stage.cache(node_id, cached[node_id])
        if node_id in fingerprints:
            stage.written_as(node_id, fingerprints[node_id])
    for node_id in mine:
        for producer in graph.predecessors(node_id):
            stage.edge(producer, node_id)
        if _is_read_outside(graph, node_id, inside):
            taps[node_id] = _TAP.format(node_id)
            stage.node(taps[node_id], Tap())
            stage.edge(node_id, taps[node_id])
    return Stage(level, stage, mine, holds, taps)


def _is_read_outside(graph: "Graph", node_id: str, inside: set[str]) -> bool:
    """Whether anybody outside this stage reads it: a leaf of the whole graph is
    read by whoever ran it."""
    after = graph.successors(node_id)
    return not after or any(other not in inside for other in after)


def around(
    graph: "Graph",
    put: dict[str, tuple[Any, Any]],
) -> tuple["Graph", set[str]]:
    """The same graph with somebody standing on each side of the nodes named.
    `put` is `{node_id: (before, after)}`, and what comes back is the graph and
    the ids of everything the three occupy — which is what `stages` has to be
    told breaks the chain.

    **The id stays with the `after`**: what the rest of the graph calls `x` is
    what `x` gives out, so nobody downstream is told anything. The original graph
    is not touched.
    """
    hosts, devices = graph.hosts(), graph.devices()
    frozen, cached, fingerprints = graph.frozen(), graph.cached(), graph.fingerprints()
    out = type(graph)()
    occupied: set[str] = set()

    def as_told(node_id: str, like: str) -> None:
        """Everything the source graph says about `like`, said about `node_id`."""
        if like in hosts:
            out.place_at(node_id, hosts[like])
        if like in devices:
            out.place(node_id, devices[like])

    for node_id in graph.topological_sort():
        if node_id not in put:
            out.node(node_id, graph.implementation(node_id))
            as_told(node_id, node_id)
            if node_id in frozen:
                out.freeze(node_id, frozen[node_id])
            if node_id in cached:
                out.cache(node_id, cached[node_id])
            if node_id in fingerprints:
                out.written_as(node_id, fingerprints[node_id])
            continue
        before, after = put[node_id]
        computing = _COMPUTES.format(node_id)
        beside = (
            (_IN.format(node_id), before),
            (computing, graph.implementation(node_id)),
            (node_id, after),
        )
        for each, what in beside:
            out.node(each, what)
            as_told(each, node_id)
            occupied.add(each)
        # What was said about a node stays with the node, not with its company.
        if node_id in frozen:
            out.freeze(computing, frozen[node_id])
        if node_id in cached:
            out.cache(computing, cached[node_id])
        if node_id in fingerprints:
            out.written_as(computing, fingerprints[node_id])
        out.edge(_IN.format(node_id), computing)
        out.edge(computing, node_id)
    for source, target in graph.edges():
        out.edge(source, _IN.format(target) if target in put else target)
    return out, occupied
