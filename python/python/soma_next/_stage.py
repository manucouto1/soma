"""Cutting a graph into stages: what runs in one `forward` and what cannot.

A pass stops being a single `forward` where the chain that joins the output to
the input breaks, and that happens for two reasons that look different and are
the same one: the value **crossed a cable** — what arrives on the other side is
data, not the graph that produced it — or the node that produced it **learns on
its own**, and a node that learns lets go of its activation by construction.

Hence the pair `(host, learns)`: an edge whose two ends do not share it is a
**cut**, and a cut is a stage boundary. Grouping by the pair is what makes local
greedy and split learning the same path through the code.

The cut is **derived, never declared**: `.at()` already says where each thing
runs, and asking for it to be said a second time would be repeating what the
graph knows.

    learns(n)   the implementation has `learn`, asked as a duck
    where(n)    `hosts().get(n)`, `None` meaning here
    cut(p, c)   `(where(p), learns(p)) != (where(c), learns(c))`
    level(n)    0 with no predecessors, else `max(level(p) + cut(p, n))`
    stage k     the nodes with `level(n) == k`

Three properties fall out, and they are what make the backward pass
demonstrable: every cut edge crosses a stage boundary, no cut edge stays inside
a stage, and no edge goes backwards — so the stages in reverse are a valid order
to walk backwards through.

**A stage is not uniform in host on purpose.** `A.at("a") | B.at("b")` is *one*
stage and a single `forward`, and `compile`/`distribute` rebuild the wave
inside it. The waves are kept by not being clever about them.

A stage is remade as a `Graph` of its own: a `Held` for each value that comes
from outside — named with **the id of the real producer**, so the fan-in map a
node receives is nailed to what it was in the whole graph —, the same node
objects, and a `Tap` for each value somebody outside reads, because `run` gives
back only the terminals and without one the value of a node that feeds inside
*and* outside never comes back.

`Held` and `Tap` are never placed: they do not show up in `hosts()`, do not go
into `_share_out` and are in no artifact. And a stage knows **whose piece it
is**, so running it tells a worker what the whole graph would have told it: half
a catalog is another catalog, and a worker has only one.

A `Held` gives back what it was handed, **verbatim**: whoever fills it says how
it crosses — an `Opaque` for a tensor staying in this process, plain data when
there is a cable ahead. A `Tap` can wrap, because it is the last one and its
value goes straight to Python.

There is no torch here on purpose: how many cuts a graph has is a fact of the
graph, not of the training.
"""

from __future__ import annotations

from soma_next._dsl import Node
from soma_next._soma_next import Done, Opaque

__all__ = ["Held", "Stage", "Tap", "learns", "stages"]

_TAP = "out:{}"
"""How a tap is named after the node it reads."""


class _Nothing:
    """That a hold was never filled, which `None` cannot say: `None` is a value
    a node can legitimately produce."""


_NOTHING = _Nothing()


class Held(Node):
    """A value an earlier stage produced, waiting to be handed on."""

    def __init__(self, node_id):
        self.node_id = node_id
        self.value = _NOTHING

    def forward(self, x, ctx):
        """What it was handed, verbatim. Fails if nobody handed it anything."""
        if self.value is _NOTHING:
            raise ValueError(
                f"`{self.node_id}` was never handed in: this stage takes it from "
                f"an earlier one, and no earlier one produced it"
            )
        return Done(self.value)


class Tap(Node):
    """A terminal, so what a stage produces comes back even when somebody inside
    reads it too."""

    def forward(self, x, ctx):
        """The same value, wrapped: a tap is always the last one and always here,
        so nothing it wraps has a cable or a store ahead of it."""
        return Done(Opaque(x))


class Stage:
    """One `forward` of a cut graph: its own graph, what it waits for and what it
    gives back."""

    def __init__(self, level, graph, nodes, holds, taps):
        self.level = level
        self.graph = graph
        self.nodes = tuple(nodes)
        self.holds = holds
        self.taps = taps

    def fill(self, produced):
        """Hands in what earlier stages produced: it takes what it holds and
        ignores the rest, because a stage reads from any stage before it."""
        for node_id, held in self.holds.items():
            if node_id in produced:
                held.value = produced[node_id]

    def read(self, out):
        """What `forward` gave back, keyed by the node that produced it instead
        of by the tap that carried it."""
        if len(self.taps) == 1:
            [node_id] = self.taps
            return {node_id: out}
        return {node_id: out[tap] for node_id, tap in self.taps.items()}

    def __repr__(self):
        return f"Stage({self.level}: {', '.join(self.nodes)})"


def stages(graph):
    """The graph cut into stages, in the order they run. One stage means there
    is no cut and the graph is its own single pass."""
    level = _levels(graph)
    if not level:
        return []
    order = graph.topological_sort()
    return [
        _stage(graph, k, [n for n in order if level[n] == k])
        for k in range(max(level.values()) + 1)
    ]


def _levels(graph):
    """How many cuts each node is behind, which is the stage it falls in."""
    hosts = graph.hosts()
    side, level = {}, {}
    for node_id in graph.topological_sort():
        side[node_id] = (hosts.get(node_id), learns(graph.implementation(node_id)))
        level[node_id] = max(
            (
                level[before] + int(side[before] != side[node_id])
                for before in graph.predecessors(node_id)
            ),
            default=0,
        )
    return level


def learns(implementation):
    """Whether the node trains itself, asked with the same duck as `parameters()`.

    Here and nowhere else: where a graph gets cut and what its optimizer leaves
    alone are the same question, and two spellings of it would drift apart.
    """
    return getattr(implementation, "learn", None) is not None


def _stage(graph, level, mine):
    """One stage remade as a graph: the holds it waits for, the same node objects
    with everything that was said about them, and a tap per output."""
    inside = set(mine)
    hosts, devices = graph.hosts(), graph.devices()
    frozen, cached, fingerprints = graph.frozen(), graph.cached(), graph.fingerprints()
    stage, holds, taps = type(graph)(), {}, {}
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


def _is_read_outside(graph, node_id, inside):
    """Whether anybody outside this stage reads it: a leaf of the whole graph is
    read by whoever ran it."""
    after = graph.successors(node_id)
    return not after or any(other not in inside for other in after)

