"""What a graph's nodes will be called before anything runs, and what an edit did::

    from somatize import foreseen

    foreseen.names(g)                        # what each answer will be called
    foreseen.unneeded(g, x, store=store)     # what would not have to run at all
    foreseen.changes(before, after)          # what the edit did
    foreseen.snapshot(g)                     # the same, kept for later

A name is a hash of the **recipe** and not of the data, so only the graph's input
is hashed by content and from there down they are hashes of hashes. The engine
already makes this pass before its first node; asking for it on its own turns
*is my cache still good?* into a millisecond instead of a run.

`changes` answers `{node: [finding, ...]}` — the shape `somatize.health` uses,
and for the same reason: what happens to a node is more than one fact.

| finding | what it says |
|---|---|
| `CHANGED` | its **shape** moved: another class, other arguments, or who feeds it |
| `RESETTLED` | it is frozen at another state — other weights, another version |
| `SALTED` | its salt moved |
| `DOWNSTREAM` | none of those moved and its name moved anyway |
| `STALE` | **its name did not move and its code did** |
| `SUSPECT` | something above it is `STALE` |
| `ADDED` / `GONE` | it is in one graph and not the other |
| `UNVERSIONED` | its answer is kept and nobody can say whether its code moved |
| `UNKNOWN` | it cannot be named on one side or the other |

The first three are one question split three ways, because two questions get
asked of one answer: *does my cache still hold* is all three, *did the code
change* is `CHANGED` alone — weights belong to a version, they are not one.

`STALE` exists because the fingerprint of the code is deliberately not in the
key: editing a `forward` renames nothing, so a diff that only looked at names
would answer *nothing changed* to the very edit being asked about. Here it is an
**opinion and not an invalidation**, and it reaches down as `SUSPECT`.
`UNKNOWN` must never be read as *unchanged*: a `.mapped()` node is named by items
nobody has yet.

Each side is a `Graph` or a `snapshot` of one, because two versions of a module
do not coexist in an interpreter. Nothing here reads or writes a store.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from somatize._graph import Graph
    from somatize._somatize import Store

#: One side of a comparison: everything `changes` reads about a graph,
#: already named. Plain JSON on purpose — it outlives the graph it came
#: from, which is the point of `snapshot`.
Snapshot = dict[str, Any]

import contextlib
import json
import tempfile

__all__ = ["FINDINGS", "changes", "names", "snapshot", "unneeded"]

FINDINGS = (
    "STALE",
    "SUSPECT",
    "CHANGED",
    "RESETTLED",
    "SALTED",
    "DOWNSTREAM",
    "ADDED",
    "GONE",
    "UNVERSIONED",
    "UNKNOWN",
)
"""Every finding there is, the ones worth looking at first at the front."""

MOVED = (("CHANGED", "shape"), ("RESETTLED", "state"), ("SALTED", "salt"))
"""The parts of a node's own recipe, each with what it is called when it moves."""


def names(
    graph: "Graph",
    input: Any | None = None,
    *,
    store: "Store | str | None" = None,
) -> dict[str, str]:
    """What each node's answer will be called — `{node: name}` — with nothing run.
    A node missing from it cannot be named in advance, which is what a
    `.mapped()` node and anything under it are.
    """
    with _somewhere(store) as place:
        return _named(graph, input, place)


def unneeded(
    graph: "Graph",
    input: Any | None = None,
    *,
    store: "Store | str",
) -> list[str]:
    """The nodes that would not have to run at all, because something below them
    is already kept. A `store` and not a temporary one: this is the only question
    here whose answer depends on what is in it."""
    said = json.loads(graph.foreseen_json(input, store=store))
    unneeded_: list[str] = said["unneeded"]
    return unneeded_


def snapshot(
    graph: "Graph",
    input: Any | None = None,
    *,
    store: "Store | str | None" = None,
) -> Snapshot:
    """Everything `changes` reads about a graph, as plain JSON, so a version can
    be compared against one that no longer exists in this process. Two are
    comparable when taken with the same `input`, which the default always is.
    """
    with _somewhere(store) as place:
        return _snapshot(graph, input, place)


def changes(
    before: "Graph | Snapshot",
    after: "Graph | Snapshot",
    input: Any | None = None,
    *,
    store: "Store | str | None" = None,
) -> dict[str, list[str]]:
    """What an edit did, as `{node: [finding, ...]}`. A node with nothing said
    about it is not in it. `input` and `store` are only used for a side that is
    still a graph, since a snapshot has been named already.
    """
    with _somewhere(store) as place:
        # Written as a pair rather than built from a generator: `_which` reads
        # `was, is_ = sides` and there being exactly two of them is the contract,
        # not an accident of how many were passed in.
        sides = (_snapshot(before, input, place), _snapshot(after, input, place))

    found: dict[str, list[str]] = {}
    for node in set(sides[0]["shape"]) | set(sides[1]["shape"]):
        if one := _which(node, sides):
            found[node] = one
    stale = [node for node, findings in found.items() if "STALE" in findings]
    for node in _below(sides[1], stale):
        found.setdefault(node, []).append("SUSPECT")
    return {node: found[node] for node in sorted(found)}


def _which(node: str, sides: tuple[Snapshot, Snapshot]) -> list[str]:
    """What became of one node, or an empty list if nothing did. The order of the
    questions is the contract: what cannot be named is asked before what its name
    says, and what its name says before what its name could not say."""
    was, is_ = sides
    if node not in was["shape"]:
        return ["ADDED"]
    if node not in is_["shape"]:
        return ["GONE"]
    if node not in was["names"] or node not in is_["names"]:
        return ["UNKNOWN"]
    if was["names"][node] != is_["names"][node]:
        moved = [
            name
            for name, part in MOVED
            if was[part].get(node) != is_[part].get(node)
        ]
        return moved or ["DOWNSTREAM"]
    versions = (was["fingerprints"].get(node), is_["fingerprints"].get(node))
    if all(versions):
        return ["STALE"] if versions[0] != versions[1] else []
    return ["UNVERSIONED"] if node in set(was["kept"]) | set(is_["kept"]) else []


def _below(side: Snapshot, roots: list[str]) -> set[str]:
    """Everything these nodes feed, however far down."""
    feeds: dict[str, list[str]] = {}
    for source, target in side["edges"]:
        feeds.setdefault(source, []).append(target)
    reached: set[str] = set()
    asking = list(roots)
    while asking:
        for node in feeds.get(asking.pop(), ()):
            if node not in reached:
                reached.add(node)
                asking.append(node)
    return reached


def _snapshot(graph: "Graph | Snapshot", input: Any, place: "Store | str") -> Snapshot:
    """One side of the comparison, whether it arrived as a graph or already as
    this. `shape` is what implements a node, what it was built with and who feeds
    it, because those are one question; `state` and `salt` are apart since they
    move a name without the code moving.
    """
    if isinstance(graph, dict):
        return graph
    identities, frozen, cached = graph.identities(), graph.frozen(), graph.cached()
    declarations = graph.declarations()
    return {
        "names": _named(graph, input, place),
        "shape": {
            node: [
                identities.get(node),
                declarations.get(node),
                graph.predecessors(node),
            ]
            for node in graph.nodes()
        },
        "state": dict(frozen),
        "salt": dict(cached),
        "kept": sorted(cached),
        "declared": _declared(graph),
        "fingerprints": graph.fingerprints(),
        "edges": [list(edge) for edge in graph.edges()],
    }


def _declared(graph: "Graph") -> dict[str, str]:
    """What each node was built with, in words. For reading and not for
    comparing: a node whose declaration could not be written down has none, and
    that is already said by its absence from the digests."""
    from somatize import _declaration

    said: dict[str, str] = {}
    for node in graph.nodes():
        try:
            said[node] = _declaration.written(graph.implementation(node))
        except _declaration.CannotDeclare:
            pass
    return said


def _named(graph: "Graph", input: Any, place: "Store | str") -> dict[str, str]:
    """The one name of each node that has one. A node named item by item is left
    out rather than flattened: the honest answer is *cannot tell*.
    """
    said = json.loads(graph.foreseen_json(input, store=place))
    return {node: keys["One"] for node, keys in said["keys"].items() if "One" in keys}


@contextlib.contextmanager
def _somewhere(store: "Store | str | None") -> Iterator["Store | str"]:
    """Where the hash comes from. Without one, a directory that is thrown away:
    the names do not depend on what is in it."""
    if store is not None:
        yield store
        return
    with tempfile.TemporaryDirectory(prefix="soma-foreseen-") as place:
        yield place
