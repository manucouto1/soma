"""What a graph's nodes will be called, before anything runs — and what an edit did.

Two questions, and only the first is about one graph::

    from soma_next import foreseen

    foreseen.names(g)                        # what each answer will be called
    foreseen.unneeded(g, x, store=store)     # what would not have to run at all
    foreseen.changes(before, after)          # what the edit did
    foreseen.snapshot(g)                     # the same, kept for later

The engine already makes this pass before its first node — a name is a hash of
the **recipe** and not of the data, so only the graph's input is hashed by
content and from there down they are hashes of hashes. Asking for it on its own
is what turns *is my cache still good?* into a question answered in a millisecond
rather than by running.

## What an edit did, as findings and not as buckets

`changes` answers `{node: [finding, ...]}`, which is the shape
`soma_next.health` already uses, and for the same reason: what happens to a node
is **more than one fact** and a node with nothing said about it is fine.

| finding | what it says |
|---|---|
| `CHANGED` | its **shape** moved: another class, or somebody else feeding it |
| `RESETTLED` | it is frozen at another state — other weights, another version |
| `SALTED` | its salt moved |
| `DOWNSTREAM` | none of those moved and its name moved anyway |
| `STALE` | **its name did not move and its code did** |
| `SUSPECT` | something above it is `STALE` |
| `ADDED` / `GONE` | it is in one graph and not the other |
| `UNVERSIONED` | its answer is kept and nobody can say whether its code moved |
| `UNKNOWN` | it cannot be named on one side or the other |

The first three are one question split three ways, because **two different
questions get asked of the same answer**. *Does my cache still hold* is all three
at once: a node frozen at another checkpoint really does produce another answer,
and its name moving is the cache being right. *Did the code change* is `CHANGED`
alone — weights belong to a version, they are not a version, and a rerun of the
same architecture on new data is the same thing trained again and not a different
thing.

`CHANGED` against `DOWNSTREAM` is what makes a list of forty nodes readable: it
says **where the edit is**, and the rest is what inherited it. Who feeds a node
is part of its shape because rewiring it moves its key without touching anything
the node is made of.

More than one can be true at once — a node reworked *and* resalted says both —
and `SUSPECT` rides on top of any of them, because reading a stale answer happens
to a node whatever became of its own name. What is exclusive is the group: a name
either moved or did not, so nothing that moved is also `STALE`, `UNVERSIONED`,
`ADDED`, `GONE` or `UNKNOWN`.

`UNKNOWN` is not an omission and must never be read as *unchanged*. A `.mapped()`
node is named by the content of its items, which nobody has yet, and nothing
under it can be named either. Saying "unchanged" there is the one answer that
costs somebody a week.

## `STALE`, which is the finding this exists for

The fingerprint of the code is **deliberately not in the key** — a cosmetic
refactor would invalidate half the store in silence, so it is kept beside the
value and compared on a hit. The cost of that decision is that editing the body
of a `forward` renames nothing, and a diff that only looked at names would answer
*nothing changed* to the very edit being asked about.

So it is looked at here, where it is an **opinion and not an invalidation**:
`STALE` is the finding that says *you should have bumped the salt*.

It needs both fingerprints, and a class with no source to read has none — a
notebook cell, an `exec`. That absence is `UNVERSIONED` and not silence, because
silence here is the exact lie this module exists to avoid: in a notebook, where
every node is defined in a cell, a graph compared with an edited copy of itself
would answer *nothing to report* about an afternoon of edits. Putting the nodes
in a module is what answers it.

It is asked **only of a node whose answer is kept**, which is the scope a version
is recorded at: parsing an AST for a node nothing is remembered about would be
paid by everyone who declares a graph. So a graph that keeps nothing gets no
opinion about its code and is not told forty times over.

And it **reaches down**, which is what `SUSPECT` is for. A stale node hits, so
everything under it goes on being fed the answer the old code gave — including
what recomputes, which recomputes from it. Leaving those silent would be saying
*checked, and fine* about the half of a graph that is quietly running last week's
encoder.

## Two graphs, or two snapshots

`changes` takes either. A `Graph` is a live object, and two versions of one
module do not coexist in an interpreter — so comparing **two commits** means
comparing what was written down, which is what `snapshot` is: a `dict` of plain
JSON with the names already worked out.

The two are interchangeable because a snapshot carries everything the comparison
reads and nothing else. Two of them are comparable when they were taken with the
**same input**, which the default — none at all — always is.

## What it costs, and what it does not need

Neither `names` nor `changes` reads or writes anything: naming is the `Keeper`'s
and the keeper is the store's — the core computes no hash — so a store is only
where the hash function comes from, and a temporary one, which is what
`store=None` opens, gives the same names. `unneeded` is the only one that looks
inside, and it looks by name and fetches nothing.

And **the input is not needed either**. Every key on both sides carries the same
hash of it, so which one it is cancels out of every comparison: `changes` with no
input at all gives the same findings as `changes` with the real batch, and does
not pay the 121 ms of weighing it. Pass one only when the names themselves are
what you want.
"""

from __future__ import annotations

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


def names(graph, input=None, *, store=None):
    """What each node's answer will be called — `{node: name}` — with nothing run.

    A node missing from it cannot be named in advance, which is what a
    `.mapped()` node and anything under it are.
    """
    with _somewhere(store) as place:
        return _named(graph, input, place)


def unneeded(graph, input=None, *, store):
    """The nodes that would not have to run at all, because something below them
    is already kept. A `store` and not a temporary one: this is the only question
    here whose answer depends on what is in it."""
    said = json.loads(graph.foreseen_json(input, store=store))
    return said["unneeded"]


def snapshot(graph, input=None, *, store=None):
    """Everything `changes` reads about a graph, as plain JSON, so a version of it
    can be kept and compared against one that no longer exists in this process.

    Two are comparable when they were taken with the same `input`, which the
    default — none at all — always is.
    """
    with _somewhere(store) as place:
        return _snapshot(graph, input, place)


def changes(before, after, input=None, *, store=None):
    """What an edit did, as `{node: [finding, ...]}` — see the findings above.
    A node with nothing said about it is not in it.

    Each side is a `Graph` or a `snapshot` of one; `input` and `store` are only
    used for the ones that are still graphs, since a snapshot has been named
    already.
    """
    with _somewhere(store) as place:
        sides = tuple(_snapshot(one, input, place) for one in (before, after))

    found = {}
    for node in set(sides[0]["shape"]) | set(sides[1]["shape"]):
        if one := _which(node, sides):
            found[node] = one
    stale = [node for node, findings in found.items() if "STALE" in findings]
    for node in _below(sides[1], stale):
        found.setdefault(node, []).append("SUSPECT")
    return {node: found[node] for node in sorted(found)}


def _which(node, sides):
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


def _below(side, roots):
    """Everything these nodes feed, however far down."""
    feeds = {}
    for source, target in side["edges"]:
        feeds.setdefault(source, []).append(target)
    reached, asking = set(), list(roots)
    while asking:
        for node in feeds.get(asking.pop(), ()):
            if node not in reached:
                reached.add(node)
                asking.append(node)
    return reached


def _snapshot(graph, input, place):
    """One side of the comparison, whether it arrived as a graph or already as
    this.

    `shape` is what implements a node and who feeds it, together, because both
    are the same question — *what is this node* — and both move its key without
    anything it is made of having moved. `state` and `salt` are apart from it
    since they move a name without the code moving at all.
    """
    if isinstance(graph, dict):
        return graph
    identities, frozen, cached = graph.identities(), graph.frozen(), graph.cached()
    return {
        "names": _named(graph, input, place),
        "shape": {
            node: [identities.get(node), graph.predecessors(node)]
            for node in graph.nodes()
        },
        "state": dict(frozen),
        "salt": dict(cached),
        "kept": sorted(cached),
        "fingerprints": graph.fingerprints(),
        "edges": [list(edge) for edge in graph.edges()],
    }


def _named(graph, input, place):
    """The one name of each node that has one.

    A node whose output is named item by item is left out rather than flattened:
    the pass never produces one today, and the honest answer if it ever does is
    *cannot tell*, which is the side everything else here gives up on.
    """
    said = json.loads(graph.foreseen_json(input, store=place))
    return {node: keys["One"] for node, keys in said["keys"].items() if "One" in keys}


@contextlib.contextmanager
def _somewhere(store):
    """Where the hash comes from. Without one, a directory that is thrown away:
    the names do not depend on what is in it."""
    if store is not None:
        yield store
        return
    with tempfile.TemporaryDirectory(prefix="soma-next-foreseen-") as place:
        yield place
