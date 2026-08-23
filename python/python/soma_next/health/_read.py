"""Diagnosing a run from what was written down.

The third of the three things CU19 split observability into, and the one that
is an **opinion**. What makes the split real is that this reads the *record*::

    diagnose(store, run="tuesday")

No graph, no torch, no training. The numbers were measured while it ran and
kept; this is somebody's view of them, and it can be taken again with other
bounds::

    diagnose(store, run="tuesday", thresholds=Thresholds(grad_low=1e-12))

> **A diagnosis has to be reproducible from the stored record, without training
> again.**

That sentence has been in `docs/use-cases.md` since CU19 and this module is
where it stops being an aspiration. It is a test: the same store, judged twice
at two bounds, answers twice — and an argument about a threshold costs a scan
rather than an afternoon of GPU.

## Which steps a verdict is taken over

`last=N` reads the last N `forward`s, which is the question worth asking of a
run in flight. Without it, all of them.

Each `health` fact already carries what the audit reduced over its own window —
the maxima that `DEAD` and `SATURATED` read are maxima over that. What happens
here is choosing which of those facts to look at, and the **latest one per
node** is what gets judged: a run that recovered is not still ill.
"""

from __future__ import annotations

from soma_next._soma_next import Thresholds, about, verdict
from soma_next.record._read import facts, forwards

__all__ = ["Thresholds", "about", "diagnose", "history", "named", "seen", "within"]


def diagnose(store, *, run, thresholds=None, last=None):
    """What is wrong, as `{where: [flag, ...]}`.

    `where` is a node, or `node.path.to.submodule` when `inside=` was asked to
    look in.

    A node with nothing wrong is **not in the answer**. Empty would say *this
    was checked and is fine*, and no flags does not mean that: a metric nobody
    measured cannot raise one. `seen` is what says what was measured.

    It costs a fetch per `forward` looked at, because the numbers are in the
    blobs. `last=N` is how a run of ten thousand steps is asked about now.
    """
    said = {}
    for node, one in seen(store, run=run, last=last).items():
        flags = verdict(one, thresholds)
        if flags:
            said[node] = flags
    return said


def seen(store, *, run, last=None):
    """The numbers a verdict would be taken over — the latest of each.

    Keyed by node, and by `node.path.to.submodule` for anything `inside=` was
    asked to look at. The dot is what lets a figure colour the **node** while
    the detail says which layer of it: a node is often a whole architecture,
    and *this node is unhealthy* is not an answer when the node is twenty
    layers deep.

    For looking at what was measured rather than at what somebody thinks of it,
    and for taking the verdict yourself with `verdict(seen[where], bounds)`.
    """
    latest = {}
    for row in _rows(store, run=run, last=last):
        for fact in row:
            if fact.get("fact") == "health" and "node" in fact:
                latest[named(fact)] = _numbers(fact)
    return latest


def named(fact):
    """Where a health fact was measured: a node, or a submodule of one."""
    inside = fact.get("inside")
    return f"{fact['node']}.{inside}" if inside else fact["node"]


def within(where):
    """The node a key belongs to, whether or not it names a submodule of it."""
    return where.split(".", 1)[0]


def history(store, *, run, node, of="grad_norm", last=None):
    # `node` is a `where`: a node, or `node.submodule`.
    """One measurement of one node over the run, as `(forward, value)` pairs.

    What a curve is drawn from — a gradient norm falling away, an update ratio
    drifting, the stable rank of an update collapsing. A fetch per `forward`,
    so `last=` is worth using.
    """
    drawn = []
    for which, row in enumerate(_rows(store, run=run, last=last, numbered=True)):
        at, row = row
        for fact in row:
            if fact.get("fact") == "health" and named(fact) == node and of in fact:
                value = _number(fact[of])
                if value is not None:
                    drawn.append((at, value))
    return drawn


def _rows(store, *, run, last=None, numbered=False):
    """The facts of each `forward`, in order."""
    steps = forwards(store, run=run)
    if last is not None:
        steps = steps[-last:]
    for step in steps:
        row = facts(store, run=run, forward=step["forward"]) or []
        yield (step["forward"], row) if numbered else row


def _numbers(fact):
    """A health fact as the numbers a verdict takes.

    `fact` and `node` are dropped: they say **which** measurement this is, and a
    verdict is about the measurement and not about where it came from.
    """
    said = {}
    for name, what in fact.items():
        if name in ("fact", "node", "inside"):
            continue
        if name in ("nan", "inf"):
            said[name] = what not in (False, "False", "false", "0", "", None)
            continue
        value = _number(what)
        if value is not None:
            said[name] = value
    return said


def _number(what):
    """Text back to a number, or `None` where it was not one.

    A record is text — that is what makes it readable with `cat` — so this is
    where it stops being text. `None` and not zero: a metric that could not be
    read must not pass for one that was measured at zero.
    """
    try:
        value = float(what)
    except (TypeError, ValueError):
        return None
    return value if value == value and abs(value) != float("inf") else None
