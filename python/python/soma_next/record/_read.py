"""Reading back what happened.

The other half of CU20. A `Recorder` writes `run/<id>/<n>`; these read it, and
they are functions over a `Store` for the same reason `gather` and `take` are:
touching the folder is not pure, and level 3 has no type. Neither has this.

# What each one costs, because that is the whole design

The record of a `forward` carries the summary and its **blob** carries the
detail, so the split is a price list::

    runs, forwards, curve      one scan, no fetches
    facts                      one fetch
    nodes, fleet               one scan and a fetch per forward

Everything a progress view asks for — how far along, how long each step took,
which broke, what the loss is doing — is on the free side. The per-node detail
is the expensive one and it is asked once, not once a step.

`curve` is free only for what the recorder was told to summarise::

    Recorder(store, run="tuesday", summarising=["loss"])

Without that the losses are in the blobs and reading ten thousand of them is ten
thousand round trips. It says which of the two it did, rather than being quietly
slow.

# Live and read back are two different paths, on purpose

While a run is going, what you want arrives at `watching=` and costs nothing to
get. These are for what is over, or for what **another machine** is doing —
where there is no connection, and a scan is the only thing there is. The rows
they answer with have the same shape either way, so whatever draws one draws the
other.
"""

from __future__ import annotations

import json

PREFIX = "run/"

#: What a record says about a `forward`, and the fields that are numbers.
NUMERIC = ("forward", "took_us", "nodes")


def runs(store):
    """Every run this store holds, newest last.

    One scan and no fetches. What comes back is one dict per run: how many
    `forward`s it has, how many broke, when the first and last were written, and
    how long they took in total.

    The one to call first, because a store holds whatever anybody put in it —
    a cache, a study, artifacts — and this is the question "what is in here".
    """
    seen = {}
    for record in _records(store):
        said = dict(record.meta)
        run = said.get("run")
        if run is None:
            continue
        it = seen.setdefault(
            run,
            {
                "run": run,
                "forwards": 0,
                "broke": 0,
                "took_us": 0,
                "first": record.when,
                "last": record.when,
            },
        )
        it["forwards"] += 1
        it["broke"] += said.get("state") == "broke"
        it["took_us"] += int(said.get("took_us", 0))
        it["first"] = min(it["first"], record.when)
        it["last"] = max(it["last"], record.when)
    return sorted(seen.values(), key=lambda it: it["first"])


def forwards(store, *, run):
    """Every `forward` of that run, in order, as the record says it.

    One scan and no fetches — which is what makes this the thing a progress view
    reads in a loop. `forward`, `took_us` and `nodes` come back as numbers;
    anything the recorder was told to summarise comes back beside them under its
    own `<kind>.<field>` name, as text.
    """
    rows = []
    for record in _records(store):
        said = dict(record.meta)
        if said.get("run") != run:
            continue
        row = {name: what for name, what in said.items() if name != "run"}
        for name in NUMERIC:
            if name in row:
                row[name] = int(row[name])
        row["when"] = record.when
        rows.append(row)
    return sorted(rows, key=lambda row: row["forward"])


def facts(store, *, run, forward):
    """Everything that happened in one `forward`, in the order it arrived.

    The detail, and the one call that costs a fetch. Each fact is the same dict
    a `watching=` callable was handed while it was running — `Fact::flattened`
    and nothing else — so what you looked at live is what you read back.

    `None` when there is no such `forward`, which is not a failure: a run that
    is still going has not written the next one yet.
    """
    bound = store.resolve(f"{PREFIX}{run}/{forward}")
    if bound is None:
        return None
    return _blob(store, bound)


def curve(store, *, run, of="loss.value"):
    """One number per `forward`, as `(forward, value)` pairs.

    What a training curve is. `of` names the field: `loss.value` when the
    recorder was told to summarise `loss`, and otherwise anything a fact carries
    — `took_us` for how long each step took.

    Free when the field is in the record, and one fetch per `forward` when it is
    not. Which of the two happened is said out loud by `curve_costs`, because a
    reader that is quietly a thousand times slower is worse than one that
    refuses.
    """
    drawn = []
    for row in forwards(store, run=run):
        if of in row:
            drawn.append((row["forward"], float(row[of])))
            continue
        kind, _, field = of.partition(".")
        for fact in facts(store, run=run, forward=row["forward"]) or []:
            if fact.get("fact") == kind and field in fact:
                drawn.append((row["forward"], float(fact[field])))
    return drawn


def curve_costs(store, *, run, of="loss.value"):
    """Whether `curve` would scan or fetch: `"scan"` or `"fetch"`.

    Asked before drawing something ten thousand steps long, and the answer is
    `"fetch"` when the recorder was not told to summarise that kind.
    """
    rows = forwards(store, run=run)
    return "scan" if rows and of in rows[0] else "fetch"


#: What a fact with no `host` on it is: the machine doing the asking. A fleet
#: view that leaves it out is missing its busiest member, and calling it by a
#: name would be inventing one nobody wrote down.
HERE = "here"


def fleet(store, *, run, last=None):
    """What each machine did, as the inverse of `nodes`.

    The record is written run → `forward` → node, and *where* is an attribute of
    a fact. This turns it the other way up, because *what is this machine doing*
    is a question nobody could ask of it and is the one somebody with three
    hosts came for.

    **There is no registry and no heartbeat.** A machine is here because it did
    something, and what it did is already written down: `left` says a slice
    crossed to it and how long the whole round trip took, and every fact from
    over there carries its `host`. The original keeps a coordinator with a
    `last_heartbeat` and a thirty-second timeout; this has no coordinator to
    keep one in, and CU18 already answered liveness a different way — not *does
    it answer* but *is it still writing*.

    `waiting_us` is the column that only exists up here: the round trip **minus**
    what actually ran over there, which is the wire, the queue and the codec. It
    is the number that says whether sending it was worth it, and no per-node view
    can produce it because neither half of the subtraction belongs to a node.

    It costs a scan and a fetch per `forward`, the same as `nodes`. `last=N`
    reads only the last N, which is the question worth asking of a fleet that is
    working now.
    """
    seen = {}
    rows = forwards(store, run=run)
    for row in rows[-last:] if last is not None else rows:
        for fact in facts(store, run=run, forward=row["forward"]) or []:
            one = seen.setdefault(
                fact.get("host", HERE),
                {
                    "host": fact.get("host", HERE),
                    "slices": 0,
                    "trip_us": 0,
                    "ran": 0,
                    "took_us": 0,
                    "failed": 0,
                    "nodes": set(),
                    "last": None,
                },
            )
            one["last"] = row["forward"]
            if fact["fact"] == "left":
                one["slices"] += 1
                one["trip_us"] += int(fact.get("took_us", 0))
                continue
            if "node" in fact:
                one["nodes"].add(fact["node"])
            if fact["fact"] in ("ran", "failed"):
                one[fact["fact"]] += 1
                one["took_us"] += int(fact.get("took_us", 0))
    for one in seen.values():
        one["nodes"] = sorted(one["nodes"])
        # What the round trip cost over and above the work: the wire, the queue
        # and the codec. Never below zero — a `left` counted on one `forward` and
        # the work it carried counted on another would otherwise read as a
        # machine that finished before it was asked.
        one["waiting_us"] = max(0, one["trip_us"] - one["took_us"])
    return sorted(seen.values(), key=lambda one: (one["host"] != HERE, one["host"]))


def nodes(store, *, run, last=None):
    """What each node did over the run, added up.

    The aggregated view, and the one that answers *is the profiling
    reasonable*: per node, how many times it ran, how long in total and on
    average, how often it was read back instead, and where it ran.

    **It costs a fetch per `forward`**, because which node did what is in the
    blobs. `last=N` reads only the last N, which is what you want when a run is
    ten thousand steps long and the question is what it is doing now.
    """
    rows = forwards(store, run=run)
    if last is not None:
        rows = rows[-last:]
    seen = {}
    for row in rows:
        for fact in facts(store, run=run, forward=row["forward"]) or []:
            node = fact.get("node")
            if node is None:
                continue
            it = seen.setdefault(
                node,
                {
                    "node": node,
                    "ran": 0,
                    "recalled": 0,
                    "failed": 0,
                    "took_us": 0,
                    "hosts": set(),
                    "devices": set(),
                },
            )
            it[fact["fact"]] = it.get(fact["fact"], 0) + 1
            it["took_us"] += int(fact.get("took_us", 0))
            if "host" in fact:
                it["hosts"].add(fact["host"])
            if "device" in fact:
                it["devices"].add(fact["device"])
    for it in seen.values():
        it["hosts"] = sorted(it["hosts"])
        it["devices"] = sorted(it["devices"])
        # Over the times it really ran: a node read back from the cache took no
        # time at all, and averaging over those would say a hit is fast rather
        # than saying it did not happen.
        it["mean_us"] = it["took_us"] / it["ran"] if it["ran"] else 0.0
    return sorted(seen.values(), key=lambda it: -it["took_us"])


def _records(store):
    """Every record of a run in this store.

    A store holds whatever anybody put in it, so belonging to a run is a
    question and not an assumption — the same guard `study` puts in front of a
    trial's name.
    """
    return [
        record
        for record in store.bound()
        if record.name.startswith(PREFIX) and _numbered(record.name) is not None
    ]


def _numbered(name):
    """The `(run, forward)` that name is, or `None` if it is not one of ours."""
    rest = name[len(PREFIX) :].rsplit("/", 1)
    if len(rest) != 2:
        return None
    try:
        return rest[0], int(rest[1])
    except ValueError:
        return None


def _blob(store, bound):
    """The facts a record points at."""
    bytes_ = store.get(bound.digest)
    if bytes_ is None:
        raise RuntimeError(
            f"`{bound.name}` points at `{bound.digest}` and this store does not "
            f"have it: the record and the bytes are two things, and one of them "
            f"is missing"
        )
    return json.loads(bytes_)
