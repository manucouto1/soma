"""What happened, read back.

A `Recorder` writes one record per `forward`; this is how it is read::

    from soma_next import Recorder, Store
    from soma_next.record import curve, forwards, nodes, runs

    store = Store("/scratch/runs")
    Trainer(g, objective=..., optimizer=...,
            watching=Recorder(store, run="tuesday", summarising=["loss"]))

    runs(store)                            # what is in here at all
    forwards(store, run="tuesday")         # step by step, one scan
    curve(store, run="tuesday")            # the losses, one scan
    nodes(store, run="tuesday", last=50)   # who spent the time, a fetch each

**Functions and not a type**, like `gather` and `take`: what is being read is a
folder, and a class around a store would only be the store with a longer name.

There are two ways to see a run and they are not rivals. While it is going, what
you want arrives at `watching=` and costs nothing; when it is over — or when it
is **another machine's** — there is no connection and a scan is all there is.
Both answer in the same shape, so whatever draws one draws the other.
"""

from soma_next.record._figure import Live, gantt, progress, spent
from soma_next.record._read import (
    curve,
    curve_costs,
    facts,
    forwards,
    nodes,
    runs,
)

__all__ = [
    "Live",
    "curve",
    "curve_costs",
    "facts",
    "gantt",
    "forwards",
    "nodes",
    "progress",
    "runs",
    "spent",
]
