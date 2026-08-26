"""Level 3: what is above one training run.

The graph is a network — one ``forward``. The ``Trainer`` is a training run — an
afternoon. This is the level above and, like a federated round, it has **no
type**: N training runs are a ``for``::

    from somatize.study import Partition, Pruner, Sampler, Space
    from somatize.torch import Trainer

    space = Space().real("lr", 1e-5, 1e-1, log=True).choice("opt", ["adam", "sgd"])
    sampler, finished = Sampler.tpe(goal="min"), []

    for trial in range(50):
        point = sampler.ask(space, trial, finished)
        g = build(**point)                       # a Point is a mapping
        t = Trainer(g, objective=cross_entropy, optimizer=Adam(parameters(g)))
        finished.append((point, t.fit(data, epochs=10).loss))

What lives here are the pieces that ``for`` asks for, all of one shape: **numbers
in, a decision out — never a tensor**. That is what lets it be Rust while the
loop stays in Python.

A ``Pruner`` says whether a trial going badly is worth another epoch, and it
**stops nothing** — it answers and the loop stops calling::

    for epoch in range(50):
        reported.append(t.fit(data, epochs=1).loss)
        if why := pruner.verdict(reported, finished):
            break

The three schemes differ in what they judge against: ``median``/``percentile``
the other trials, ``threshold`` a constant, ``patience`` the trial itself. The
samplers differ in what they **look at**: ``grid`` at the space's shape and it is
the one that runs out, ``random``/``halton``/``sobol`` at nothing, ``tpe`` at what
already happened. All but ``tpe`` derive their point from the seed and the
**index**, so a machine that claimed trial 7 gets the same point without
replaying six.

``Partition`` is five schemes and not sklearn's fifteen, because stratifying is a
k-fold inside each class and grouping a k-fold over the groups; the rest are
parameters. It is not called ``Split``: ``somatize.torch.Split`` is already split
learning.
"""

from somatize._somatize import Partition, Point, Pruner, Sampler, Space
from somatize.study._figure import coordinates, influence, table
from somatize.study._run import (
    DONE,
    FAILED,
    MAX,
    MIN,
    PRUNED,
    RUNNING,
    STALE,
    abandoned,
    curves,
    direction,
    finished,
    importance,
    in_flight,
    report,
    take,
    trials,
)

__all__ = [
    "DONE",
    "FAILED",
    "MAX",
    "MIN",
    "PRUNED",
    "RUNNING",
    "STALE",
    "Partition",
    "Point",
    "Pruner",
    "Sampler",
    "Space",
    "abandoned",
    "coordinates",
    "curves",
    "direction",
    "finished",
    "importance",
    "in_flight",
    "influence",
    "report",
    "table",
    "take",
    "trials",
]
