"""Level 3: what is above one training run.

The graph is a network — the scale of one ``forward``. The ``Trainer`` is a
training run — the scale of an afternoon. This is the level above, and like a
federated round it has **no type**: N training runs are a ``for``::

    from soma_next.study import Partition
    from soma_next.torch import Trainer

    space = Space().real("lr", 1e-5, 1e-1, log=True).choice("opt", ["adam", "sgd"])
    sampler, finished = Sampler.tpe(goal="min"), []

    for trial in range(50):
        point = sampler.ask(space, trial, finished)
        g = build(**point)                       # a Point is a mapping
        t = Trainer(g, objective=cross_entropy, optimizer=Adam(parameters(g)))
        finished.append((point, t.fit(data, epochs=10).loss))

What lives here are the pieces that ``for`` asks for, and they all have the same
shape: **numbers in, a decision out — never a tensor**. That is what lets all of
it be Rust while the loop stays in Python, where torch is.

A ``Pruner`` says whether a trial that is going badly is worth another epoch,
and it **stops nothing**: it answers, and the loop stops calling the trainer::

    pruner = Pruner.median(goal="min", warmup=2, startup=5)
    finished = []

    for config in configs:
        t = Trainer(build(config), objective=cross_entropy, optimizer=...)
        reported = []
        for epoch in range(50):
            reported.append(t.fit(data, epochs=1).loss)
            if why := pruner.verdict(reported, finished):
                print(f"dropped at epoch {epoch}: {why}")
                break
        finished.append(reported)

``Trainer.step`` was already the primitive and ``fit`` sugar over it, so none of
this added a line to level 2. The three schemes differ in what they judge
against: ``median``/``percentile`` the other trials, ``threshold`` a constant,
``patience`` the trial itself.

The three samplers differ in **what they look at**: ``grid`` at the space's
shape — and it is the one that runs out, so ``ask`` answering ``None`` is how a
``for`` stops without being told a number —, ``random`` at nothing, and ``tpe``
at what already happened. The first two derive their point from the seed and the
**index**, so a machine that claimed trial 7 out of a shared folder gets the same
point without replaying six; ``tpe`` cannot, and that is what being guided means.

``Partition`` is five schemes and not sklearn's fifteen, because stratifying and
grouping are not different algorithms — stratifying is a k-fold inside each
class, grouping is a k-fold over the groups. What is left over is parameters:
``LeaveOneOut`` is ``kfold(n)``, a holdout of one part in k is fold 0 of a
k-fold, and purged cross-validation is ``time_series(k, gap=...)``.

It is not called ``Split``: ``soma_next.torch.Split`` is already split learning.
"""

from soma_next._soma_next import Partition, Point, Pruner, Sampler, Space
from soma_next.study._run import (
    DONE,
    FAILED,
    PRUNED,
    RUNNING,
    curves,
    finished,
    report,
    take,
    trials,
)

__all__ = [
    "DONE",
    "FAILED",
    "PRUNED",
    "RUNNING",
    "Partition",
    "Point",
    "Pruner",
    "Sampler",
    "Space",
    "curves",
    "finished",
    "report",
    "take",
    "trials",
]
