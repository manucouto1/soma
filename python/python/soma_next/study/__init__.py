"""Level 3: what is above one training run.

The graph is a network — the scale of one ``forward``. The ``Trainer`` is a
training run — the scale of an afternoon. This is the level above, and like a
federated round it has **no type**: N training runs are a ``for``::

    from soma_next.study import Partition
    from soma_next.torch import Trainer

    scores = []
    for train, test in Partition.stratified(5).folds(len(y), classes=y.tolist()):
        t = Trainer(g, objective=cross_entropy, optimizer=Adam(parameters(g)))
        t.fit(data[train], epochs=10)
        scores.append(evaluate(g, data[test]))

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

``Partition`` is five schemes and not sklearn's fifteen, because stratifying and
grouping are not different algorithms — stratifying is a k-fold inside each
class, grouping is a k-fold over the groups. What is left over is parameters:
``LeaveOneOut`` is ``kfold(n)``, a holdout of one part in k is fold 0 of a
k-fold, and purged cross-validation is ``time_series(k, gap=...)``.

It is not called ``Split``: ``soma_next.torch.Split`` is already split learning.
"""

from soma_next._soma_next import Partition, Pruner

__all__ = ["Partition", "Pruner"]
