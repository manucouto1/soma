"""Putting together what several training runs learnt apart.

The level-3 question, and the one CU11 put off and changed the shape of on the
way: not *is a node's state a `Value`* but **what does a training run export**.
The answer here is the smallest one that is true — its weights, node by node —
and what to do with several of them is a **function**.

# Why a function and not a node, and not a graph either

A federated round is: every client trains on what it has, somebody averages the
weights, everybody starts the next round from the average. Written down:

```python
for _ in range(rounds):
    for client in clients:
        client.fit(client.data)
    average = fedavg([client.export() for client in clients])
    for client in clients:
        client.load(average)
```

That is a `for` over a list, and it stays a `for` over a list. A graph earns its
keep when there are **dependencies to declare** — this has none: the clients do
not read each other, and the order they run in is nobody's business. The original
soma put training runs and graph slices in the same enum and that was the
mistake; the three levels exist so that this one can be a list.

`fedavg`, `fedprox`, `fedyogi` and `scaffold` differ in **arithmetic**, which is
what a function is for. The day one of them needs a topology that is not flat —
hierarchical, gossip — that day it is a graph, and not before.

# What is not in here

**The optimizer's state.** Momentum is local to a client and averaging it is not
FedAvg; whoever wants it says so by exporting it themselves.
"""

from __future__ import annotations

from typing import Iterable, Sequence

#: What one training run exported: the state of each node, by node id.
Export = dict[str, dict[str, "torch.Tensor"]]

import torch


def fedavg(
    exports: Iterable[Export],
    sizes: Sequence[float] | None = None,
) -> Export:
    """The average of what several training runs exported, weight for weight.

    `sizes` is how many samples each one saw, which is what FedAvg weights by —
    a client with ten times the data pulls ten times as hard. Left out, they
    weigh the same.

    ```python
    average = fedavg([client.export() for client in clients], sizes=[900, 100])
    ```

    What is **not** averaged is whatever is not a floating-point number: a
    `num_batches_tracked` is a count, and the mean of two counts is not a count.
    The first one's is kept, which is what every implementation of this does and
    none of them says out loud.
    """
    each = list(exports)
    if not each:
        raise ValueError("there is nothing to average: no training run exported")
    _check_they_are_the_same_shape(each)
    shares = _shares(len(each), sizes)
    return {
        node_id: {
            key: _mean([export[node_id][key] for export in each], shares)
            for key in state
        }
        for node_id, state in each[0].items()
    }


def _mean(
    values: Sequence["torch.Tensor"],
    shares: Sequence[float],
) -> "torch.Tensor":
    """These, weighted. What cannot be halved is not: the first one's stands."""
    if not values[0].is_floating_point():
        return values[0].clone()
    total = torch.zeros_like(values[0], dtype=torch.float64)
    for value, share in zip(values, shares):
        total += value.to(total.device, torch.float64) * share
    return total.to(values[0].device, values[0].dtype)


def _shares(how_many: int, sizes: Sequence[float] | None) -> list[float]:
    """What each one is worth, adding up to one."""
    if sizes is None:
        return [1.0 / how_many] * how_many
    each = [float(size) for size in sizes]
    if len(each) != how_many:
        raise ValueError(
            f"there are {how_many} training runs to average and {len(each)} "
            f"sizes to weigh them by"
        )
    if any(size < 0 for size in each):
        raise ValueError("a training run cannot have seen a negative number of samples")
    total = sum(each)
    if total == 0:
        raise ValueError("every training run is said to have seen nothing")
    return [size / total for size in each]


def _check_they_are_the_same_shape(exports: Sequence[Export]) -> None:
    """That they are exports of the same network.

    Averaging two different nets is not a thing that fails later: it is a thing
    that produces numbers, so it is refused here with the name of whatever does
    not line up.
    """
    first = exports[0]
    for which, export in enumerate(exports[1:], start=1):
        if set(export) != set(first):
            missing = sorted(set(first) - set(export)) or sorted(set(export) - set(first))
            raise ValueError(
                f"training run {which} exported different nodes from the first: "
                f"`{missing[0]}` is in one and not the other"
            )
        for node_id, state in first.items():
            if set(export[node_id]) != set(state):
                raise ValueError(
                    f"training run {which} exported a different `{node_id}` from "
                    f"the first: they do not have the same weights in them"
                )
            for key, value in state.items():
                if export[node_id][key].shape != value.shape:
                    raise ValueError(
                        f"training run {which} exported a `{node_id}` whose "
                        f"`{key}` is {tuple(export[node_id][key].shape)} and the "
                        f"first's is {tuple(value.shape)}"
                    )
