"""Whether the model is learning **what you think** it is learning.

    from soma_next.data import contribution, leaning

    said = contribution(g, batches, objective=mse, over=("symptoms", "text"))
    leaning(said)          # {"symptoms": ["IGNORED_INPUT(symptoms)"], ...}

## Why this exists

From a real research project. Symptom channels for detecting a mental-health
condition, where symptomatic interpretability and performance could be had one
at a time and never together. **Months** went into diagnosing the architecture.
The problem was in the data: the predictive signal was in the *self-disclosure*
and not in the presence of symptoms, and no amount of looking at gradients was
ever going to say so.

What says so takes one afternoon: take an input away and score it again. If the
channel the whole study was built around costs nothing to remove, it is not what
the model is using — and every hour spent on the network was an hour spent in
the wrong place.

This is a different question from `soma_next.health`, and they are easy to
confuse. That one asks whether a network is **learning**; this asks whether it
is learning what you meant.

## Shuffled and not zeroed

A zero is a value, and often an unusually informative one: a network that has
never seen an all-zero channel falls over for a reason that has nothing to do
with importance. Shuffling keeps the distribution of the channel and destroys
only its **correspondence with the answer**, which is exactly the thing being
asked about.

Across the batch, so what is broken is which row goes with which — the same
thing permutation importance has always done, and for the same reason.

## What it is not

It is a **ranking**, not an attribution: it says an input orders the results,
not that it is worth so many points. Two inputs carrying the same signal both
look unimportant, because removing either leaves the other. That is a true
thing about the data and not a flaw in the method, and it is worth knowing
before somebody concludes that neither matters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, ContextManager, Iterable, Sequence

if TYPE_CHECKING:
    import random as _random

    from soma_next._graph import Graph
    from soma_next._soma_next import Thresholds, Worker

#: One example: what goes in, keyed by input, and what should come out.
Batch = tuple[dict[str, Any], Any]

import random

from soma_next._soma_next import leaning as _leaning

__all__ = ["contribution", "leaning", "shares", "shuffled"]


def contribution(
    graph: "Graph",
    batches: Iterable[Batch],
    *,
    objective: Callable[[Any, Any], float],
    over: Iterable[str] | None = None,
    repeats: int = 3,
    seed: int = 0,
    workers: "dict[str, Worker] | None" = None,
) -> dict[str, float]:
    """How much worse the score gets without each input, as `{name: drop}`.

    `batches` is an iterable of `(input, target)` — the same shape a `Trainer`
    takes — and `input` is a mapping, because a graph with two branches is fed
    a map and the keys are what gets taken away one at a time.

    `over` names which keys to try. Without it, all of them.

    `repeats` is how many shuffles each input gets, averaged. Three is enough to
    stop one unlucky permutation deciding an afternoon's conclusion, and it
    costs three forward passes per input per batch — which is why this is a
    thing you run once and not a thing you run every step.

    Nothing is trained and nothing is changed: the model is asked the same
    question with one of its inputs scrambled.
    """
    each: list[Batch] = [(dict(one), target) for one, target in batches]
    if not each:
        return {}
    names = list(over) if over is not None else list(each[0][0])
    intact = _scored(graph, each, objective, workers)
    said: dict[str, float] = {}
    for name in names:
        worse = [
            _scored(graph, _shuffling(each, name, random.Random(seed + which)), objective, workers)
            for which in range(repeats)
        ]
        said[name] = sum(worse) / len(worse) - intact
    return said


def leaning(
    drops: dict[str, float],
    thresholds: "Thresholds | None" = None,
) -> dict[str, list[str]]:
    """What is wrong with what the model is leaning on.

    `{name: [flag, ...]}`, and a name with nothing wrong is not in it — for the
    same reason a healthy node is not in a diagnosis: no flags is not a clean
    bill, it is *nothing tripped*.
    """
    shares, flags = _leaning(dict(drops), thresholds)
    said: dict[str, list[str]] = {}
    for flag in flags:
        name = flag[flag.index("(") + 1 : -1]
        said.setdefault(name, []).append(flag)
    del shares
    return said


def shares(
    drops: dict[str, float],
    thresholds: "Thresholds | None" = None,
) -> dict[str, float]:
    """What each input is worth, as `{name: share}` — the drops divided by what
    they add up to, so they read as *how much of what matters is this*."""
    said, _ = _leaning(dict(drops), thresholds)
    return {name: share for name, share, _ in said}


def shuffled(what: Any, order: Sequence[int]) -> Any:
    """One input, with its rows put in that order.

    An `Opaque` is unwrapped and wrapped again, because that is how a tensor
    reaches a graph at all and shuffling the wrapper rather than the tensor
    would be a very quiet way of measuring nothing.

    Torch tensors, lists and tuples. What is **not** touched is anything else —
    a string, a number, something this library has never seen — because
    shuffling a thing whose first axis is not the batch is not a measurement,
    it is a different question asked by accident.
    """
    from soma_next import Opaque

    try:
        import torch
    except ImportError:
        # The idiom for an optional import, and the `ignore` is the price: a
        # module name rebound to `None` is exactly what a checker is there to
        # object to, and it is what "torch may not be here" looks like.
        torch = None  # type: ignore[assignment]
    if isinstance(what, Opaque):
        return Opaque(shuffled(what.value, order))
    if torch is not None and isinstance(what, torch.Tensor):
        return what[torch.as_tensor(order, dtype=torch.long)]
    if isinstance(what, (list, tuple)):
        return type(what)(what[at] for at in order)
    return what


def _shuffling(
    batches: list[Batch],
    name: str,
    rolling: "_random.Random",
) -> list[Batch]:
    """The same batches with one input's rows permuted."""
    out: list[Batch] = []
    for one, target in batches:
        if name not in one:
            out.append((one, target))
            continue
        order = list(range(_rows(one[name])))
        rolling.shuffle(order)
        out.append(({**one, name: shuffled(one[name], order)}, target))
    return out


def _rows(what: Any) -> int:
    """How many rows an input has, which is what gets permuted."""
    from soma_next import Opaque

    if isinstance(what, Opaque):
        return _rows(what.value)
    return int(what.shape[0]) if hasattr(what, "shape") else len(what)


def _scored(
    graph: "Graph",
    batches: list[Batch],
    objective: Callable[[Any, Any], float],
    workers: "dict[str, Worker] | None",
) -> float:
    """The mean objective over these batches. Nothing is trained.

    Under `no_grad` where torch is here: nothing is going to be differentiated
    and a graph kept alive for a backward pass that never comes is memory spent
    on nothing — on a real model, enough of it to matter.
    """
    try:
        import torch

        # Annotated as the shape both arms share, so the fallback is not read as
        # the wrong type for a variable the first arm already fixed.
        held: ContextManager[Any] = torch.no_grad()
    except ImportError:
        import contextlib

        held = contextlib.nullcontext()
    total = 0.0
    with held:
        for one, target in batches:
            total += float(objective(graph.forward(one, workers=workers), target))
    return total / len(batches)
