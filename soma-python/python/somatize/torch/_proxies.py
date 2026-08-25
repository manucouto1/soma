"""Scoring a candidate without training it.

    from somatize.torch import proxies
    from somatize.study import Space, ask

    for trial in range(200):                       # the loop is level 3's
        g = build(ask(space, trial))               # and it has no type
        score[trial] = proxies(g, x)["synflow"]

**A proxy is not a `Flag`, and it never was.** `synflow` of one network is a
number with no meaning; it only means something next to another network's. That
puts it at level 3 — where a study is a `for` and there is no type at all — and
not in the vocabulary of a diagnosis, which is about *this* network.

Which leaves one question worth asking of any of them, and it is not "does it
correlate with the score":

> **Does it beat counting parameters?**

Size is free. Abdelfattah et al. (ICLR 2021) report `synflow` at 0.76 rank
correlation with parameter count across NAS-Bench-201, which is close to saying
it measures size — so a proxy that costs a forward and a backward has to earn
the difference. What that came to when it was measured here is in
`health/tests/proxies.py`, and the honest answer is beside each one below.

## The five

| proxy | what it reads | what it needs |
|---|---|---|
| `synflow` | `sum(abs(w * dR/dw))` with every weight made positive | nothing but a shape |
| `snip` | the same product against a real loss | a batch and a target |
| `grasp` | `-sum(w * H g)`, so a second backward through the gradient | a batch and a target |
| `zen` | how far the output moves when the input is nudged | a batch |
| `naswot` | how differently the units switch across a batch | a batch |

Three of them never see a label, which is the point of the family: a candidate
can be scored before there is anything to train it on.

## What it takes

A **graph**, the same as `probe` and `architecture`, because a candidate
architecture in this library is a graph. A bare module becomes one by being a
node's, which is one line and is how everything else here works.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Iterable

if TYPE_CHECKING:
    import torch as _torch

    from somatize._graph import Graph

#: What turns an output and a target into a number to minimise.
Objective = Callable[[Any, Any], "_torch.Tensor"]

import math
import warnings

try:
    import torch
except ImportError:  # pragma: no cover - the trainer module already needs torch
    torch = None  # type: ignore[assignment]

from somatize.torch._inside import _held, _worth_drawing
from somatize.torch._params import parameters
from somatize.torch._probe import _crossable, _tensor, _unwrapped

#: Every proxy there is, in the order they are cheapest to take.
EVERY = ("synflow", "zen", "naswot", "snip", "grasp")

#: Which of them never sees a label.
FREE = ("synflow", "zen", "naswot")


def proxy(
    graph: "Graph",
    example: Any,
    of: str,
    *,
    target: Any = None,
    objective: Objective | None = None,
) -> float:
    """One cheap score for one candidate, higher being better.

    `of` names which — see `EVERY`. `snip` and `grasp` want a `target` and an
    `objective` because they read a real loss; the other three do not, and
    handing them one is not an error, it is ignored.

    The units are nobody's and comparing two runs of the same proxy is the only
    thing it is for, which is why every one of these that spans decades comes
    back as a logarithm: a rank correlation does not care and a person reading
    the numbers does.
    """
    if torch is None:
        raise RuntimeError("`proxy` needs torch")
    if of not in EVERY:
        raise ValueError(f"`{of}` is not a proxy; there are: {', '.join(EVERY)}")
    if of not in FREE and (target is None or objective is None):
        raise ValueError(f"`{of}` reads a loss, so it needs `target=` and `objective=`")
    taken: dict[str, Callable[..., float]] = {
        "synflow": _synflow,
        "snip": _snip,
        "grasp": _grasp,
        "zen": _zen,
        "naswot": _naswot,
    }
    return taken[of](graph, example, target, objective)


def proxies(
    graph: "Graph",
    example: Any,
    *,
    target: Any = None,
    objective: Objective | None = None,
) -> dict[str, float]:
    """Every proxy that can be taken with what it was given, as `{name: score}`.

    Without a `target` and an `objective` the three that read a loss are **not
    in the answer**, rather than being in it as `None`. A score that is missing
    and a score that is bad have to look different, which is the same rule
    `Seen` keeps on the other side of the library.
    """
    said: dict[str, float] = {}
    for name in EVERY:
        if name not in FREE and (target is None or objective is None):
            continue
        try:
            said[name] = proxy(graph, example, name, target=target, objective=objective)
        except RuntimeError as why:  # pragma: no cover - a model that will not differentiate
            warnings.warn(f"`{name}` could not be taken of this graph: {why}", stacklevel=2)
    return said


# ── The five ──


def _synflow(
    graph: "Graph",
    example: Any,
    _target: Any,
    _objective: Objective | None,
) -> float:
    """Tanaka et al. (2020), and the only one that never sees data at all.

    Every weight is made positive and a batch of ones is pushed through, so what
    comes back is a property of the **topology** — how much of the network a
    signal can reach — with the values taken out of it. That is the whole idea
    and it is also the reason to be suspicious: a bigger network reaches more.
    """
    held = parameters(graph)
    signs = [p.sign() for p in held]
    with torch.no_grad():
        for p in held:
            p.abs_()
    try:
        # A batch of ones **of the example's shape**, so there has to be a
        # tensor in the example to take a shape from. Without one there is no
        # `synflow` to take, which is a score that is missing and not a bad one.
        ones = _tensor(_unwrapped(example))
        if ones is None:
            return float("nan")
        made = _ran(graph, torch.ones_like(ones))
        if made is None:
            return float("nan")
        score = _by_parameter(made.sum(), held)
    finally:
        # Put back before anything can raise past here: a proxy that leaves the
        # candidate's weights in absolute value has scored the next one too.
        with torch.no_grad():
            for p, sign in zip(held, signs):
                p.mul_(sign)
    return math.log(score) if score > 0 else float("-inf")


def _snip(
    graph: "Graph",
    example: Any,
    target: Any,
    objective: Objective,
) -> float:
    """Lee et al. (2019). The same product as `synflow`, against a real loss:
    how much each weight is holding up the answer on data it has seen."""
    made = _ran(graph, example)
    if made is None:
        return float("nan")
    score = _by_parameter(objective(made, target), parameters(graph))
    return math.log(score) if score > 0 else float("-inf")


def _grasp(
    graph: "Graph",
    example: Any,
    target: Any,
    objective: Objective,
) -> float:
    """Wang et al. (2020). `-w . H g`: whether a step would make the gradient
    bigger or smaller, which needs the gradient to stay differentiable and is
    why this one costs a second backward through the first."""
    made = _ran(graph, example)
    if made is None:
        return float("nan")
    held = [p for p in parameters(graph) if p.requires_grad]
    g = torch.autograd.grad(objective(made, target), held, create_graph=True,
                            allow_unused=True)
    kept = [(p, one) for p, one in zip(held, g) if one is not None]
    if not kept:
        return float("nan")
    halved = torch.stack([(one * one).sum() for _, one in kept]).sum() / 2
    hessian = torch.autograd.grad(halved, [p for p, _ in kept], allow_unused=True)
    return -sum(float((p.detach() * h).sum()) for (p, _), h in zip(kept, hessian) if h is not None)


def _zen(
    graph: "Graph",
    example: Any,
    _target: Any,
    _objective: Objective | None,
    eps: float = 1e-2,
) -> float:
    """Lin et al. (2021), the expressivity half: how far the output moves when
    the input is nudged. Forward only, twice, and it never sees a label."""
    into = _tensor(_unwrapped(example))
    if into is None:
        return float("nan")
    with torch.no_grad():
        here, there = _ran(graph, into), _ran(graph, into + eps * torch.randn_like(into))
        if here is None or there is None:
            return float("nan")
        moved = float((there - here).norm())
    return math.log(moved / eps) if moved > 0 else float("-inf")


def _naswot(
    graph: "Graph",
    example: Any,
    _target: Any,
    _objective: Objective | None,
) -> float:
    """Mellor et al. (2021). Two inputs that switch the same units the same way
    are two inputs this network cannot tell apart; the log determinant of the
    Hamming kernel is how many it can.

    The code is the **sign** of each activation's output rather than a `relu`
    mask, so a `tanh` network gets a code too. Saying that out loud matters: the
    paper is about rectifiers, and this is the obvious extension rather than the
    paper's claim.
    """
    codes: list[Any] = []
    hooks: list[Any] = []
    for node in graph.nodes():
        for name, module in _held(graph.implementation(node)):
            for _, one in _worth_drawing(module):
                if _switches(one):
                    hooks.append(one.register_forward_hook(
                        lambda _m, _a, out, kept=codes: kept.append(out)))
    try:
        with torch.no_grad():
            made = _ran(graph, example)
    finally:
        for hook in hooks:
            hook.remove()
    rows = made.shape[0] if made is not None and made.dim() else 0
    kept = [(one > 0).float().reshape(rows, -1) for one in codes
            if torch.is_tensor(one) and one.dim() and one.shape[0] == rows]
    if rows < 2 or not kept:
        return float("nan")
    code = torch.cat(kept, dim=1)
    agree = code @ code.t() + (1 - code) @ (1 - code).t()
    sign, value = torch.linalg.slogdet(
        agree.double() + 1e-3 * torch.eye(rows, dtype=torch.float64))
    return float(value) if sign > 0 else float("-inf")


# ── Odds and ends ──


def _ran(graph: "Graph", example: Any) -> Any:
    """The graph's output as a tensor, or nothing."""
    return _tensor(_unwrapped(graph.forward(_crossable(example))))


def _by_parameter(
    scalar: "_torch.Tensor",
    held: Iterable["_torch.nn.Parameter"],
) -> float:
    """`sum(abs(w * dL/dw))`, which is what `synflow` and `snip` both are.

    One function because they are one measurement asked of two different
    scalars, and writing it twice is how the two would slowly stop agreeing.
    """
    for p in held:
        p.grad = None
    scalar.backward()
    score = sum(float((p.detach() * p.grad).abs().sum()) for p in held if p.grad is not None)
    for p in held:
        p.grad = None
    return score


def _switches(module: Any) -> bool:
    """Whether this is a thing whose units are on or off — the only kind that
    has a code to compare."""
    from somatize.torch._inside import kind_of

    return kind_of(module) == "activation"
