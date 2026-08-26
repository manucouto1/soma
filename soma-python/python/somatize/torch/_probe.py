"""What can be measured before a step is taken.

    from somatize.torch import probe
    from somatize.health import diagnose

    probe(g, x, watching=Recorder(store, run="before"))
    diagnose(store, run="before")

The static half, and the same half of the same wall: this **measures** and
decides nothing. A probe is **one `forward` that was recorded and never trained**
— literally `run/<id>/0`, which is why `diagnose`, `seen`, `profile`, `overlaid`
and `alerts` all read one without knowing it exists.

| what | how | why not something else |
|---|---|---|
| `signal_gain` | the scale here against the last normalisation upstream | the drift is geometric, so what matters is the ratio |
| `jacobian_gain` | `sqrt(E‖Jᵀv‖²)` from here to the output | the backward signal, scale-free at every depth |
| `jacobian_spread` | `s_max / s_rms` of the sketch `JᵀV` | isometry is about the spectrum's *shape* |

There is deliberately no `grad_norm`: at initialisation there is no loss, so a
parameter gradient would be taken against a target somebody made up and would
land in the same field the audit fills from a real one.

Two forwards and `k` backwards, and the `k` are over the whole network and **not
`k` per layer**. The second forward is `architecture`'s, under `no_grad`, and it
is what decides **which** layers are measured — walking the modules here would
break the invariant the layer rests on, *every layer that can carry a flag has a
box*. Nothing here changes what the network computes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from somatize._typing import Fact

if TYPE_CHECKING:
    import torch as _torch

    from somatize._graph import Graph
    from somatize._remote import Broker

#: What names a measurement: the node, and the path inside it — or `""`
#: for the node itself. A pair and not the joined string, because these are
#: written down apart and only a reader joins them.
Key = tuple[str, str]

#: What one hook kept: what went in, what came out, and the module, so that
#: *is this the normalisation* is asked of the thing rather than looked up.
Caught = tuple["_torch.Tensor | None", "_torch.Tensor", Any]

import math
import warnings

try:
    import torch
except ImportError:  # pragma: no cover - the trainer module already needs torch
    torch = None  # type: ignore[assignment]

from somatize.torch._audit import _of_activation
from somatize.torch._inside import _held, architecture, kind_of

#: How many random probes the Jacobian sketch is taken over. The sketch's own
#: spread is what `jacobian_spread` is measured against, so this is a floor on
#: how fine a spectrum it can see and not a precision knob: below about eight
#: the ratio is noise, and above about thirty-two the backward passes are the
#: cost of a training step.
PROBES = 24


def probe(
    graph: "Graph",
    example: Any,
    *,
    depth: int = 0,
    most: int = 48,
    probes: int = PROBES,
    watching: Any = None,
    broker: "Broker | None" = None,
) -> dict[str, dict[str, Any]]:
    """What this graph looks like at initialisation, as `{where: numbers}`.

    `where` is a node, or `node.path.to.submodule` — the same keys the audit uses
    and the same scope the figure draws, so a finding from a probe lands on the
    box a finding from a run would. The answer is the shape
    `somatize.health.seen` returns from a store. `watching=` takes a `Recorder`
    or anything callable.
    """
    if torch is None:
        raise RuntimeError("`probe` needs torch")
    watched = _watching(graph, example, depth, most, broker)

    seen: dict[Key, Caught] = {}
    order: list[Key] = []
    hooks: list[Any] = []
    for key, module in watched.items():
        hooks.append(module.register_forward_hook(_caught(seen, order, key)))
    try:
        output = graph.forward(_crossable(example), watching=watching, broker=broker)
    finally:
        for hook in hooks:
            hook.remove()

    numbers = {key: _of_one(one) for key, one in seen.items()}
    _signal(numbers, seen, order, example)
    _isometry(numbers, seen, order, _unwrapped(output), probes)

    told = _telling(watching)
    said: dict[str, dict[str, Any]] = {}
    for key, one in numbers.items():
        if not one:
            continue
        node, inside = key
        said[f"{node}.{inside}" if inside else node] = one
        if told is not None:
            told({"fact": "health", "node": node, "inside": inside,
                  **{name: str(what) for name, what in one.items()}})
    return said


def _watching(
    graph: "Graph",
    example: Any,
    depth: int,
    most: int,
    broker: "Broker | None",
) -> dict[Key, Any]:
    """Which module each key names, taken from what the figure will draw.

    Not from `_worth_drawing`, which is what the audit does: **the scope has to
    be the drawing's scope**. `architecture` splices the symbolic view over the
    module walk, so at `depth=1` a walk opens a composite the figure keeps whole
    — and a finding on a layer with no box lands nowhere.

    Folded paths are watched too: six identical blocks are drawn once and
    measured six times, because measuring one and calling it the other five is a
    diagnosis of a network nobody built.
    """
    said = {}
    # Wrapped here as well: `architecture` obeys the rule every input obeys and
    # a probe is friendlier than that, taking a bare tensor or an `Opaque`.
    for node, inside in architecture(graph, _crossable(example), depth=depth, most=most,
                                     broker=broker).items():
        held = dict(_held(graph.implementation(node)))
        for path in {one.path for one in inside.layers} | set(inside.folded):
            module = _module_at(held, path)
            # An `fx` node that is not a module — a `+`, a `concat` — has a box
            # and nothing to hook. It is drawn and not measured, which is the
            # allowed direction.
            if module is not None:
                said[(node, path)] = module
    return said


def _module_at(held: dict[str, Any], path: str) -> Any:
    """The module a drawn path names, or nothing.

    A path is the attribute the module hangs off a node by, then its own path
    inside it — the prefix being what stops a node with two modules on it having
    two `0`s and the second quietly overwriting the first.
    """
    name, _, rest = path.partition(".")
    module = held.get(name)
    if module is None:
        return None
    try:
        return module.get_submodule(rest) if rest else module
    except AttributeError:
        return None


def _caught(
    seen: dict[Key, Caught],
    order: list[Key],
    key: Key,
) -> Callable[[Any, Any, Any], None]:
    """A forward hook that keeps what went in and what came out.

    The tensors themselves and not statistics of them: the Jacobian half needs to
    differentiate through them, and a number cannot be differentiated.
    """

    def saw(module: Any, args: Any, out: Any) -> None:
        into, made = _tensor(args[0] if args else None), _tensor(out)
        if made is None:
            return
        # The module travels with its tensors. It is what says whether this is
        # the normalisation the next gain is measured from, and looking it up
        # again later would mean keeping a second map in step with this one.
        seen[key] = (into, made, module)
        order.append(key)

    return saw


def _of_one(one: Caught) -> dict[str, Any]:
    """What a layer's output says about itself, on the one step there is."""
    _, made, _module = one
    said = _of_activation(made)
    # A window of one is still a window, and the maximum over it is the value.
    # The names have to be the ones a verdict reads, because the whole point is
    # that the same bound answers a probe and a run.
    for name in ("zero_frac", "sat_frac"):
        if name in said:
            said[f"{name}_max"] = said.pop(name)
    said.pop("act_abs_mean", None)
    return said


def _signal(
    numbers: dict[Key, dict[str, Any]],
    seen: dict[Key, Caught],
    order: list[Key],
    example: Any,
) -> None:
    """The scale of the signal, against where the last normalisation left it.

    A norm resets the scale, so drift measured from the input would blame a layer
    for what happened three norms ago — resetting the reference is **structure
    and not a bound**. In execution order, which is what a chain means: a branch
    running hot and added to something bigger has not moved the signal, and the
    layer that consumes the sum is where the drift shows.
    """
    reference = _scale(_tensor(_unwrapped(example)))
    for key in order:
        into, made, module = seen[key]
        scale = _scale(made)
        if scale is None:
            continue
        if reference is None:
            reference = _scale(into) or scale
        # A normalisation sets the reference and does not report a gain of its
        # own. Changing the scale is its **job**, so reading that change as
        # drift would put the loudest number in the run on the one layer that
        # is doing the thing being asked about.
        if kind_of(module) == "norm":
            reference = scale
            continue
        if reference > 0 and math.isfinite(scale):
            numbers.setdefault(key, {})["signal_gain"] = scale / reference


def _isometry(
    numbers: dict[Key, dict[str, Any]],
    seen: dict[Key, Caught],
    order: list[Key],
    output: Any,
    probes: int,
) -> None:
    """The Jacobian from each layer to the output, sketched with random probes:
    `k` unit probes pushed into the output and one backward each, every layer
    reading its own `Jᵀv` off the same pass.

    Two numbers out of the `k × n` sketch — `jacobian_gain`, the root mean square
    of `‖Jᵀv‖`, which is the vanishing picture with no optimizer and no target;
    and `jacobian_spread`, `s_max / s_rms`, Pennington et al. (2017)'s claim that
    the shape matters and not only the mean.
    """
    if output is None or not torch.is_tensor(output) or not output.requires_grad:
        warnings.warn(
            "this graph does not end in one differentiable tensor, so there is no "
            "Jacobian to sketch: `signal_gain` and the activation statistics are "
            "measured and `jacobian_gain` and `jacobian_spread` are not",
            stacklevel=3,
        )
        return
    # `dict.fromkeys` and not a `set`: order is the whole point, and a module
    # that ran twice is in `order` twice while `seen` holds only its last run.
    wanted = [key for key in dict.fromkeys(order) if seen[key][1].requires_grad]
    if not wanted:
        return
    made = [seen[key][1] for key in wanted]
    v = torch.randn(probes, *output.shape, device=output.device, dtype=output.dtype)
    v = v / v.reshape(probes, -1).norm(dim=1).reshape(probes, *([1] * output.dim())).clamp_min(1e-12)
    grads = _sketched(output, made, v, probes)
    if grads is None:
        return
    for key, g in zip(wanted, grads):
        if g is None:
            continue
        flat = g.reshape(probes, -1).detach().float()
        gain = float(flat.norm(dim=1).pow(2).mean().sqrt())
        one = numbers.setdefault(key, {})
        if math.isfinite(gain):
            one["jacobian_gain"] = gain
        try:
            s = torch.linalg.svdvals(flat.double())
        except Exception:  # pragma: no cover - a degenerate sketch says nothing
            continue
        rms = float(s.pow(2).mean().sqrt())
        if rms > 0 and math.isfinite(rms):
            one["jacobian_spread"] = float(s[0]) / rms


def _sketched(
    output: "_torch.Tensor",
    made: list["_torch.Tensor"],
    v: "_torch.Tensor",
    probes: int,
) -> Any:
    """`k` vector-Jacobian products, batched where `vmap` can follow the model.

    The fallback is a plain loop and it is `k` times slower, said out loud rather
    than discovered: a probe that quietly costs a training step is one nobody runs.
    """
    try:
        return torch.autograd.grad(output, made, grad_outputs=v, is_grads_batched=True,
                                   retain_graph=True, allow_unused=True)
    except (RuntimeError, NotImplementedError) as why:
        warnings.warn(
            f"this model cannot be differentiated in a batch ({why}), so the "
            f"{probes} probes run one at a time — the probe costs {probes} backward "
            f"passes instead of one",
            stacklevel=3,
        )
    kept = None
    for one in v:
        got = torch.autograd.grad(output, made, grad_outputs=one, retain_graph=True,
                                  allow_unused=True)
        if kept is None:
            kept = [[] if g is None else [g] for g in got]
        else:
            for held, g in zip(kept, got):
                if g is not None:
                    held.append(g)
    return [torch.stack(held) if held else None for held in (kept or [])]


def _scale(what: Any) -> float | None:
    """How big the signal is, as a standard deviation.

    The deviation and not the mean magnitude: what a normalisation sets is the
    variance, so that is the thing a gain is a gain of.
    """
    if what is None or not torch.is_tensor(what) or what.numel() < 2:
        return None
    with torch.no_grad():
        said = float(what.detach().float().std())
    return said if math.isfinite(said) else None


def _tensor(what: Any) -> "_torch.Tensor | None":
    """The tensor in whatever arrived, or nothing."""
    if torch.is_tensor(what):
        return what
    if isinstance(what, (tuple, list)) and what:
        return _tensor(what[0])
    return None


def _unwrapped(what: Any) -> Any:
    """An `Opaque` holds the tensor; everything else is already itself."""
    from somatize import Opaque

    return what.value if isinstance(what, Opaque) else what


def _crossable(what: Any) -> Any:
    """A tensor is wrapped to cross an edge; everything else passes as it is."""
    from somatize import Opaque

    return Opaque(what) if torch.is_tensor(what) else what


def _telling(watching: Any) -> Callable[[Fact], None] | None:
    """Whatever `watching=` was given, as one callable — or `None`."""
    from somatize.torch._trainer import _telling as the_same_one

    return the_same_one(watching)
