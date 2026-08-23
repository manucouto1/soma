"""What can be measured before a step is taken.

    from soma_next.torch import probe
    from soma_next.health import diagnose

    probe(g, x, watching=Recorder(store, run="before"))
    diagnose(store, run="before")

The static half of CU21, and it is the same half of the same wall: this
**measures** and decides nothing. What comes out is a `health` fact per layer —
numbers, in the same record as everything else, under the same keys — and
whether those numbers are bad is `soma_next.health`'s opinion.

A probe is **one `forward` that was recorded and never trained**. That is not a
metaphor for the record's benefit: it is literally `run/<id>/0`, which is why
`diagnose`, `seen`, `profile`, `overlaid` and `alerts` all read a probe without
knowing one exists. Nothing new was added to the record's shape and nothing new
had to learn to read it.

## The three numbers, and why none of them is a gradient norm

| what | how | why not something else |
|---|---|---|
| `signal_gain` | the scale here against the last normalisation upstream | the drift is geometric, so what matters is the ratio and not the size |
| `jacobian_gain` | `sqrt(E‖Jᵀv‖²)` from here to the output, over random probes | the **backward** signal, scale-free, so it means the same thing at every depth |
| `jacobian_spread` | `s_max / s_rms` of the sketch `JᵀV` | dynamical isometry is about the spectrum's *shape*, and a mean cannot see it |

There is deliberately no `grad_norm` here. At initialisation there is no loss,
so a parameter gradient would have to be taken against a target somebody made
up, and the number would land in the same field the audit fills from a real
loss — at a different scale, judged by the same bound. Two things with one name
is how a threshold quietly stops meaning anything. The backward direction is
`jacobian_gain`, which needs no target and is a ratio.

## What it costs

Two forwards and `k` backwards — and the `k` are over the whole network, **not
`k` per layer**. Every layer reads its own `Jᵀv` off the same backward, which is
what puts this in front of a training run rather than instead of one.
`is_grads_batched` does the `k` in a single call where `vmap` can follow the
model, and falls back to a loop where it cannot, saying so.

The second forward is `architecture`'s, under `no_grad`, and it is what decides
**which** layers are measured. Walking the modules here instead would be one
forward cheaper and would break the invariant the whole layer rests on — *every
layer that can carry a flag has a box* — because a module walk and what the
figure draws are not the same set once `fx` has had its say.

Nothing here changes what the network computes, and no weight is moved.
"""

from __future__ import annotations

import math
import warnings

try:
    import torch
except ImportError:  # pragma: no cover - the trainer module already needs torch
    torch = None

from soma_next.torch._audit import _of_activation
from soma_next.torch._inside import _held, architecture, kind_of

#: How many random probes the Jacobian sketch is taken over. The sketch's own
#: spread is what `jacobian_spread` is measured against, so this is a floor on
#: how fine a spectrum it can see and not a precision knob: below about eight
#: the ratio is noise, and above about thirty-two the backward passes are the
#: cost of a training step.
PROBES = 24


def probe(graph, example, *, depth=0, most=48, probes=PROBES, watching=None, workers=None):
    """What this graph looks like at initialisation, as `{where: numbers}`.

    `where` is a node, or `node.path.to.submodule` — the same keys the audit
    uses and the same scope the figure draws, so a finding from a probe lands on
    exactly the box a finding from a run would.

    The answer is the shape `soma_next.health.seen` returns from a store, which
    is what lets the same numbers be judged either way::

        from soma_next._soma_next import verdict
        {where: verdict(numbers) for where, numbers in probe(g, x).items()}

    `watching=` takes a `Recorder` or anything callable, the same as a
    `Trainer`'s, and writing them down is what makes the diagnosis re-askable
    tomorrow at another bound.
    """
    if torch is None:
        raise RuntimeError("`probe` needs torch")
    watched = _watching(graph, example, depth, most, workers)

    seen, order, hooks = {}, [], []
    for key, module in watched.items():
        hooks.append(module.register_forward_hook(_caught(seen, order, key)))
    try:
        output = graph.forward(_crossable(example), watching=watching, workers=workers)
    finally:
        for hook in hooks:
            hook.remove()

    numbers = {key: _of_one(one) for key, one in seen.items()}
    _signal(numbers, seen, order, example)
    _isometry(numbers, seen, order, _unwrapped(output), probes)

    told = _telling(watching)
    said = {}
    for key, one in numbers.items():
        if not one:
            continue
        node, inside = key
        said[f"{node}.{inside}" if inside else node] = one
        if told is not None:
            told({"fact": "health", "node": node, "inside": inside,
                  **{name: str(what) for name, what in one.items()}})
    return said


def _watching(graph, example, depth, most, workers):
    """Which module each key names, taken from what the figure will draw.

    Not from `_worth_drawing` directly, which is what the audit does and what
    this did first. **The scope has to be the drawing's scope**, and those two
    are not the same thing: `architecture` runs the module walk and then splices
    the symbolic view over it, so at `depth=1` a walk opens a composite the
    figure keeps whole — and a finding on a layer with no box lands nowhere.

    > Every layer that can carry a flag has a box.

    Folded paths are watched too. Six identical blocks are drawn once and
    measured six times, because measuring one of them and calling it the other
    five is a diagnosis of a network nobody built; a finding on any of them
    lands on the box that stands for them, which is what `folded` is for.

    It costs a second forward, and this one is under `no_grad`. At
    initialisation that is worth an invariant being true by construction rather
    than by coincidence.
    """
    said = {}
    # Wrapped here as well: `architecture` obeys the rule every input obeys and
    # a probe is friendlier than that, taking a bare tensor or an `Opaque`.
    for node, inside in architecture(graph, _crossable(example), depth=depth, most=most,
                                     workers=workers).items():
        held = dict(_held(graph.implementation(node)))
        for path in {one.path for one in inside.layers} | set(inside.folded):
            module = _module_at(held, path)
            # An `fx` node that is not a module — a `+`, a `concat` — has a box
            # and nothing to hook. It is drawn and not measured, which is the
            # allowed direction.
            if module is not None:
                said[(node, path)] = module
    return said


def _module_at(held, path):
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


# ── One forward, and what each layer said while it ran ──


def _caught(seen, order, key):
    """A forward hook that keeps what went in and what came out.

    The tensors themselves and not statistics of them: the Jacobian half needs
    to differentiate through them afterwards, and a number cannot be
    differentiated.
    """

    def saw(module, args, out):
        into, made = _tensor(args[0] if args else None), _tensor(out)
        if made is None:
            return
        # The module travels with its tensors. It is what says whether this is
        # the normalisation the next gain is measured from, and looking it up
        # again later would mean keeping a second map in step with this one.
        seen[key] = (into, made, module)
        order.append(key)

    return saw


def _of_one(one):
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


def _signal(numbers, seen, order, example):
    """The scale of the signal, against where the last normalisation left it.

    A norm layer resets the scale, so drift measured from the input would blame
    a layer for what happened three norms ago. Resetting the reference at one is
    **structure and not a bound** — which is why it belongs here, on the
    measuring side of the wall, and the threshold does not.

    In execution order, which is what a hook gives and what a chain means. A
    layer inside a branch is measured against the trunk it hangs off, and that
    is the honest reading: a branch running hot and added to something bigger
    has not moved the signal, and the layer that consumes the sum is where the
    drift shows up.
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


def _isometry(numbers, seen, order, output, probes):
    """The Jacobian from each layer to the output, sketched with random probes.

    `k` unit probes pushed into the output and one backward each; every layer
    reads its own `Jᵀv` off the same pass. What comes back is a `k × n` sketch
    of `Jᵀ`, and two numbers come out of it:

    - `jacobian_gain`, the root mean square of `‖Jᵀv‖` — the factor a gradient
      at the output arrives here by. Its profile over depth is the vanishing
      picture, measured with no optimizer, no target and no step.
    - `jacobian_spread`, `s_max / s_rms` of the sketch. A flat spectrum makes
      the `k` columns look Gaussian and the ratio sits near one; a spectrum with
      a long tail does not. Pennington et al. (2017) is the claim that the shape
      matters and not only the mean.
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


def _sketched(output, made, v, probes):
    """`k` vector-Jacobian products, batched where `vmap` can follow the model.

    The fallback is a plain loop and it is `k` times slower, which is worth
    saying out loud rather than discovering: a probe that quietly costs a
    training step is a probe nobody runs.
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


# ── Odds and ends ──


def _scale(what):
    """How big the signal is, as a standard deviation.

    The deviation and not the mean magnitude: what a normalisation sets is the
    variance, so that is the thing a gain is a gain of.
    """
    if what is None or not torch.is_tensor(what) or what.numel() < 2:
        return None
    with torch.no_grad():
        said = float(what.detach().float().std())
    return said if math.isfinite(said) else None


def _tensor(what):
    """The tensor in whatever arrived, or nothing."""
    if torch.is_tensor(what):
        return what
    if isinstance(what, (tuple, list)) and what:
        return _tensor(what[0])
    return None


def _unwrapped(what):
    """An `Opaque` holds the tensor; everything else is already itself."""
    from soma_next import Opaque

    return what.value if isinstance(what, Opaque) else what


def _crossable(what):
    """A tensor is wrapped to cross an edge; everything else passes as it is."""
    from soma_next import Opaque

    return Opaque(what) if torch.is_tensor(what) else what


def _telling(watching):
    """Whatever `watching=` was given, as one callable — or `None`."""
    from soma_next.torch._trainer import _telling as the_same_one

    return the_same_one(watching)
