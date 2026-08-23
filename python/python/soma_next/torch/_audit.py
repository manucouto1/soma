"""Measuring what a verdict needs, from inside the nodes.

    t = Trainer(g, objective=..., optimizer=..., auditing=True,
                watching=Recorder(store, run="tuesday"))

This is the **measuring** half of CU21 and it decides nothing. What it produces
is a `health` fact per node per audited step — numbers, in the same record as
everything else — and whether those numbers are bad is `soma_next.health`'s
opinion, taken later and re-takeable with other bounds.

That split is the invariant of the whole layer written as two files: *a
diagnosis has to be reproducible from the stored record, without training
again*. If the thresholds lived here, they would be baked into the measurement
and an argument about a bound would cost an afternoon of GPU.

## Where the numbers come from

| what | from |
|---|---|
| gradient norm, parameter norm, update ratio | the node's `parameters()`, after `backward` |
| zero fraction, saturated fraction, NaN, Inf | a forward hook on the node's modules |
| dead / dormant / ignored channels | per-channel means over the window, opt-in |
| effective rank, group CKA | a snapshot pass, on a cadence |
| the update's stable rank | two snapshots of a weight, on a cadence |

## What it costs, and why almost all of it is opt-in

The per-step half is O(numel) reductions on tensors that are already in memory:
a norm, a comparison, a mean. The **snapshot** half is an SVD, and it runs every
`snapshot` steps rather than every step for exactly that reason. Channels are
off unless asked for, because a per-channel reduction is another pass.

Nothing here changes what the network computes. Hooks read; they do not write.
"""

from __future__ import annotations

import math

try:
    import torch
except ImportError:  # pragma: no cover - the trainer module already needs torch
    torch = None


SATURATED_AT = 50.0
"""What counts as pinned, matching the default threshold on the other side.

It is here **and** there because they are two different jobs: this one decides
what to count, that one decides how much counting is too much. They agree today
and the day somebody moves the bound, the record still says what was counted."""

DEAD_EPS = 1e-7
"""And what counts as off."""


class Audit:
    """Watches the nodes of a graph and says what it saw.

    Built by the `Trainer` when `auditing=` is given; there is little reason to
    make one by hand except to choose a cadence::

        Trainer(g, ..., auditing=Audit(every=10, channels=True))

    `groups` declares channel partitions a node is meant to keep apart, so that
    two of them carrying the same information can be found::

        Audit(groups={"encoder": {"audio": range(0, 64), "text": range(64, 128)}})
    """

    def __init__(
        self,
        *,
        every=1,
        snapshot=50,
        channels=False,
        groups=None,
        window=20,
        inside=None,
        most=32,
    ):
        #: How often a step is measured at all. Every step is affordable for the
        #: cheap half and is the default; a long run can afford less.
        self.every = every
        #: How often the SVD half runs. Separate from `every` because they are
        #: two costs, and conflating them would price the cheap one like the
        #: expensive one.
        self.snapshot = snapshot
        #: Per-channel statistics, which cost another pass over each activation.
        self.channels = channels or bool(groups)
        #: Which channels a node means to keep apart, per node.
        self.groups = dict(groups or {})
        #: How many measured steps a `Seen` is reduced over. The maxima that
        #: `DEAD` and `SATURATED` read are maxima over **this**.
        self.window = window
        #: Whether to look **inside** a node, and how far. A node is often a
        #: whole architecture, and *this node is unhealthy* is not an answer
        #: when the node is twenty layers. `True` is the automatic scope;
        #: `{"encoder": 2}` is a depth; `{"head": ["attn.*"]}` are patterns.
        self.inside = inside
        #: How many submodules of one node to hook. A cap and not a policy:
        #: what is dropped is said out loud.
        self.most = most
        self._hooks = []
        self._seen = {}
        self._history = {}
        self._weights = {}
        self._step = 0

    # ── Attaching ──

    def watch(self, graph):
        """Hooks every node of this graph, and — with `inside=` — what is in it.

        What comes back is keyed by node, and by `node.path.to.submodule` for
        anything inside one. The dot is not decoration: it is what lets a
        figure colour the **node** while the hover says which layer of it.
        """
        self.release()
        self.watching = {}
        for node in graph.nodes():
            held = graph.implementation(node)
            for name, module in _modules(held):
                self.watching[(node, None)] = module
                for path, inner in _scoped(module, self.inside, node, self.most, depth=0):
                    # Prefixed with the attribute it hangs off. A node with two
                    # modules on it would otherwise have two `0`s, and the
                    # second would quietly overwrite the first. An empty path is
                    # the module itself, and `stem.` is not a name.
                    self.watching[(node, f"{name}.{path}" if path else name)] = inner
        for key, module in self.watching.items():
            self._hooks.append(
                module.register_forward_hook(_forward_of(self, key), always_call=False)
            )
        return self

    def release(self):
        """Takes the hooks off. A hook nobody removed is a graph nobody can
        garbage-collect, and a second `watch` would double every count."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    # ── Measuring ──

    def observed(self, graph):
        """One `health` fact per node, for the step that just finished.

        Called by the `Trainer` after `backward`, which is the only moment when
        the activations of this step and the gradients for them are both in
        hand. Empty on a step this audit is not measuring.
        """
        self._step += 1
        if (self._step - 1) % self.every:
            self._seen.clear()
            return []
        snapping = (self._step - 1) % self.snapshot == 0
        said = []
        for key, module in getattr(self, "watching", {}).items():
            node, inside = key
            held = module if inside else graph.implementation(node)
            one = self._seen.pop(key, {})
            one.update(_of_parameters(self, key, held))
            if snapping:
                one.update(_of_snapshot(self, key, held))
            if not one:
                continue
            one["node"] = node
            if inside:
                one["inside"] = inside
            said.append(self._windowed(key, one))
        self._seen.clear()
        return said

    def _windowed(self, key, one):
        """This step's numbers, reduced over the window a verdict is taken on.

        The maxima are maxima over the window and the rest are the latest,
        which is not an inconsistency: `DEAD` asks *did this ever happen* and a
        gradient norm asks *what is it now*.
        """
        kept = self._history.setdefault(key, [])
        kept.append(one)
        del kept[: -self.window]
        said = {"fact": "health", **one}
        for name in ("zero_frac", "sat_frac"):
            over = [step[name] for step in kept if name in step]
            if over:
                said[f"{name}_max"] = max(over)
                said.pop(name, None)
        said["nan"] = any(step.get("nan") for step in kept)
        said["inf"] = any(step.get("inf") for step in kept)
        for name in ("eff_rank", "param_norm"):
            said.update(_slope(name, kept))
        usual = sorted(step["update_rank"] for step in kept if "update_rank" in step)
        if len(usual) >= 3:
            said["update_rank_usual"] = usual[len(usual) // 2]
        return said


# ── What one hook sees ──


def _forward_of(audit, key):
    """A forward hook that writes this step's activation statistics down."""

    def saw(_module, _args, output):
        tensor = _tensor(output)
        if tensor is None:
            return
        one = audit._seen.setdefault(key, {})
        one.update(_of_activation(tensor))
        if audit.channels:
            one.update(_of_channels(audit, key, tensor))

    return saw


def _of_activation(t):
    """What a tensor says about itself. Cheap: three reductions."""
    with torch.no_grad():
        f = t.detach().float()
        said = {"nan": bool(torch.isnan(f).any()), "inf": bool(torch.isinf(f).any())}
        finite = f[torch.isfinite(f)]
        if finite.numel() == 0:
            return said
        magnitude = finite.abs()
        said["act_abs_mean"] = float(magnitude.mean())
        said["zero_frac"] = float((magnitude < DEAD_EPS).float().mean())
        said["sat_frac"] = float((magnitude > SATURATED_AT).float().mean())
        return said


def _of_channels(audit, key, t):
    """Per-channel means, accumulated across the window.

    The channel axis is the last one, which is what a `Linear` produces and what
    a transformer carries. A convolution's is not, and saying so is better than
    guessing: this measures what it can name.
    """
    if t.dim() < 2:
        return {}
    with torch.no_grad():
        f = t.detach().float()
        flat = f.reshape(-1, f.shape[-1])
        magnitude = flat.abs()
        per_channel = magnitude.mean(dim=0)
        zero = (magnitude < DEAD_EPS).float().mean(dim=0)
        held = audit._weights.setdefault(key, {})
        held["act_per_channel"] = per_channel.cpu()
        held["zero_per_channel"] = zero.cpu()
        held["act_matrix"] = flat.cpu()
        layer = float(per_channel.mean())
        said = {"dead_channels": int((zero > 0.95).sum())}
        if layer > 0:
            # Sokar et al. (ICML 2023): dormancy is a **normalised** score, so a
            # channel attenuated a thousandfold is dormant while being perfectly
            # alive. Dead and dormant are two findings and conflating them loses
            # the useful one.
            said["dormancy_frac"] = float((per_channel / layer <= 0.1).float().mean())
        return said


def _of_parameters(audit, key, held):
    """What the gradients say, and how big a step they are about to take."""
    params = list(held.parameters()) if hasattr(held, "parameters") else []
    grads = [p.grad for p in params if p.grad is not None]
    if not params:
        return {}
    with torch.no_grad():
        said = {}
        if grads:
            norm = math.sqrt(sum(float(g.detach().float().pow(2).sum()) for g in grads))
            said["grad_norm"] = norm
            said["nan"] = any(bool(torch.isnan(g).any()) for g in grads)
            said["inf"] = any(bool(torch.isinf(g).any()) for g in grads)
        weights = math.sqrt(sum(float(p.detach().float().pow(2).sum()) for p in params))
        said["param_norm"] = weights
        # The update against the weights it moved, which practice puts near
        # 1e-3. It is the ratio and not either half: a big step on big weights
        # is an ordinary step.
        before = audit._weights.get(key, {}).get("flat")
        now = torch.cat([p.detach().float().reshape(-1) for p in params]).cpu()
        if before is not None and before.shape == now.shape and weights > 0:
            said["update_ratio"] = float((now - before).norm()) / weights
        audit._weights.setdefault(key, {})["flat"] = now
        return said


def _of_snapshot(audit, key, held):
    """The expensive pass: what the representation and the update look like.

    Every `snapshot` steps, because both of these are an SVD and the rest of the
    audit is a handful of reductions.
    """
    said = {}
    kept = audit._weights.setdefault(key, {})
    matrix = kept.pop("act_matrix", None)
    if matrix is not None and min(matrix.shape) >= 2:
        said["eff_rank"] = _eff_rank(matrix)
        groups = audit.groups.get(key[0] if key[1] is None else f"{key[0]}.{key[1]}")
        if groups:
            said["group_cka"] = _leakage(matrix, groups)
    if audit.channels:
        said.update(_ignored(kept))
    said.update(_update_rank(kept, held))
    return said


def _eff_rank(matrix):
    """Effective rank (Roy & Vetterli, 2007): the exponential of the entropy of
    the normalised singular values. How many directions the representation is
    really using, which a plain rank cannot say."""
    with torch.no_grad():
        try:
            values = torch.linalg.svdvals(matrix)
        except Exception:
            return math.nan
        share = values / values.sum().clamp_min(1e-12)
        share = share[share > 0]
        return float(torch.exp(-(share * share.log()).sum()))


def _leakage(matrix, groups):
    """The largest linear CKA between two declared groups of channels.

    Kornblith et al. (2019). The centring is not decoration: CKA is defined on
    centred features, and without it two representations that merely share an
    offset read as the same one.
    """
    named = list(groups)
    features = {}
    for name in named:
        index = torch.as_tensor(list(groups[name]), dtype=torch.long)
        features[name] = matrix.index_select(1, index)
    worst = math.nan
    for i, a in enumerate(named):
        for b in named[i + 1 :]:
            said = _cka(features[a], features[b])
            worst = said if math.isnan(worst) else max(worst, said)
    return worst


def _cka(x, y):
    with torch.no_grad():
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)
        top = (y.T @ x).norm() ** 2
        below = (x.T @ x).norm() * (y.T @ y).norm()
        return float(top / below) if below > 0 else math.nan


def _ignored(kept):
    """Channels alive in the forward pass that no gradient ever comes back for.

    Gradient starvation: the network computes something and never asks for it.
    A **dormant** channel is not ignored — it is not computing anything to be
    ignored — so the two are counted apart.
    """
    per_channel = kept.get("act_per_channel")
    grads = kept.get("grad_per_channel")
    if per_channel is None or grads is None or per_channel.shape != grads.shape:
        return {}
    layer = float(per_channel.mean())
    if layer <= 0:
        return {}
    alive = (per_channel / layer) > 0.1
    return {"ignored_channels": int((alive & (grads < 1e-9)).sum())}


def _update_rank(kept, held):
    """The stable rank of `W_t - W_{t-d}`: how many directions this node moved
    in between two snapshots.

    `||A||_F^2 / ||A||_2^2`, which needs one singular value rather than all of
    them. Recorded and drawn; **not** flagged by default — see
    `health/tests/narrowing.py` for the measurement that decided that.
    """
    weight = _biggest(held)
    if weight is None:
        return {}
    now = weight.detach().float().cpu().clone()
    before = kept.get("snapshot")
    kept["snapshot"] = now
    if before is None or before.shape != now.shape:
        return {}
    with torch.no_grad():
        moved = now - before
        top = torch.linalg.matrix_norm(moved, 2) ** 2
        if float(top) <= 0:
            return {}
        return {"update_rank": float(moved.norm() ** 2 / top)}


def _biggest(held):
    """The widest 2-D parameter a node holds, which is the one worth watching.

    One and not all of them: this is an SVD, and a bias vector has nothing to
    say about how many directions a layer moved in.
    """
    if not hasattr(held, "parameters"):
        return None
    two = [p for p in held.parameters() if p.dim() == 2]
    return max(two, key=lambda p: p.numel(), default=None)


def _modules(held):
    """The torch modules a node holds, as `[(attribute, module)]`.

    A node is the user's class and there is no protocol for *give me your
    layers* — `parameters()` is all a node promises. Its attributes are where
    the modules are, which is the same place the original looked, and the
    attribute's name is what keeps two of them apart.
    """
    if torch is None or isinstance(held, type):
        return []
    return [(name, one) for name, one in vars(held).items() if isinstance(one, torch.nn.Module)]


def _tensor(output):
    """The tensor a node produced, out of whatever it returned."""
    from soma_next import Opaque

    if isinstance(output, Opaque):
        output = output.value
    if torch is not None and isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        return _tensor(output[0])
    return None


def _slope(name, kept):
    """How a metric is moving over the window, relative to its own size.

    Relative, so that a rank of 40 falling by one and a norm of 0.4 falling by
    a hundredth are the same finding. Silent under three points: two make a
    line through anything.
    """
    over = [step[name] for step in kept if name in step and math.isfinite(step[name])]
    if len(over) < 3 or over[0] == 0:
        return {}
    return {f"{name}_slope": (over[-1] - over[0]) / (abs(over[0]) * len(over))}


def _scoped(root, inside, node, most, depth=0):
    """Which submodules of a node to look at, as `[(path, module)]`.

    Three ways of saying it, because three questions get asked. `True` is *look
    inside and work out where*; an `int` is *this many levels down*; a list of
    patterns is *these, by name*. The root is never in the answer — it is
    already audited under the node's own id.

    The automatic one is the original's, and the heuristic is worth keeping:
    **direct children that own parameters, descending one extra level through a
    single-child wrapper.** That last clause is the `nn.Sequential` case, which
    is what almost everybody writes, and without it the automatic scope answers
    *the sequential* and nothing useful.
    """
    import fnmatch

    said = inside.get(node, False) if isinstance(inside, dict) else inside
    if said is None or said is False:
        return []
    named = [(name, one) for name, one in root.named_modules() if name]
    has = lambda one: any(True for _ in one.parameters(recurse=True))  # noqa: E731

    if isinstance(said, (list, tuple)):
        chosen = [
            (name, one)
            for name, one in named
            if any(fnmatch.fnmatchcase(name, pattern) for pattern in said)
        ]
    elif isinstance(said, int) and said is not True:
        chosen = [(n, m) for n, m in named if n.count(".") + 1 <= said and has(m)]
    else:
        # **The same scope the figure draws**, and that is not a convenience: a
        # finding lands on a box only if the thing it was measured on has one.
        # They were two rules for a while, and what that looked like was every
        # flag piled into the node's label with the layers underneath unmarked.
        from soma_next.torch._inside import _worth_drawing

        chosen = [(path, one) for path, one in _worth_drawing(root, depth) if has(one)]

    if len(chosen) > most:
        # Said out loud, because a cap that quietly drops half a network is a
        # diagnosis of the half it kept.
        import warnings

        warnings.warn(
            f"`{node}` has {len(chosen)} submodules to look at and the cap is {most}: "
            f"dropping {[n for n, _ in chosen[most:]]}. Raise `Audit(most=...)` or "
            f"narrow it with `inside={{'{node}': ['...']}}`",
            stacklevel=3,
        )
        chosen = chosen[:most]
    return chosen
