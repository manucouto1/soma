"""The measurement behind `MISSING_NORMALISATION`, before the flag was written.

The claim on trial: **the scale of the signal drifting over a stretch nobody is
normalising is worth saying at initialisation**, before a step is taken. The
quantity is the gain against the last normalisation upstream — a norm layer sets
the variance, so drift measured from the input would blame a layer for what
happened three norms ago.

It is a conjunction on purpose, the shape `LOSING_PLASTICITY` already has:
drifting alone is a network that is fine, and having no normalisation alone is a
network that is fine. What is being asked is whether the two together separate.

There is a false positive this has to stay quiet on, and it was already written
down. `examples/07-a-real-architecture.ipynb` dropped the normalisation from a
three-block residual trunk and the un-normalised version scored **better** —
0.0050 against 0.0103 — with activations that grew *less* through the stack. A
bound that fires there is a bound that teaches somebody else's lesson.

What is measured here is the definition the probe ships, and not a stand-in for
it: every **leaf** module's output against the output of the last normalisation
before it, in execution order. On a chain the two are the same number, because
the local ratios telescope; on a residual trunk they are not, and it is the
shipped one that has to be argued with.

Run it with `python health/tests/normalisation.py` (it needs torch, which is why
it is not a `#[test]`). What it printed on 23 August 2026:

    a network that learnt nothing at all scores 0.0780

    -- a residual trunk, the shape notebook 07 measured --
    res  3 blocks  norm    up     0.58  down     0.21 | loss    0.0029 (0.0029-0.0029)
    res  3 blocks  none    up     0.61  down     0.21 | loss    0.0019 (0.0019-0.0019)
    res  8 blocks  norm    up     0.59  down     0.21 | loss    0.0031 (0.0031-0.0031)
    res  8 blocks  none    up     0.69  down     0.21 | loss    0.0020 (0.0020-0.0021)
    res 20 blocks  norm    up     0.61  down     0.20 | loss    0.0033 (0.0033-0.0034)
    res 20 blocks  none    up     0.89  down     0.21 | loss    0.0023 (0.0023-0.0023)

    -- a plain stack, and what its init does to the scale --
    plain 12  g=0.50  none up     0.70  down  4.0e-06 | loss    0.0485 (0.0464-0.0500)
    plain 12  g=0.71  none up     1.00  down  5.5e-04 | loss    0.0297 (0.0257-0.0320)
    plain 12  g=1.00  none up     1.63  down     0.38 | loss    0.0267 (0.0217-0.0279)
    plain 12  g=1.41  none up   101.61  down     1.18 | loss    0.0746 (0.0730-0.0757)
    plain 12  g=2.00  none up  9.9e+03  down     1.66 | loss    9.4430 (0.7635-11.0378)
    plain 12  g=2.00  norm up     2.81  down     0.54 | loss    0.0596 (0.0532-0.0618)

    -- how deep an unnormalised trunk gets before a decade would fire --
    res   3 blocks  none   up     0.61
    res   8 blocks  none   up     0.69
    res  20 blocks  none   up     0.89
    res  40 blocks  none   up     1.42
    res  80 blocks  none   up     4.00

Three readings, and the middle one is the reason the flag has one side.

**Amplifying separates.** Everything that learnt sits at 2.81 or below;
everything that did not is at 100 or above. A bound of one decade sits between
them with 3.6x of margin below and 10x above, and it is a decade because the
drift is geometric — the useful signal is an order of magnitude, not a
percentage.

**Attenuating does not, and it is not close.** A plain stack whose signal
arrives five ten-thousandths of the size it went in trains as well as the
healthy one — the two ranges overlap. Adam is scale-invariant per parameter, so
a signal that shrank does not stop a step from being taken. Attenuation costs
something at the extreme, though still well under the floor, and there is no
bound that separates *costs a little* from *fine*. Inventing one is what
`NARROWING` is a standing lesson against, so `MISSING_NORMALISATION` fires
**only upwards**.

**And it needs both halves.** The same badly-initialised stack *with*
normalisation drifts 2.81x and trains. Structure alone — "there is no norm layer
in this stretch" — would have flagged every plain row including the ones that
trained best.

The last table is the margin: an unnormalised residual trunk grows like the
square root of its depth, so it takes about eighty blocks to reach 4x and would
need some five hundred to trip a decade. That is a network worth asking about.
"""
import torch, math

WIDTH = 64


class Residual(torch.nn.Module):
    """`x + f(x)`, the one operation `fx` sees and a module walk does not."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return x + self.inner(x)


def trunk(blocks, normalised):
    made = []
    for _ in range(blocks):
        if normalised:
            made.append(torch.nn.LayerNorm(WIDTH))
        made.append(Residual(torch.nn.Sequential(
            torch.nn.Linear(WIDTH, WIDTH), torch.nn.GELU(),
            torch.nn.Linear(WIDTH, WIDTH))))
    return torch.nn.Sequential(*made, torch.nn.Linear(WIDTH, WIDTH))


def plain(depth, gain, normalised):
    made = []
    for _ in range(depth):
        made.append(torch.nn.Linear(WIDTH, WIDTH))
        if normalised:
            made.append(torch.nn.LayerNorm(WIDTH))
        made.append(torch.nn.GELU())
    net = torch.nn.Sequential(*made, torch.nn.Linear(WIDTH, WIDTH))
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, torch.nn.Linear):
                # He, times whatever gain is being tried. `g=1` is the init
                # everybody ships; the rows either side of it are the mistake.
                m.weight.normal_(0, gain * math.sqrt(2 / WIDTH))
                m.bias.zero_()
    return net


def gains(net, x):
    """How far up and how far down the scale gets from the last normalisation.

    Leaf modules in execution order, which is what a forward hook gives and what
    the probe does for real. A norm layer resets the reference: that is
    structure and not a threshold, which is why it belongs in the measurement
    rather than beside the bound.

    Both directions come back, because which of them is worth a flag is the
    question and not the assumption.
    """
    seen = []
    hooks = [m.register_forward_hook(lambda mod, _a, out, mm=m: seen.append((mm, out)))
             for m in net.modules() if not any(m.children())]
    with torch.no_grad():
        net(x)
    for h in hooks:
        h.remove()
    reference, up, down = float(x.std()), 0.0, math.inf
    for module, out in seen:
        scale = float(out.float().std())
        # The same rule `somatize.torch.kind_of` uses, written out rather than
        # imported: this crate has no dependencies and its measurements keep it.
        # A normalisation sets the reference and reports no gain of its own —
        # changing the scale is its job, and reading that as drift would put the
        # loudest number on the one layer doing the thing being asked about.
        if type(module).__name__.endswith("Norm"):
            reference = scale
            continue
        if reference > 0 and math.isfinite(scale):
            up, down = max(up, scale / reference), min(down, scale / reference)
    return up, down


def trained(make, steps=1500, seed=0):
    """One net, made and trained under this seed. The seed moves the init and
    the data both: a bound that only holds for one draw of the weights is not a
    bound."""
    torch.manual_seed(1)
    teacher = torch.nn.Sequential(torch.nn.Linear(WIDTH, WIDTH), torch.nn.Tanh(),
                                  torch.nn.Linear(WIDTH, WIDTH))
    for p in teacher.parameters():
        p.requires_grad_(False)
    torch.manual_seed(seed)
    net = make()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    losses = []
    for _ in range(steps):
        x = torch.randn(256, WIDTH)
        with torch.no_grad():
            y = teacher(x)
        loss = torch.nn.functional.mse_loss(net(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.detach().item())
        if not math.isfinite(losses[-1]):
            break
    kept = [l for l in losses[-20:] if math.isfinite(l)]
    return sum(kept) / len(kept) if kept else float("nan")


def shown(value):
    return f"{value:8.2f}" if 1e-2 <= abs(value) < 1e3 else f"{value:8.1e}"


def report(name, make):
    torch.manual_seed(0)
    up, down = gains(make(), torch.randn(256, WIDTH))
    # Three seeds and the median, because one seed either side of a bound is how
    # a threshold gets invented.
    losses = sorted(trained(make, seed=seed) for seed in (0, 1, 2))
    print(f"{name:22} up {shown(up)}  down {shown(down)} | loss {losses[1]:9.4f} "
          f"({losses[0]:.4f}-{losses[2]:.4f})")


torch.manual_seed(1)
with torch.no_grad():
    NOTHING = float(torch.nn.Sequential(torch.nn.Linear(WIDTH, WIDTH), torch.nn.Tanh(),
                                        torch.nn.Linear(WIDTH, WIDTH))(
        torch.randn(4096, WIDTH)).var())
print(f"a network that learnt nothing at all scores {NOTHING:.4f}\n")

print("-- a residual trunk, the shape notebook 07 measured --")
for blocks in (3, 8, 20):
    for normalised in (True, False):
        report(f"res {blocks:2d} blocks  {'norm' if normalised else 'none'}",
               lambda b=blocks, n=normalised: trunk(b, n))

print("\n-- a plain stack, and what its init does to the scale --")
for gain in (0.5, 0.71, 1.0, 1.41, 2.0):
    report(f"plain 12  g={gain:.2f}  none", lambda g=gain: plain(12, g, False))
report("plain 12  g=2.00  norm", lambda: plain(12, 2.0, True))

print("\n-- how deep an unnormalised trunk gets before a decade would fire --")
for blocks in (3, 8, 20, 40, 80):
    torch.manual_seed(0)
    up, _ = gains(trunk(blocks, False), torch.randn(256, WIDTH))
    print(f"res {blocks:3d} blocks  none   up {shown(up)}")
