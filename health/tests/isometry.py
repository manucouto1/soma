"""The measurement behind the isometry half of the static probe.

The claim on trial: **at initialisation, the singular spectrum of the Jacobian
from a layer to the output says whether the network will train**, before a step
is taken. Not its size — that is the gradient norm, and `VANISHING` already
reads it — but its *shape*. Pennington, Schoenholz and Ganguli (NeurIPS 2017)
show that a spectrum concentrated near one trains dramatically faster than a
spectrum with the same mean and a long tail, and that orthogonal initialisation
through a `tanh` achieves it while Gaussian initialisation and `relu` cannot.

Two numbers come out of the same probes, and only one of them is new:

- **gain** — `sqrt(E||J^T v||^2)`, the backward signal from here to the output.
  Its profile over depth is the vanishing picture, measured without an
  optimizer, a loss or a step.
- **spread** — `s_max / s_rms` of the sketch `J^T V`, with `V` a batch of `k`
  random probes. A flat spectrum makes those `k` columns look Gaussian and the
  ratio sits near one; a spiked one shows up as a ratio that does not.

The cost is `k` backwards for the whole network, not `k` per layer: every layer
sees its own `J^T v` on the same backward, which is what makes this affordable
enough to put in front of a training run.

Run it with `python health/tests/isometry.py` (it needs torch, which is why it
is not a `#[test]`). What it printed on 23 August 2026:

    predicting nothing scores 0.0780
                                 gain first  gain last spread max       loss
    -- at criticality, where the literature says the shape is what differs --
         orthogonal-tanh depth 8   2.98e-01   1.00e+00       1.10    0.0016-0.0017
           gaussian-tanh depth 8   2.86e-01   1.01e+00       1.27    0.0021-0.0022
                 he-relu depth 8   1.41e+00   1.43e+00       1.41    0.0444-0.0467
        orthogonal-tanh depth 20   1.94e-01   1.00e+00       1.11    0.0017-0.0018
          gaussian-tanh depth 20   1.86e-01   1.00e+00       1.36    0.0064-0.0069
                he-relu depth 20   9.80e-01   1.42e+00       1.87    0.0451-0.0458
        orthogonal-tanh depth 40   1.35e-01   1.00e+00       1.11    0.0022-0.0022
          gaussian-tanh depth 40   7.82e-02   1.02e+00       1.41    0.0163-0.0178
                he-relu depth 40   7.45e-01   1.44e+00       1.88    0.0549-0.0602
    -- and walking off it, which is where a network stops being able to train --
    gaussian-tanh g=0.5 depth 20   3.81e-07   5.01e-01       1.33    0.0635-0.0655
    gaussian-tanh g=0.8 depth 20   4.73e-03   8.01e-01       1.33    0.0173-0.0175
    gaussian-tanh g=1.25 depth 20   1.23e+00   1.25e+00       1.47    0.0337-0.0361
    gaussian-tanh g=1.5 depth 20   3.89e+00   1.50e+00       1.63    0.0605-0.0670
    gaussian-tanh g=2.0 depth 20   3.75e+01   2.00e+00       1.76    0.0685-0.0716
    gaussian-tanh g=4.0 depth 20   1.04e+05   4.00e+00       4.72    0.0977-0.1027
    gaussian-tanh g=0.5 depth 40   1.67e-13   5.10e-01       1.37    0.0695-0.0720
    gaussian-tanh g=0.8 depth 40   2.65e-05   8.15e-01       1.37    0.0579-0.0601
    gaussian-tanh g=1.25 depth 40   1.95e+00   1.27e+00       1.90    0.0692-0.0706
    gaussian-tanh g=1.5 depth 40   1.66e+01   1.53e+00       2.20    0.0694-0.0721
    gaussian-tanh g=2.0 depth 40   7.86e+02   2.04e+00       2.90    0.0689-0.0720
    gaussian-tanh g=4.0 depth 40   8.44e+07   4.08e+00       3.30    0.0996-0.1074

**Neither of them earned an alarm, and the second table is why.** Both rank: the
nine rows at criticality come out almost in order, with orthogonal-`tanh` at a
spread of 1.10 and the best loss, and `he-relu` at 1.88 and the worst. Ranking
is not separating, and a flag has to separate.

Walking the gain across criticality is what shows it. On the upper side the
worst network that still trains — `he-relu` at depth 8 — has a first-layer gain
of **1.41**, and the best network that does not — `gaussian-tanh` at `g=1.25`
and depth 40 — has **1.95**. A factor of 1.4 between them is not a bound, it is
where the sampling happened to land; one more row at `g=1.35` would sit inside
it. The spread does worse still: `he-relu` at depth 20 reads **1.87** and trains
to 0.0451, while `g=2.0` at depth 20 reads **1.76** and does not train at all.
The failing network has the *tighter* spectrum.

So both numbers are **recorded and drawn and neither raises anything**, which is
what `NARROWING` established as the thing to do when a measurement does not
support an alarm. A profile of `jacobian_gain` over depth is a picture somebody
can read, and that is a weaker and honest claim.

There is a rule underneath the three measurements of this slice, and it is worth
more than any of them. `MISSING_NORMALISATION` separates because the forward
scale is a **runaway**: a geometric process either stays where it was or leaves
by decades, and there is nothing in between to be wrong about. These two vary
**continuously** with how well the network turns out. Something continuous is a
ranking, and a ranking belongs at level 3 beside the proxies — where a number
only ever means something next to another candidate's — and not in a vocabulary
of findings about one network.
"""
import torch, math

WIDTH = 64
PROBES = 24


def sketch(net, x, probes=PROBES):
    """Per layer, the gain to the output and the spread of its spectrum.

    One backward per probe, and every layer reads its own `J^T v` off the same
    one. `is_grads_batched` does the `k` of them in a single call, which is
    where the affordability comes from.
    """
    kept = {}
    hooks = []
    for name, child in net.named_children():
        def caught(_mod, args, out, n=name):
            kept[n] = args[0]
        hooks.append(child.register_forward_hook(caught))
    into = x.detach().requires_grad_(True)
    out = net(into)
    for h in hooks:
        h.remove()
    wanted = [t for t in kept.values() if t.requires_grad]
    names = [n for n, t in kept.items() if t.requires_grad]
    if not wanted:
        return {}
    v = torch.randn(probes, *out.shape)
    v = v / v.reshape(probes, -1).norm(dim=1).reshape(probes, *([1] * out.dim()))
    grads = torch.autograd.grad(out, wanted, grad_outputs=v, is_grads_batched=True,
                                retain_graph=True)
    said = {}
    for name, g in zip(names, grads):
        flat = g.reshape(probes, -1)
        gain = float(flat.norm(dim=1).pow(2).mean().sqrt())
        s = torch.linalg.svdvals(flat.double())
        rms = float(s.pow(2).mean().sqrt())
        said[name] = (gain, float(s[0]) / rms if rms > 0 else float("nan"))
    return said


def stack(depth, how, gain=1.0):
    """A deep stack, initialised the way the literature says matters.

    `gain` is the departure from criticality: Poole et al. (2016) put the
    edge of chaos for a `tanh` network at unit gain, and either side of it the
    signal either collapses onto a point or decorrelates into noise. It is the
    knob that makes a network that genuinely cannot train, which is what a bound
    has to be shown against.
    """
    made = []
    for _ in range(depth):
        made.append(torch.nn.Linear(WIDTH, WIDTH))
        made.append(torch.nn.Tanh() if "tanh" in how else torch.nn.ReLU())
    net = torch.nn.Sequential(*made, torch.nn.Linear(WIDTH, WIDTH))
    with torch.no_grad():
        for m in net.modules():
            if not isinstance(m, torch.nn.Linear):
                continue
            m.bias.zero_()
            if how.startswith("orthogonal"):
                # The critical point: an orthogonal map has every singular
                # value equal, which is dynamical isometry by construction.
                torch.nn.init.orthogonal_(m.weight, gain=gain)
            elif how.startswith("gaussian"):
                m.weight.normal_(0, gain * math.sqrt(1 / WIDTH))
            else:
                m.weight.normal_(0, gain * math.sqrt(2 / WIDTH))
    return net


def trained(make, steps=1500, seed=0):
    torch.manual_seed(seed + 1)
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


torch.manual_seed(1)
NOTHING = float(torch.nn.Sequential(torch.nn.Linear(WIDTH, WIDTH), torch.nn.Tanh(),
                                    torch.nn.Linear(WIDTH, WIDTH))(torch.randn(4096, WIDTH)).var())
print(f"predicting nothing scores {NOTHING:.4f}\n")
def report(label, depth, how, gain=1.0):
    torch.manual_seed(0)
    seen = sketch(stack(depth, how, gain), torch.randn(64, WIDTH))
    if not seen:
        return
    gains = [g for g, _ in seen.values()]
    spreads = [s for _, s in seen.values()]
    losses = sorted(trained(lambda: stack(depth, how, gain), seed=seed) for seed in (0, 1))
    print(f"{label:>28} {gains[0]:10.2e} {gains[-1]:10.2e} "
          f"{max(spreads):10.2f} {losses[0]:9.4f}-{losses[-1]:.4f}")


print(f"{'':28} {'gain first':>10} {'gain last':>10} {'spread max':>10} {'loss':>10}")
print("-- at criticality, where the literature says the shape is what differs --")
for depth in (8, 20, 40):
    for how in ("orthogonal-tanh", "gaussian-tanh", "he-relu"):
        report(f"{how} depth {depth}", depth, how)

print("-- and walking off it, which is where a network stops being able to train --")
for depth in (20, 40):
    for gain in (0.5, 0.8, 1.25, 1.5, 2.0, 4.0):
        report(f"gaussian-tanh g={gain} depth {depth}", depth, "gaussian-tanh", gain)
