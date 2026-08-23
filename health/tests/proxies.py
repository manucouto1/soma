"""The measurement behind the zero-cost proxies, and the question is one thing.

A proxy does **not** diagnose a network. `synflow` of one network is a number
with no meaning; it only means something next to another network's. So a proxy
is not a `Flag` and never was — it is a **cheap objective**, the thing a study's
loop scores with instead of training, which puts it at level 3 where there is no
type at all.

Which leaves exactly one question worth measuring, and it is not "does it
correlate with the score":

> **Does it beat counting parameters?**

Abdelfattah et al. (ICLR 2021) report `synflow` at 0.76 rank correlation with
parameter count across NAS-Bench-201, which is close to saying it measures size.
Size is free. A proxy that costs a forward and a backward has to earn the
difference, so every row below carries the parameter-count baseline beside it
and the only column that matters is the gap.

The five, and what each costs:

| proxy | what it reads | data |
|---|---|---|
| `synflow` | `sum(abs(theta * dR/dtheta))` with all weights made positive | none |
| `snip` | the same product against a real loss | one batch |
| `grasp` | `-sum(theta * H g)`, so a double backward | one batch |
| `zen` | how far the output moves when the input is nudged | noise |
| `naswot` | how differently the units switch across a batch | one batch |

Run it with `python health/tests/proxies.py` (it needs torch, which is why it is
not a `#[test]`). What it printed on 23 August 2026 — four of the twenty-four
candidates, and then the answer:

    relu-d2-w16      loss   0.0081 synflow=     7.02  snip=    -1.37  grasp=    -0.08  zen=     1.53  naswot=  -131.90  parameters=  1344.00
    relu-d8-w128     loss   0.0049 synflow=    20.61  snip=    -2.60  grasp=    -0.00  zen=    -3.96  naswot=   320.87  parameters=123936.00
    tanh-d2-w128     loss   0.0005 synflow=     5.21  snip=    -0.36  grasp=    -0.03  zen=     1.87  naswot=   282.59  parameters= 24864.00
    tanh-d8-w16      loss   0.0226 synflow=     4.32  snip=    -1.35  grasp=    -0.02  zen=    -1.36  naswot=   133.62  parameters=  2976.00

    24 candidates, 3 seeds each, median loss
      parameters  rho vs score   0.59   (the baseline: size is free)
         synflow  rho vs score  -0.16   rho vs parameters   0.42   gap -0.75
            snip  rho vs score   0.61   rho vs parameters  -0.02   gap +0.02
           grasp  rho vs score  -0.08   rho vs parameters   0.57   gap -0.67
             zen  rho vs score   0.45   rho vs parameters  -0.39   gap -0.14
          naswot  rho vs score   0.69   rho vs parameters   0.97   gap +0.10

**Only `snip` earns its keep, and it is not the one that scored highest.**

`naswot` has the best rank correlation with the trained score, 0.69 against the
free baseline's 0.59 — and a correlation of **0.97 with parameter count**. It is
size, with noise on top. A tenth of a rank correlation is what it adds over a
number that costs nothing to compute, and reporting the 0.69 on its own would be
a claim nobody could check.

`snip` is the only one that beats counting **and** is uncorrelated with size:
0.61 against 0.59, at -0.02 with parameter count. Two hundredths is not much,
but it is two hundredths of something **orthogonal** to what size already says,
which is the only kind of gain a proxy can honestly claim.

`synflow` comes out at **-0.16, worse than nothing**, and the reason is worth
more than the number. On this family it reads **depth**: a `relu` stack at depth
eight scores 12 to 21 where the same widths at depth two score 7 to 9, and depth
is what hurts on this task. Abdelfattah et al. report 0.76 with parameter count
on NAS-Bench-201, where depth and size move together; here they do not, and the
proxy follows the half that is wrong.

Which is the whole reason this library ships **all five and picks none**. Which
proxy is worth anything depends on the family being searched, and that is a
question with a cheap answer — this file — rather than a default somebody has to
discover is wrong.

Two things this cannot show, and they are why it is not the last word. Twenty-four
MLPs on one teacher task is not a benchmark, and every candidate here trains: the
spread of scores is a fortieth to a two-thousandth, so what is being ranked is
*better and worse* rather than *works and does not*.
"""
import torch, math

WIDTH_OF = (16, 32, 64, 128)
DEPTH_OF = (2, 4, 8)
INPUT = 32
SEEDS = (0, 1, 2)


def candidate(depth, width, how):
    made = [torch.nn.Linear(INPUT, width),
            torch.nn.ReLU() if how == "relu" else torch.nn.Tanh()]
    for _ in range(depth - 1):
        made += [torch.nn.Linear(width, width),
                 torch.nn.ReLU() if how == "relu" else torch.nn.Tanh()]
    return torch.nn.Sequential(*made, torch.nn.Linear(width, INPUT))


# ── The five ──


def synflow(net, x):
    """Tanaka et al. (2020). Data-free: every weight made positive, a batch of
    ones through it, and the sum of what each parameter contributes."""
    signs = {}
    with torch.no_grad():
        for name, p in net.named_parameters():
            signs[name] = p.sign()
            p.abs_()
    net.zero_grad()
    net(torch.ones(1, INPUT)).sum().backward()
    score = sum(float((p * p.grad).abs().sum()) for p in net.parameters() if p.grad is not None)
    with torch.no_grad():
        for name, p in net.named_parameters():
            p.mul_(signs[name])
    net.zero_grad()
    return math.log(score) if score > 0 else float("-inf")


def snip(net, x, y):
    """Lee et al. (2019). The same product, against a loss that saw data."""
    net.zero_grad()
    torch.nn.functional.mse_loss(net(x), y).backward()
    score = sum(float((p * p.grad).abs().sum()) for p in net.parameters() if p.grad is not None)
    net.zero_grad()
    return math.log(score) if score > 0 else float("-inf")


def grasp(net, x, y):
    """Wang et al. (2020). `-theta . H g`: how the gradient's own size responds
    to the weights, which needs the gradient to stay differentiable."""
    net.zero_grad()
    held = [p for p in net.parameters() if p.requires_grad]
    g = torch.autograd.grad(torch.nn.functional.mse_loss(net(x), y), held, create_graph=True)
    hessian = torch.autograd.grad(sum((one * one).sum() for one in g) / 2, held)
    score = -sum(float((p * h).sum()) for p, h in zip(held, hessian))
    net.zero_grad()
    return score


def zen(net, x, eps=1e-2):
    """Lin et al. (2021), the expressivity half: how far the output moves when
    the input is nudged. Forward only, and it never sees a label."""
    with torch.no_grad():
        noise = torch.randn_like(x)
        moved = float((net(x + eps * noise) - net(x)).norm())
    return math.log(moved / eps) if moved > 0 else float("-inf")


def naswot(net, x):
    """Mellor et al. (2021). Two inputs that switch the same units the same way
    are two inputs this network cannot tell apart; the log determinant of the
    Hamming kernel is how many it can.

    The code is the **sign** of each unit's output rather than a `relu` mask, so
    a `tanh` network gets a code too. Saying that out loud matters: the paper is
    about rectifiers and this is the obvious extension, not the paper's claim.
    """
    codes = []
    hooks = [m.register_forward_hook(lambda _m, _a, out: codes.append((out > 0).float()))
             for m in net.modules()
             if isinstance(m, (torch.nn.ReLU, torch.nn.Tanh))]
    with torch.no_grad():
        net(x)
    for h in hooks:
        h.remove()
    if not codes:
        return float("nan")
    code = torch.cat([c.reshape(x.shape[0], -1) for c in codes], dim=1)
    agree = code @ code.t() + (1 - code) @ (1 - code).t()
    sign, value = torch.linalg.slogdet(agree.double() + 1e-3 * torch.eye(x.shape[0]).double())
    return float(value) if sign > 0 else float("-inf")


# ── What they are being compared against ──


def trained(make, steps=1500, seed=0):
    torch.manual_seed(1)
    teacher = torch.nn.Sequential(torch.nn.Linear(INPUT, 64), torch.nn.Tanh(),
                                  torch.nn.Linear(64, INPUT))
    for p in teacher.parameters():
        p.requires_grad_(False)
    torch.manual_seed(seed)
    net = make()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    losses = []
    for _ in range(steps):
        x = torch.randn(256, INPUT)
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


def spearman(a, b):
    """Rank correlation, the same thirty lines `study.importance` uses.

    Ranks and not values, because a proxy is only ever asked which of two
    networks is better and its units are nobody's.
    """
    kept = [(x, y) for x, y in zip(a, b) if math.isfinite(x) and math.isfinite(y)]
    if len(kept) < 3:
        return float("nan")
    n = len(kept)

    def ranked(values):
        order = sorted(range(n), key=lambda i: values[i])
        rank = [0.0] * n
        at = 0
        while at < n:
            same = at
            while same + 1 < n and values[order[same + 1]] == values[order[at]]:
                same += 1
            for i in range(at, same + 1):
                rank[order[i]] = (at + same) / 2
            at = same + 1
        return rank

    ra, rb = ranked([x for x, _ in kept]), ranked([y for _, y in kept])
    mean_a, mean_b = sum(ra) / n, sum(rb) / n
    top = sum((x - mean_a) * (y - mean_b) for x, y in zip(ra, rb))
    left = math.sqrt(sum((x - mean_a) ** 2 for x in ra))
    right = math.sqrt(sum((y - mean_b) ** 2 for y in rb))
    return top / (left * right) if left and right else float("nan")


torch.manual_seed(1)
TEACHER = torch.nn.Sequential(torch.nn.Linear(INPUT, 64), torch.nn.Tanh(),
                              torch.nn.Linear(64, INPUT))
for p in TEACHER.parameters():
    p.requires_grad_(False)

named = ["synflow", "snip", "grasp", "zen", "naswot", "parameters"]
taken = {name: [] for name in named}
scored, labels = [], []
for how in ("relu", "tanh"):
    for depth in DEPTH_OF:
        for width in WIDTH_OF:
            make = lambda d=depth, w=width, h=how: candidate(d, w, h)
            torch.manual_seed(0)
            net = make()
            x = torch.randn(64, INPUT)
            with torch.no_grad():
                y = TEACHER(x)
            taken["synflow"].append(synflow(make(), x))
            taken["snip"].append(snip(make(), x, y))
            taken["grasp"].append(grasp(make(), x, y))
            taken["zen"].append(zen(make(), x))
            taken["naswot"].append(naswot(make(), x))
            taken["parameters"].append(sum(p.numel() for p in net.parameters()))
            # Lower is better, so the sign is flipped: every correlation below
            # is against **how good the network turned out**.
            losses = sorted(trained(make, seed=seed) for seed in SEEDS)
            scored.append(-losses[1])
            labels.append(f"{how}-d{depth}-w{width}")
            print(f"{labels[-1]:16} loss {-scored[-1]:8.4f} "
                  + "  ".join(f"{n}={taken[n][-1]:9.2f}" for n in named))

print(f"\n{len(labels)} candidates, {len(SEEDS)} seeds each, median loss\n")
base = spearman(taken["parameters"], scored)
print(f"{'parameters':>12}  rho vs score {base:6.2f}   (the baseline: size is free)")
for name in named[:-1]:
    against = spearman(taken[name], scored)
    size = spearman(taken[name], taken["parameters"])
    print(f"{name:>12}  rho vs score {against:6.2f}   rho vs parameters {size:6.2f}"
          f"   gap {against - base:+.2f}")
