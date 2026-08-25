"""The measurement behind `Thresholds::narrowing_of_usual`, kept so it can be
argued with.

Huang et al. (2026) monitor the spectrum of `dW = W_t - W_{t-d}` and find it
collapses thousands of steps before the loss does. Their certificate is the
deviation from a **healthy baseline run**; a single training run has no
baseline, so the substitution on trial here is the update's own recent median.

Run it with `python health/tests/narrowing.py` (it needs torch, which is why it
is not a `#[test]`). What it printed on 23 August 2026:

    -- healthy --
    lr=3e-4 s0    srank 2.3 -> 4.4 | lowest vs own median 0.69 | loss broke @ None
    lr=3e-4 s1    srank 1.9 -> 5.2 | lowest vs own median 0.71 | loss broke @ None
    lr=3e-4 s2    srank 2.2 -> 4.4 | lowest vs own median 0.70 | loss broke @ None
    -- hot --
    lr=0.05 s0    srank 2.6 -> 2.4 | lowest vs own median 0.65 | loss broke @ 1
    lr=0.05 s1    srank 1.7 -> 2.2 | lowest vs own median 0.43 | loss broke @ 2
    lr=0.2  s0    srank 2.5 -> 1.3 | lowest vs own median 0.52 | loss broke @ 1
    lr=0.2  s1    srank 2.0 -> 1.3 | lowest vs own median 0.86 | loss broke @ 1
    lr=0.5  s0    srank 2.3 -> 1.0 | lowest vs own median 0.43 | loss broke @ 1
    lr=0.5  s1    srank 2.1 -> 1.2 | lowest vs own median 0.46 | loss broke @ 1

Healthy dips to 0.69-0.71 and destabilised ranges 0.43-0.86: **they overlap in
both directions**, so no bound separates them, and the flag is off by default.

Two things this run cannot show, and they are the reason it is not the last
word rather than the reason to ignore it. The hot runs broke at step 1, so
there is no lead time here to demonstrate — which is the monitor's whole claim.
And the trend does separate cleanly (healthy 2.3 -> 4.4, hot 2.5 -> 1.3), but
against runs that were already broken, so it is not evidence of early warning
either.
"""
import torch, math

def srank(a):
    f = a.norm() ** 2
    top = torch.linalg.matrix_norm(a, 2) ** 2
    return float(f / top) if top > 0 else float("nan")

def run(lr, steps=1500, delta=10, width=128, seed=0):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(width, width), torch.nn.GELU(),
        torch.nn.Linear(width, width), torch.nn.GELU(),
        torch.nn.Linear(width, width), torch.nn.GELU(),
        torch.nn.Linear(width, width),
    )
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    teacher = torch.nn.Sequential(torch.nn.Linear(width, width), torch.nn.Tanh(),
                                  torch.nn.Linear(width, width))
    for p in teacher.parameters():
        p.requires_grad_(False)
    watched = net[4].weight
    history, losses, snaps = [], [], {}
    for step in range(steps):
        x = torch.randn(256, width)
        with torch.no_grad():
            y = teacher(x)
        loss = torch.nn.functional.mse_loss(net(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.detach().item())
        if step % delta == 0:
            snaps[step] = watched.detach().clone()
            if step - delta in snaps:
                history.append((step, srank(snaps[step] - snaps.pop(step - delta))))
        if not math.isfinite(losses[-1]):
            break
    return history, losses

def report(name, history, losses):
    ranks = [r for _, r in history]
    worst, at = 1.0, None
    for i in range(20, len(ranks)):
        usual = sorted(ranks[i - 20:i])[10]
        if usual > 0 and ranks[i] / usual < worst:
            worst, at = ranks[i] / usual, history[i][0]
    first = min(losses[:50])
    broke = next((i for i, l in enumerate(losses)
                  if not math.isfinite(l) or l > 5 * first), None)
    print(f"{name:20} srank {ranks[0]:6.1f} -> {ranks[-1]:6.1f} | "
          f"lowest vs own median {worst:.2f} @ {at} | loss broke @ {broke}")
    return worst, broke

print("-- healthy --")
healthy = [report(f"lr=3e-4 s{s}", *run(3e-4, seed=s)) for s in (0, 1, 2)]
print("-- hot --")
for lr in (0.05, 0.2, 0.5):
    for s in (0, 1):
        report(f"lr={lr} s{s}", *run(lr, seed=s))
