---
title: Gradient Health Audit
description: Diagnose vanishing/exploding gradients, dead units, NaNs, and saturation per filter without manual hooks.
---

When a training loop misbehaves — loss plateaus, NaNs appear, a layer
goes dead — the question is always *which node*. Soma installs the
plumbing to answer that:

```python
with g.gradient_audit() as audit:
    for x, y in batches:
        with g.context() as ctx:
            g.zero_grad()
            out, aux = g.forward(x)
            loss = compute_loss(out, y, aux)
            g.backward(ctx, loss)
        g.step(ctx)

print(audit.report().pretty())
audit.assert_healthy()        # raises GradientHealthError on any flag
```

`gradient_audit()` registers forward + backward hooks on every live
`DifferentiableFilter._module` on entry. Param-grad statistics are
captured by `Graph.backward` after `loss.backward()` returns, so
gradients are fully accumulated when read (the standard
`register_full_backward_hook` fires too early for early modules).
Hooks are removed on exit, even on exception.

## What gets measured

Per filter, per training step:

- **Activations** — mean (|·|), std, min, max, %zero, %NaN, %Inf,
  %saturated (`|x| > saturation_threshold`).
- **Output gradient** — L2 norm, max |grad|, NaN/Inf flags. This is
  the gradient flowing *into* this filter from the next.
- **Parameter gradient** — L2 norm, max |grad|, NaN/Inf flags,
  %zero entries.
- **Parameter norm** and **`∂/θ` ratio** — proxy for relative update
  magnitude. Adam-tuned models typically sit in `1e-3 … 1e-1`.

Aggregated across all steps inside the with-block (audit a whole
epoch). The report sorts filters in topological order so the failing
node is right next to the one feeding it.

## Flags

| Flag | Trigger |
|---|---|
| `HEALTHY` | None of the others fired |
| `VANISHING` | `param_grad_norm < thresholds.grad_lo` |
| `EXPLODING` | `param_grad_norm > thresholds.grad_hi` |
| `NAN` / `INF` | Any tensor (act / out_grad / param_grad) had NaN or Inf |
| `DEAD` | `> thresholds.dead_frac` of activations are below `dead_eps` |
| `SATURATED` | `> thresholds.saturation_frac` of activations exceed `activation_saturation` |
| `NO_DATA` | The filter never saw a forward inside the audit |

Defaults are tuned for typical CV/NLP scales (LayerNorm-ish
activations, Adam steps). Override per call:

```python
from soma import Thresholds

with g.gradient_audit(thresholds=Thresholds(
    grad_lo=1e-9, grad_hi=1e2,
    dead_frac=0.99,
    saturation_frac=0.7,
)) as audit:
    ...
```

## Reading a report

```python
rep = audit.report()
print(rep.pretty())
# filter   steps  act|μ|     act σ      |out∂|     |θ∂|       |θ|        ∂/θ        flags
# -----------------------------------------------------------------------------------------
# encoder  10     2.137e-01  4.011e-01  6.842e-04  3.011e-05  3.402e+01  8.853e-07  VANISHING
# pooler   10     1.772e-01  3.998e-01  4.881e-02  6.230e-02  9.111e+00  6.838e-03  HEALTHY
# head     10     8.114e-02  3.022e-01  1.123e+00  9.001e-01  4.220e+00  2.133e-01  HEALTHY
```

The `encoder` here has param-grad norm 5 orders of magnitude smaller
than its parameter norm — a textbook vanishing signal localised to one
node. Fixes are surgical: bump that filter's LR, rescale init, swap
activation, or unfreeze deeper layers.

`audit.report().dataframe()` returns a pandas DataFrame for plotting
or persisting alongside other metrics:

```python
df = audit.report().dataframe()
df.to_csv("audit.csv", index=False)
```

## Standalone (no `Graph`)

For users who haven't migrated to the `Graph` orchestrator yet,
`audit_modules` works on a list of `(name, module)` pairs:

```python
from soma import audit_modules

with audit_modules([("encoder", encoder), ("head", head)]) as audit:
    for x, y in batches:
        opt.zero_grad()
        out = head(encoder(x))
        loss = ce(out, y)
        loss.backward()
        audit._snapshot_after_backward()      # explicit when no Graph
        opt.step()

audit.assert_healthy()
```

Without `Graph` driving `backward`, you call `_snapshot_after_backward`
yourself once gradients have accumulated.

## When to use it

- **Bring-up of a new pipeline.** Run one epoch under audit to
  confirm gradients reach every filter.
- **Regression tests.** `audit.assert_healthy()` in a smoke test
  catches a layer that silently went dead after a refactor.
- **Comparing pipelines.** Persist `audit.dataframe()` next to
  metrics so a drop in F1 has a *health* explanation.
- **Mixed-precision / fine-tuning debugging.** NaN / saturation
  flags localise the layer where overflow originates.

## RPC-readiness

Like the [training-loop primitives](../design/gradients/#native-training-loop-python),
the audit API is shaped so the same call sites work once filters live
on remote workers. Hook installation and snapshot collection move
worker-side; the aggregator reads records via `rpc.rpc_sync`. User
code does not change.
