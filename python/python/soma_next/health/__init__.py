"""Whether what happened was healthy. **An opinion, and it says so.**

    from soma_next.health import diagnose, about
    from soma_next.torch import Audit

    t = Trainer(g, ..., auditing=True, watching=Recorder(store, run="tuesday"))
    ...                                      # train
    diagnose(store, run="tuesday")           # {"body": ["VANISHING"]}
    about("VANISHING")                       # what to do about it

Three things get called observability and they are not the same: the
declaration **drawn**, the record of what **happened**, and this — a judgement
*about* that record, taken at thresholds somebody chose. Mixing them is what
gave the original an `enum Event` of 37 variants with `NodeStarted` beside
`HealthFlag`.

The line between the second and the third is an invariant, and it is a test:

> **A diagnosis has to be reproducible from the stored record, without training
> again.**

So the verdict lives in a Rust crate with **no dependencies at all** — numbers
in, flags out, no measuring and no clock — and this module is what carries a
store to it. Change a bound and ask again; the record has not moved.

## What it can say

| flag | what it means |
|---|---|
| `NAN` / `INF` | a number stopped being one; nothing below it means anything |
| `VANISHING` / `EXPLODING` | the parameter gradients are too small to train on, or too big to step on |
| `DEAD` / `SATURATED` | the output is off, or pinned where the derivative is nothing |
| `STALLED` / `OVERSTEPPING` | the update is tiny, or enormous, next to the weights it moves |
| `DEAD_CHANNELS(n)` / `IGNORED_CHANNELS(n)` | part of the width does nothing, or does something nobody asks for |
| `LEAKAGE` | two groups meant to stay apart carry the same information |
| `NARROWING` | the update has collapsed into a few directions |
| `LOSING_PLASTICITY` | weights growing, rank falling and units going quiet, all at once |

`about(flag)` says what to do about each, and it lives beside the thresholds
because they are one opinion: splitting them is how a dashboard ends up saying
something this library never said.

## Drawn

`profile(store, run=...)` is the picture, because vanishing is a **shape over
depth** and not a property of a layer; `flags(store, run=...)` is the table of
what tripped with its advice beside it. They are the only figures in this
library where colour is allowed to mean good-or-bad, because they are the only
ones drawing opinions.
"""

from soma_next.health._figure import flags, profile
from soma_next.health._read import Thresholds, about, diagnose, history, seen

__all__ = ["Thresholds", "about", "diagnose", "flags", "history", "profile", "seen"]
