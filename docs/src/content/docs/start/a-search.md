---
title: A search, and the machines running it
description: A study end to end — the loop, a second machine against the same folder, and reading the trials back. No coordinator, no port, nothing to keep alive.
---

Searching hyper-parameters is the level above a training run, and here it has
**no type**: N runs are a `for` loop. What `somatize.study` provides are the
pieces that loop asks for — where to look, when to give up, and how two
machines share the work without talking to each other.

Everything below was run. The numbers are the ones it printed.

## The loop

```python
import math, os, socket, sys
from somatize import Store
from somatize.study import (Pruner, Sampler, Space, curves, finished,
                            in_flight, report, take)

store = Store(sys.argv[1])
me = f"{socket.gethostname()}/{os.getpid()}"
STUDY, TRIALS = "spam", 24

space = (Space()
         .real("lr", 1e-5, 1e-1, log=True)
         .int("width", 16, 256)
         .choice("opt", ["adam", "sgd"]))
sampler = Sampler.tpe(goal="min")
pruner = Pruner.median(goal="min", warmup=3)

for trial in range(TRIALS):
    point = sampler.ask(space, trial, finished(store, space, study=STUDY)
                        + in_flight(store, space, study=STUDY))
    if not take(store, point, study=STUDY, trial=trial, me=me, goal="min"):
        continue                      # somebody else has that one

    reported, why = [], None
    for epoch in range(12):
        reported.append(train(point, epoch))
        report(store, point, reported, study=STUDY, trial=trial, me=me,
               state="running", goal="min")
        if why := pruner.verdict(reported, curves(store, study=STUDY)):
            break

    report(store, point, reported, study=STUDY, trial=trial, me=me,
           state="pruned" if why else "done", score=reported[-1],
           because=why, goal="min")
```

That is the whole thing — and `train` is yours, whatever it is. A pruner
**stops nothing**: it answers, and the loop stops calling. Which is why none of
this asked the `Trainer` for a single new method.

## The second machine is the same file

Run it twice against one directory:

```console
$ python search.py /scratch/study &   python search.py /scratch/study &   wait

yilliqiya/1354201: 17 of 24 trials were mine
yilliqiya/1354200: 7 of 24 trials were mine
```

Seventeen and seven. **Every trial claimed exactly once**, and nothing
coordinated it — no server, no port, no lock daemon, nothing to keep alive.

It works because a trial is a **number** and `ask` is a function of that
number: a machine that claims trial 7 derives the same point as everybody else
would have, without replaying six. So the only thing left to settle is *who has
7*, and `take` settles it with one conditional write. **The state is the
queue.**

The share is uneven, and that is the point rather than a flaw. Nobody was
assigned anything; the machine that got free first took the next number. Slurm
distributes, and a shared folder is all the machines have in common.

`in_flight` is what stops the two proposing neighbours: `ask` takes a score
that **may be absent**, and absent means *running*. Those points are avoided
without voting on how big the good pile is — handing a guided sampler a made-up
bad score for a running trial moves the threshold the wrong way, which was
[measured](/soma/searching/handing-out-trials/).

## Reading it back

```python
from somatize.study import direction, finished, importance, trials

trials(store, space, study="spam")     # every trial, whatever state
```

```console
24 trials, goal: min
states: {'pruned': 17, 'done': 7}
who:    {'…/1354201': 17, '…/1354200': 7}

  0.0549  lr=0.004373620971140737,width=240,opt=adam
  0.0573  lr=0.0009029392293863634,width=200,opt=adam
  0.0662  lr=0.00009610717139068465,width=245,opt=adam
```

Each row is one scan and no fetches, because the configuration is kept as
**text beside the score**. Only a curve costs a fetch, and `curves` says so in
its own docstring. `goal` is written per trial rather than per study: a score is
a number and the number does not say which way is better, so a reader who never
saw this script has no other way to find out.

## A number that would be misread

```python
importance(store, space, study="spam")    # Spearman's ρ, biggest first
```

```console
[('width', 0.29), ('lr', 0.11), ('opt', 0.0)]
```

`opt` scored **0.00**, and it does not mean the optimizer is irrelevant. All
seven trials that ran to the end used `adam` — TPE had converged on it and the
rest were pruned — so there is nothing for a rank correlation to correlate
with. The library says this itself:

> `0.0` where a knob never varied: no evidence, which is not no effect.

Which is the shape of the whole family: pruned and finished are **never ranked
together**, because a pruned trial's score was measured after fewer epochs and
is not comparable. A figure that put them on one axis would be a figure that
lies.

## The pictures

```python
from somatize.study import coordinates, influence, table

table(store, space, study="spam")         # every trial, sortable
influence(store, space, study="spam")     # a knob against the score
coordinates(store, space, study="spam")   # parallel coordinates
```

They return plotly figures, so a notebook shows them and a script saves them.
[Notebook 04](/soma/tutorials/04-a-study/) runs all three on a real problem.

## Where to go next

[A study](/soma/searching/a-study/) is the design underneath: the three
families, why `ask` is a function of the index, and where the line between Rust
and Python falls. [Handing trials out of a folder](/soma/searching/handing-out-trials/)
is the distributed half — `claim` against a directory and against an S3 bucket,
what a folder can and cannot tell you, and the measurement behind `in_flight`.
The full surface is [`somatize.study`](/soma/reference/python/study/).
