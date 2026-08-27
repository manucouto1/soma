---
title: Handing trials out of a folder
description: A directory everybody mounted, a conditional write, and no coordinator — plus the measurement that says what to tell a guided sampler about a running trial.
---

N machines search one space together, and **none of them ever speaks to
another**. What they have in common is a directory they all mounted.

```python
for trial in range(100):
    point = sampler.ask(space, trial, finished(store, space, study="spam"))
    if not take(store, point, study="spam", trial=trial, me=me, goal="min"):
        continue                                   # somebody else has that one
    ...
    report(store, point, reports, study="spam", trial=trial, me=me,
           state="done", goal="min")
```

Same script everywhere. Slurm hands out `me`.

## `claim` is one method, and it is on the trait for a reason

```rust
/// Points a name at some bytes **only if nobody has**, and says whether it did.
fn claim(&self, name: &str, digest: &Digest, meta: Meta) -> Result<bool, StoreError>;
```

Not `resolve` then `bind`. Between those two somebody else does the same, and
two machines train the same round while nobody trains the next. So it is on the
trait **with no default implementation** — one written out of the other two
would be a race with a doc comment on it.

That single method is the whole coordination layer. There is no server, no
port, no protocol and no lease to renew.

## Because a trial is a number

A trial is an index. `ask` is a function of that index. So **nothing is handed
out**: a machine that claims trial 7 derives its point without replaying six,
and the claim settles who gets it.

The state *is* the queue, and a claim is exactly-once by construction.

## A directory or a bucket, and they are the same store

```python
store = Store("/mnt/shared/studies")
store = Store.on_bucket("http://127.0.0.1:9000", "studies")   # `s3` feature, off by default
```

Same layout, same JSON. What differs is that on a bucket `claim` is a
**conditional PUT**.

This is why the bucket implementation went in when it did. Of the three uses of
a store, only one demands a genuinely shared disk: a cache degrades to a miss
and an artifact degrades to a miss, but **handing out work does not degrade** —
it silently duplicates.

An endpoint that accepts `If-None-Match: *` and writes anyway would give every
trial to every machine and say nothing. So `Bucket::at` spends **two round
trips proving it does not** before handing the store over.

Nothing above learned a new word: `take`, `report` and `gather` never asked
what kind of store they had.

## What a folder can and cannot tell

```python
from somatize.study import abandoned, in_flight

in_flight(store, space, study="spam")   # what other machines are holding, no score
abandoned(store, study="spam")          # what has stopped moving
```

A machine dies mid-trial and the folder cannot tell that from a long epoch.
Nothing can, from the outside. So `abandoned` **reports and the loop chooses**
— it does not decide.

Being wrong is cheap, and deliberately so: too eager costs a trial run twice,
and a claim still cannot collide, because `claim` uses a link and the highest
attempt wins. A trial whose machine died is rescued by claiming the next
attempt.

`STALE` is one hour, and it is an argument to change rather than a constant to
respect.

## The measured part: what to tell a guided sampler

A guided sampler spread over machines needs to know what the others are
holding, or two of them propose neighbours. The obvious fix is to hand it the
running trials with a made-up bad score, so it avoids them.

**Measured, that gets it backwards.** A bad score does not merely fail to help:
it moves the threshold that separates the good pile from the bad one, and it
moves it the wrong way.

So `ask` takes a score that **may be absent**, and absent means *running*.
Those points are kept away from **without voting on how big the good pile is**.

That is the whole reason `in_flight` returns trials each with no score rather
than with a placeholder.

## Why one record and not five events

A trial lives at `<study>/trial/<n>/<attempt>`, as **one record rewritten as it
goes**.

A bus would emit `TrialStarted`, `TrialPruned`, `TrialCompleted`. Those three
are the **diff** of this record: derivable from it, while the record is not
derivable from them without replaying all three in order. And a scan of the
folder already carries what a sampler needs, so the cheap question stays cheap.

| where | what | cost |
|---|---|---|
| the record | `state`, `point`, `score`, `who`, `goal` | one scan, no fetches |
| the blob | the whole curve, and why it stopped | a fetch each |

That is why a sampler's entire history is one scan: the configuration is kept
as **text** beside the score. Only a pruner pays.

## The same shape one level up

Federation reuses all of it. A training run exports its weights node by node,
`fedavg` is a function, and a federated round is a `for` — with the folder
doing the same job:

```python
from somatize.torch import fedavg, gather
```

`gather` waits for the round, and **whoever finds it complete claims the
averaging**, so there is still no coordinator to keep alive. See
[training](/soma/running/training/).

## It has been run for real

`soma-python/tests/cluster/test_searching.py` searches hyper-parameters over
real SMS messages with the graph cut across containers — tokenising where there
is no torch at all — **and** the study cut across machines.

Two distributions at once, and they are not the same one: one cuts a `forward`
across hosts, the other cuts a `for` across machines. Nothing in either knows
about the other.
