---
title: The record of what happened
description: Eleven facts, a hole called Watcher, and a reader that is a price list — with the one rule about where facts come back from.
---

`run()` cannot hand the facts back at the end. That was the requirement the
whole design came out of: somebody training in a notebook, with half the graph
on other machines, **keeps being told** what is going on.

So there is a hole:

```python
g.forward(x, watching=print)
```

`watching=` takes anything callable. It is the Python side of `Watcher`, the
fifth hole in the core — the core says what a `Fact` is and has nowhere to put
one, exactly as it does with `Keeper`.

## Emitting is synchronous. Delivering is the implementor's

`saw` is called from the walk and returns. What the implementor does with the
fact — write it, drop it, push it onto a channel another thread drains into a
figure — is where anything asynchronous belongs.

That is why live costs no runtime, and it matters more than it looks: an
`async` there would be `async` in **every caller of the engine**, and it would
drag `Store` along with it. That objection is what has twice kept a message bus
out of this project. A bus is not refused, it is deferred to where it earns its
place.

## Eleven facts, and each level keeps its own vocabulary

```rust
pub enum Fact {
    Ran { .. }, Failed { .. }, Spared { .. }, Recalled { .. },
    Kept { .. }, Items { .. }, Left { .. }, Elsewhere { .. },
    Said { .. }, Finished { .. }, Broke { .. },
}
```

An enum of facts is fine, and the original's thirty-seven variants were not the
mistake. **The mistake was that they were three vocabularies in one** — a
`NodeStarted` beside a `HealthFlag`, a fact beside an opinion about facts.

Here each level keeps its own: the engine's is this enum, a training run's
(`loss`, `updated`) lives where the loss is, and a trial's has been a record on
disk since the study work. **They do not meet in Rust, they meet in the
record.**

That meeting is the shape a fact is written in: a **name and text-to-text
pairs**. So what you print is what you would find in the store, and a level
that is not the engine says its own things through the same door:

```python
recorder({"fact": "loss", "value": 2.0 * 0.93**step + 0.1})
```

`Fact::Said { kind, pairs }` is the carrier for that on the wire — a carrier
and not a vocabulary, so the core never learns what a loss is, or a load
average.

Four of the eleven are worth knowing by name:

| | |
|---|---|
| `Recalled` | not advanced at all: what it would have produced was already kept |
| `Spared` | not run because **nobody needed** what it makes. A fact and not an absence — a node missing from a record cannot otherwise be told from one that was never in the graph |
| `Left` / `Elsewhere` | a slice crossed to another machine and came back, and this is what happened over there. `Elsewhere` is recursive |
| `Items` | a `.mapped()` node, item by item: it runs the new ones |

## Every measurement is a duration, never an instant

A duration from another machine is worth reading. Two wall clocks disagree.
**When** something was written down is the store's business, and the store
stamps it.

That is also what makes `gantt` compose: every fact carries how far into the
`forward` it began, so a `Wave` draws as overlapping bars and a remote slice
sits inside the round trip it arrived under. An offset into a slice is a fact
about the slice.

## Where facts come back from

> Where a connection is open, facts come back down it. Where there is none,
> they go to the store and whoever wants them scans.

A remote slice's facts come back down the connection that was **already open**:
`dispatch` was blocked reading it anyway, so `Answer` gained one non-terminal
variant, `Saw(Fact)`, and reading one answer became reading until one is
terminal. No second connection, no port, no bus. See
[across machines](/soma/running/across-machines/).

A relay attributes nothing. The client wraps what arrives, because the host's
*name* belongs to the graph and a worker has never heard of it.

## Writing it down

```python
from somatize import Recorder, Store

store = Store(tempfile.mkdtemp())
recorder = Recorder(store, run="tuesday", summarising=["loss"])

for step in range(40):
    g.forward(x, watching=recorder)
    recorder({"fact": "loss", "value": loss})
```

One record per `forward`, at `run/<id>/<n>`. A loss is computed **after** the
forward that made it, so it goes into the record that closed last, rewritten —
and there is no guessing about which, because the two vocabularies come through
different doors.

`summarising=` is what decides which kinds get lifted onto the cheap side of
the reader below. Which kinds those are is the caller's business: the store
still does not learn what a loss is.

## Reading it back is a price list

`somatize.record` is functions over a `Store`, in the same shape as `gather`
and `take`:

| call | cost |
|---|---|
| `runs(store)` | one scan |
| `forwards(store, run=…)` | one scan |
| `curve(store, run=…)` | one scan — *because of* `summarising=` |
| `facts(store, run=…, forward=n)` | one fetch |
| `nodes(store, run=…)` | **a fetch per forward** |

So everything a progress view asks for is free, and the per-node breakdown is
asked once rather than once a step.

```python
curve_costs(store, run="tuesday")   # "scanned" or "fetched"
```

That function exists because **a reader that is quietly a thousand times slower
is worse than one that says so.**

## And it draws

```python
from somatize.record import Live, gantt, machines, progress, spent

progress(store, run="tuesday")   # what happened, from the store
spent(store, run="tuesday")      # where the time went
gantt(store, run="tuesday")      # the timeline, waves overlapping
```

`Live` is handed facts as they happen. It and `progress` fill **one drawing
function**, and they can, because a fact read back is the very dict a watcher
was given. A live view and a report written twice are two things that slowly
stop agreeing.

The colours come from one table in `somatize._theme`, which is also what this
site is painted from. One fact per channel: hue says *where*, never
good-or-bad, and the only red marks a `forward` that broke — which is a fact.

The smooth line is a **centred rolling mean and not a spline**. A spline
through measured points invents the values between them, and a loss dipping
below a minimum that never happened is a figure that lies.

## What it is not

It is not a judgement. Whether 400 ms is slow, or a gradient is dying, is an
**opinion about the record** — and the invariant that keeps the two apart is a
test: *a diagnosis has to be reproducible from the stored record, without
training again*. See [the health of a network](/soma/looking/health/).
