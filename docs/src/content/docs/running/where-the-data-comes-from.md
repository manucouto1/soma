---
title: Where the data comes from
description: "A source is a node — there is no Source trait — and what the graph is handed is a coordinate, not a batch: 121 ms of hashing against 0.027 ms."
---

There is **no `Source` trait**, and that is the whole design.

A source takes something and answers with something, which is `Node`. A second
trait with a method that does what `forward` does would be a hole with one
tenant *and* the `E0034` the rules warn about — two traits with a same-named
method in scope make that name unusable.

Being a node is the point. `.at()`, `.cached()`, `.mapped()`, the record and
the figure all reach a dataset with **nothing written for them**.

```python
from somatize.data import Parquet

g = Graph.somatize(
    Parquet("data/train").named("rows").frozen()
    >> Tokenize().named("tok")
    >> Model().named("model")
)
```

## What the graph is handed is a coordinate

Not a batch. A `Span` — `at` and `take`, a position and a length.

The reason is arithmetic. The input is the one value a cache has to hash **by
content**, because there is nothing else to name it by. Measured on 24 August
2026, release build, with one node returning a constant so that only the input
grows:

| what is handed to `forward` | with a store behind it |
|---|---|
| 1 MB of tensor | 6.1 ms |
| 19 MB of tensor (32×3×224×224) | **121 ms**, every step, hit or miss |
| a `Span` | **0.027 ms** |

Nothing there is pathological — `torch.save` runs at 1 ms/MB and sha256 at
2 ms/MB. **Which is why the answer is not a faster hash.** There is no faster
hash to reach for. The answer is not to weigh the batch.

A span is a position, and a position can be asked for twice. That is what makes
a source *settled*: what moves is which spans exist, not the source's state.

## The other half of the name was free

A source has to state its version **without reading itself**, or the saving
above is spent again.

Against a `Store` that costs nothing: a name resolves to a digest, and the
digest **is** the hash of the content. So `Parquet::version` is one `resolve`
and no bytes, through `Memory::freeze` — which is the call made twice on
purpose.

That closed a silent bug. A source declared `.frozen()` looked exactly like a
tokenizer, so its version stayed out of the key and **two different datasets
shared a name**.

## Arrow is the type; polars is the tool

`arrow` and `parquet` are in `soma-data`. An expression engine is not, and
whoever wants one brings their own — around 370 crates when that was measured,
charged to a worker that only tokenizes.

**No runtime came in.** Zero `tokio`, which is exactly why **SQL is not here**,
and that is decided rather than delayed: every Rust driver worth using carries a
runtime, and `Store` is synchronous on purpose. An `async` `Store` would be
`async` in every caller of it.

Also not here: **ranged reads**, which need the store to learn to read a range
first.

`Ipc` is the **second implementor of `Codec`**, and it comes from another crate
— which is what promoted `Codec` into the core in the first place. So a frame
is kept, and a frame crosses a wire.

## Streaming is not a mode

There are no stream semantics to declare, no `FixedState` / `Evolving` /
`Barrier` to pick between, and no second code path.

> **The difference between training and deploying is how many rows the frame
> brings.**

4096 from a folder of parquet, or one from a topic. Same graph, same nodes,
same plan.

## Asking what an input was worth

```python
from somatize.data import contribution

contribution(g, batches, objective=accuracy, over=["text", "symptoms"])
```

It shuffles one input and scores again; the drop is what that input was worth.

**Shuffled and not zeroed** — a zero is a value, and what is being asked about
is the correspondence with the answer. See
[the health of a network](/soma/looking/health/), where `IgnoredInput` and
`SoleReliance` are the two findings that come out of it.

## And it composes with everything else

Because it is a node and nothing more:

```python
Parquet("s3://bucket/train").named("rows").frozen().at("loader").mapped()
```

`.at()` sends the reading to the machine that has the disk. `.mapped()` names
each item by its content rather than its place. `.frozen()` puts the version in
the key. None of that was written for data.
