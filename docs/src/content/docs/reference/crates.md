---
title: The crates
description: Eight published crates and one that is not, what each one is for, and what each one costs to depend on.
---

The workspace is nine crates. Eight are published to crates.io as
`somatize-*`; the ninth is the extension module and carries `publish = false`,
because its API is Python's.

The directory is named after what it holds and the package after what it is
called on crates.io — `soma-core/` builds `somatize-core`. The names diverge
because `soma` and `soma-core` on crates.io belong to other people, and
`somatize` was already ours.

| crate | directory | what it is for |
|---|---|---|
| `somatize-core` | `soma-core/` | the graph, the plan of how to run it, and the engine that walks that plan |
| `somatize-store` | `soma-store/` | where a computed value is kept: bytes by content, and names that point at them |
| `somatize-data` | `soma-data/` | where the data comes from, and what it is once it arrives |
| `somatize-study` | `soma-study/` | a search over configurations, and what each trial did |
| `somatize-health` | `soma-health/` | whether what a run did is healthy — an opinion, and it says so |
| `somatize-tree` | `soma-tree/` | what an edit did to a graph, said before anybody runs it |
| `somatize-fabric-wire` | `soma-fabric/wire/` | carrying a slice of a plan to another process, and bringing back what it produced |
| `somatize-fabric-broker` | `soma-fabric/broker/` | the name a graph gave a host, turned into a way of reaching it |
| `_somatize` | `soma-python/` | the PyO3 module. Not published as a crate |

## What each one costs

The column that decides most of the layout. Counted with
`cargo tree --edges normal`, default features, the crate itself included:

| crate | crates pulled in |
|---|---|
| `somatize-core` | **1** |
| `somatize-study` | **1** |
| `somatize-health` | **1** |
| `somatize-store` | 24 |
| `somatize-fabric-wire` | 25 |
| `somatize-fabric-broker` | 26 |
| `somatize-tree` | 55 |
| `somatize-data` | 64 |
| `_somatize` | 152 |

Three of them are **one**, which is to say they have no dependencies at all.

For the core that is a position: it provides five holes and fills none of them,
so there is nothing for it to depend on. `serde` is behind a feature and off by
default.

For `study/` and `health/` it is stronger than a position, it is what makes an
invariant testable. `health/` is numbers in and flags out — no measuring, no
clock, and **not even the core**. That is what makes *a diagnosis has to be
reproducible from the stored record, without training again* a test rather than
an aspiration: change a bound and ask again, and the record has not moved.

## Why `fabric` is two crates and not one with modules

The dependency between them runs **one way only**: `broker` depends on
`wire`, because resolving a name has to end in something that can carry a
slice. Nothing in `wire` reaches for a broker — grep it and there are no hits.
That asymmetry is what makes *the cable knows nothing about the rendezvous* a
fact the compiler checks rather than a sentence in a document, and it is why a
worker needs no broker at all in order to serve a slice: it is being talked
to, and finding out where it is was somebody else's problem.

## Why there is no runtime

There is no `tokio` anywhere in this list, and `Store` is synchronous on
purpose. That is the reason SQL is not in the data layer: every driver worth
using carries a runtime, and an `async` `Store` would be `async` in every
caller of it.

`data/` takes Arrow as the type and leaves the tool to whoever wants one — an
expression engine was measured at around 370 crates when that decision was
taken, and the worker that only tokenizes has no use for expressions. Sixty-four
is what Arrow and parquet cost on their own.

## Features, and where they stop

| crate | feature | default | what it adds |
|---|---|---|---|
| `somatize-core` | `serde` | off | the plan and the keys, serialisable |
| `somatize-store` | `s3` | off | `Store.on_bucket(...)`, the second implementor |
| `somatize-tree` | `cli` | **on** | the `somatize-tree` binary, and `clap` with it |
| `_somatize` | `remote` | **on** | the wire and the broker |

Cargo features **do not reach the wheel.** `pip install somatize[viz]` adds
dependencies *of Python*; the Rust is compiled in. What the features are for is
the worker's image, which is built `--no-default-features`, and not for anyone
installing the package.

## The ten that are not here

The original's workspace is thirteen crates. Three of those names survive —
core, store and the Python module — and ten have no counterpart at all: the
`soma` facade, plus `-compiler`, `-runtime`, `-macros`, `-worker`, `-memory`,
`-agent`, `-llm`, `-mcp` and `-coordinator`.

Nothing had to be done with them. They simply stopped publishing versions, and
versions on crates.io are immutable, so `0.5.1` still resolves for anyone who
pinned it.

`-compiler` and `-worker` are the two worth a sentence, because their absence
is a decision and not an omission. Compiling a graph is `compile`, a free
function in `somatize-core`; and a worker is a binary in the Python package,
`python -m somatize.worker`, because what it needs is an interpreter and not a
crate.

For what is inside each one, see the
[Rust API reference](/soma/api/rust/).
