---
title: The rules it is written under
description: Vertical slices, a type taxonomy that avoids dyn Trait soup, and the rule that a hole with no tenant is deleted.
---

These are not aspirations. Each one was paid for by something that went wrong
in the version before this one, and each one is enforced by something — the
compiler, a test, or a use case that closed with a decision written down.

## Vertical slices, not crate layers

Nothing is written without a real consumer **today**. Building a whole crate
before anything uses it is how the original ended up with fourteen traits with
a single implementor and two with none.

The consequence is visible in the shape of the repository: `study/` was the
first crate with no dependencies at all, not even the core's, and it was
written when there was a loop that needed it. `data/` came in when a source had
to be a node. Neither existed in anticipation.

## The type taxonomy

The rule that avoids the `dyn Trait` soup, and it decides three ways:

**An `enum`** when the set is closed and you know it. It is the domain's
language, and the compiler keeps track when a variant is added.

**A `trait`** only when the implementation is supplied by *someone else* — the
user. If two real implementors cannot be named today, it is a struct. Two
traits with a same-named method in scope also make that name unusable
(`error[E0034]`), even when the signatures differ.

**A `struct` with typestate** for the invariants, so an impossible state cannot
be written rather than being validated.

And the thing that does not survive the trip from Java: behaviour does not
belong to a type, it belongs to the **(type, trait) pair**. `Codec` is declared
in the core, and `impl Codec for Ipc` lives in `soma-data` while
`impl Codec for Codecs` lives in `soma-python` — neither of them anywhere near
the file that declares the trait. Implementations scatter out of necessity and
are looked up by the trait, not by the type.

## A hole with no tenant is deleted

The core provides five holes and fills none of them: `Node` is the user's,
`Transport` carries a slice elsewhere, `Keeper` hashes a recipe and keeps what
it names, `Watcher` is told what happened, and `Codec` writes down what only
exists in one process.

There have been six. `Driver` served what a suspended node asked for, and after
eighteen use cases it had **no consumer outside its own tests** — its own
docstring said it was there to keep the agentic layer out of the core, which is
a hole justifying itself. It went, and with it went `Transition` and `Await`: a
node is a function.

What it left behind is the channel. `Ctx` is where whoever executes hands a
node what it knows, so something that wants a value injected puts it there and
**no node signature changes**.

`Watcher` arrived with two implementors in two crates on the first day. That is
the bar.

## One file per type

The type, its inherent `impl`s and the errors its operations produce, together.
An inherent `impl` is **never** split across files: if it feels like it should
be, the operation was probably not a method of that type. That was the case of
`run`, which needed the graph *and* the catalog *and* the input, and ended up a
function.

A family of types gets a folder with a `mod.rs` — never a `family.rs` sitting
next to a `family/`. The folder is created when the family already has members,
never in anticipation. The tests mirror the same shape.

## The tests are another crate

They live outside `src/`, so they only see the public API and cannot pass by
leaning on anything private. One test binary per type, not one per file.

## The core knows nothing about Python

`#[pyclass]` does not go into `core/`. The moment a core type carries one, it
can no longer be used without an interpreter loaded. `python/` translates; if a
domain rule ends up written there, it is in the wrong place.

## Rules the use cases produced

Three more, each one the conclusion of a slice rather than a starting position:

> **What separates is a runaway. What ranks is a proxy.**

The forward scale is geometric — it stays put or it leaves by decades, with
nothing in between to be wrong about, so it earns an alarm. Everything
continuous is a ranking, and a ranking belongs where a number only means
something next to another candidate's. That is why two of the three things a
probe measures raise nothing at all: they rank, and the network with the
tighter spectrum was the one that failed.

> **If it can be recalculated it is record. If somebody thought it, it is
> reasoning.**

A pruned line is derived and never stored; a verdict is written down and never
guessed at. It is what keeps standing a *derived* property rather than a field:
a hypothesis goes back to the previous fact when the commit its refutation
rested on is judged invalid, with nobody saying anything again.

> **One fact per channel.**

In every figure this library draws, hue says *where* something ran or *which*
series it is, and it never doubles as good-or-bad. The only colour that means a
judgement is the one marking a `forward` that broke, which is a fact. The
colours are one table, and this site is painted out of it.

## Rust core, Python interface

Rust for the engine: no GC pauses in the middle of a wave, an ownership model
that makes data races in parallel execution a compile error, and serde, which
is what lets a slice of a plan travel to another machine at all.

Python for the surface, because that is where the work happens and where torch
is. The line between the two is drawn by **shape and not by language**: Rust
keeps what is pure, deterministic and hashable; the loop stays in Python. No
callback crosses — the original's `TrialExecutor` had one implementor and it
was a closure wrapper. So nothing is asked of a `Trainer`: a pruner answers,
and the loop stops calling.
