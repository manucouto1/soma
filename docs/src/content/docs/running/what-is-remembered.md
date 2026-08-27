---
title: What is remembered
description: The key a node's answer gets, the one rule about when it may be kept, and the fingerprint that is deliberately not in the key.
---

Two suffixes settle it, and neither changes a `forward`:

```python
embed.cached()    # the answer is worth keeping
embed.frozen()    # this node's state does not change while the graph runs
```

They are two of the four maps in `Memory`, the fifth of
[the five facts](/soma/model/overview/), and they are independent: a node can
be frozen without being cached, or named without being frozen.

## The key

```text
key(root) = H(content)
key(node) = H(identity, declaration, state, keys of its predecessors)
```

The root is **the only place data is hashed**. Everything below is named from
names, which is what makes asking cost nothing.

Each part is in there because leaving it out broke something:

**Identity** — what implements the node. Without it, two different nodes both
called `embed` collide in a shared store.

**Declaration** — what it was built with. `Embed(512)` and `Embed(64)` are one
class, one identity and, before this part existed, **one name**: the second run
was handed the first one's answer with no error and no warning. See
[a node is one thing](/soma/model/a-node/) for why it is captured at `__init__`
and never read off the object.

**State** — the digest of what the node is settled at, when it has one.

**The keys of its predecessors**, and not the predecessors themselves. That is
what makes an upstream node which recomputes to the same value stop the
invalidation right there.

## The fingerprint of the code is deliberately not in it

Editing a `forward` renames nothing. That is a decision, not an oversight: a
cosmetic refactor would otherwise invalidate half the store in silence.

So the fingerprint is kept **beside** the value and compared on a hit. The line
it draws is what the caller **said** against how it is **written** — and only
the first can be pinned down identically in every process, which a key computed
on a client and again on a worker has to be.

What it costs is that a real change to a `forward` renames nothing either, and
that is paid for one layer up rather than undone. `somatize.foreseen` looks at
the fingerprint where it is an **opinion and not an invalidation**: `STALE`
says *you should have bumped the salt*, and `SUSPECT` is everything under it,
which goes on being fed the old code's answer whatever became of its own name.
See [what an edit did](/soma/tutorials/11-what-an-edit-did/).

`.cached("v2")` is the salt, and it is how you say so by hand.

## One rule about when something may be kept

> A node's output can be kept if nothing upstream of it can change — itself
> included.

`cacheable(graph, memory)` checks it, and it is a free function rather than a
method for the same reason `compile` is one: it needs the graph **and** the
table.

Freezing the node alone is not enough, and the reason is a bug that is silent
in exactly the worst way: what is restored from a store is a **leaf**, so the
backward pass stops there and everything above it quietly stops training.

The same rule falls out again without mentioning gradients at all. The digest
of the state is in the key, so a node that keeps changing never hits and only
fills the store. Being named is checked in the same walk, since a chain with a
hole in it delivers no key below it.

The two ways it refuses:

| | |
|---|---|
| `Unsettled` | something upstream of a cached node is not frozen |
| `Nameless` | something upstream has no identity, so no key reaches down |

### And a third way, which does not refuse

Both of those raise, and stop you. There is one that does not: **an `Opaque`
whose type has no `Codec` cannot be written down**, so the value is computed,
not kept, and computed again next time. The run stays green. What you get is a
line on `stderr` that interrupts nothing:

```
what `enc` produced could not be kept: ValueError: a `dict` cannot leave this
process: nothing says how to write one down. Register it with
`codec("a name", dict, dump=..., load=...)`
```

Measured on a node that sleeps 0.4 s and returns `Opaque({...})`: the second
`forward` cost **0.40 s** without a codec and **0.00 s** with one. At real
scale — a sentence encoder in front of a training loop — that is every epoch
paying for a prefix that was supposed to be paid for once.

It does not refuse because it cannot: whether a value is writable is not a fact
about the graph, and `cacheable` runs before anything has produced one. So the
rule to carry is that `.cached()` on a node returning an `Opaque` is **half a
declaration** — the other half is the codec, and `somatize.torch` registers the
one for a tensor on being imported, which is why this never comes up until you
return something of your own.

## Frozen is information, and somebody else obeys

The core holds *this node's state does not change while the graph runs* as
inert information and reasons over it. It cannot make it true.
`somatize.torch` is what makes it true, with `requires_grad_(False)`.

This is the same shape as the device: the plan carries where a node was said to
run, and the node is the one that moves anything.

## The pre-pass, and what is not run at all

`key_for` promises the name a node's output will have **before it has one**, so
the engine can name the whole plan with nothing executed, ask the `Keeper`
whether those names are present — a scan, no fetches — and work **backwards
from the leaves**: a node whose answer is kept does not need its inputs.

A slice nobody needs is **not sent**, and `Fact::Spared` says so out loud,
because a node missing from a record cannot otherwise be told from one that was
never in the graph. It gives up towards keeping a node in the two cases it
cannot foresee: a `.mapped()` node, and one with no key.

The regression that caught, and only a notebook caught it: the pre-pass named
the root and the walk named it again, so the batch was hashed twice and asking
early cost exactly what it saves.

## Why a source is handed a coordinate

Naming a 19 MB batch costs **121 ms on every step**, hit or miss, because a
cache has to look at all of a value to name it. A span costs **0.027 ms**.
There is no faster hash to reach for, so the answer is not to weigh the batch —
it is to hand the graph a **coordinate** instead of one.

The other half of the name is free: a name in a `Store` resolves to a digest,
and the digest **is** the hash of the content, so a source states its version
with one lookup and no bytes. See
[where the data comes from](/soma/running/where-the-data-comes-from/).
