---
title: Overview
description: Five orthogonal facts, five holes the core provides and fills none of, and three levels none of which knows the one above exists.
---

There are three things to hold, and everything else follows from them.

## Five facts, and confusing them is the easy mistake

| | says | lives in |
|---|---|---|
| `Graph` | **what** exists | `soma-core/src/graph.rs` |
| `Catalog` | **who** executes it | `soma-core/src/catalog.rs` |
| `Placement` | **where** | `soma-core/src/placement.rs` |
| `Plan` | **when** | `soma-core/src/plan.rs` |
| `Memory` | **what is remembered** of each node | `soma-core/src/memory.rs` |

They are five types and not one struct with five fields, and each separation
was forced by something.

**`Graph` is topology only** — identities and edges, nothing about what a node
does. That is why it is *data*: it serialises, it compares, it gets sent
somewhere else. Creating a graph does not need to know what any node does, so
it does not know, and **that is the reason the core depends on nothing**.

**`Catalog` is the half that is not data.** An implementation does not
serialise and does not travel. What joins the two is the node id and nothing
else. When a subgraph goes to another machine, the graph travels and the
implementations do not — which is exactly why they had to be apart.

**`Placement` is two maps and not a pair**, because the two halves are obeyed
by different people. `distribute` reads the **host** when deciding the shape of
the plan; the node reads the **device** through `ctx.device` when executing. A
node can have either, both, or neither.

**`Plan` says when**, and it deliberately does not carry the device. A device
is inert for the traversal — it changes nothing about who waits for whom — and
crossing a wire is not, so crossing a wire is a named step and a device is not.

**`Memory` is four maps**, independent of one another: a node can be frozen
without being cached, named without being frozen, and any combination of the
rest. What it settles is the key:

```text
key(root) = H(content)                    ← the only place data is hashed
key(node) = H(identity, declaration, state, keys of its predecessors)
```

The **identity** is in there or two different nodes called `embed` collide in a
shared store. The **declaration** is in there because `Embed(512)` and
`Embed(64)` are one class and one identity — and, before that part existed,
one name, so the second run was handed the first one's answer with no error and
no warning.

## Five holes, provided and never filled

The core defines five traits. It implements none of them, and that is the whole
of its extension model — no plugin system, no registry, no configuration:

| trait | what somebody else supplies |
|---|---|
| `Node` | the work. Yours |
| `Transport` | carrying a slice of a plan somewhere else |
| `Keeper` | hashing a recipe, and keeping what it names |
| `Watcher` | being told what happened |
| `Codec` | writing down what only exists in one process |

A trait is only a trait here when the implementation comes from **someone
else**, and if two real implementors cannot be named today it is a struct
instead. `Codec` earned the promotion late: it came into the core when a third
tenant showed it was not the wire's.

There was a sixth. `Driver` served what a suspended node asked for, and after
eighteen use cases it had no consumer outside its own tests — its own docstring
said it existed to keep the agentic layer out of the core, which is a hole
justifying itself. It was deleted, along with the suspension it existed for: a
node is a function.

What it left behind is worth more than what it was. `Ctx` is the channel
whoever executes hands a node, so anything that wants a value injected puts it
there and **no node signature changes**. See
[the rules it is written under](/soma/model/philosophy/).

## Three levels, and none knows the one above exists

| level | scale | what it is |
|---|---|---|
| the graph | one `forward` | a network |
| the `Trainer` | an afternoon | a training run |
| N training runs | a campaign | **a Python list** |

```mermaid Downward is a call. Upward is only ever a return value — no level is handed a callback into the one above it.
flowchart TB
    L3["N training runs · a campaign<br/>a Python list. No type at all"]
    L2["Trainer · an afternoon<br/>one training run"]
    L1["Graph · one forward<br/>a network"]

    L3 -->|"calls"| L2
    L2 -->|"calls"| L1
    L1 -.->|"what it produced"| L2
    L2 -.->|"a loss, a verdict"| L3
```

The third one has no type, and that is on purpose. A graph earns its keep when
there are dependencies to declare; N runs have none, so `fedavg` is a function
and a federated round is a `for` loop. Making level 3 a graph would have been
the same mistake one level up.

The line between them is what decides where a thing lives. Micro-batches are
level 2 — the batch belongs to the caller, so `torch.chunk` reaches it — while
`.mapped()` is the engine's, because caching item by item has to **name** each
item, and a name comes from its content and not its place.

And nothing is asked of a level below. A pruner answers and the training loop
stops calling; the loop is never handed a callback to run. See
[the rules](/soma/model/philosophy/#rust-core-python-interface) for what that
cost the version before this one.

## Next

[The plan](/soma/running/the-plan/) is where the five facts turn into an
execution: what `compile` does, why it decomposes rather than flattens, and
what `distribute` adds afterwards.
