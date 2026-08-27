---
title: A node is one thing
description: One method, one shape, and the two types that used to exist beside it — with what crosses an edge and why a bool does not.
---

```python
from somatize import Node


class Tokenize(Node):
    def forward(self, text, ctx):
        return [float(len(word)) for word in text.split()]
```

That is the whole contract. In Rust it is a trait with one method:

```rust
pub trait Node: Send + Sync {
    fn forward(&self, input: &Value, ctx: &Ctx<'_>) -> Result<Value, NodeError>;
}
```

`input` is what arrived along the edges. It **runs to the end**: whatever it
takes happens inside, and the engine neither counts nor bounds it. A node that
needs something from the world — a model, a tool, an index, three rounds of
something, a retry — **calls it**, holding whatever client that takes.

## There used to be more, and there is a lesson in each

**There was a `Filter` and a `Step`.** One was deterministic and memoised by
content; the other was effectful and journalled instead. Two node types, two
sets of rules, and a compiler that had to know which was which. They are one
type now, and nothing was lost: a node that calls a model is a node that calls
a model, and whether its answer is worth keeping is said by `.cached()` on the
declaration rather than by its class.

**There was a two-variant return value**, so a node could say *I produced this*
or *I need something first*. It turned out not to be needed either.

**There was a `Driver`**, which served what a suspended node asked for, and a
`Transition` and an `Await` to suspend with. All three went after eighteen use
cases, because `Driver` had no consumer outside its own tests.

What survives that deletion is the useful half: `Ctx`.

## `Ctx` is a channel, and that is why it is a type

```rust
pub struct Ctx<'a> {
    pub device: Option<&'a Device>,
}
```

One field today. It is a struct rather than an argument because **adding to it
is additive**: something that wants a value injected — an agentic layer, a
tracer, a budget — puts it there, and every node ever written keeps its
signature. That is the channel `Driver` left behind, and it is worth more than
`Driver` was.

Note what `device` is: **information**, not an instruction. The core cannot
move anything to a GPU. The one that obeys is the node.

## What crosses an edge

A closed set of seven shapes, because the engine is in Rust and the data has to
have a shape Rust understands:

| | |
|---|---|
| `Null` | what a root node gets when you pass no input |
| `Number` | an `f64` |
| `Text` | |
| `Bytes` | |
| `List` | |
| `Map` | what a node with several predecessors is handed |
| `Opaque` | something the core **does not look at** |

There is no `Json` — it would pull in `serde_json` for a core that depends on
nothing — and no shaped `Tensor`, because nobody produces one.

`Opaque` is of another nature. Some values cannot be converted without being
destroyed: a torch tensor mid-autograd-graph, round-tripped through numbers,
comes back without its `grad_fn`. So it travels as itself, and when it has to
cross a wire a `Codec` writes it down — which is the fifth hole, and `Ipc` in
`soma-data` is its second implementor. Filling it for a type of your own is
[four lines](/soma/running/writing-a-codec/).

Everything heavy is behind an `Arc`, because a value is cloned on every edge
and cloning must not copy.

**An `Opaque` is only visible from outside the graph.** You return one, and the
next node down is handed **the value**, unwrapped — whether it has one
predecessor or five, because the engine opens it on the way in. So `.value`
inside a `forward` is an `AttributeError`, and `.value` is right when you call
`node.forward(x, ctx)` yourself, which is what a unit test of one node does:

```python
class Encode(Node):
    def forward(self, x, ctx):
        return Opaque(self.net(x))       # wrapped on the way out

class Next(Node):
    def forward(self, x, ctx):
        return self.net(x)               # x is the tensor, not the Opaque

Encode().forward(batch, None).value      # …and outside, it is the Opaque
```

Which is worth knowing before you write your first aggregator: it is the same
rule, and reaching for `.value` on what arrived is the mistake it invites.

**A `bool` is not one of them**, and the refusal is deliberate:

```python
def forward(self, values, ctx):
    # `1.0` and not `True`: what crosses an edge is a closed set of shapes,
    # and a `bool` is not one of them. Turning it into `1.0` behind your back
    # is the library deciding what you meant.
    return 1.0 if sum(values) / len(values) > self.threshold else 0.0
```

## A fan-in is a map, so an aggregator is an ordinary node

A node with several incoming edges is handed a `Map` keyed by who sent what.
There is no aggregator type and no reducer interface:

```python
class Vote(Node):
    """An aggregator is a node that reads a map. There is no type behind it."""

    def forward(self, said, ctx):
        return sum(said.values()) / len(said)
```

The same holds in the other direction: a node with three successors is read by
three of them, and none of that needs a variant in the [plan](/soma/running/the-plan/).

## What a node was built with

`Node.__init_subclass__` remembers each instance's **declaration** — the
arguments it was constructed with, bound against the signature so that
`Layer(64, 32)` and `Layer(in_=64, out=32)` are one declaration. It is half of
the node's key.

It is captured at `__init__` and never read off the object afterwards, and that
distinction cost a run to learn. Reading `vars(obj)` put a call counter in a
key and the encoder ran three times instead of once: **what a node holds is not
what it was built with**. A node that counts, caches a client or moves a tensor
has attributes that move *while the graph runs*.

Two ways of getting it wrong, and they are not symmetric. *Unstable* — one
declaration, two texts, which is what a memory address does — costs a cache
that misses forever. *Lossy* — two declarations, one text, which is what
scrubbing an address to `<Helper>` does — costs the wrong value in silence.
Neither is accepted, which is why scrubbing is not the answer: a declaration
that can be neither raises `CannotDeclare`, and the graph is refused before the
first node, naming the attribute.

## Naming one

```python
Tokenize().named("tokenize")
```

Without it a node gets the `snake_case` of its class. The id is what joins a
node to everything else — the catalog looks it up, the placement keys on it,
the record files under it — so two nodes of the same class in one graph need
two names.

## Next

[The plan](/soma/running/the-plan/) is how these get walked, and
[what is remembered](/soma/running/what-is-remembered/) is what `.cached()`
and `.frozen()` settle.
