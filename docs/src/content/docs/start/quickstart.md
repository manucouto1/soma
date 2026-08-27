---
title: Quickstart
description: From an empty file to a graph that runs across machines, keeps what is worth keeping, and tells you what it did.
---

Everything below is runnable top to bottom.

## Install

```bash
pip install somatize          # the graph, the engine, the store, the record
pip install 'somatize[viz]'   # + plotly, which every figure needs
```

The Rust is compiled inside the wheel — there is no toolchain to install and no
cargo feature to pick. One command needs building separately, and only notebook
13 uses it: see [the notebooks](/soma/start/notebooks/).

## 1. A node

A node is **one thing**. It has one method, `forward(input, ctx)`: it takes
what arrived along its incoming edges and returns what it produced. There is no
`fit`, no filter type and no step type — whatever a node needs to answer, a
retry, a model, three rounds of something, happens **inside it**.

```python
from somatize import Graph, Node


class Tokenize(Node):
    """Words to numbers. No torch here on purpose: a node is a function."""

    def forward(self, text, ctx):
        return [float(len(word)) for word in text.split()]


class Embed(Node):
    def __init__(self, scale):
        self.scale = scale

    def forward(self, counts, ctx):
        return [n * self.scale for n in counts]


class Score(Node):
    def forward(self, values, ctx):
        return sum(values) / len(values)
```

`ctx` is the channel: whoever executes the graph hands a node what it knows
through it, so something that wants a value injected puts it there and **no
node signature changes**.

## 2. A graph

`>>` chains and `|` forks. This is the normal way to write one, in both
languages — `Graph.somatize(a >> (b | c) >> d)` in Python and
`(a >> (b | c) >> d).somatize()` in Rust:

```python
g = Graph.somatize(
    Tokenize().named("tokenize")
    >> Embed(0.5).named("embed")
    >> Score().named("score")
)

g.forward("a graph declared and then run")   # 2.0
```

`node()` and `edge()` are still there for when the topology is built in a loop
or comes from outside:

```python
n = Graph()
for who in ("a", "b", "c", "d"):
    n.node(who, Tokenize())
n.edge("a", "c")
n.edge("a", "d")
n.edge("b", "d")
```

A node with several incoming edges is handed a **map** keyed by who sent what,
so an aggregator is an ordinary node and not a type of its own.

## 3. Where it runs, and what is kept

Four suffixes, and none of them changes a `forward`:

```python
spread = Graph.somatize(
    Tokenize().named("tokenize").at("worker1").mapped()
    >> Embed(0.5).named("embed").on("cuda:0").cached().frozen()
    >> (Score().named("strict") | Score().named("loose").at("worker2"))
)

spread.hosts()       # {'tokenize': 'worker1', 'loose': 'worker2'}
spread.devices()     # {'embed': 'cuda:0'}
spread.cached()      # {'embed': None}   ← the value None is the salt
spread.frozen()      # {'embed': None}
spread.mapped_nodes()
```

They are four different facts and mixing them up is the easy mistake. The graph
says **what** exists, `.at()` says **where**, `.on()` says on which device,
`.cached()` says **what is remembered**, and the plan says **when** — which you
can read before running anything:

```python
print(spread.plan())
```

`.frozen()` matters more than it looks: it is what puts a node's *version* in
the key of what it produced, so two datasets that share a name do not share an
answer.

## 4. Watch it

`watching=` takes anything callable. Every fact the engine emits goes through
it, synchronously, including the facts that come back from other machines —
they return down the connection that was already open.

```python
g.forward("the quick brown fox", watching=print)
# {'fact': 'ran', 'node': 'tokenize', ...}
# {'fact': 'ran', 'node': 'embed', ...}
# {'fact': 'ran', 'node': 'score', ...}
# {'fact': 'finished', ...}
```

A `Recorder` is the implementation that writes them to a store, and a level
above the engine says its own things through the same door — the core never
learns what a loss is:

```python
import tempfile
from somatize import Recorder, Store

store = Store(tempfile.mkdtemp())
recorder = Recorder(store, run="tuesday", summarising=["loss"])

for step in range(40):
    g.forward("the quick brown fox jumps", watching=recorder)
    recorder({"fact": "loss", "value": 2.0 * 0.93**step + 0.1})
```

## 5. Read it back

`somatize.record` is functions over a store, and it is a **price list**:
`runs`, `forwards` and `curve` are one scan, `facts` is one fetch, and `nodes`
is a fetch per forward. So everything a progress view asks for is free and the
per-node breakdown is asked once, not once a step.

```python
from somatize.record import curve, curve_costs, facts, forwards, nodes, runs

runs(store)                          # [{'run': 'tuesday', 'forwards': 40, 'broke': 0, ...}]
curve(store, run="tuesday")          # [(0, 2.1), (1, 1.96), (2, 1.83), ...]
curve_costs(store, run="tuesday")    # whether that scanned or fetched
nodes(store, run="tuesday")          # per node: ran, recalled, failed, took_us, host
```

`curve_costs` exists because a reader that is quietly a thousand times slower
is worse than one that says so.

## 6. Draw it

Two figures, and they share one table of colours with everything else this
library draws:

```python
spread                 # in a notebook: the graph draws itself, having never run
spread.figure()        # the same, explicitly

from somatize.record import Live, progress, spent

progress(store, run="tuesday")   # what happened, from the store
spent(store, run="tuesday")      # where the time went
```

`Live` is handed facts as they happen and fills the **same drawing function**
as `progress` — they can, because a fact read back is the very dict a watcher
was given. A live view and a report written twice are two things that slowly
stop agreeing.

## Where to go next

- **[The notebooks](/soma/start/notebooks/)** — thirteen of them, shipped
  executed with their outputs, so opening one shows what it does.
- **[The plan: what runs when](/soma/running/the-plan/)** — waves, and what
  leaves the machine.
- **[Across machines](/soma/running/across-machines/)** — `.at()`, a broker,
  and a worker that is your own code or a bare one.
- **[The health of a network](/soma/looking/health/)** — whether it is
  learning, taken from the record and not from a second training run.
