---
title: Across machines
description: A suffix on the declaration, a broker that says where a host is, and a wire that carries a whole slice — with no coordinator anywhere.
---

```python
Tokenize().named("tokenize").at("worker1") >> Embed().named("embed")
```

`.at()` is the same kind of thing as `.on()` and `.cached()`: a suffix on the
declaration that changes no `forward`. It names a **host**, not an address —
the same graph spreads over two processes here or two machines there without
touching what is written.

What turns a name into a wire is a `Broker`:

```python
import sys
from somatize import Broker, Worker

broker = Broker.embedded({
    "w1": Worker.spawn([sys.executable, "-m", "somatize.worker"], mode="network")
})

g.forward(x, broker=broker)
```

## Three things, and each one is a separate decision

**`.at()` is the graph's.** It is topology-adjacent and it travels with the
graph. It says `worker1`, and `worker1` is a word only this graph knows.

**A `Worker` is a declaration.** An address or a command, plus how to pack for
it. **It opens nothing.** A graph that names a host on a run that never reaches
it costs nothing for naming it.

**A `Broker` resolves the name**, and the wire is opened the first time
somebody actually sends work.

There is no registry, no heartbeat and no coordinator. A machine does work here
in exactly two ways — a worker serving slices to the client that is talking to
it right now, and a machine claiming trials from a shared folder, which is
watched by *is it still writing*. There is no third case, so there is nothing
to keep alive.

## Control and cargo go by different routes

That is the one decision worth knowing before anything else. What crosses to a
broker is a **rendezvous**: tens of bytes, once per host per session. What
crosses the wire next door is an **activation**. The broker is in the first and
steps out of the second — which is why an embedded broker can be a thread
exchanging real serialised messages and still cost nothing anyone can measure.

The broker answers with a `Path`, and there are four:

| | |
|---|---|
| `InProcess` | both ends are this process. **Never inferred** — whoever registers a host says so, because a broker that worked it out by comparing addresses would quietly undo the reason a worker is a separate process, which is the GIL |
| `Mount` | both ends see the same filesystem; a path is written and read. Free, and a cluster already has one |
| `Direct` | the two ends reach each other and speak, over a socket or over a child's pipes. One crossing, lowest latency, broker gone |
| `Relayed` | neither can reach the other, so the bytes stream through the broker. No disk, no durability, never more than a window in flight |

What the table cannot show is the part that matters: **where the bytes
actually go**. The broker is in the first exchange and out of the second —
except in the one case where nothing else is left.

```mermaid Where the bytes go. The broker is asked once per host per session; after that only Relayed keeps it in the path.
flowchart TB
    subgraph ask["Rendezvous · tens of bytes, once per host per session"]
        direction LR
        C0["client"] -. "where is w1?" .-> B0(["broker"])
        B0 -. "a Path" .-> C0
    end

    subgraph one["InProcess · nothing is transferred"]
        direction LR
        C1["client and worker, one process"]
    end

    subgraph two["Mount · both ends see the same filesystem"]
        direction LR
        C2["client"] ==> |written| D2[("a shared directory")]
        D2 ==> |read| W2["worker"]
    end

    subgraph three["Direct · lowest latency, broker gone"]
        direction LR
        C3["client"] ==> |"a socket, or a child's pipes"| W3["worker"]
    end

    subgraph four["Relayed · neither end can reach the other"]
        direction LR
        C4["client"] ==> B4(["broker"])
        B4 ==> W4["worker"]
    end

    ask ~~~ one ~~~ two ~~~ three ~~~ four
```

`InProcess` still runs the slice where it was placed — it is the `.at()` that
never left home and pays for a trip anyway. And `Relayed` is the only one that
keeps the broker in the path, which is why it is the last resort rather than
the default: no disk, no durability, and never more than a window in flight.

A client always does the same thing: it talks to a broker. The only thing that
changes between having a platform account, having a head node and having
neither is **which** broker, which is a URL. There is no second code path and
no degraded mode. `Broker.embedded` is the deployment that exists today, and it
is what makes this work with nothing else installed.

## What crosses the wire

`Plan::Remote` is a **whole plan**, not a step — see
[the plan](/soma/running/the-plan/). So a chain of five nodes that all live on
`w1` is one message, not five.

The conversation is a `Request` and an `Answer`, in MessagePack:

```rust
enum Request {
    Hello { runtime: String, offering: Option<Label> },
    Provision { bytes: Vec<u8> },
    Work { plan, input, known, keys, placement, memory },
}
```

`Work` carries `keys` — what each already-computed value is called — so what
runs over there **can name what it produces** and the chain of keys does not
stop at the wire. That is what lets a remote node's answer land in a cache the
client can hit next time.

```rust
enum Answer {
    Ready,
    Send,
    Refused(String),
    Done(Outcome),
    Failed(String),
    Saw(Fact),      // ← the only non-terminal one
}
```

`Saw` is how a run on another machine keeps telling you what it is doing.
`dispatch` was blocked reading that connection anyway, so reading one answer
became reading until one is terminal: **no second connection, no port, no bus.**
The rule it sets down is *where a connection is open, facts come back down it;
where there is none, they go to the store and whoever wants them scans.*

A relay attributes nothing. The client wraps what arrives, because the host's
**name** is the graph's and a worker has never heard of it.

## Two ways to pack, and they are not interchangeable

```python
Worker.at("10.0.0.4:7000", mode="project")
Worker.spawn(argv, mode="network")
```

**`project`** means the worker can already import your code, so a node travels
as a **name**. That is the normal case for a cluster where everyone has the
repo.

**`network`** means it cannot, so the node travels as **bytes** — a
`cloudpickle` artifact. This is what a node defined in a notebook cell needs:
nothing over there could possibly import it.

`send=` names the modules that must travel **inside** the artifact rather than
by reference. cloudpickle serialises by reference anything that comes from an
importable module, which leaves out exactly the case a generic worker is for:
your nodes in `my_package/net.py`, and a worker that cannot import
`my_package`.

Packing is the one part that could not be in Rust: a pickle artifact is made by
cloudpickle, which is Python's, and **the wire deliberately does not look at
what it carries**. `Provision` is a hole in the wire, filled from `python/`: it
turns an artifact into a catalog.

## An `Opaque` crossing

What only exists in one process — a torch tensor with its autograd graph —
crosses with a `Codec` in front of it. That is the fifth hole. A value nobody
registered a codec for is the one that does not cross, and the message says so
rather than arriving mangled.

## The worker itself

```bash
python -m somatize.worker --listen 0.0.0.0:7000
```

It is a binary in the Python package rather than a crate, because what it needs
is an interpreter. It is also deliberately **stupid**: it holds no state you
care about and knows nothing about your graph. You always have the code; what
does not fit on one machine is the state, not the code.

Two things it says about itself, in a vocabulary of its own, crossing as
`Fact::Said { kind, pairs }` — a carrier and not a vocabulary, so the core
never learns what a load average is. And `--reporting SECONDS` makes an **idle**
worker write to the store on a clock, one name per machine, rewritten: an idle
worker's connection is one nobody is reading, so there is nowhere for a fact to
come back down.

## Reading it back

```python
from somatize.record import fleet, machines

fleet(store, run="cheap")     # per machine: slices, took_us, waiting_us
machines(store, run="cheap")  # the same, drawn
```

The column that earns that view is **`waiting_us`** — the round trip *minus*
what actually ran over there, which is the wire and the queue. Neither half of
that subtraction belongs to a node, which is why no per-node view can produce
it and why it is the answer to *was sending it worth it*.

See [a fleet](/soma/tutorials/09-a-fleet/) for what that looks like on a real
run.
