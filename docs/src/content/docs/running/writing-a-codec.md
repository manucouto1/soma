---
title: Writing a codec
description: Four lines that say how your type is written down — where to register them, why the kind is named after the type, and the one failure here that does not raise.
---

```python
from somatize import codec

codec("myproject.Reading", Reading, dump=..., load=...)
```

`Codec` is the fifth [hole](/soma/model/overview/), and it is the one that came
into the core last: a third tenant showed it was not the wire's. Everything else
about a value is the engine's business — when it runs, where it runs, whether it
is worth keeping. **What it weighs written down is nobody's but yours.**

## What it buys, and what it moves

An `Opaque` is the wrapper for something that only exists in this process: a
torch tensor with its autograd graph, a tokenizer, a connection. Without a codec
it cannot go anywhere — not across a wire, not into a store. With one, the
frontier does not disappear, it **moves**: from *an opaque does not travel* to
*an opaque nobody registered a codec for does not travel*.

Which is a much better place for it to be, because the second one is a sentence
you can act on.

## The four lines

```python
import json
from dataclasses import dataclass, asdict

from somatize import codec

@dataclass
class Reading:
    hz: float
    samples: list[float]

codec(
    "myproject.Reading",
    Reading,
    dump=lambda one: json.dumps(asdict(one)).encode(),
    load=lambda raw: Reading(**json.loads(raw)),
)
```

`dump(obj) -> bytes` and `load(bytes) -> obj`, and that is the whole contract.
There is no base class to inherit from and no registry object to hold: a codec
is two functions and a name, so the type it is about does not have to know this
library exists.

## Where you register it is the point

**A codec is registered by importing whoever calls `codec(...)`.** That is not a
convention, it is the mechanism — `somatize.torch` registers the one for a
tensor on being imported, and there is no other way in.

So put the call in the module that **defines the type**. Then whoever can build
one can also write one down, and the two never come apart:

```python
# myproject/reading.py
@dataclass
class Reading: ...

codec("myproject.Reading", Reading, dump=..., load=...)
```

This matters most where you cannot see it. A [worker](/soma/running/across-machines/)
starts **empty** — it does not know what your nodes are, and it may never
mention `Reading` itself while one goes past. With `project` provisioning it
builds its catalog out of its own clone, so importing your node imports your
type, which registers your codec. Nothing had to be shipped and nothing had to
be configured.

What does *not* happen for you: this library summons its own codecs by name when
bytes arrive written by one it has not imported yet, and **only its own** — it
does not know what your `kind` means, so it cannot guess which module to import.
Yours is yours to import, in every process that reads it.

## The `kind` is named after the type, never after the run

```python
codec("myproject.Reading", ...)   # yes
codec("readings-v2-final", ...)   # no
```

The `kind` is what gets written beside the bytes, so it is what a store keeps
**forever**. It is the only word somebody has when they open that record by hand
in two years, and it is also what makes the second chance above work at all:
when an object is in hand, its type is the name; when bytes arrive, the `kind`
is the only name. That those two meet is not luck.

## What `load` gives back is a leaf

Whatever graph produced the value is gone. A tensor read out of a store is a
tensor and not a node in a backward pass, whatever `dump` did about
`requires_grad` — so carrying it across preserves nothing and costs something.

That is the whole reason a cached prefix has to be settled:

```python
embed.cached().frozen()
```

`.frozen()` is not paperwork next to `.cached()`. It is the promise that makes
handing back a leaf sound, and [`Unsettled`](/soma/running/what-is-remembered/)
is what you get when it is missing — checked before anything runs, rather than
discovered as a network that quietly stopped training.

Two more things that read as style and are not:

- **Read back where it will be used, not where it was written.** The torch codec
  loads `map_location="cpu"`, because a store is shared between machines and one
  that only reads back on the box that wrote it is not shared at all. Whoever
  receives it moves it, which is what a placed node already does with its input.
- **Do not trust the bytes.** The torch codec passes `weights_only=True`.
  Unpickling arbitrary objects out of a store several machines can write to
  turns a cache into a way in.

## The failure that does not raise

This is why the page exists. Everything else about caching stops you where you
made the mistake — a node that is cached but not frozen raises `Unsettled`, one
with no identity raises `Nameless`, and both do it **before** anything runs.

A missing codec does neither. The value is computed, cannot be written down, and
is computed again next time. The run is green and the answer is right; what you
lost is the cache, silently, and the only sign is one line on `stderr` that
interrupts nothing:

```text
what `sense` produced could not be kept: ValueError: a `Reading` cannot leave
process: nothing says how to write one down. Register it with
`codec("a name", Reading, dump=..., load=...)`, which is what `somatize.torch`
does for a tensor on being imported
```

It does not raise on purpose: refusing to run because something could not be
*kept* would turn an optimisation into a requirement, and a graph that ran
yesterday would stop running because somebody added `.cached()`. But a hint that
costs a whole cache deserves to be looked for, so when a `.cached()` node seems
never to hit, this is the first thing to check.

A codec that returns the wrong thing fails the same quiet way, and says so with
both types:

```text
the codec for `bad` has to return bytes, and it returned a `str`
```

## What is already registered

```python
from somatize import codecs_registered

codecs_registered()      # ['torch.Tensor']  — after `import somatize.torch`
```

Two ship with this library, and they arrive by different doors:

| | who registers it | where it lives |
|---|---|---|
| `torch.Tensor` | `import somatize.torch` | `soma-python/`, in Python |
| `arrow.RecordBatch` | nobody — it is not in the registry | `Ipc`, in `soma-data/` |

The second is the trait's other implementor and it comes from **another crate**,
which is what makes `Codec` a hole rather than a Python detail: a
[frame](/soma/running/where-the-data-comes-from/) did not come from Python and
does not need a registry, because what it weighs is Arrow IPC and `data/` says
so. It is also asked **before** the registry on the way back, so registering
something of your own under `arrow.RecordBatch` would be overriding what a frame
is rather than adding to it. Name yours after your type and the question never
comes up.
