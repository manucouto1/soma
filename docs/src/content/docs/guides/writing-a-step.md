---
title: Writing a Step
description: The poll contract, the five transitions, dynamic fan-out, suspend and resume, and running a pipeline from inside a step.
---

A step is the effectful counterpart to a filter: it calls models, runs
tools, decides what happens next, and may pause for a person. In Python a
step is **any object with `poll(ctx)`**, duck-typed exactly the way a
filter's `forward` is — no base class, no registration ceremony.

```python
from soma.agentic import Done, Await, Llm

class AskOnce:
    _cache_version = "1"

    def poll(self, ctx):
        if ctx.turn == 0:
            return Await(Llm("ollama/qwen2.5", prompt=ctx.input))
        return Done(ctx.result()["text"])

g = soma.Graph()
g.node("ask", AskOnce())
g.forward("why is the sky blue?")
```

## The poll contract

`poll` advances one turn: it looks at where it is, and returns a
*transition* saying what should happen next. It is called with
`ctx.turn == 0` and no results to start; thereafter with the results of
whatever it last asked for. The runtime performs the effects — `poll`
itself never touches the network.

The one rule that everything else rests on: **`poll` must be
deterministic given the same context.** The journal records each effect's
result once; on resume it re-polls the step from turn 0 and serves the
recorded results, so a deterministic step retakes the identical path and
ends up exactly where it was. Put anything nondeterministic — a model, a
tool, a clock — in an effect, where it gets recorded.

The corollary: a step keeps **no state of its own**. Anything it
accumulates — a conversation, a running total — it rebuilds from
`ctx.history`, because replay feeds back identical results and a field on
`self` would drift from them.

## What `ctx` carries

| Field | Type | Meaning |
|---|---|---|
| `node_id` | `str` | This node's id in the graph |
| `run_id` | `str` | The run this belongs to |
| `input` | `Any` | Input resolved from predecessors |
| `turn` | `int` | Which turn this is, counting from 0 |
| `results` | `list[dict]` | Results of last turn's effects, in request order; empty on turn 0 |
| `history` | `list[list[dict]]` | Every turn's results, oldest first |
| `result()` | `dict \| None` | Convenience: the single result of a one-effect turn |

Nothing else, on purpose.

## The five transitions

Transitions are plain dicts built by helpers from `soma.agentic`, so what
crosses into Rust is data rather than a class hierarchy:

| Helper | Dict shape | Meaning |
|---|---|---|
| `Done(value=None)` | `{"transition": "done", "value": ...}` | Finished, with this output |
| `Await(*effects)` | `{"transition": "await", "effects": [...]}` | Perform these effects **concurrently**; poll again with the results in order |
| `Spawn(specs, join="all")` | `{"transition": "spawn", "specs": [...], "join": ...}` | Create and run these nodes now; poll again with their outputs |
| `Goto(target, carry=None)` | `{"transition": "goto", "target": ..., "carry": ...}` | Hand control to another node; this step is done |
| `Suspend(reason="waiting")` | `{"transition": "suspend", "reason": ...}` | Stop and persist the run |

Effects an `Await` can carry: `Llm(model, prompt, system=None)`,
`ToolCall(name, args=None)`, `RunGraph(graph, input=None, mode="forward")`,
`Sleep(seconds)`, `Custom(kind, payload=None)`. Each result arrives as a
dict with a `"kind"` key — `"llm"` carries `text`; `"tool"`, `"graph"`,
`"node"` and `"custom"` carry `output`; `"failed"` carries `message`; all
carry `is_error`. A failed effect is handed to the step rather than
raised, so it can retry, fall back, or give up deliberately.

## `register_step` vs `node`, and `handoff`

Two ways to put a step in a graph, and they are not interchangeable:

- **`g.node(id, step)`** adds a node. It has edges, receives input from
  its predecessors, and runs when the graph reaches it.
- **`g.register_step(id, step)`** registers a **spawn target without
  adding a node**. A node with no incoming edges is a root, and a root
  also runs once on the graph's own input — which is exactly what you do
  not want for a worker that only exists to be spawned.

`Goto` needs a declared control edge: `g.handoff(a, b)` says node `a` may
hand control to node `b`. A `Goto` naming a target the graph never
declared is an error, not a silent jump.

## Dynamic fan-out: `Spawn`

The width of a fan-out is often a property of the data — a plan with two
tasks wants two workers, one with nine wants nine — and that is the one
thing a static topology cannot say:

```python
from soma.agentic import Done, Spawn, Run

class Fanout:
    _cache_version = "1"

    def poll(self, ctx):
        if ctx.turn == 0:
            return Spawn([Run("worker", task, label=f"w{i}")
                          for i, task in enumerate(ctx.input)])
        return Done([r["output"] for r in ctx.results])

g.node("fanout", Fanout())
g.register_step("worker", Worker())   # spawnable, and not a root
```

`Run(runs, input, label=None)` names a registered step (or filter) and
what to feed it. The `join` policy decides how the children recombine:

- `"all"` (default) — wait for all; a failure fails the join.
- `"all_settled"` — wait for all, keep whatever succeeded; failures
  arrive as `{"kind": "failed"}` results the step can read.
- `"first"` — take the first success and cancel the rest.

`soma.agentic.orchestrate(planner, worker, synthesizer)` is this pattern
packaged: the pool is sized from the plan at runtime.

## Suspend and resume

`Suspend` stops the run and persists it — a pause, not a failure. From
`fit`/`forward` it surfaces as the `soma.SomaSuspended` exception, which
carries `run_id`, `node_id`, `turn`, `kind` and `reason`: exactly the
arguments `Graph.resume(...)` takes.

```python
try:
    g.forward(doc, run_id="review-1")
except soma.SomaSuspended as s:
    answer = get_human_answer(s.reason)                     # later, elsewhere
    g.resume(s.run_id, s.node_id, s.turn, s.reason, answer)
    result = g.forward(doc, run_id="review-1")
```

There is no separate checkpoint format: the journal *is* the checkpoint.
`resume` records the answer at the exact site the step suspended; the
next run under the same `run_id` re-polls from the start, every prior
effect is served from the journal, and the suspension point now has its
answer in `ctx.results`. This survives a process exit — the journal lives
in the persistent cache.

## Running a pipeline from a step

`RunGraph` makes a computational pipeline a tool for a step — it runs
through the ordinary compiler and executor, with the ordinary cache:

```python
from soma.agentic import Await, Done, RunGraph

class Planner:
    _cache_version = "1"

    def __init__(self, sub):
        self._sub = sub            # underscored — see the trap below

    def poll(self, ctx):
        if ctx.turn == 0:
            return Await(RunGraph(self._sub, input=ctx.input))
        result = ctx.result()
        if result["kind"] == "failed":
            return Done("could not run it: " + result["message"])
        return Done(result["output"])

g.node("planner", Planner(sub))
g.register_graph(sub)              # make sub's implementations runnable here
```

The effect carries the sub-graph's *structure*; `g.register_graph(sub)`
merges its node implementations into the outer graph's catalog. The same
id behind a different implementation is a `ValueError` — whichever one
lost would answer for the other's cache entries. `mode="fit"` fits the
sub-graph instead; a pipeline that fails comes back as
`{"kind": "failed", "message": ...}`, because an unfittable configuration
is a finding, not a crash. Sub-graphs may themselves contain steps —
agent → pipeline → agent — capped at nesting depth 8.

## Declaring schemas

Optional class attributes make the compiler's edge check fire from
Python — a mismatch raises `SomaSchemaMismatch` at `compile()`, not
mid-run:

```python
class Summarize:
    _cache_version = "1"
    _input_schema = "text"          # "text" | "json" | "messages" | "bytes"
    _output_schema = {"dtype": "json", "shape": []}   # or the full mapping
    def poll(self, ctx): ...
```

## The trap: underscore what is not config

A step's public attributes are its configuration and enter its cache
identity, exactly as a filter's do. A live `soma.Graph` — or anything
else unhashable — **cannot** be part of that identity, so storing one
under a public name (`self.sub = sub`) fails identity hashing. Store it
underscored (`self._sub`): underscored attributes never enter the key,
and nothing is lost — the journal keys a `RunGraph` effect by the graph's
own content anyway. If the object genuinely *is* configuration, give the
step a `_cache_version` that you bump when it changes.

## Checklist

- `poll(ctx)` returns one of the five transitions, always.
- Deterministic given the same context; nondeterminism goes in effects.
- No state on `self`; rebuild from `ctx.history`.
- `_cache_version` set (mandatory in notebooks, wise everywhere).
- Spawn targets registered with `register_step`, not `node`.
- `Goto` targets declared with `handoff`.
- Live graphs and other unhashable attributes underscored.
