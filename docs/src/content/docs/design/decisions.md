---
title: Architecture Decisions
description: Boundaries the code is expected to keep, why they were chosen, and what was deliberately not done.
---

## Why this page exists

A comment in `soma-core/src/graph/mod.rs` pointed at an `architecture-review.md`
that did not exist. This is that file, and it is a decision record rather
than a plan: each entry says what was chosen, what it was chosen *over*,
and what would have to change for the answer to be different.

Decisions live here because a boundary nobody wrote down is a boundary
that erodes on the next commit that finds it inconvenient.

## Errors: typed at the edges, shared at the seams

**Decided.** `soma-worker`, `soma-llm` and `soma-coordinator` define their
own error types with `thiserror` and convert to `SomaError` at the
boundary. `soma-core`, `soma-compiler` and `soma-runtime` keep sharing
`SomaError`.

The problem was not that one error type existed; it was that
`SomaError::Other(String)` had become half of all error construction in
the workspace. `soma-worker` builds 51 errors and *every one* is `Other`;
`soma-llm` 27, `soma-runtime` 24. So nothing downstream could tell a
transport failure from a Python failure from a bad plan — a caller that
wants to retry a dropped socket but not a broken pipeline has no way to
ask which it got.

Rejected: **per-crate enums everywhere**, with a `From` lattice. It is the
textbook answer and it is wrong here for the core three. `soma-core`,
`soma-compiler` and `soma-runtime` share one domain — a graph, a plan and
its execution — so an error crossing between them is not crossing a
domain boundary, and making `?` convert at every hop buys nothing but
plumbing.

Rejected: **enriching the shared enum** with `Transport`, `Venv`,
`Provider` variants. Cheapest, and it would make errors matchable — but it
puts worker and LLM-client concepts inside `soma-core`, which is the
boundary this page exists to restore.

The test of the decision: a variant belongs in `SomaError` if
`soma-core` could plausibly raise it. `Venv` and `WebSocket` cannot.

## `soma-core` holds contracts, not execution

**Decided.** Execution and I/O move out; types and pure rendering stay.

| Moves | To | Why |
|---|---|---|
| `impl StrategyExecutor for TrainingStrategy` and the aggregator impls (~90 lines) | `soma-runtime` | A distributed training loop that shards inputs and calls workers is execution |
| `store/s3.rs`, `store/zarr.rs` (1259 lines) | new `soma-store` | Each owns a `tokio::runtime::Runtime` and `block_on`s network I/O, so anything depending on `soma-core` inherits a runtime |
| `Study::save` / `Study::load` | `soma-runtime` | `std::fs` |

The line is not "no I/O in `soma-core`" — it is **no runtime, no network,
no optional heavy dependency**. What made the stores wrong was not that
they touched a disk; it was that depending on the contract crate handed
you a tokio runtime. So:

- **`LocalDataStore` stays**, `std::fs` and all. It is the reference
  implementation that makes `DataStore` usable out of the box, and it
  costs a caller nothing.
- **`TrainingStrategy` the type stays.** It is a graph-level attribute —
  part of what a graph *is* — so it is a contract. Only its `impl` is
  execution. The type and its behaviour split; that is the whole point.
- **`summary.rs`, `svg.rs`, `viz.rs` stay** (1106 lines). Pure
  `data → String`, no I/O, no runtime. Rendering a graph to text is
  serialization in the broad sense, and moving them would be churn in the
  name of purity: 24–29 external call sites, no problem being solved.

The check that the split worked is one command: `cargo tree -p
somatize-core` no longer mentions tokio.

Rejected: **document the drift and move nothing**. The stores are the
reason not to. `tokio` arriving transitively through a contract crate is
exactly the kind of coupling a contract crate exists to prevent.

## `NodeId` stays a `String`

**Decided, deferred deliberately.** Promoting `NodeId` to a newtype is
workspace-wide churn — every node id, edge id, branch label, handoff
target and worker tag — for a class of bug that has not actually
occurred. The `__state_`/`__input_` prefixes that shared the same
namespace were the real hazard there, and they are now behind
`somatize_core::keys`, which is the only thing that knows how they are
spelled.

Revisit if a collision between a node id and a reserved key ever happens
in practice, or if id-shaped strings start being passed to the wrong
parameter.

## `/submit` places work; it does not proxy it

**Decided.** The coordinator answers *which worker*, takes a lease, and
the client then talks to that worker directly. It used to deserialize a
whole `SerializedPlan` — cloudpickled filters and inline tensors
included — and throw it away, which made it look like it executed
something.

Rejected: **forwarding the plan over the coordinator's WebSocket.** Soma
moves tensors; proxying doubles the traffic across that hop, and the
worker already serves WebSocket directly. Placement is the job that
cannot be done anywhere else.

Revisit if workers ever need to be unreachable from clients (a private
subnet), which is the case that makes a proxy worth its cost.

## A differentiable fan-in is keyed by name

**Decided.** A filter reading several predecessors receives a dict keyed
by predecessor node id, and opts in with `_multi_input = True`.

Rejected: **a positional tuple**, which is more torch-idiomatic but makes
the meaning depend on the order edges were declared in. This audit spent
most of its time removing order-dependence; adding some back at a new API
would be a poor trade.

Combining the inputs stays the filter's job. Concatenating, adding and
attending are different models, and a framework that picks for you is
picking wrong for someone.

## One `NodeCatalog`, not two registries and an adapter

**Decided.** Every node a graph can execute — filter or step — lives in
one registry, `NodeCatalog`, which also holds the trained states. It
implements the compiler's `NodeRegistry` port, and the executor reads it
directly.

Rejected: **separate filter and step registries joined by a borrow-pair
adapter a caller had to remember to build.** That was the previous shape,
and it meant three ways to answer "what is this node". The concrete bug
it caused is the reason for the entry: a caller passing the filter half
alone got the graph compiled with every step edge unchecked — which is
how `.compile()` came to skip schemas that `.run()` then enforced, the
worst ordering possible.

The test of the decision: there is exactly one place to ask
`node_meta(id)`, and it answers for both kinds.

## `Step` beside `Filter`, not a second engine

**Decided.** An effectful node is a peer node kind in the same graph,
executed through the same single site (`run_node`), scheduled by the same
compiler, reported on the same event bus. The difference survives as
*data* — `NodeMeta { effectful, cacheable, deterministic }`, where
`From<StepMeta>` sets `cacheable: false` — so the executor's existing
cache guard skips a step without any `if is_step`.

Rejected: **a separate agent runtime beside the pipeline runtime.** Every
framework that built one ended up with two schedulers, two caches, two
event systems and a bridge; meanwhile everything agentic flows are
missing everywhere else — schema validation, content caching, search
spaces, lineage — is exactly what the existing runtime already does.
Also rejected: **a catalog of agent node types** (the production engine
examined for the agentic design carries 24 closed-enum node variants,
three of which are unimplemented but still documented). Soma keeps five
*structural* kinds; every behaviour is library.

## `poll` is synchronous; the driver owns the concurrency

**Decided.** `Step::poll(ctx) -> Transition` is a cheap, synchronous
decision. It describes what it needs (an `Effect`); a driver performs it.

Rejected: **`async fn poll`**, which every other Rust agent framework
chooses. An async trait colours the whole runtime and complicates the GIL
story; a sync one buys three things at once: the Python bridge stays a
plain call (Python decides, Rust performs, no GIL held across I/O), the
journal is trivial (a step is a pure function of its inputs and the
recorded results, so feeding it the journal replays the identical path),
and steps compose with filters because both are just nodes.

The corollary is that a step holds no state of its own: `StepCtx` carries
`results` and `history`, and anything a step accumulates it rebuilds from
those — a field on `self` would drift from what replay feeds back.

## Transitions cross the Python bridge as dicts

**Decided.** A Python step returns plain dicts built by helpers —
`Done(v)` is `{"transition": "done", "value": v}` — and `soma.agentic`
exports the five constructors.

Rejected: **a class hierarchy mirrored on both sides.** What crosses into
Rust should be data: the Rust side gets one thing to parse
(`parse_transition`), Python keeps ordinary values that print, compare
and pickle like anything else, and a step can be written without
importing anything but five names. A hierarchy would also need versioning
in lockstep with the extension module; a dict shape is checked where it
is read, with an error message that names the helper to use.

## Patterns are functions returning graphs

**Decided.** `react`, `route`, `refine`, `debate`, `board`,
`parallel_vote`, `self_consistency` and `orchestrate` are functions in
`soma.agentic`, each returning an ordinary `Graph` built from the same
`node`, `edge`, `branch` and `loop` anyone can call.

Rejected: **patterns as engine variants or node types.** A framework
whose patterns are enum variants has to grow its core for every idea
anyone has; the dead node types in every such framework are the
evidence. A pattern that is a function costs nothing to add and nothing
to keep — and because the result is a plain graph, schema checking, the
persistent cache, `search()` and the experiment pool apply to it
unchanged, which is the entire value proposition.

## `Effect::Graph` purity follows the graph's content, and nesting stops at 8

**Decided.** Whether a sub-graph run may be memoized by content is
decided by what the graph *contains* and the mode it runs in:
`is_pure()` is true only for a filter-only `forward`. A sub-graph
containing a step, or any `fit`-mode run, is journaled per
`(run, node, turn, index)` like a model call. And agent → pipeline →
agent nesting is capped at `MAX_GRAPH_DEPTH = 8`, failing with a message
that names the cap.

Rejected: **treating every graph effect as pure**, which was the original
behaviour. A filter-only forward genuinely inherits Soma's determinism —
its nodes are content-cached already — but a step-containing sub-graph
calls a model, so content-keying it would replay the first answer
forever: the `_deterministic = False` foot-gun one level removed. `Fit`
is impure for a second reason: a replay must re-write the fitted states,
and serving the recorded summary alone would leave the graph unfitted
for the effects after it.

Rejected: **unbounded recursion.** Each nesting level is a step whose
sub-graph contains another step; real flows are one or two levels deep,
and a graph that reaches eight is almost certainly recursing on itself.
Stopping with a readable failure beats a stack that grows until the OS
ends the process.
