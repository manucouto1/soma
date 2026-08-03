---
title: Architecture Decisions
description: Boundaries the code is expected to keep, why they were chosen, and what was deliberately not done.
---

## Why this page exists

A comment in `soma-core/src/graph.rs` pointed at an `architecture-review.md`
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

Staying, deliberately:

- **`TrainingStrategy` the type.** It is a graph-level attribute — part of
  what a graph *is* — so it is a contract. Only its `impl` is execution.
  The type and its behaviour split; that is the whole point.
- **`summary.rs`, `svg.rs`, `viz.rs`** (1106 lines). Pure `data → String`,
  no I/O, no runtime. Rendering a graph to text is serialization in the
  broad sense, and moving them would be churn in the name of purity: 24–29
  external call sites, no problem being solved.

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
