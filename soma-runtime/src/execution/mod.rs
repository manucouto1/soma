//! Running a compiled plan.
//!
//! The crate's centre of gravity, and the shortest description of it is a
//! stack. [`GraphSession`](graph_session::GraphSession) binds a graph to a
//! catalog, a cache and an event bus, and is what a caller holds.
//! [`Runner`](runner::Runner) says *where* a plan runs.
//! [`execute`](executor::execute) walks the plan tree, and every leaf of
//! that walk reaches [`run_node`](executor) — the one execution site, for
//! filters and steps alike.
//!
//! `run_node`'s guts are three primitives: `output_key` derives the
//! memoization key (and refuses to for a node whose metadata says it is not
//! cacheable), `compute_node` runs the thing inside `catch_unwind`, and
//! `store_output` records the value with its provenance. [`stream`]
//! composes those same three per chunk, which is what makes streaming the
//! same execution site rather than a second one — see D-11 for how far that
//! claim can be trusted today.
//!
//! [`node_catalog`] is the registry all of this reads: every node, filter
//! or step, plus the states `fit` produced. There is one, deliberately —
//! filters and steps used to live in two registries joined by an adapter,
//! which is how `.compile()` came to skip every step's schema validation
//! while `.run()` checked them.

pub mod executor;
pub mod forward;
pub mod graph_session;
pub mod node_catalog;
pub mod runner;
pub mod stream;
