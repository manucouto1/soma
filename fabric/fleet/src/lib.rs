//! Where the workers are, what they can do, and what they are doing.
//!
//! The management side of the fabric. Nothing here goes near a graph: a
//! `Reaching` knows one host, a `Session` knows the hosts of one run, and
//! neither outlives the client that made it. This is the thing that outlives
//! them and can be asked *what is out there*.
//!
//! # The one idea it is built on
//!
//! **A worker does not know the name the graph gave it.** It calls itself
//! `node3-4127`; `w1` is the client's word. So what this reports are states of
//! the **pair** — see [`Standing`] — and a machine that is perfectly alive and
//! belongs to nobody is a row and not a gap.
//!
//! # What it reads, and what that costs
//!
//! Two sources that answer different questions, kept apart on purpose:
//!
//! | source | what it says | when it exists |
//! |---|---|---|
//! | `machine/<id>` in a store | up, busy, memory, cores, slices served | always, **even when nobody is using that machine** |
//! | the record of a run | which graph called it what | only for machines somebody talked to |
//!
//! The first is a scan and no fetches, because the whole of a reading is in its
//! record — that is what the wire's idle reporting was shaped for. The second
//! is a fetch per record and is bounded by whoever asks.
//!
//! # What it does not do
//!
//! It does not pack — what travels to a worker is built by whoever knows what a
//! `cloudpickle` is. It does not judge — a machine at 0.88 is a machine at
//! 0.88, and whether that is bad is an opinion at a bound, which lives in
//! `health/` where it can be argued with. And it does not attribute: guessing
//! that `node3-4127` is probably `w1` because the ports match is a guess that
//! is usually right, which is a bug that is occasionally silent.
//!
//! # It has no registry and no heartbeat, and that is inherited
//!
//! The original kept a coordinator with a `last_heartbeat` and a thirty-second
//! timeout. There is no coordinator here to keep one in, and liveness was
//! already answered another way: the store stamps every write and keeps one
//! reading per machine, rewritten, so **a reading that has not moved is a
//! machine that has stopped**. What that costs to find out is a scan.
//!
//! Which has a consequence worth saying before somebody looks for it: there is
//! **no load history**. A rewritten name is a snapshot and not a series, so a
//! curve of an idle machine cannot be drawn from here and is not drawn. What
//! has history is the record.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod fleet;
mod listing;
mod naming;
mod ran;
pub mod seed;
mod seen;
mod serving;

pub use fleet::Fleet;
pub use listing::{Listed, Listing, Named, Rung, Trouble, Wire, Wires};
pub use naming::names;
pub use ran::{Did, Ran, Run, ran, runs};
pub use seen::{Seen, Standing};
pub use serving::{Serving, routes};

// Re-exported because they are this crate's subject: the name a graph gave, and
// what a machine says about itself.
pub use somatize_core::Host;
pub use somatize_fabric_wire::Machine;
