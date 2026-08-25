//! What an edit did to a graph, said before anybody runs it.
//!
//! Three pieces and one direction between them: [`revision`] puts a commit
//! somewhere it can be imported from, [`snapshot`] asks a subprocess what the
//! graph there is called, and [`difference`] compares two of those answers.
//!
//! Nothing here holds a graph. A graph is Python, it exists for the length of
//! the probe's process, and what crosses back is a `Snapshot`.

pub mod bench;
pub mod data;
pub mod findings;
pub mod journal;
pub mod moves;
pub mod revision;
pub mod serving;
pub mod snapshot;
pub mod trials;
pub mod walk;
