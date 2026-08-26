//! What an edit did to a graph, said before anybody runs it.
//!
//! [`revision`] puts a commit somewhere it can be imported from, [`snapshot`]
//! asks a subprocess what the graph there is called, and [`findings`] reads
//! what comparing two of those answers said. Around them, what a question was:
//! [`moves`] and [`journal`] hold it, [`trials`] runs it, [`walk`] and
//! [`data`] read the store back, and [`serving`] puts all of it behind HTTP.
//!
//! Nothing here holds a graph. A graph is Python, it exists for the length of
//! the probe's process, and what crosses back is a [`snapshot::Snapshot`].

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
