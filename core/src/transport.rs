//! Who carries a slice of plan elsewhere and brings back what it produced.
//!
//! The core does not know what a wire is: to it a [`Host`](crate::Host) is a
//! name and a `Transport` is someone who knows how to reach it. It neither
//! serializes, nor spawns processes, nor knows about sockets — just as it does
//! not know what a node's request is, which is why the
//! [`Driver`](crate::Driver) exists.
//!
//! The same old division of labour, and the reason the core still has no
//! dependencies: **the core provides the hole; whoever knows what goes in it is
//! a library.** A wire format would require `serde`.
//!
//! Declared versus injected, for the third time: a [`Node`](crate::Node) is put
//! there by whoever declares the graph; a `Driver` and a `Transport` by whoever
//! **executes**. That is why a [`Host`](crate::Host) is a name and not an
//! address — the same graph spreads across two processes here or two machines
//! there without touching a line of what was declared.

use crate::{Key, Memory, NodeId, Placement, Plan, Value};
use std::fmt;

/// Knows how to execute a plan elsewhere.
pub trait Transport: Send + Sync {
    /// Executes `plan` over there, with what it needs in order to do so.
    ///
    /// A [`Value::Opaque`] has to fail here: it carries something that only
    /// exists in this process.
    fn dispatch(&self, plan: &Plan, cargo: &Cargo<'_>) -> Result<Outcome, TransportError>;
}

/// What a plan needs beyond itself in order to run elsewhere.
pub struct Cargo<'a> {
    /// The graph's input, for the steps over there that read from nobody.
    pub input: &'a Value,
    /// What was already produced **here** that the plan over there reads and
    /// does not produce. Only that: the wire is the expensive part.
    pub known: &'a [(NodeId, Value)],
    /// What each of those is called. Without them the slice over there can name
    /// nothing it produces, and a cache that stops at the process boundary is a
    /// cache nobody can rely on.
    pub keys: &'a [(NodeId, Key)],
    /// Where each node runs. It travels because a placement is data and the
    /// catalog is not.
    pub placement: &'a Placement,
    /// What is remembered about each node, for the same reason and by the same
    /// rule: it is data. Without it the other side does not know what is frozen,
    /// what is worth keeping, or what any of it is called.
    pub memory: &'a Memory,
}

/// What came back from executing a plan elsewhere.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Outcome {
    /// What the last step returned, exactly as it would have returned it here.
    /// Not "the last one in the map": a wave has no single output.
    pub last: Value,
    /// What each node produced, to be merged with what is here.
    pub produced: Vec<(NodeId, Value)>,
    /// And what each of those is called, so the chain of keys carries on below
    /// the slice that went away.
    pub keys: Vec<(NodeId, Key)>,
}

impl Outcome {
    /// The same outcome with whatever cannot leave this process left out of
    /// `produced`.
    ///
    /// An intermediate value of a slice is read by the steps of that slice,
    /// which ran where it did: sending it back was never the point, and
    /// refusing the whole answer over one is refusing the case this exists for —
    /// two steps on one host, with something live in between them. What is
    /// **not** filtered is `last`, which is the value of the slice itself: that
    /// one has a reader here by definition.
    ///
    /// Whoever does read one of the dropped values gets [`RunError::Lost`],
    /// naming both ends.
    pub fn travelling(mut self) -> Self {
        self.produced.retain(|(_, value)| value.travels());
        self
    }
}

/// What a transport can answer when it cannot carry something.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransportError(String);

impl TransportError {
    /// A failure described by a message.
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }

    /// The message.
    pub fn message(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for TransportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for TransportError {}
