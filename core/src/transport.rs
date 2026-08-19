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

use crate::{NodeId, Placement, Plan, Value};
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
    /// Where each node runs. It travels because a placement is data and the
    /// catalog is not.
    pub placement: &'a Placement,
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
