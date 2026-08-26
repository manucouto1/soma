//! The contract for what a node executes. Just the one.
//!
//! A node is a function: a [`Value`] in, a [`Value`] out. There is no second
//! kind and no second shape — a filter and a step are one type, and the
//! two-variant return value that once carried the distinction turned out not to
//! be needed either.
//!
//! A node that needs something from the world — a model, a tool, an index —
//! **calls it**, holding whatever client that takes. What is kept for whoever
//! wants something injected instead is the **channel**: [`Ctx`] is where the
//! executor hands a node what it knows, and adding to it changes no signature.

use crate::{Device, Value};

/// Something a node knows how to do. `Send + Sync` because a Python `Graph` is
/// a pyclass — which PyO3 requires to be `Send` — and it carries the catalog.
pub trait Node: Send + Sync {
    /// Runs it. `input` is what arrived along the edges. It runs to the end:
    /// whatever it takes happens inside, and the engine neither counts nor
    /// bounds it.
    fn forward(&self, input: &Value, ctx: &Ctx<'_>) -> Result<Value, NodeError>;
}

/// What a node knows beyond its input, which travels separately. A type rather
/// than an argument because it is the **channel**: adding to it is additive, and
/// every node ever written has this signature.
#[derive(Debug, Clone, Copy)]
pub struct Ctx<'a> {
    /// Where this node was said to run, if it was said. It arrives as
    /// **information**: the core cannot move anything to a GPU, so the one that
    /// obeys is the node.
    pub device: Option<&'a Device>,
}

/// What a node can answer when it cannot advance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeError(String);

impl NodeError {
    /// A failure described by a message.
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }

    /// The message.
    pub fn message(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for NodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for NodeError {}
