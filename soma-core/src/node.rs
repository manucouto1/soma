//! The contract for what a node executes. Just the one.
//!
//! A node is a function: a [`Value`] in, a [`Value`] out. There is no second
//! kind and no second shape.
//!
//! **A filter and a step are one type**, which is what CU6 set out to get. It
//! took a two-variant return value to get there and it turned out not to need
//! one: with a single shape the distinction has nowhere left to live. Two traits
//! had duplicated in the type system something that was already in the return
//! value, and propagated upwards — catalog, plan, engine, errors — the
//! obligation to know which of the two each node was. None of that comes back.
//!
//! # What is not here, and where it went
//!
//! A node that needs something from the world — a model, a tool, an index —
//! **calls it**, holding whatever client that takes. The core used to offer a
//! way to ask instead: return `Await(requests)`, have an injected `Driver` serve
//! them, and be asked again. It was the seam of an agentic layer, it cost every
//! node the `Done(...)` around its answer, and after eighteen use cases it had
//! **no consumer outside the tests**. A hole with no tenant is what this project
//! exists not to build, so it went.
//!
//! What is kept is the **channel**: [`Ctx`] is where the executor hands a node
//! what it knows. An agentic layer that wants something injected puts it there,
//! and no node signature changes.

use crate::{Device, Value};

/// Something a node knows how to do.
///
/// `Send + Sync` because a Python `Graph` is a pyclass — which PyO3 requires to
/// be `Send` — and it carries the catalog.
pub trait Node: Send + Sync {
    /// Runs it. `input` is what arrived along the edges.
    ///
    /// It runs to the end: whatever it takes — a retry, a model, three rounds
    /// of something — happens inside, and the engine neither counts it nor
    /// bounds it. A node that does not come back does not come back, the same
    /// way a function that does not return does not return.
    fn forward(&self, input: &Value, ctx: &Ctx<'_>) -> Result<Value, NodeError>;
}

/// What a node knows beyond its input, which travels separately.
///
/// One field today, and a type rather than an argument on purpose: this is the
/// **channel** by which whoever executes hands a node what it knows. Adding to
/// it is additive; passing the same thing as an argument would not be, and
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
