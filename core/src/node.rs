//! The contract for what a node executes. Just the one.
//!
//! A node advances one turn and says how things continue:
//! [`Transition::Done`] if it is finished, [`Transition::Await`] if it needs
//! something from the world first.
//!
//! **The difference between a filter and a step is that variant, not a type.**
//! Two traits duplicated in the type system a distinction that was already in
//! the return value, and propagated upwards — catalog, plan, engine, errors —
//! the obligation to know which of the two each node was. The side effect that
//! earns its keep on its own: a node can **evolve**, gaining an `Await` branch
//! in the same body instead of being rewritten as another type.
//!
//! What a node asks for is **opaque to the core**: a [`Value`] the
//! [`Driver`](crate::Driver) knows how to interpret. That is why there are no
//! LLMs here, no tools, no effect log.

use crate::{Device, Value};

/// Something a node knows how to do.
///
/// `Send + Sync` because a Python `Graph` is a pyclass — which PyO3 requires to
/// be `Send` — and it carries the catalog.
pub trait Node: Send + Sync {
    /// Advances one turn.
    ///
    /// Called with `ctx.turn == 0` and no results; afterwards, with whatever
    /// the driver returned for the previous turn's requests, in the same
    /// order. `input` is always the same: what arrived along the edges.
    fn forward(&self, input: &Value, ctx: &Ctx<'_>) -> Result<Transition, NodeError>;
}

/// What a node knows beyond its input, which travels separately.
#[derive(Debug, Clone, Copy)]
pub struct Ctx<'a> {
    /// How many times it has already been asked; starts at 0.
    pub turn: usize,
    /// What the driver returned for the previous turn's requests, in order.
    /// Empty on turn 0.
    pub results: &'a [Value],
    /// Where this node was said to run, if it was said. It arrives as
    /// **information**: the core cannot move anything to a GPU, so the one that
    /// obeys is the node.
    pub device: Option<&'a Device>,
}

/// How things continue after a turn.
///
/// No `#[non_exhaustive]`: whoever executes a node has to decide what to do
/// with each variant, so adding one *should* break everyone.
#[derive(Debug, Clone, PartialEq)]
pub enum Transition {
    /// Finished, with this output.
    Done(Value),
    /// Needs someone to do this before continuing. It will be asked again with
    /// the results.
    Await(Vec<Value>),
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
