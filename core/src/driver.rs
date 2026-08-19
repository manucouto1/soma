//! Who does what a step asks for.
//!
//! The core does not know what a request is: to it, it is a `Value`. The one
//! who interprets it is the driver — calling a model, running a tool, querying
//! an index. That ignorance is what keeps the agentic layer out of the core.

use crate::Value;

/// Serves what a node asked for with [`Transition::Await`](crate::Transition).
///
/// Declared versus injected: a [`Node`](crate::Node) is put there by whoever
/// declares the graph; a driver by whoever **executes**, and what it returns
/// crosses no edge.
pub trait Driver: Send + Sync {
    /// Serves the requests and returns one result per request, in order.
    fn perform(&self, requests: &[Value]) -> Result<Vec<Value>, DriverError>;
}

/// What a driver can answer when it cannot serve a request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DriverError(String);

impl DriverError {
    /// A failure described by a message.
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }

    /// The message.
    pub fn message(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for DriverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for DriverError {}
