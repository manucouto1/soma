//! Error types for the Soma runtime.

use thiserror::Error;

/// The one error type the whole workspace returns.
///
/// A single enum rather than per-crate errors so `?` composes across every
/// crate boundary without conversion layers. `#[non_exhaustive]` on purpose:
/// most callers act on one or two variants — [`Suspended`](Self::Suspended),
/// [`Pruned`](Self::Pruned) — and pass the rest along, so adding a variant
/// should not break them.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum SomaError {
    /// `fit` was called without `y` on a filter that learns from labels.
    #[error("filter requires labels (y) but none were provided")]
    RequiresLabels,

    /// The cache store failed to read, write, or resolve an entry.
    #[error("cache error: {0}")]
    Cache(String),

    /// The compiler rejected the graph: a schema mismatch across an edge, a
    /// loop or branch it cannot claim, a reference to nothing.
    #[error("compilation error: {0}")]
    Compilation(String),

    /// A node failed while running. The workhorse variant: filter panics
    /// (caught in `run_node`), step errors, and effect failures a step chose
    /// not to absorb all surface here, named after the node they came from.
    #[error("execution error at node `{node_id}`: {message}")]
    Execution {
        /// The node that failed.
        node_id: String,
        /// What went wrong.
        message: String,
    },

    /// A pruner stopped the trial early. Closer to control flow than to a
    /// fault: the study runner records the trial as pruned, not failed, and
    /// the Python bindings surface it as its own exception type.
    #[error("trial pruned at step {step}: {reason}")]
    Pruned {
        /// The intermediate-report step at which the pruner intervened.
        step: usize,
        /// Which rule fired, in the pruner's words.
        reason: String,
    },

    /// The run stopped at `node_id`, waiting for something outside it.
    ///
    /// Not a failure: the work so far is journaled and the run continues
    /// where it left off once the answer is supplied. It travels as an error
    /// so that `?` unwinds the whole plan — a suspended run must not have
    /// its later nodes execute — while callers that care can match on it.
    /// `reason` stays typed. It used to be
    /// `serde_json::to_string(&reason).unwrap_or("unknown")` — the shape a
    /// caller needs in order to answer, flattened into a string that
    /// nothing ever parsed back, which is why resuming was unreachable
    /// from anywhere but Rust.
    #[error("run `{run_id}` suspended at node `{node_id}` (turn {turn}): {}", reason.label())]
    Suspended {
        /// The run that stopped — the id to resume with.
        run_id: String,
        /// The node whose step suspended.
        node_id: String,
        /// The step's turn at the moment it suspended; resume replays the
        /// journal up to here and re-polls with the answer.
        turn: usize,
        /// Boxed: `SomaError` is returned from nearly every function in
        /// the workspace, and this is the only variant with a payload
        /// worth more than a pointer.
        reason: Box<crate::agentic::effect::SuspendReason>,
    },

    /// A value did not have the shape its consumer declared — raised both by
    /// the compiler checking edges and at runtime when data arrives.
    #[error("schema mismatch: expected {expected}, got {got}")]
    SchemaMismatch {
        /// What the consumer's schema demands.
        expected: String,
        /// What actually arrived.
        got: String,
    },

    /// Something referenced a node id — an edge endpoint, a `Goto` target, a
    /// spawn spec — that the graph does not contain.
    #[error("node `{0}` not found in graph")]
    NodeNotFound(String),

    /// The graph's data edges form a cycle, so no execution order exists.
    /// Iteration is expressed as a declared loop the compiler claims, never
    /// by wiring a data edge back around.
    #[error("cycle detected in graph")]
    CycleDetected,

    /// Encoding or decoding failed — canonical CBOR for identities and
    /// journal keys, JSON for values, states, and the wire.
    #[error("serialization error: {0}")]
    Serialization(String),

    /// A [`crate::data::store::DataStore`] backend failed to put, get, or move data.
    #[error("data store error: {0}")]
    DataStore(String),

    /// An underlying filesystem error, converted via `?`.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// An error that fits no other variant — mostly host code and bindings.
    #[error("{0}")]
    Other(String),
}

/// Shorthand for `std::result::Result` with [`SomaError`], used across the
/// workspace.
pub type Result<T> = std::result::Result<T, SomaError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_messages() {
        let err = SomaError::RequiresLabels;
        assert_eq!(
            err.to_string(),
            "filter requires labels (y) but none were provided"
        );

        let err = SomaError::Execution {
            node_id: "scaler_1".into(),
            message: "dimension mismatch".into(),
        };
        assert_eq!(
            err.to_string(),
            "execution error at node `scaler_1`: dimension mismatch"
        );

        let err = SomaError::Pruned {
            step: 5,
            reason: "below median".into(),
        };
        assert_eq!(err.to_string(), "trial pruned at step 5: below median");
    }

    #[test]
    fn result_type_alias_works() {
        fn ok_fn() -> Result<i32> {
            Ok(42)
        }
        fn err_fn() -> Result<i32> {
            Err(SomaError::CycleDetected)
        }
        assert_eq!(ok_fn().unwrap(), 42);
        assert!(err_fn().is_err());
    }
}
