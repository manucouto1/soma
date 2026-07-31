//! Error types for the Soma runtime.

use thiserror::Error;

#[derive(Error, Debug)]
#[non_exhaustive]
pub enum SomaError {
    #[error("filter requires labels (y) but none were provided")]
    RequiresLabels,

    #[error("cache error: {0}")]
    Cache(String),

    #[error("compilation error: {0}")]
    Compilation(String),

    #[error("execution error at node `{node_id}`: {message}")]
    Execution { node_id: String, message: String },

    #[error("trial pruned at step {step}: {reason}")]
    Pruned { step: usize, reason: String },

    /// The run stopped at `node_id`, waiting for something outside it.
    ///
    /// Not a failure: the work so far is journaled and the run continues
    /// where it left off once the answer is supplied. It travels as an error
    /// so that `?` unwinds the whole plan — a suspended run must not have
    /// its later nodes execute — while callers that care can match on it.
    #[error("run `{run_id}` suspended at node `{node_id}` (turn {turn}): {reason}")]
    Suspended {
        run_id: String,
        node_id: String,
        turn: usize,
        reason: String,
    },

    #[error("schema mismatch: expected {expected}, got {got}")]
    SchemaMismatch { expected: String, got: String },

    #[error("node `{0}` not found in graph")]
    NodeNotFound(String),

    #[error("cycle detected in graph")]
    CycleDetected,

    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("data store error: {0}")]
    DataStore(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("{0}")]
    Other(String),
}

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
