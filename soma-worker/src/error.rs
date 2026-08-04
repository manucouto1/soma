//! What can go wrong in a worker.
//!
//! Every error this crate produced used to be `SomaError::Other(String)` —
//! all 51 of them. So a caller could not tell a dropped socket from a
//! Python subprocess that died from a plan it could not decode, and the
//! three want different responses: retry, restart the interpreter, give
//! up. A string that reads differently is not a distinction a program can
//! act on.
//!
//! These variants are the worker's domain, which is why they live here and
//! not in [`SomaError`]: `soma-core` describes graphs and their execution,
//! and it has no business knowing what a venv is. At the boundary — the
//! traits this crate implements for the runtime — a `WorkerError` converts
//! into `SomaError`, keeping its message.
//!
//! See the "Errors: typed at the edges, shared at the seams" entry in the
//! design decisions.

use somatize_core::error::SomaError;

/// A failure inside a worker.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum WorkerError {
    /// The socket or the HTTP endpoint. Usually worth retrying.
    #[error("transport: {0}")]
    Transport(String),

    /// The Python subprocess: it would not start, would not answer, or
    /// answered something that was not a reply.
    #[error("python subprocess: {0}")]
    Python(String),

    /// Building or updating an isolated environment.
    #[error("environment: {0}")]
    Env(String),

    /// A payload that would not encode or decode. Distinct from
    /// [`WorkerError::Transport`]: the bytes arrived, they were wrong.
    #[error("encoding: {0}")]
    Encoding(String),

    /// A lock was poisoned or a worker thread panicked. Means some other
    /// error already happened and took a thread with it.
    #[error("worker state: {0}")]
    Concurrency(String),

    /// The remote worker ran the plan and reported a failure. The message
    /// is the worker's, not ours.
    #[error("remote worker: {0}")]
    Remote(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// A failure that came from the shared core — a cache miss that
    /// mattered, a bad graph — and is passed through unchanged.
    #[error(transparent)]
    Core(#[from] SomaError),
}

/// The seam. A worker error crossing into the runtime keeps its message
/// and its prefix, so `transport: WS connect: …` still says which layer
/// failed even after the type is gone.
impl From<WorkerError> for SomaError {
    fn from(e: WorkerError) -> Self {
        match e {
            WorkerError::Core(inner) => inner,
            WorkerError::Io(inner) => SomaError::Io(inner),
            other => SomaError::Other(other.to_string()),
        }
    }
}

/// `Result` with this crate's error.
pub type Result<T> = std::result::Result<T, WorkerError>;

#[cfg(test)]
mod tests {
    use super::*;

    /// The distinction survives as far as the shared type allows.
    ///
    /// Before this existed, all 51 errors this crate produced were
    /// `SomaError::Other`, so "the socket dropped" and "the interpreter
    /// died" were the same value with different prose. Inside the crate
    /// they are now different variants; crossing the seam they keep the
    /// prefix that says which layer failed.
    #[test]
    fn a_worker_error_keeps_its_layer_when_it_crosses_the_seam() {
        let transport: SomaError = WorkerError::Transport("WS connect refused".into()).into();
        assert!(transport.to_string().contains("transport:"), "{transport}");

        let python: SomaError = WorkerError::Python("interpreter exited".into()).into();
        assert!(
            python.to_string().contains("python subprocess:"),
            "{python}"
        );
        assert_ne!(transport.to_string(), python.to_string());
    }

    /// A core error passing through is not re-wrapped: it would gain a
    /// second layer of prefix and stop matching what it was.
    #[test]
    fn a_core_error_passes_through_unchanged() {
        let original = SomaError::NodeNotFound("scaler".into());
        let round_tripped: SomaError = WorkerError::Core(original).into();
        assert!(matches!(round_tripped, SomaError::NodeNotFound(id) if id == "scaler"));
    }
}
