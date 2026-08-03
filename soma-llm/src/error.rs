//! What can go wrong reaching a model or a tool server.
//!
//! Everything this crate raised used to be `SomaError::Other(String)`. The
//! failures behind those strings are not alike: an endpoint that returned
//! 503, an MCP server whose process would not start, and a catalog naming
//! a provider that was never registered need a retry, a restart and a
//! config fix respectively.
//!
//! They stay out of [`SomaError`] because they are this crate's domain —
//! `soma-core` has no concept of a provider or an MCP server. At the
//! seams (the [`EffectHandler`] and [`LlmProvider`] impls) they become
//! `SomaError`, keeping the prefix that says which layer failed.
//!
//! Retry classification is a separate, narrower question and already has
//! its own type: see `Pushback` in `openai_compat`. That decides whether
//! to try again *inside* one call; this describes what finally escaped.
//!
//! See the "Errors: typed at the edges, shared at the seams" entry in the
//! design decisions.
//!
//! [`EffectHandler`]: somatize_core::effect::EffectHandler
//! [`LlmProvider`]: crate::LlmProvider

use somatize_core::error::SomaError;

/// A failure reaching a model or a tool server.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum LlmError {
    /// The endpoint: refused, timed out, answered with a status, or
    /// answered something that was not a completion. Already past the
    /// retry policy — this is what was left after it gave up.
    #[error("provider `{provider}`: {message}")]
    Provider { provider: String, message: String },

    /// An MCP server: would not start, would not answer, or its client
    /// handle was poisoned by an earlier panic.
    #[error("mcp server `{server}`: {message}")]
    Mcp { server: String, message: String },

    /// The provider catalog or the router: a name that resolves to
    /// nothing. A configuration fix, not a runtime failure.
    #[error("configuration: {0}")]
    Config(String),

    /// A handler was given an effect it does not handle. Means the
    /// driver's routing is wrong, not the provider.
    #[error("routing: {0}")]
    UnexpectedEffect(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// A failure from the shared core, passed through unchanged.
    #[error(transparent)]
    Core(#[from] SomaError),
}

impl LlmError {
    /// A provider failure, with the provider named.
    pub fn provider(provider: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Provider {
            provider: provider.into(),
            message: message.into(),
        }
    }

    /// An MCP failure, with the server named.
    pub fn mcp(server: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Mcp {
            server: server.into(),
            message: message.into(),
        }
    }
}

/// The seam.
impl From<LlmError> for SomaError {
    fn from(e: LlmError) -> Self {
        match e {
            LlmError::Core(inner) => inner,
            LlmError::Io(inner) => SomaError::Io(inner),
            other => SomaError::Other(other.to_string()),
        }
    }
}

/// `Result` with this crate's error.
pub type Result<T> = std::result::Result<T, LlmError>;

#[cfg(test)]
mod tests {
    use super::*;

    /// Which layer failed survives the crossing, and the provider or
    /// server is named in it — the two things a caller reads first.
    #[test]
    fn an_llm_error_names_its_layer_and_its_peer() {
        let provider: SomaError = LlmError::provider("nvidia", "503 Service Unavailable").into();
        assert!(
            provider.to_string().contains("provider `nvidia`"),
            "{provider}"
        );

        let mcp: SomaError = LlmError::mcp("filesystem", "no stdout").into();
        assert!(mcp.to_string().contains("mcp server `filesystem`"), "{mcp}");
        assert_ne!(provider.to_string(), mcp.to_string());
    }

    #[test]
    fn a_core_error_passes_through_unchanged() {
        let original = SomaError::NodeNotFound("agent".into());
        let crossed: SomaError = LlmError::Core(original).into();
        assert!(matches!(crossed, SomaError::NodeNotFound(id) if id == "agent"));
    }
}
