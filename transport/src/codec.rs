//! Who writes down what only exists in one process, so a wire can carry it.
//!
//! The second hole of this crate, next to [`Provision`](crate::Provision), and
//! for the same reason: a wire format is ours, and what a
//! [`Value::Opaque`](soma_next_core::Value::Opaque) *carries* never is. One
//! holds a live Python object; another will hold something else. Neither is
//! something this crate can learn without learning an interpreter.
//!
//! It is **not** a fifth hole in the core. The core's four —`Node`, `Driver`,
//! `Transport`, `Keeper`— are what the core provides and does not fill; this one
//! is about the wire's alphabet, and the wire is this crate's.
//!
//! # It does not move the frontier, it moves what falls on which side
//!
//! [`Value::travels`](soma_next_core::Value::travels) does not change and stays
//! true: what comes out of `packed` **does** travel, being maps and bytes. A
//! codec does not relax the limit — it turns a value that could not cross into
//! one that can, before anybody asks. The frontier goes from "the variant" to
//! "the variant nobody registered a codec for", which is the more precise
//! statement of the two.
//!
//! # The same pair at both ends, in mirror image
//!
//! | | on the way out | on the way back |
//! |---|---|---|
//! | the client, in [`Worker::dispatch`](crate::Worker) | the input and what is already known, **packed** | what came back, **unpacked** |
//! | the worker, in [`Serving`](crate::Serving) | what it produced, **packed** | the input and what is already known, **unpacked** |
//!
//! Packing happens **before** anything is put into a message, so
//! [`Answer::to_bytes`](crate::Answer)'s refusal is untouched and still guards:
//! by the time it looks, whatever had a codec is already bytes.
//!
//! # Failing is not the same in both directions, and that is not an oversight
//!
//! Going **out from here**, a value nobody can write down is an error: somebody
//! over there is waiting for it and there is no second answer. Coming **back**,
//! it is left behind exactly as before —
//! [`Outcome::travelling`](soma_next_core::Outcome::travelling) — and named by
//! `RunError::Lost` if anybody reads it. And unlike a
//! [`Keeper`](soma_next_core::Keeper), a codec that fails **does** stop the run:
//! a cache that cannot answer recomputes, and a wire that cannot carry has
//! nothing to fall back on.

use soma_next_core::{NodeId, Value};
use std::fmt;

/// Writes down what cannot leave a process, and reads it back.
pub trait Codec: Send + Sync {
    /// This value in a shape that can leave the process, at any depth.
    ///
    /// Whatever carries nothing opaque comes back as it was: this is asked of
    /// every value that crosses, so the ordinary case has to be cheap.
    fn packed(&self, value: &Value) -> Result<Value, CodecError>;

    /// And the live one back, on the side that will use it.
    fn unpacked(&self, value: &Value) -> Result<Value, CodecError>;
}

/// Every one of these packed, or the first that cannot be.
pub(crate) fn packing(
    codec: &dyn Codec,
    values: &[(NodeId, Value)],
) -> Result<Vec<(NodeId, Value)>, CodecError> {
    values
        .iter()
        .map(|(id, value)| Ok((id.clone(), codec.packed(value)?)))
        .collect()
}

/// And every one of these unpacked.
pub(crate) fn unpacking(
    codec: &dyn Codec,
    values: &[(NodeId, Value)],
) -> Result<Vec<(NodeId, Value)>, CodecError> {
    values
        .iter()
        .map(|(id, value)| Ok((id.clone(), codec.unpacked(value)?)))
        .collect()
}

/// Why something could not be written down, or read back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodecError(String);

impl CodecError {
    /// A failure described by a message.
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }

    /// The message.
    pub fn message(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for CodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for CodecError {}
