//! Who writes down what only exists in one process, so a wire can carry it.
//!
//! The second hole of this crate, next to [`Provision`](crate::Provision), and
//! for the same reason: a wire format is ours, and what a
//! [`Value::Opaque`](somatize_core::Value::Opaque) *carries* never is. One
//! holds a live Python object; another will hold something else. Neither is
//! something this crate can learn without learning an interpreter.
//!
//! It **is** a hole of the core, and it took a third tenant to see it. It was
//! written here when the wire was its only consumer, and filed under the wire —
//! but a `Store` asks an opaque value the same question a socket does, and
//! `data/` asks it a third time for Arrow IPC. What decides where a hole lives
//! is what it serves, not who reached for it first: `Packing` next door is the
//! proof, since it is a `Keeper` and `Keeper` was always the core's.
//!
//! # It does not move the frontier, it moves what falls on which side
//!
//! [`Value::travels`](somatize_core::Value::travels) does not change and stays
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
//! [`Outcome::travelling`](somatize_core::Outcome::travelling) — and named by
//! `RunError::Lost` if anybody reads it. And unlike a
//! [`Keeper`](somatize_core::Keeper), a codec that fails **does** stop the run:
//! a cache that cannot answer recomputes, and a wire that cannot carry has
//! nothing to fall back on.

use crate::{NodeId, Value};
use std::fmt;
use std::sync::Arc;

/// The reserved key that says a map is not a map.
///
/// **Here and not in an implementor**, for the same reason a store's record is
/// in the store and not in the directory: this shape is what makes two codecs
/// the same codec. A frame written down by the Rust side and read back by the
/// Python one meet in these two keys and nowhere else, and two copies of them
/// would drift the day one of them changed.
const PACKED: &str = "__soma_opaque__";

/// Where the bytes are, next to it.
const BYTES: &str = "bytes";

/// Something written down: what kind it was, and the bytes it became.
///
/// The `kind` is what a store keeps forever, so it is named after the **type or
/// the format** and never after the run — `torch.Tensor`, `arrow.RecordBatch`.
/// It is also how the far end knows who to ask to read it back.
pub fn written_down(kind: impl Into<String>, bytes: Vec<u8>) -> Value {
    Value::map(vec![
        (PACKED.to_string(), Value::text(kind.into())),
        (BYTES.to_string(), Value::Bytes(Arc::new(bytes))),
    ])
}

/// What was written down in there, if that is what it is.
pub fn as_written(value: &Value) -> Option<(&str, &[u8])> {
    let Value::Map(pairs) = value else {
        return None;
    };
    let kind = pairs.iter().find(|(key, _)| key == PACKED)?;
    let bytes = pairs.iter().find(|(key, _)| key == BYTES)?;
    match (&kind.1, &bytes.1) {
        (Value::Text(kind), Value::Bytes(bytes)) => Some((kind, bytes)),
        _ => None,
    }
}

/// Whether there is anything written down in there at all, at any depth.
///
/// Asked before doing the walk, so that the ordinary value — which is most of
/// them — costs one look and nothing else.
pub fn anything_written(value: &Value) -> bool {
    if as_written(value).is_some() {
        return true;
    }
    match value {
        Value::Map(pairs) => pairs.iter().any(|(_, value)| anything_written(value)),
        Value::List(items) => items.iter().any(anything_written),
        _ => false,
    }
}

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
pub fn packed_all(
    codec: &dyn Codec,
    values: &[(NodeId, Value)],
) -> Result<Vec<(NodeId, Value)>, CodecError> {
    values
        .iter()
        .map(|(id, value)| Ok((id.clone(), codec.packed(value)?)))
        .collect()
}

/// And every one of these unpacked.
pub fn unpacked_all(
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
