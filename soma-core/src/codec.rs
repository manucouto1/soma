//! Who writes down what only exists in one process, so it can be kept or sent.
//!
//! A hole of the core, and it took a third tenant to see it: it was written next
//! to the wire when that was its only consumer, but a `Store` asks an opaque
//! value the same question a socket does, and `data/` asks it a third time for
//! Arrow IPC. What decides where a hole lives is what it serves.
//!
//! It does not move the frontier, it moves what falls on which side.
//! [`Value::travels`] stays true — what comes out of `packed` **does** travel,
//! being maps and bytes. The frontier goes from *the variant* to *the variant
//! nobody registered a codec for*.
//!
//! The same pair sits at both ends in mirror image: the client packs the input
//! and unpacks what came back, the worker unpacks the input and packs what it
//! produced. Packing happens **before** a message is built, so `Answer`'s
//! refusal is untouched and still guards.
//!
//! Failing is not symmetric. Going out, a value nobody can write down is an
//! error: somebody over there is waiting. Coming back it is left behind and
//! named if anybody reads it. And unlike a [`Keeper`](crate::Keeper) a codec
//! that fails **does** stop the run — a cache recomputes, a wire has nothing to
//! fall back on.

use crate::{NodeId, Value};
use std::fmt;
use std::sync::Arc;

/// The reserved key that says a map is not a map.
///
/// Here and not in an implementor: this shape is what makes two codecs the same
/// codec, and two copies of it would drift the day one changed.
const PACKED: &str = "__soma_opaque__";

/// Where the bytes are, next to it.
const BYTES: &str = "bytes";

/// Something written down: what kind it was, and the bytes it became. The
/// `kind` is named after the **type or the format** and never after the run —
/// `torch.Tensor`, `arrow.RecordBatch` — since it is also how the far end knows
/// who to ask to read it back.
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

/// Whether there is anything written down in there at all, at any depth. Asked
/// before the walk, so the ordinary value costs one look.
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
    /// This value in a shape that can leave the process, at any depth. Whatever
    /// carries nothing opaque comes back as it was: this is asked of every value
    /// that crosses.
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
