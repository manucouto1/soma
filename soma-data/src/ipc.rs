//! The codec for frames: Arrow IPC, in both directions.

use crate::Frame;
use somatize_core::Value;
use somatize_core::{Codec, CodecError, as_written, written_down};

/// Writes frames down so they can be kept or sent, and reads them back.
///
/// The **second implementor of `Codec`, and from another crate** — the first was
/// `python/`'s registry of `dump`/`load` pairs — which is what keeps the hole a
/// hole. A unit struct because the format is the format and there is nothing to
/// configure; whoever runs the engine hands it in, and a store and a wire get
/// the same bytes because what a frame weighs has one answer.
///
/// It refuses an opaque that is not a frame rather than guessing. A graph
/// carrying tensors **and** rows wants both codecs, which is what `python/`'s
/// does: a frame comes here, anything else goes to its registry.
pub struct Ipc;

impl Codec for Ipc {
    fn packed(&self, value: &Value) -> Result<Value, CodecError> {
        Ok(match value {
            Value::Opaque(_) => {
                let frame = Frame::of(value).ok_or_else(|| {
                    CodecError::new(
                        "this opaque value is not a frame, and Arrow IPC is all this codec \
                         knows how to write down",
                    )
                })?;
                written_down(
                    Frame::KIND,
                    frame.written().map_err(|e| CodecError::new(e.message()))?,
                )
            }
            Value::Map(pairs) => Value::map(
                pairs
                    .iter()
                    .map(|(key, value)| Ok((key.clone(), self.packed(value)?)))
                    .collect::<Result<Vec<_>, CodecError>>()?,
            ),
            Value::List(items) => Value::list(
                items
                    .iter()
                    .map(|item| self.packed(item))
                    .collect::<Result<Vec<_>, CodecError>>()?,
            ),
            other => other.clone(),
        })
    }

    fn unpacked(&self, value: &Value) -> Result<Value, CodecError> {
        if let Some((kind, bytes)) = as_written(value) {
            // Somebody else's kind is left exactly as it arrived: it is not
            // ours to read and it is not ours to lose, and the process that
            // does know it may be one hop further on.
            if kind != Frame::KIND {
                return Ok(value.clone());
            }
            return Ok(Frame::read(bytes)
                .map_err(|e| CodecError::new(e.message()))?
                .value());
        }
        Ok(match value {
            Value::Map(pairs) => Value::map(
                pairs
                    .iter()
                    .map(|(key, value)| Ok((key.clone(), self.unpacked(value)?)))
                    .collect::<Result<Vec<_>, CodecError>>()?,
            ),
            Value::List(items) => Value::list(
                items
                    .iter()
                    .map(|item| self.unpacked(item))
                    .collect::<Result<Vec<_>, CodecError>>()?,
            ),
            other => other.clone(),
        })
    }
}
