//! The codec for frames: Arrow IPC, in both directions.

use crate::Frame;
use soma_next_core::Value;
use soma_next_transport::{Codec, CodecError, as_written, written_down};

/// Writes frames down so they can be kept or sent, and reads them back.
///
/// # The second implementor of `Codec`, and from another crate
///
/// The first was `python/`'s, holding a registry of `dump`/`load` pairs the user
/// filled. This one knows exactly one thing and needs no registry: a
/// [`Frame`] is Arrow IPC. Which is what a hole is for — the trait stays a trait
/// because the implementations really do come from elsewhere, and neither of the
/// two could have been written in `transport`.
///
/// # A unit struct, because it holds nothing
///
/// There is nothing to configure: the format is the format. Whoever runs the
/// engine hands it in — `Packing::over(&cache, &Ipc)` for a store,
/// `Worker::dispatch` for a wire — and the same bytes come out either way,
/// because what a frame weighs is one question with one answer.
///
/// # What it refuses
///
/// An opaque that is not a frame. It says so rather than guessing, which is the
/// same frontier every codec draws: *what nobody registered a codec for does not
/// travel*.
///
/// A graph carrying tensors **and** rows wants both codecs at once, and that is
/// what `python/`'s does: a frame is handed here, anything else goes to the
/// registry it holds. One that asks each in turn, rather than a bigger one that
/// knows both.
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
