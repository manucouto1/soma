//! `SOMA1` framed binary codec for [`Value`].
//!
//! The persistent cache stored values as JSON in Phase 1 — an f64
//! tensor round-tripped through decimal text at ~3× the size. This
//! codec writes tensors as raw little-endian f64, keeping payloads at
//! ~1× raw size, and hashes the encoded bytes with BLAKE3 in the same
//! pass.
//!
//! Frame layout:
//!
//! ```text
//! [0..6)  magic  b"SOMA1\0"
//! [6]     variant tag: 0=Empty 1=Tensor 2=Json 3=Bytes 4=Object
//! [7]     compression: 0=raw (other values reserved)
//! [8..]   payload
//! ```
//!
//! Tensor payload: `u32 ndim` + `ndim × u64` dims + `u64 count` +
//! `count × f64` little-endian values. Json payload: `serde_json`
//! bytes (deterministic — the default map is sorted). Bytes/Object:
//! raw bytes.

use crate::action::ContentHash;
use crate::error::{Result, SomaError};
use crate::value::Value;
use std::sync::Arc;

pub const MAGIC: &[u8; 6] = b"SOMA1\0";

const TAG_EMPTY: u8 = 0;
const TAG_TENSOR: u8 = 1;
const TAG_JSON: u8 = 2;
const TAG_BYTES: u8 = 3;
const TAG_OBJECT: u8 = 4;
const TAG_TEXT: u8 = 5;

const COMPRESSION_RAW: u8 = 0;

/// Encode a value into a `SOMA1` frame.
pub fn encode_value(value: &Value) -> Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(64);
    buf.extend_from_slice(MAGIC);
    match value {
        Value::Empty => {
            buf.push(TAG_EMPTY);
            buf.push(COMPRESSION_RAW);
        }
        Value::Tensor { values, shape } => {
            buf.push(TAG_TENSOR);
            buf.push(COMPRESSION_RAW);
            buf.reserve(4 + shape.len() * 8 + 8 + values.len() * 8);
            buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());
            for dim in shape {
                buf.extend_from_slice(&(*dim as u64).to_le_bytes());
            }
            buf.extend_from_slice(&(values.len() as u64).to_le_bytes());
            for v in values.iter() {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        Value::Text(s) => {
            buf.push(TAG_TEXT);
            buf.push(COMPRESSION_RAW);
            buf.extend_from_slice(s.as_bytes());
        }
        Value::Json(v) => {
            buf.push(TAG_JSON);
            buf.push(COMPRESSION_RAW);
            let bytes = serde_json::to_vec(v.as_ref())
                .map_err(|e| SomaError::Cache(format!("codec: json encode: {e}")))?;
            buf.extend_from_slice(&bytes);
        }
        Value::Bytes(b) => {
            buf.push(TAG_BYTES);
            buf.push(COMPRESSION_RAW);
            buf.extend_from_slice(b);
        }
        Value::Object(b) => {
            buf.push(TAG_OBJECT);
            buf.push(COMPRESSION_RAW);
            buf.extend_from_slice(b);
        }
    }
    Ok(buf)
}

/// Encode and content-hash in one step.
pub fn encode_and_hash(value: &Value) -> Result<(Vec<u8>, ContentHash)> {
    let bytes = encode_value(value)?;
    let hash = ContentHash::blake3(&bytes);
    Ok((bytes, hash))
}

/// Decode a `SOMA1` frame back into a value.
pub fn decode_value(bytes: &[u8]) -> Result<Value> {
    if bytes.len() < 8 || &bytes[..6] != MAGIC {
        return Err(SomaError::Cache("codec: not a SOMA1 frame".into()));
    }
    let tag = bytes[6];
    if bytes[7] != COMPRESSION_RAW {
        return Err(SomaError::Cache(format!(
            "codec: unknown compression byte {}",
            bytes[7]
        )));
    }
    let payload = &bytes[8..];
    match tag {
        TAG_EMPTY => Ok(Value::Empty),
        TAG_TENSOR => decode_tensor(payload),
        TAG_JSON => {
            let v: serde_json::Value = serde_json::from_slice(payload)
                .map_err(|e| SomaError::Cache(format!("codec: json decode: {e}")))?;
            Ok(Value::json(v))
        }
        TAG_TEXT => {
            let s = std::str::from_utf8(payload)
                .map_err(|e| SomaError::Cache(format!("codec: text decode: {e}")))?;
            Ok(Value::text(s))
        }
        TAG_BYTES => Ok(Value::bytes(payload.to_vec())),
        TAG_OBJECT => Ok(Value::object(payload.to_vec())),
        other => Err(SomaError::Cache(format!("codec: unknown tag {other}"))),
    }
}

fn decode_tensor(payload: &[u8]) -> Result<Value> {
    let err = || SomaError::Cache("codec: truncated tensor frame".into());
    let mut at = 0usize;
    let take = |at: &mut usize, n: usize| -> Result<&[u8]> {
        let slice = payload.get(*at..*at + n).ok_or_else(err)?;
        *at += n;
        Ok(slice)
    };

    let ndim = u32::from_le_bytes(take(&mut at, 4)?.try_into().unwrap()) as usize;
    if ndim > 64 {
        return Err(SomaError::Cache(format!("codec: implausible ndim {ndim}")));
    }
    let mut shape = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        shape.push(u64::from_le_bytes(take(&mut at, 8)?.try_into().unwrap()) as usize);
    }
    let count = u64::from_le_bytes(take(&mut at, 8)?.try_into().unwrap()) as usize;
    let data = take(&mut at, count.checked_mul(8).ok_or_else(err)?)?;
    let mut values = Vec::with_capacity(count);
    for chunk in data.chunks_exact(8) {
        values.push(f64::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(Value::Tensor {
        values: Arc::new(values),
        shape,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn roundtrip_all_variants() {
        let values = vec![
            Value::Empty,
            Value::tensor(vec![1.0, -2.5, f64::MAX, 0.0], vec![2, 2]),
            Value::tensor(vec![], vec![0]),
            Value::json(json!({"a": [1, 2.5, "x"], "b": null})),
            Value::bytes(vec![0, 255, 128]),
            Value::object(vec![0x80, 0x04]),
            Value::text(""),
            Value::text("Summarize the following in one sentence."),
            // Non-ASCII must survive the byte-level frame intact.
            Value::text("resumen: ¿qué pasó? — 数字 🧬"),
        ];
        for v in values {
            let (bytes, hash) = encode_and_hash(&v).unwrap();
            assert!(hash.verify(&bytes));
            assert_eq!(decode_value(&bytes).unwrap(), v, "roundtrip failed for {v}");
        }
    }

    /// Text and a JSON string holding the same characters are different
    /// values, and must hash differently — otherwise a prompt and a JSON
    /// document quoting it would share a cache line.
    #[test]
    fn text_and_json_string_are_distinct() {
        let (text_bytes, text_hash) = encode_and_hash(&Value::text("hello")).unwrap();
        let (json_bytes, json_hash) = encode_and_hash(&Value::json(json!("hello"))).unwrap();

        assert_ne!(text_bytes, json_bytes);
        assert_ne!(text_hash, json_hash);
        assert_eq!(decode_value(&text_bytes).unwrap(), Value::text("hello"));
        assert_eq!(
            decode_value(&json_bytes).unwrap(),
            Value::json(json!("hello"))
        );
    }

    /// Invalid UTF-8 in a text frame is reported, not silently replaced.
    #[test]
    fn text_frame_rejects_invalid_utf8() {
        let mut frame = Vec::from(MAGIC);
        frame.push(TAG_TEXT);
        frame.push(COMPRESSION_RAW);
        frame.extend_from_slice(&[0xff, 0xfe]);
        assert!(decode_value(&frame).is_err());
    }

    #[test]
    fn tensor_size_is_near_raw() {
        let n = 10_000;
        // Full-precision mantissas — the realistic case for model
        // weights/features (short decimals like 0.7 flatter JSON).
        let v = Value::tensor((0..n).map(|i| (i as f64).sin()).collect(), vec![n]);
        let encoded = encode_value(&v).unwrap();
        let raw = n * 8;
        assert!(
            encoded.len() <= raw + raw / 10 + 64,
            "encoded {} bytes vs raw {} — must stay within ~1.1×",
            encoded.len(),
            raw
        );
        // And meaningfully smaller than the JSON text form.
        let json_len = serde_json::to_vec(&v).unwrap().len();
        assert!(
            encoded.len() * 2 < json_len,
            "binary ({}) should be well under half of JSON ({})",
            encoded.len(),
            json_len
        );
    }

    #[test]
    fn identical_values_hash_identically() {
        let a = Value::tensor(vec![1.0, 2.0], vec![2]);
        let b = Value::tensor(vec![1.0, 2.0], vec![2]);
        assert_eq!(
            encode_and_hash(&a).unwrap().1,
            encode_and_hash(&b).unwrap().1
        );
        let c = Value::tensor(vec![1.0, 2.0], vec![2, 1]);
        assert_ne!(
            encode_and_hash(&a).unwrap().1,
            encode_and_hash(&c).unwrap().1
        );
    }

    #[test]
    fn garbage_rejected() {
        assert!(decode_value(b"").is_err());
        assert!(decode_value(b"NOTSOMA1xxxx").is_err());
        let mut frame = encode_value(&Value::Empty).unwrap();
        frame[7] = 9; // unknown compression
        assert!(decode_value(&frame).is_err());
    }
}
