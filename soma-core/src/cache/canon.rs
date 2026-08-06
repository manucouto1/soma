//! Canonical, deterministic serialization for cache-key hashing.
//!
//! Cache keys must be identical across processes, machines, and library
//! versions. Ordinary serializer output is not: `serde_json` encodes
//! `HashMap`s in random iteration order, and pickle-style formats depend
//! on traversal order and library version. This module produces a
//! canonical CBOR encoding (RFC 8949 §4.2 core deterministic encoding)
//! with dCBOR-style float canonicalization:
//!
//! - map keys sorted bytewise by their encoded form (duplicates are an error)
//! - a single canonical NaN bit pattern
//! - `-0.0` normalized to `+0.0`
//!
//! Use [`canonical_bytes`] / [`hash_canonical`] for anything that feeds a
//! [`CacheKey`]; never hash raw serializer output.

use crate::cache::CacheKey;
use crate::error::{Result, SomaError};
use ciborium::value::Value as Cbor;
use serde::Serialize;

/// Encode `value` to canonical CBOR bytes.
///
/// Errors when the value cannot be serialized — for cache keys that must
/// mean "uncacheable", never a silent fallback encoding.
pub fn canonical_bytes<T: Serialize + ?Sized>(value: &T) -> Result<Vec<u8>> {
    let mut raw = Vec::new();
    ciborium::ser::into_writer(value, &mut raw)
        .map_err(|e| SomaError::Cache(format!("not canonically serializable: {e}")))?;
    let decoded: Cbor = ciborium::de::from_reader(raw.as_slice())
        .map_err(|e| SomaError::Cache(format!("canonical re-decode failed: {e}")))?;
    let canon = canonicalize(decoded)?;
    encode(&canon)
}

/// Hash `value`'s canonical CBOR encoding.
pub fn hash_canonical<T: Serialize + ?Sized>(value: &T) -> Result<CacheKey> {
    Ok(CacheKey::hash_data(&canonical_bytes(value)?))
}

fn encode(v: &Cbor) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    ciborium::ser::into_writer(v, &mut buf)
        .map_err(|e| SomaError::Cache(format!("canonical encode failed: {e}")))?;
    Ok(buf)
}

fn canonicalize(v: Cbor) -> Result<Cbor> {
    Ok(match v {
        Cbor::Float(f) => Cbor::Float(canonical_float(f)),
        Cbor::Array(items) => {
            Cbor::Array(items.into_iter().map(canonicalize).collect::<Result<_>>()?)
        }
        Cbor::Map(entries) => {
            let mut encoded: Vec<(Vec<u8>, Cbor, Cbor)> = Vec::with_capacity(entries.len());
            for (key, value) in entries {
                let key = canonicalize(key)?;
                let value = canonicalize(value)?;
                let key_bytes = encode(&key)?;
                encoded.push((key_bytes, key, value));
            }
            encoded.sort_by(|a, b| a.0.cmp(&b.0));
            for pair in encoded.windows(2) {
                if pair[0].0 == pair[1].0 {
                    return Err(SomaError::Cache(
                        "duplicate map key in canonical encoding".into(),
                    ));
                }
            }
            Cbor::Map(encoded.into_iter().map(|(_, k, v)| (k, v)).collect())
        }
        Cbor::Tag(tag, inner) => Cbor::Tag(tag, Box::new(canonicalize(*inner)?)),
        other => other,
    })
}

/// dCBOR float rules: one NaN bit pattern, no negative zero.
fn canonical_float(f: f64) -> f64 {
    if f.is_nan() {
        f64::NAN
    } else if f == 0.0 {
        0.0
    } else {
        f
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn hashmap_encoding_is_order_independent() {
        // The historical bug: serde_json::to_vec of a HashMap is
        // iteration-order dependent. Canonical CBOR must not be.
        let mut reference: Option<Vec<u8>> = None;
        for i in 0..100 {
            let mut map = HashMap::new();
            // Insert in varying orders to shuffle bucket layouts.
            let keys = ["alpha", "beta", "gamma", "delta", "epsilon"];
            for (j, _k) in keys.iter().enumerate() {
                let idx = (i + j) % keys.len();
                map.insert(keys[idx].to_string(), idx as i64);
            }
            let bytes = canonical_bytes(&map).unwrap();
            match &reference {
                None => reference = Some(bytes),
                Some(r) => assert_eq!(r, &bytes, "iteration {i} diverged"),
            }
        }
    }

    #[test]
    fn nested_structures_canonicalize() {
        #[derive(serde::Serialize)]
        struct Config {
            name: String,
            params: HashMap<String, f64>,
            layers: Vec<u32>,
        }
        let mut params = HashMap::new();
        params.insert("lr".into(), 0.001);
        params.insert("momentum".into(), 0.9);
        let a = Config {
            name: "m".into(),
            params: params.clone(),
            layers: vec![64, 32],
        };
        let b = Config {
            name: "m".into(),
            params,
            layers: vec![64, 32],
        };
        assert_eq!(hash_canonical(&a).unwrap(), hash_canonical(&b).unwrap());
    }

    #[test]
    fn negative_zero_normalizes() {
        assert_eq!(
            canonical_bytes(&(-0.0f64)).unwrap(),
            canonical_bytes(&0.0f64).unwrap()
        );
    }

    #[test]
    fn nan_payloads_collapse_to_one_encoding() {
        let quiet = f64::NAN;
        let payload = f64::from_bits(0x7ff8_0000_0000_0001);
        assert!(payload.is_nan());
        assert_eq!(
            canonical_bytes(&quiet).unwrap(),
            canonical_bytes(&payload).unwrap()
        );
    }

    #[test]
    fn distinct_values_distinct_hashes() {
        assert_ne!(
            hash_canonical(&vec![1.0f64, 2.0]).unwrap(),
            hash_canonical(&vec![2.0f64, 1.0]).unwrap()
        );
        assert_ne!(hash_canonical(&"a").unwrap(), hash_canonical(&"b").unwrap());
    }

    #[test]
    fn float_and_int_do_not_collide() {
        // 1u64 and 1.0f64 encode differently in CBOR (major type 0 vs 7).
        assert_ne!(
            canonical_bytes(&1u64).unwrap(),
            canonical_bytes(&1.0f64).unwrap()
        );
    }
}
