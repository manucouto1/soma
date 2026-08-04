//! Content-addressable caching — keys, traits, and metadata.
//!
//! [`CacheKey`] is a SHA-256 hash of computation inputs. Two cache keys:
//! - **State key**: `hash(config + training_data)` — for fit() results
//! - **Output key**: `hash(config + state + input)` — for forward() results
//!
//! [`CacheStore`] is the K/V interface; implementations live in soma-runtime.

use crate::error::Result;
use crate::value::Value;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;

/// Content-addressable hash identifying a computation.
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CacheKey(pub [u8; 32]);

impl CacheKey {
    /// Create a cache key by hashing arbitrary byte slices.
    pub fn from_parts(parts: &[&[u8]]) -> Self {
        let mut hasher = Sha256::new();
        for part in parts {
            // Length-prefix each part to avoid collisions between
            // concat("ab", "c") and concat("a", "bc")
            hasher.update((part.len() as u64).to_le_bytes());
            hasher.update(part);
        }
        Self(hasher.finalize().into())
    }

    /// Create a cache key for a filter's trained state.
    /// key = hash(filter_config_hash + x_hash [+ y_hash])
    ///
    /// The labels `y` are part of the key: the same features trained
    /// against different labels must never collide. `None` and
    /// `Some(...)` always produce distinct keys (different part counts,
    /// and every part is length-prefixed).
    pub fn for_state(config_hash: &CacheKey, x_hash: &CacheKey, y_hash: Option<&CacheKey>) -> Self {
        match y_hash {
            Some(y) => Self::from_parts(&[&config_hash.0, &x_hash.0, b"y", &y.0]),
            None => Self::from_parts(&[&config_hash.0, &x_hash.0]),
        }
    }

    /// Create a cache key for a filter's output.
    /// key = hash(filter_config_hash + state_hash + input_data_hash)
    pub fn for_output(
        config_hash: &CacheKey,
        state_hash: &CacheKey,
        input_hash: &CacheKey,
    ) -> Self {
        Self::from_parts(&[&config_hash.0, &state_hash.0, &input_hash.0])
    }

    /// Hash arbitrary serializable data.
    pub fn hash_data(data: &[u8]) -> Self {
        Self::from_parts(&[data])
    }

    /// Hash a [`Value`] for use as cache-key material.
    ///
    /// Not `hash_data(serde_json::to_vec(value))`, which is what the
    /// runtime used to do. JSON has no way to write a non-finite float:
    /// `serde_json` turns NaN *and* every infinity into `null`, silently.
    /// A tensor of NaN and a tensor of +∞ therefore serialized to the same
    /// bytes, hashed to the same key, and the second one was answered with
    /// the first one's cached output.
    ///
    /// Floats are hashed by their bit pattern instead, so every distinct
    /// value gets a distinct key. Two consequences worth knowing: the two
    /// NaN encodings are different keys (they are different bit patterns),
    /// and `0.0` and `-0.0` are different keys too. Both are the safe
    /// direction — a redundant miss costs a recomputation, a false hit
    /// costs a wrong answer.
    pub fn for_value(value: &Value) -> Self {
        let mut hasher = Sha256::new();
        Self::absorb(&mut hasher, value);
        Self(hasher.finalize().into())
    }

    fn absorb(hasher: &mut Sha256, value: &Value) {
        // A leading tag per variant keeps `Bytes(b"x")` and `Object(b"x")`
        // apart, and a length prefix keeps concatenations apart.
        match value {
            Value::Tensor { values, shape } => {
                hasher.update([0u8]);
                hasher.update((shape.len() as u64).to_le_bytes());
                for dim in shape {
                    hasher.update((*dim as u64).to_le_bytes());
                }
                hasher.update((values.len() as u64).to_le_bytes());
                for v in values.iter() {
                    hasher.update(v.to_bits().to_le_bytes());
                }
            }
            Value::Text(text) => {
                hasher.update([5u8]);
                hasher.update((text.len() as u64).to_le_bytes());
                hasher.update(text.as_bytes());
            }
            Value::Json(json) => {
                hasher.update([1u8]);
                // `serde_json::Value` cannot hold a non-finite number, and
                // its object maps are ordered, so this round-trip is both
                // lossless and deterministic.
                let bytes = serde_json::to_vec(json.as_ref()).unwrap_or_default();
                hasher.update((bytes.len() as u64).to_le_bytes());
                hasher.update(&bytes);
            }
            Value::Bytes(bytes) => {
                hasher.update([2u8]);
                hasher.update((bytes.len() as u64).to_le_bytes());
                hasher.update(bytes.as_slice());
            }
            Value::Object(bytes) => {
                hasher.update([3u8]);
                hasher.update((bytes.len() as u64).to_le_bytes());
                hasher.update(bytes.as_slice());
            }
            Value::Empty => hasher.update([4u8]),
            // No catch-all on purpose. `Value` is `#[non_exhaustive]`
            // downstream but not here, so a new variant fails to compile
            // until someone decides how it hashes — which beats it
            // silently sharing a key with whatever the fallback picked.
        }
    }

    /// Returns the hex representation.
    pub fn to_hex(&self) -> String {
        self.0.iter().map(|b| format!("{b:02x}")).collect()
    }
}

impl fmt::Debug for CacheKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CacheKey({}...)", &self.to_hex()[..12])
    }
}

impl fmt::Display for CacheKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", &self.to_hex()[..16])
    }
}

/// Which storage tier a cached entry lives in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheTier {
    Memory,
    Local,
    Remote,
}

/// Where a cached value originated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Origin {
    Computed {
        node_id: String,
        run_id: String,
    },
    Ingested {
        source: String,
    },
    Streamed {
        window_start: DateTime<Utc>,
        window_end: DateTime<Utc>,
    },
}

/// Metadata about a cached entry, queryable without loading the value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntryMeta {
    pub key: CacheKey,
    pub size_bytes: u64,
    pub created_at: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
    pub ttl: Option<std::time::Duration>,
    pub origin: Origin,
}

/// The K/V cache store interface.
///
/// Implementations may be in-memory, on-disk (RocksDB/sled),
/// or remote (S3). The tiered cache composes multiple stores.
pub trait CacheStore: Send + Sync {
    fn get(&self, key: &CacheKey) -> Result<Option<Value>>;
    fn put(&self, key: &CacheKey, value: &Value) -> Result<()>;
    fn exists(&self, key: &CacheKey) -> Result<bool>;
    fn remove(&self, key: &CacheKey) -> Result<()>;
    fn metadata(&self, key: &CacheKey) -> Result<Option<EntryMeta>>;

    /// Store a value together with its provenance. Stores that persist
    /// metadata should override this; the default discards the origin.
    fn put_with_origin(&self, key: &CacheKey, value: &Value, origin: &Origin) -> Result<()> {
        let _ = origin;
        self.put(key, value)
    }

    /// Store a freshly-computed value with its full provenance record:
    /// origin, wall-clock compute cost, and the producer's determinism
    /// declaration. Cost-aware eviction needs the compute time — a tiny
    /// value that took days must outlive a huge one that took seconds.
    /// The default discards the extra metadata.
    fn put_computed(
        &self,
        key: &CacheKey,
        value: &Value,
        origin: &Origin,
        compute: std::time::Duration,
        deterministic: bool,
    ) -> Result<()> {
        let _ = (compute, deterministic);
        self.put_with_origin(key, value, origin)
    }

    /// Which tier this store is, for reporting.
    ///
    /// A single-tier store answers with its own kind. [`CacheTier::Memory`]
    /// is the default because the in-memory store is the one people write
    /// by hand; a store that is anything else should say so.
    fn tier(&self) -> CacheTier {
        CacheTier::Memory
    }

    /// Like [`CacheStore::get`], but also reports which tier served the value.
    ///
    /// A composed store overrides this — that is the whole point. Without
    /// it, a hit served from disk is indistinguishable from one served from
    /// RAM, and the numbers that are supposed to tell you whether the disk
    /// tier is earning its keep say only that the cache was used.
    fn get_located(&self, key: &CacheKey) -> Result<Option<(Value, CacheTier)>> {
        Ok(self.get(key)?.map(|value| (value, self.tier())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_key_deterministic() {
        let k1 = CacheKey::from_parts(&[b"hello", b"world"]);
        let k2 = CacheKey::from_parts(&[b"hello", b"world"]);
        assert_eq!(k1, k2);
    }

    #[test]
    fn cache_key_sensitive_to_content() {
        let k1 = CacheKey::from_parts(&[b"hello", b"world"]);
        let k2 = CacheKey::from_parts(&[b"hello", b"world!"]);
        assert_ne!(k1, k2);
    }

    #[test]
    fn cache_key_sensitive_to_part_boundaries() {
        // "ab" + "c" must differ from "a" + "bc"
        let k1 = CacheKey::from_parts(&[b"ab", b"c"]);
        let k2 = CacheKey::from_parts(&[b"a", b"bc"]);
        assert_ne!(k1, k2);
    }

    #[test]
    fn cache_key_for_state() {
        let config = CacheKey::hash_data(b"scaler_config");
        let data = CacheKey::hash_data(b"training_data");
        let state_key = CacheKey::for_state(&config, &data, None);

        // Same inputs → same key
        let state_key2 = CacheKey::for_state(&config, &data, None);
        assert_eq!(state_key, state_key2);

        // Different data → different key
        let data2 = CacheKey::hash_data(b"different_data");
        let state_key3 = CacheKey::for_state(&config, &data2, None);
        assert_ne!(state_key, state_key3);
    }

    #[test]
    fn cache_key_for_state_sensitive_to_labels() {
        let config = CacheKey::hash_data(b"config");
        let x = CacheKey::hash_data(b"features");
        let y1 = CacheKey::hash_data(b"labels_a");
        let y2 = CacheKey::hash_data(b"labels_b");

        let unsupervised = CacheKey::for_state(&config, &x, None);
        let supervised_a = CacheKey::for_state(&config, &x, Some(&y1));
        let supervised_b = CacheKey::for_state(&config, &x, Some(&y2));

        assert_ne!(unsupervised, supervised_a);
        assert_ne!(supervised_a, supervised_b);
    }

    #[test]
    fn for_value_deterministic_and_sensitive() {
        let v1 = Value::tensor(vec![1.0, 2.0], vec![2]);
        let v2 = Value::tensor(vec![1.0, 2.0], vec![2]);
        let v3 = Value::tensor(vec![1.0, 2.0], vec![1, 2]);

        assert_eq!(CacheKey::for_value(&v1), CacheKey::for_value(&v2));
        // Same data, different shape → different hash
        assert_ne!(CacheKey::for_value(&v1), CacheKey::for_value(&v3));
    }

    /// JSON writes NaN and both infinities as `null`, so hashing a value's
    /// JSON gave three distinct tensors one key — and the second one was
    /// answered with the first one's cached output.
    #[test]
    fn for_value_separates_non_finite_floats() {
        let nan = Value::tensor(vec![f64::NAN], vec![1]);
        let pos = Value::tensor(vec![f64::INFINITY], vec![1]);
        let neg = Value::tensor(vec![f64::NEG_INFINITY], vec![1]);

        assert_eq!(
            serde_json::to_vec(&nan).unwrap(),
            serde_json::to_vec(&pos).unwrap(),
            "the premise: JSON really does flatten these together"
        );

        assert_ne!(CacheKey::for_value(&nan), CacheKey::for_value(&pos));
        assert_ne!(CacheKey::for_value(&pos), CacheKey::for_value(&neg));
        // And -0.0 is not 0.0, for the same reason.
        assert_ne!(
            CacheKey::for_value(&Value::tensor(vec![0.0], vec![1])),
            CacheKey::for_value(&Value::tensor(vec![-0.0], vec![1]))
        );
    }

    /// A tag per variant: two variants that wrap the same bytes are
    /// different values and must not share a key.
    #[test]
    fn for_value_separates_variants_holding_the_same_bytes() {
        assert_ne!(
            CacheKey::for_value(&Value::Bytes(std::sync::Arc::new(b"x".to_vec()))),
            CacheKey::for_value(&Value::Object(std::sync::Arc::new(b"x".to_vec())))
        );
    }

    #[test]
    fn cache_key_for_output() {
        let config = CacheKey::hash_data(b"config");
        let state = CacheKey::hash_data(b"state");
        let input = CacheKey::hash_data(b"input");
        let key = CacheKey::for_output(&config, &state, &input);

        // Different state → different key
        let state2 = CacheKey::hash_data(b"state2");
        let key2 = CacheKey::for_output(&config, &state2, &input);
        assert_ne!(key, key2);
    }

    #[test]
    fn cache_key_hex_and_display() {
        let key = CacheKey::hash_data(b"test");
        let hex = key.to_hex();
        assert_eq!(hex.len(), 64); // 32 bytes = 64 hex chars

        let display = format!("{key}");
        assert_eq!(display.len(), 16); // truncated display

        let debug = format!("{key:?}");
        assert!(debug.starts_with("CacheKey("));
    }

    #[test]
    fn cache_key_serde_roundtrip() {
        let key = CacheKey::hash_data(b"test_data");
        let json = serde_json::to_string(&key).unwrap();
        let deserialized: CacheKey = serde_json::from_str(&json).unwrap();
        assert_eq!(key, deserialized);
    }
}
