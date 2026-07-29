//! Content-addressable caching — keys, traits, and metadata.
//!
//! [`CacheKey`] is a SHA-256 hash of computation inputs. Two cache keys:
//! - **State key**: `hash(config + training_data)` — for fit() results
//! - **Output key**: `hash(config + state + input)` — for forward() results
//!
//! [`CacheStore`] is the K/V interface; implementations live in soma-runtime.

use crate::error::{Result, SomaError};
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

    /// Hash a [`Value`]'s serialized content.
    ///
    /// Errors when the value cannot be serialized — an unhashable value
    /// must mean "uncacheable", never a silent hash of empty bytes
    /// (which would make two different unserializable values collide).
    pub fn for_value(value: &Value) -> Result<Self> {
        let bytes = serde_json::to_vec(value)
            .map_err(|e| SomaError::Cache(format!("value not hashable: {e}")))?;
        Ok(Self::hash_data(&bytes))
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

        assert_eq!(
            CacheKey::for_value(&v1).unwrap(),
            CacheKey::for_value(&v2).unwrap()
        );
        // Same data, different shape → different hash
        assert_ne!(
            CacheKey::for_value(&v1).unwrap(),
            CacheKey::for_value(&v3).unwrap()
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
