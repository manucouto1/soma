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
    /// key = hash(filter_config_hash + training_data_hash)
    pub fn for_state(config_hash: &CacheKey, data_hash: &CacheKey) -> Self {
        Self::from_parts(&[&config_hash.0, &data_hash.0])
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
        let state_key = CacheKey::for_state(&config, &data);

        // Same inputs → same key
        let state_key2 = CacheKey::for_state(&config, &data);
        assert_eq!(state_key, state_key2);

        // Different data → different key
        let data2 = CacheKey::hash_data(b"different_data");
        let state_key3 = CacheKey::for_state(&config, &data2);
        assert_ne!(state_key, state_key3);
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
