//! Two-table cache model: action records + content-addressed blobs.
//!
//! Following Bazel's action-cache/CAS split and Nectar's
//! data-computation duality:
//!
//! - An **action record** ([`ActionResult`]) is a small JSON document
//!   keyed by the computation's provenance ([`crate::cache::CacheKey`]:
//!   config + state + input hashes). It names the output by *content*
//!   hash and carries the metadata GC needs (compute cost, size,
//!   provenance, timestamps).
//! - A **blob** is the output's bytes, stored once under its
//!   [`ContentHash`] — identical outputs from different actions
//!   deduplicate automatically.
//!
//! Because the record is tiny and the blob is regenerable (re-run the
//! action), eviction can delete blobs while keeping records: a later
//! run recomputes the value, re-fills the same content address, and
//! every other record pointing at it becomes servable again. Eviction
//! degrades performance, never correctness.

use crate::cache::{CacheKey, Origin};
use crate::error::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Hash algorithm of a [`ContentHash`], self-describing (multihash
/// lesson from IPFS: bake the algorithm into the address so a future
/// migration needs no flag day).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum HashAlgo {
    /// BLAKE3 — default for payload hashing (parallel tree hashing,
    /// several GB/s single-threaded).
    Blake3,
    /// SHA-256 — for interop where required.
    Sha256,
}

impl HashAlgo {
    /// Short directory-safe prefix used in store layouts.
    pub fn prefix(&self) -> &'static str {
        match self {
            HashAlgo::Blake3 => "b3",
            HashAlgo::Sha256 => "s2",
        }
    }
}

/// Address of a blob: the hash of its bytes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ContentHash {
    pub algo: HashAlgo,
    pub digest: [u8; 32],
}

impl ContentHash {
    /// Hash bytes with BLAKE3 (the default payload algorithm).
    pub fn blake3(bytes: &[u8]) -> Self {
        Self {
            algo: HashAlgo::Blake3,
            digest: *blake3::hash(bytes).as_bytes(),
        }
    }

    /// Hash bytes with SHA-256.
    pub fn sha256(bytes: &[u8]) -> Self {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        Self {
            algo: HashAlgo::Sha256,
            digest: hasher.finalize().into(),
        }
    }

    /// Verify `bytes` against this address (content addresses are
    /// self-verifying — no signatures needed).
    pub fn verify(&self, bytes: &[u8]) -> bool {
        let recomputed = match self.algo {
            HashAlgo::Blake3 => Self::blake3(bytes),
            HashAlgo::Sha256 => Self::sha256(bytes),
        };
        recomputed.digest == self.digest
    }

    pub fn to_hex(&self) -> String {
        self.digest.iter().map(|b| format!("{b:02x}")).collect()
    }
}

impl std::fmt::Debug for ContentHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ContentHash({}:{}...)",
            self.algo.prefix(),
            &self.to_hex()[..12]
        )
    }
}

/// The record of one completed computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionResult {
    /// Provenance key: `hash(config + state + input)` (or the state-key
    /// form for `fit` results).
    pub key: CacheKey,
    /// Output name → content hash. The runtime uses a single `"output"`
    /// entry today; the map form leaves room for multi-output actions.
    pub outputs: BTreeMap<String, ContentHash>,
    /// Total encoded size of the outputs in bytes.
    pub output_bytes: u64,
    /// Wall-clock cost of the computation, for cost-aware eviction
    /// (a tiny value that took days must outlive a huge one that took
    /// seconds).
    pub compute_ms: u64,
    /// Whether the producing filter declared its forward deterministic.
    pub deterministic: bool,
    pub origin: Origin,
    pub created_at: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
}

/// Store of action records (the small table).
pub trait ActionCache: Send + Sync {
    fn get_action(&self, key: &CacheKey) -> Result<Option<ActionResult>>;
    fn put_action(&self, result: &ActionResult) -> Result<()>;
}

/// Store of content-addressed blobs (the big table).
pub trait BlobStore: Send + Sync {
    /// Store bytes under their content hash. Idempotent: storing the
    /// same bytes twice is a no-op.
    fn put_bytes(&self, bytes: &[u8]) -> Result<ContentHash>;
    fn get_bytes(&self, hash: &ContentHash) -> Result<Option<Vec<u8>>>;
    fn contains(&self, hash: &ContentHash) -> Result<bool>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blake3_content_hash_roundtrip() {
        let h = ContentHash::blake3(b"hello world");
        assert!(h.verify(b"hello world"));
        assert!(!h.verify(b"hello worlds"));
        assert_eq!(h, ContentHash::blake3(b"hello world"));
        assert_ne!(h, ContentHash::blake3(b"other"));
    }

    #[test]
    fn algo_is_part_of_identity() {
        let b = ContentHash::blake3(b"data");
        let s = ContentHash::sha256(b"data");
        assert_ne!(b, s);
        assert_eq!(b.algo.prefix(), "b3");
        assert_eq!(s.algo.prefix(), "s2");
    }

    #[test]
    fn action_result_serde_roundtrip() {
        let mut outputs = BTreeMap::new();
        outputs.insert("output".to_string(), ContentHash::blake3(b"payload"));
        let record = ActionResult {
            key: CacheKey::hash_data(b"action"),
            outputs,
            output_bytes: 7,
            compute_ms: 123_456,
            deterministic: true,
            origin: Origin::Computed {
                node_id: "n".into(),
                run_id: "r".into(),
            },
            created_at: Utc::now(),
            last_accessed: Utc::now(),
        };
        let json = serde_json::to_string(&record).unwrap();
        let back: ActionResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.key, record.key);
        assert_eq!(back.outputs, record.outputs);
        assert_eq!(back.compute_ms, 123_456);
    }
}
