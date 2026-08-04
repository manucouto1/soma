//! Cost-aware garbage collection for [`FsActionStore`].
//!
//! Evicts **blobs only** — action records are always retained, so an
//! evicted entry is regenerable: the next run recomputes it and
//! re-fills the same content address (Nectar: eviction degrades
//! performance, never correctness).
//!
//! Eviction order is by *value density*, ascending:
//!
//! ```text
//! score(blob) = max over records naming it of
//!               (compute_ms + 1) × recency_weight ÷ size_bytes
//! ```
//!
//! so a 100-byte state that took two days to fit outlives a 10 GB
//! intermediate that took two minutes, and never the other way around
//! (plain LRU gets this exactly wrong for research pipelines). Blobs
//! referenced by pinned actions are GC roots and never evicted.

use crate::cache::fs_store::FsActionStore;
use chrono::Utc;
use somatize_core::action::{BlobStore, ContentHash};
use somatize_core::error::Result;
use std::collections::{HashMap, HashSet};

/// When to evict and how much to keep. Defaults: 20 GiB ceiling,
/// one-hour minimum age.
#[derive(Debug, Clone)]
pub struct GcPolicy {
    /// Target ceiling for total CAS bytes.
    pub max_bytes: u64,
    /// Blobs younger than this are never evicted (avoids racing an
    /// in-flight run that just wrote them).
    pub min_age: std::time::Duration,
}

impl Default for GcPolicy {
    fn default() -> Self {
        Self {
            max_bytes: 20 * 1024 * 1024 * 1024, // 20 GiB
            min_age: std::time::Duration::from_secs(3600),
        }
    }
}

/// What one [`collect`] pass did, for `soma cache gc` output.
#[derive(Debug, Clone, Default)]
pub struct GcReport {
    /// Total CAS bytes before the pass.
    pub bytes_before: u64,
    /// Total CAS bytes after the pass.
    pub bytes_after: u64,
    /// Blobs deleted by this pass.
    pub blobs_evicted: usize,
    /// Blobs that survived (including roots).
    pub blobs_kept: usize,
    /// Blobs protected as outputs of pinned actions.
    pub pinned_blobs: usize,
}

/// Run one collection pass. Safe to run while writers are active:
/// blob puts are idempotent, and an evict-then-immediate-reput just
/// re-creates the file.
pub fn collect(store: &FsActionStore, policy: &GcPolicy) -> Result<GcReport> {
    let bytes_before = store.cas_bytes()?;
    let mut report = GcReport {
        bytes_before,
        bytes_after: bytes_before,
        ..Default::default()
    };
    if bytes_before <= policy.max_bytes {
        return Ok(report);
    }

    // Roots: outputs of pinned actions.
    let pinned_keys: HashSet<_> = store.pinned()?.into_iter().collect();
    let mut roots: HashSet<ContentHash> = HashSet::new();

    // Best score per blob across all records naming it.
    let now = Utc::now();
    let mut scores: HashMap<ContentHash, (f64, u64, chrono::DateTime<Utc>)> = HashMap::new();
    for record in store.actions()? {
        let pinned = pinned_keys.contains(&record.key);
        let age_days = (now - record.last_accessed).num_seconds().max(0) as f64 / 86_400.0;
        let recency = 1.0 / (1.0 + age_days);
        let size = record.output_bytes.max(1) as f64;
        let score = (record.compute_ms as f64 + 1.0) * recency / size;
        for hash in record.outputs.values() {
            if pinned {
                roots.insert(*hash);
            }
            let entry =
                scores
                    .entry(*hash)
                    .or_insert((f64::MIN, record.output_bytes, record.created_at));
            if score > entry.0 {
                entry.0 = score;
            }
            if record.created_at > entry.2 {
                entry.2 = record.created_at;
            }
        }
    }
    report.pinned_blobs = roots.len();

    let min_age = chrono::Duration::from_std(policy.min_age).unwrap_or(chrono::Duration::zero());
    let mut candidates: Vec<(f64, u64, ContentHash)> = scores
        .iter()
        .filter(|(hash, (_, _, created))| !roots.contains(hash) && now - *created >= min_age)
        .map(|(hash, (score, size, _))| (*score, *size, *hash))
        .collect();
    // Lowest value density first.
    candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut bytes = bytes_before;
    for (_, size, hash) in candidates {
        if bytes <= policy.max_bytes {
            break;
        }
        if store.contains(&hash)? {
            store.evict_blob(&hash)?;
            bytes = bytes.saturating_sub(size);
            report.blobs_evicted += 1;
        }
    }
    report.blobs_kept = scores.len() - report.blobs_evicted;
    report.bytes_after = store.cas_bytes()?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use somatize_core::cache::{CacheKey, CacheStore, Origin};
    use somatize_core::value::Value;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_root() -> PathBuf {
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("soma_gc_{}_{id}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn origin() -> Origin {
        Origin::Computed {
            node_id: "n".into(),
            run_id: "r".into(),
        }
    }

    fn policy(max_bytes: u64) -> GcPolicy {
        GcPolicy {
            max_bytes,
            min_age: Duration::ZERO,
        }
    }

    #[test]
    fn evicts_cheap_large_before_expensive_small() {
        let root = temp_root();
        let store = FsActionStore::new(&root).unwrap();

        // Expensive tiny state (2 days of compute, ~100 bytes).
        let expensive_key = CacheKey::hash_data(b"expensive");
        store
            .put_computed(
                &expensive_key,
                &Value::tensor(vec![1.0; 8], vec![8]),
                &origin(),
                Duration::from_secs(2 * 86_400),
                true,
            )
            .unwrap();

        // Cheap huge intermediate (2 seconds, ~80 KB).
        let cheap_key = CacheKey::hash_data(b"cheap");
        store
            .put_computed(
                &cheap_key,
                &Value::tensor(vec![2.0; 10_000], vec![10_000]),
                &origin(),
                Duration::from_secs(2),
                true,
            )
            .unwrap();

        // Budget forces evicting roughly one blob.
        let report = collect(&store, &policy(10_000)).unwrap();
        assert_eq!(report.blobs_evicted, 1);

        assert!(
            store.get(&expensive_key).unwrap().is_some(),
            "the expensive-per-byte state must survive"
        );
        assert!(
            store.get(&cheap_key).unwrap().is_none(),
            "the cheap-per-byte bulk must go first"
        );
        // Records survive eviction — regenerable, not lost.
        use somatize_core::action::ActionCache;
        assert!(store.get_action(&cheap_key).unwrap().is_some());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn pinned_blobs_are_roots() {
        let root = temp_root();
        let store = FsActionStore::new(&root).unwrap();
        let key = CacheKey::hash_data(b"best");
        store
            .put_computed(
                &key,
                &Value::tensor(vec![3.0; 10_000], vec![10_000]),
                &origin(),
                Duration::from_millis(1),
                true,
            )
            .unwrap();
        store.pin("best-model", &key).unwrap();

        let report = collect(&store, &policy(1)).unwrap();
        assert_eq!(report.blobs_evicted, 0);
        assert!(store.get(&key).unwrap().is_some());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn under_budget_is_a_noop() {
        let root = temp_root();
        let store = FsActionStore::new(&root).unwrap();
        let key = CacheKey::hash_data(b"small");
        store.put(&key, &Value::tensor(vec![1.0], vec![1])).unwrap();

        let report = collect(&store, &policy(u64::MAX)).unwrap();
        assert_eq!(report.blobs_evicted, 0);
        assert_eq!(report.bytes_before, report.bytes_after);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn min_age_protects_fresh_blobs() {
        let root = temp_root();
        let store = FsActionStore::new(&root).unwrap();
        let key = CacheKey::hash_data(b"fresh");
        store
            .put_computed(
                &key,
                &Value::tensor(vec![1.0; 10_000], vec![10_000]),
                &origin(),
                Duration::from_millis(1),
                true,
            )
            .unwrap();

        let fresh_policy = GcPolicy {
            max_bytes: 1,
            min_age: Duration::from_secs(3600),
        };
        let report = collect(&store, &fresh_policy).unwrap();
        assert_eq!(
            report.blobs_evicted, 0,
            "freshly-written blobs are protected"
        );

        let _ = std::fs::remove_dir_all(&root);
    }
}
