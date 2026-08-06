//! `FsActionStore` — the persistent two-table cache (action records + CAS).
//!
//! Layout under the store root:
//!
//! ```text
//! format.json                      {"version": 2}
//! actions/<aa>/<key-hex>.json      ActionResult records (small)
//! cas/<algo>/<aa>/<hash-hex>.bin   SOMA1-encoded blobs (deduplicated)
//! pins/<name>                      GC roots: files containing an action key hex
//! ```
//!
//! Commit protocol (crash-safe by construction):
//! 1. blob written first (idempotent temp+fsync+rename — concurrent
//!    same-content writers race benignly),
//! 2. action record renamed into place **last** — the record is the
//!    commit point. A crash in between leaves an orphan blob, never a
//!    record pointing at missing required state.
//!
//! Eviction deletes blobs only; records are retained so an evicted
//! entry is recomputed and re-fills the same content address
//! (Nectar: eviction degrades performance, never correctness).

use chrono::Utc;
use somatize_core::cache::action::{ActionCache, ActionResult, BlobStore, ContentHash};
use somatize_core::cache::{CacheKey, CacheStore, EntryMeta, Origin};
use somatize_core::data::codec::{decode_value, encode_and_hash};
use somatize_core::data::value::Value;
use somatize_core::error::{Result, SomaError};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// On-disk layout version, recorded in `format.json`. A mismatched dir is
/// refused with a pointer to `soma cache purge-v1` — silently reading an
/// old layout would corrupt it.
pub const FORMAT_VERSION: u32 = 2;

/// The persistent two-table store: action records + CAS blobs. See the
/// module docs for the layout and the crash-safe commit protocol.
pub struct FsActionStore {
    root: PathBuf,
}

impl FsActionStore {
    /// Open (or initialize) a store rooted at `root`.
    ///
    /// Creates the directory skeleton and stamps `format.json` on first
    /// use; refuses a root whose recorded version is not
    /// [`FORMAT_VERSION`].
    pub fn new(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        fs::create_dir_all(&root)?;

        let format_path = root.join("format.json");
        if format_path.exists() {
            let raw = fs::read_to_string(&format_path)?;
            let format: serde_json::Value = serde_json::from_str(&raw)
                .map_err(|e| SomaError::Cache(format!("unreadable cache format.json: {e}")))?;
            let version = format.get("version").and_then(|v| v.as_u64()).unwrap_or(0);
            if version != FORMAT_VERSION as u64 {
                return Err(SomaError::Cache(format!(
                    "cache dir {} has format version {version}, expected {FORMAT_VERSION}; \
                     run `soma cache purge-v1` or point SOMA_CACHE_DIR elsewhere",
                    root.display()
                )));
            }
        } else {
            write_atomic(
                &format_path,
                serde_json::json!({ "version": FORMAT_VERSION })
                    .to_string()
                    .as_bytes(),
            )?;
        }
        fs::create_dir_all(root.join("actions"))?;
        fs::create_dir_all(root.join("cas"))?;
        fs::create_dir_all(root.join("pins"))?;
        Ok(Self { root })
    }

    /// The store's root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn action_path(&self, key: &CacheKey) -> PathBuf {
        let hex = key.to_hex();
        self.root
            .join("actions")
            .join(&hex[..2])
            .join(format!("{hex}.json"))
    }

    fn blob_path(&self, hash: &ContentHash) -> PathBuf {
        let hex = hash.to_hex();
        self.root
            .join("cas")
            .join(hash.algo.prefix())
            .join(&hex[..2])
            .join(format!("{hex}.bin"))
    }

    /// Pin an action as a GC root under a human-readable name.
    pub fn pin(&self, name: &str, key: &CacheKey) -> Result<()> {
        if name.contains(['/', '\\']) || name.starts_with('.') {
            return Err(SomaError::Cache(format!("invalid pin name: {name:?}")));
        }
        write_atomic(&self.root.join("pins").join(name), key.to_hex().as_bytes())
    }

    /// All pinned action keys.
    pub fn pinned(&self) -> Result<Vec<CacheKey>> {
        let mut keys = Vec::new();
        let pins = self.root.join("pins");
        if !pins.exists() {
            return Ok(keys);
        }
        for entry in fs::read_dir(&pins)? {
            let entry = entry?;
            let hex = fs::read_to_string(entry.path())?;
            if let Some(key) = key_from_hex(hex.trim()) {
                keys.push(key);
            }
        }
        Ok(keys)
    }

    /// Iterate every action record in the store.
    pub fn actions(&self) -> Result<Vec<ActionResult>> {
        let mut out = Vec::new();
        let actions = self.root.join("actions");
        if !actions.exists() {
            return Ok(out);
        }
        for shard in fs::read_dir(&actions)? {
            let shard = shard?.path();
            if !shard.is_dir() {
                continue;
            }
            for entry in fs::read_dir(&shard)? {
                let path = entry?.path();
                if path.extension().is_some_and(|e| e == "json")
                    && let Ok(raw) = fs::read_to_string(&path)
                    && let Ok(record) = serde_json::from_str::<ActionResult>(&raw)
                {
                    out.push(record);
                }
            }
        }
        Ok(out)
    }

    /// Remove a blob (eviction). The action records naming it stay —
    /// the value is regenerable by re-running the computation.
    pub fn evict_blob(&self, hash: &ContentHash) -> Result<()> {
        let path = self.blob_path(hash);
        if path.exists() {
            fs::remove_file(&path)?;
        }
        Ok(())
    }

    /// Total bytes currently stored in the CAS.
    pub fn cas_bytes(&self) -> Result<u64> {
        fn dir_size(dir: &Path) -> u64 {
            let Ok(entries) = fs::read_dir(dir) else {
                return 0;
            };
            entries
                .filter_map(|e| e.ok())
                .map(|e| {
                    let p = e.path();
                    if p.is_dir() {
                        dir_size(&p)
                    } else {
                        e.metadata().map(|m| m.len()).unwrap_or(0)
                    }
                })
                .sum()
        }
        Ok(dir_size(&self.root.join("cas")))
    }

    fn store_computed(
        &self,
        key: &CacheKey,
        value: &Value,
        origin: &Origin,
        compute: std::time::Duration,
        deterministic: bool,
    ) -> Result<()> {
        let (bytes, hash) = encode_and_hash(value)?;
        let output_bytes = bytes.len() as u64;
        // Blob first, record last: the record is the commit point.
        self.put_bytes_prehashed(&bytes, &hash)?;
        let mut outputs = BTreeMap::new();
        outputs.insert("output".to_string(), hash);
        self.put_action(&ActionResult {
            key: key.clone(),
            outputs,
            output_bytes,
            compute_ms: compute.as_millis() as u64,
            deterministic,
            origin: origin.clone(),
            created_at: Utc::now(),
            last_accessed: Utc::now(),
        })
    }

    fn put_bytes_prehashed(&self, bytes: &[u8], hash: &ContentHash) -> Result<()> {
        let path = self.blob_path(hash);
        if path.exists() {
            return Ok(()); // dedup: identical content already stored
        }
        write_atomic(&path, bytes)
    }
}

fn key_from_hex(hex: &str) -> Option<CacheKey> {
    if hex.len() != 64 {
        return None;
    }
    let mut digest = [0u8; 32];
    for (i, byte) in digest.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).ok()?;
    }
    Some(CacheKey(digest))
}

/// Temp file in the same directory → fsync → rename. Entry existence is
/// the commit point; renames are idempotent for concurrent writers.
fn write_atomic(path: &Path, data: &[u8]) -> Result<()> {
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};
    static WRITE_SEQ: AtomicU64 = AtomicU64::new(0);

    let parent = path
        .parent()
        .ok_or_else(|| SomaError::Cache("store path has no parent".into()))?;
    fs::create_dir_all(parent)?;
    let seq = WRITE_SEQ.fetch_add(1, Ordering::Relaxed);
    let tmp = path.with_extension(format!("tmp-{}-{seq}", std::process::id()));
    {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(data)?;
        f.sync_all()?;
    }
    if let Err(e) = fs::rename(&tmp, path) {
        let _ = fs::remove_file(&tmp);
        return Err(e.into());
    }
    Ok(())
}

impl BlobStore for FsActionStore {
    fn put_bytes(&self, bytes: &[u8]) -> Result<ContentHash> {
        let hash = ContentHash::blake3(bytes);
        self.put_bytes_prehashed(bytes, &hash)?;
        Ok(hash)
    }

    fn get_bytes(&self, hash: &ContentHash) -> Result<Option<Vec<u8>>> {
        let path = self.blob_path(hash);
        if !path.exists() {
            return Ok(None);
        }
        let bytes = fs::read(&path)?;
        if !hash.verify(&bytes) {
            // Torn or corrupted blob: treat as absent (regenerable).
            tracing::warn!(hash = %hash.to_hex(), "corrupt CAS blob, ignoring");
            return Ok(None);
        }
        Ok(Some(bytes))
    }

    fn contains(&self, hash: &ContentHash) -> Result<bool> {
        Ok(self.blob_path(hash).exists())
    }
}

impl ActionCache for FsActionStore {
    fn get_action(&self, key: &CacheKey) -> Result<Option<ActionResult>> {
        let path = self.action_path(key);
        if !path.exists() {
            return Ok(None);
        }
        let raw = fs::read_to_string(&path)?;
        match serde_json::from_str(&raw) {
            Ok(record) => Ok(Some(record)),
            Err(e) => {
                tracing::warn!(key = %key, error = %e, "corrupt action record, ignoring");
                Ok(None)
            }
        }
    }

    fn put_action(&self, result: &ActionResult) -> Result<()> {
        let raw = serde_json::to_string(result)
            .map_err(|e| SomaError::Cache(format!("action record encode: {e}")))?;
        write_atomic(&self.action_path(&result.key), raw.as_bytes())
    }
}

impl CacheStore for FsActionStore {
    fn get(&self, key: &CacheKey) -> Result<Option<Value>> {
        let Some(record) = self.get_action(key)? else {
            return Ok(None);
        };
        let Some(hash) = record.outputs.get("output") else {
            return Ok(None);
        };
        let Some(bytes) = self.get_bytes(hash)? else {
            return Ok(None); // blob evicted → miss, caller recomputes
        };
        decode_value(&bytes).map(Some)
    }

    fn put(&self, key: &CacheKey, value: &Value) -> Result<()> {
        self.store_computed(
            key,
            value,
            &Origin::Ingested {
                source: "unknown".into(),
            },
            std::time::Duration::ZERO,
            true,
        )
    }

    fn put_with_origin(&self, key: &CacheKey, value: &Value, origin: &Origin) -> Result<()> {
        self.store_computed(key, value, origin, std::time::Duration::ZERO, true)
    }

    fn put_computed(
        &self,
        key: &CacheKey,
        value: &Value,
        origin: &Origin,
        compute: std::time::Duration,
        deterministic: bool,
    ) -> Result<()> {
        self.store_computed(key, value, origin, compute, deterministic)
    }

    fn exists(&self, key: &CacheKey) -> Result<bool> {
        let Some(record) = self.get_action(key)? else {
            return Ok(false);
        };
        match record.outputs.get("output") {
            Some(hash) => self.contains(hash),
            None => Ok(false),
        }
    }

    fn remove(&self, key: &CacheKey) -> Result<()> {
        // Removes the record only; shared blobs are left for GC
        // (another record may name the same content).
        let path = self.action_path(key);
        if path.exists() {
            fs::remove_file(&path)?;
        }
        Ok(())
    }

    fn metadata(&self, key: &CacheKey) -> Result<Option<EntryMeta>> {
        Ok(self.get_action(key)?.map(|record| EntryMeta {
            key: record.key,
            size_bytes: record.output_bytes,
            created_at: record.created_at,
            last_accessed: record.last_accessed,
            ttl: None,
            origin: record.origin,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_root() -> PathBuf {
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("soma_fs_store_{}_{id}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn roundtrip_and_dedup() {
        let root = temp_root();
        let store = FsActionStore::new(&root).unwrap();
        let value = Value::tensor(vec![1.0; 1000], vec![1000]);

        let k1 = CacheKey::hash_data(b"action-1");
        let k2 = CacheKey::hash_data(b"action-2");
        store.put(&k1, &value).unwrap();
        store.put(&k2, &value).unwrap(); // same content, different action

        assert_eq!(store.get(&k1).unwrap().unwrap(), value);
        assert_eq!(store.get(&k2).unwrap().unwrap(), value);

        // Both records point at ONE deduplicated blob.
        let r1 = store.get_action(&k1).unwrap().unwrap();
        let r2 = store.get_action(&k2).unwrap().unwrap();
        assert_eq!(r1.outputs["output"], r2.outputs["output"]);
        let blob_count = walk_count(&root.join("cas"), "bin");
        assert_eq!(blob_count, 1, "identical outputs must share one blob");

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn eviction_keeps_record_and_refills() {
        let root = temp_root();
        let store = FsActionStore::new(&root).unwrap();
        let key = CacheKey::hash_data(b"expensive");
        let value = Value::tensor(vec![2.0; 64], vec![64]);
        store
            .put_computed(
                &key,
                &value,
                &Origin::Computed {
                    node_id: "n".into(),
                    run_id: "r".into(),
                },
                std::time::Duration::from_secs(3600),
                true,
            )
            .unwrap();

        let record = store.get_action(&key).unwrap().unwrap();
        assert_eq!(record.compute_ms, 3_600_000);
        let hash = record.outputs["output"];

        // Evict the blob: record survives, lookup misses (recompute path).
        store.evict_blob(&hash).unwrap();
        assert!(store.get_action(&key).unwrap().is_some());
        assert!(store.get(&key).unwrap().is_none());
        assert!(!store.exists(&key).unwrap());

        // Recompute re-fills the SAME content address; entry live again.
        store.put(&key, &value).unwrap();
        assert_eq!(store.get(&key).unwrap().unwrap(), value);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn corrupt_blob_is_a_miss_not_an_error() {
        let root = temp_root();
        let store = FsActionStore::new(&root).unwrap();
        let key = CacheKey::hash_data(b"c");
        store.put(&key, &Value::tensor(vec![1.0], vec![1])).unwrap();

        let hash = store.get_action(&key).unwrap().unwrap().outputs["output"];
        fs::write(store.blob_path(&hash), b"garbage").unwrap();

        assert!(store.get(&key).unwrap().is_none());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn format_version_guard() {
        let root = temp_root();
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("format.json"), r#"{"version": 99}"#).unwrap();
        assert!(FsActionStore::new(&root).is_err());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn pins_roundtrip() {
        let root = temp_root();
        let store = FsActionStore::new(&root).unwrap();
        let key = CacheKey::hash_data(b"best-model");
        store.pin("best-model", &key).unwrap();
        assert_eq!(store.pinned().unwrap(), vec![key.clone()]);
        assert!(store.pin("../escape", &key).is_err());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn concurrent_writers_same_key() {
        let root = temp_root();
        let store = Arc::new(FsActionStore::new(&root).unwrap());
        let key = CacheKey::hash_data(b"contended");
        let value = Value::tensor(vec![7.0; 512], vec![512]);

        std::thread::scope(|s| {
            for _ in 0..8 {
                let store = store.clone();
                let key = key.clone();
                let value = value.clone();
                s.spawn(move || store.put(&key, &value).unwrap());
            }
        });
        assert_eq!(store.get(&key).unwrap().unwrap(), value);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn survives_restart() {
        let root = temp_root();
        let key = CacheKey::hash_data(b"persist");
        let value = Value::json(serde_json::json!({"w": [1, 2, 3]}));
        {
            let store = FsActionStore::new(&root).unwrap();
            store.put(&key, &value).unwrap();
        }
        {
            let store = FsActionStore::new(&root).unwrap();
            assert_eq!(store.get(&key).unwrap().unwrap(), value);
        }
        let _ = fs::remove_dir_all(&root);
    }

    fn walk_count(dir: &Path, ext: &str) -> usize {
        let Ok(entries) = fs::read_dir(dir) else {
            return 0;
        };
        entries
            .filter_map(|e| e.ok())
            .map(|e| {
                let p = e.path();
                if p.is_dir() {
                    walk_count(&p, ext)
                } else if p.extension().is_some_and(|x| x == ext) {
                    1
                } else {
                    0
                }
            })
            .sum()
    }
}
