use chrono::Utc;
use somatize_core::cache::{CacheKey, CacheStore, EntryMeta, Origin};
use somatize_core::error::{Result, SomaError};
use somatize_core::value::Value;
use std::fs;
use std::path::{Path, PathBuf};

/// Filesystem-based cache store.
///
/// Each entry is stored as a JSON file named by the cache key's hex.
/// Suitable for persistent local caching across process restarts.
pub struct LocalCache {
    base_dir: PathBuf,
}

impl LocalCache {
    pub fn new(base_dir: impl Into<PathBuf>) -> Result<Self> {
        let base_dir = base_dir.into();
        fs::create_dir_all(&base_dir)?;
        Ok(Self { base_dir })
    }

    fn key_path(&self, key: &CacheKey) -> PathBuf {
        let hex = key.to_hex();
        // Shard into subdirectories: first 2 chars / next 2 chars / full key
        self.base_dir
            .join(&hex[..2])
            .join(&hex[2..4])
            .join(format!("{hex}.json"))
    }

    fn meta_path(&self, key: &CacheKey) -> PathBuf {
        let hex = key.to_hex();
        self.base_dir
            .join(&hex[..2])
            .join(&hex[2..4])
            .join(format!("{hex}.meta.json"))
    }

    /// Number of cached entries (scans filesystem).
    pub fn len(&self) -> usize {
        walkdir_count(&self.base_dir)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Remove all cached entries.
    pub fn clear(&self) -> Result<()> {
        if self.base_dir.exists() {
            fs::remove_dir_all(&self.base_dir)?;
            fs::create_dir_all(&self.base_dir)?;
        }
        Ok(())
    }

    /// Write `data` to `path` atomically: temp file in the same directory
    /// → fsync → rename. A crash mid-write leaves only an orphan temp
    /// file (never read back: lookups go through the exact entry path),
    /// so entry existence is the commit point. Renames are idempotent,
    /// which makes concurrent same-key writers from multiple processes
    /// safe on POSIX filesystems.
    fn write_atomic(path: &Path, data: &[u8]) -> Result<()> {
        use std::io::Write;
        use std::sync::atomic::{AtomicU64, Ordering};
        static WRITE_SEQ: AtomicU64 = AtomicU64::new(0);

        let parent = path
            .parent()
            .ok_or_else(|| SomaError::Cache("cache path has no parent".into()))?;
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

    fn write_entry(&self, key: &CacheKey, value: &Value, origin: &Origin) -> Result<()> {
        let data = serde_json::to_string(value)
            .map_err(|e| SomaError::Cache(format!("serialize error: {e}")))?;
        let size = data.len() as u64;

        // Meta first, value last: the value file's existence commits the
        // entry (`exists()`/`get()` check only the value path).
        let meta = EntryMeta {
            key: key.clone(),
            size_bytes: size,
            created_at: Utc::now(),
            last_accessed: Utc::now(),
            ttl: None,
            origin: origin.clone(),
        };
        let meta_data = serde_json::to_string(&meta)
            .map_err(|e| SomaError::Cache(format!("meta serialize error: {e}")))?;
        Self::write_atomic(&self.meta_path(key), meta_data.as_bytes())?;
        Self::write_atomic(&self.key_path(key), data.as_bytes())?;
        Ok(())
    }
}

impl CacheStore for LocalCache {
    fn tier(&self) -> somatize_core::cache::CacheTier {
        somatize_core::cache::CacheTier::Local
    }

    fn get(&self, key: &CacheKey) -> Result<Option<Value>> {
        let path = self.key_path(key);
        if !path.exists() {
            return Ok(None);
        }
        let data = fs::read_to_string(&path)?;
        let value: Value = serde_json::from_str(&data)
            .map_err(|e| SomaError::Cache(format!("deserialize error: {e}")))?;
        Ok(Some(value))
    }

    fn put(&self, key: &CacheKey, value: &Value) -> Result<()> {
        self.write_entry(
            key,
            value,
            &Origin::Ingested {
                source: "unknown".into(),
            },
        )
    }

    fn put_with_origin(&self, key: &CacheKey, value: &Value, origin: &Origin) -> Result<()> {
        self.write_entry(key, value, origin)
    }

    fn exists(&self, key: &CacheKey) -> Result<bool> {
        Ok(self.key_path(key).exists())
    }

    fn remove(&self, key: &CacheKey) -> Result<()> {
        let path = self.key_path(key);
        if path.exists() {
            fs::remove_file(&path)?;
        }
        let meta = self.meta_path(key);
        if meta.exists() {
            fs::remove_file(&meta)?;
        }
        Ok(())
    }

    fn metadata(&self, key: &CacheKey) -> Result<Option<EntryMeta>> {
        let path = self.meta_path(key);
        if !path.exists() {
            return Ok(None);
        }
        let data = fs::read_to_string(&path)?;
        let meta: EntryMeta = serde_json::from_str(&data)
            .map_err(|e| SomaError::Cache(format!("meta deserialize error: {e}")))?;
        Ok(Some(meta))
    }
}

fn walkdir_count(dir: &Path) -> usize {
    if !dir.exists() {
        return 0;
    }
    fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .map(|e| {
                    if e.path().is_dir() {
                        walkdir_count(&e.path())
                    } else if e.path().extension().is_some_and(|ext| ext == "json")
                        && !e.path().to_string_lossy().contains(".meta.")
                    {
                        1
                    } else {
                        0
                    }
                })
                .sum()
        })
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::env;

    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_dir() -> PathBuf {
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = env::temp_dir().join(format!("soma_test_cache_{}_{id}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn put_and_get() {
        let dir = temp_dir();
        let cache = LocalCache::new(&dir).unwrap();
        let key = CacheKey::hash_data(b"test");
        let value = Value::tensor(vec![1.0, 2.0, 3.0], vec![3]);

        cache.put(&key, &value).unwrap();
        let retrieved = cache.get(&key).unwrap().unwrap();
        assert_eq!(retrieved, value);

        cache.clear().unwrap();
    }

    #[test]
    fn get_missing() {
        let dir = temp_dir();
        let cache = LocalCache::new(&dir).unwrap();
        assert!(cache.get(&CacheKey::hash_data(b"nope")).unwrap().is_none());
        cache.clear().unwrap();
    }

    #[test]
    fn exists_check() {
        let dir = temp_dir();
        let cache = LocalCache::new(&dir).unwrap();
        let key = CacheKey::hash_data(b"test");
        assert!(!cache.exists(&key).unwrap());
        cache.put(&key, &Value::Empty).unwrap();
        assert!(cache.exists(&key).unwrap());
        cache.clear().unwrap();
    }

    #[test]
    fn remove_entry() {
        let dir = temp_dir();
        let cache = LocalCache::new(&dir).unwrap();
        let key = CacheKey::hash_data(b"test");
        cache.put(&key, &Value::json(json!(42))).unwrap();
        assert!(cache.exists(&key).unwrap());
        cache.remove(&key).unwrap();
        assert!(!cache.exists(&key).unwrap());
        cache.clear().unwrap();
    }

    #[test]
    fn metadata_persists() {
        let dir = temp_dir();
        let cache = LocalCache::new(&dir).unwrap();
        let key = CacheKey::hash_data(b"test");
        cache
            .put(&key, &Value::tensor(vec![1.0; 50], vec![50]))
            .unwrap();

        let meta = cache.metadata(&key).unwrap().unwrap();
        assert!(meta.size_bytes > 0);
        cache.clear().unwrap();
    }

    #[test]
    fn orphan_tmp_files_are_not_entries() {
        let dir = temp_dir();
        let cache = LocalCache::new(&dir).unwrap();
        let key = CacheKey::hash_data(b"torn");

        // Simulate a crash mid-write: a temp file exists, the entry doesn't.
        let entry_path = cache.key_path(&key);
        fs::create_dir_all(entry_path.parent().unwrap()).unwrap();
        fs::write(entry_path.with_extension("tmp-999-0"), b"garbage{{{").unwrap();

        assert!(!cache.exists(&key).unwrap());
        assert!(cache.get(&key).unwrap().is_none());

        // A subsequent put commits normally despite the orphan.
        cache.put(&key, &Value::tensor(vec![1.0], vec![1])).unwrap();
        assert!(cache.exists(&key).unwrap());
        cache.clear().unwrap();
    }

    #[test]
    fn concurrent_same_key_puts_are_safe() {
        let dir = temp_dir();
        let cache = std::sync::Arc::new(LocalCache::new(&dir).unwrap());
        let key = CacheKey::hash_data(b"contended");
        let value = Value::tensor(vec![1.0; 256], vec![256]);

        std::thread::scope(|s| {
            for _ in 0..8 {
                let cache = cache.clone();
                let key = key.clone();
                let value = value.clone();
                s.spawn(move || cache.put(&key, &value).unwrap());
            }
        });

        assert_eq!(cache.get(&key).unwrap().unwrap(), value);
        cache.clear().unwrap();
    }

    #[test]
    fn put_with_origin_records_provenance() {
        let dir = temp_dir();
        let cache = LocalCache::new(&dir).unwrap();
        let key = CacheKey::hash_data(b"prov");
        let origin = Origin::Computed {
            node_id: "scaler".into(),
            run_id: "run_42".into(),
        };
        cache.put_with_origin(&key, &Value::Empty, &origin).unwrap();

        let meta = cache.metadata(&key).unwrap().unwrap();
        match meta.origin {
            Origin::Computed { node_id, run_id } => {
                assert_eq!(node_id, "scaler");
                assert_eq!(run_id, "run_42");
            }
            other => panic!("expected Computed origin, got {other:?}"),
        }
        cache.clear().unwrap();
    }

    #[test]
    fn survives_restart() {
        let dir = temp_dir();
        let key = CacheKey::hash_data(b"persist");
        let value = Value::tensor(vec![42.0], vec![1]);

        {
            let cache = LocalCache::new(&dir).unwrap();
            cache.put(&key, &value).unwrap();
        }
        // "restart": create a new instance pointing to same dir
        {
            let cache = LocalCache::new(&dir).unwrap();
            let retrieved = cache.get(&key).unwrap().unwrap();
            assert_eq!(retrieved, value);
        }

        let _ = fs::remove_dir_all(&dir);
    }
}
