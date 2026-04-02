use somatize_core::cache::{CacheKey, CacheStore, CacheTier, EntryMeta};
use somatize_core::error::Result;
use somatize_core::value::Value;

/// Multi-level cache with automatic promotion.
///
/// Checks tiers in order: Memory → Local → (Remote, future).
/// On a hit in a lower tier, promotes the entry to faster tiers.
pub struct TieredCache {
    tiers: Vec<(CacheTier, Box<dyn CacheStore>)>,
}

impl TieredCache {
    /// Create a tiered cache from a list of (tier, store) pairs.
    /// Tiers should be ordered fastest-first.
    pub fn new(tiers: Vec<(CacheTier, Box<dyn CacheStore>)>) -> Self {
        Self { tiers }
    }

    /// Create a two-level cache: Memory + Local.
    pub fn memory_and_local(memory: Box<dyn CacheStore>, local: Box<dyn CacheStore>) -> Self {
        Self {
            tiers: vec![(CacheTier::Memory, memory), (CacheTier::Local, local)],
        }
    }
}

impl CacheStore for TieredCache {
    fn get(&self, key: &CacheKey) -> Result<Option<Value>> {
        for (i, (_, store)) in self.tiers.iter().enumerate() {
            if let Some(value) = store.get(key)? {
                // Promote to faster tiers
                for (_, faster_store) in &self.tiers[..i] {
                    let _ = faster_store.put(key, &value);
                }
                return Ok(Some(value));
            }
        }
        Ok(None)
    }

    fn put(&self, key: &CacheKey, value: &Value) -> Result<()> {
        // Write to all tiers
        for (_, store) in &self.tiers {
            store.put(key, value)?;
        }
        Ok(())
    }

    fn exists(&self, key: &CacheKey) -> Result<bool> {
        for (_, store) in &self.tiers {
            if store.exists(key)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn remove(&self, key: &CacheKey) -> Result<()> {
        for (_, store) in &self.tiers {
            store.remove(key)?;
        }
        Ok(())
    }

    fn metadata(&self, key: &CacheKey) -> Result<Option<EntryMeta>> {
        for (_, store) in &self.tiers {
            if let Some(meta) = store.metadata(key)? {
                return Ok(Some(meta));
            }
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::local::LocalCache;
    use crate::cache::memory::MemoryCache;
    use std::env;
    use std::fs;
    use std::path::PathBuf;

    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_dir() -> PathBuf {
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = env::temp_dir().join(format!("soma_tiered_test_{}_{id}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    fn make_tiered() -> (TieredCache, PathBuf) {
        let dir = temp_dir();
        let memory = Box::new(MemoryCache::default());
        let local = Box::new(LocalCache::new(&dir).unwrap());
        (TieredCache::memory_and_local(memory, local), dir)
    }

    #[test]
    fn put_writes_to_all_tiers() {
        let (cache, dir) = make_tiered();
        let key = CacheKey::hash_data(b"test");
        let value = Value::tensor(vec![1.0, 2.0], vec![2]);

        cache.put(&key, &value).unwrap();

        // Both tiers should have it
        assert!(cache.tiers[0].1.exists(&key).unwrap()); // memory
        assert!(cache.tiers[1].1.exists(&key).unwrap()); // local

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn get_from_memory_first() {
        let (cache, dir) = make_tiered();
        let key = CacheKey::hash_data(b"test");
        let value = Value::tensor(vec![1.0], vec![1]);

        cache.put(&key, &value).unwrap();
        let result = cache.get(&key).unwrap().unwrap();
        assert_eq!(result, value);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn promotes_from_local_to_memory() {
        let dir = temp_dir();
        let memory = Box::new(MemoryCache::default());
        let local = Box::new(LocalCache::new(&dir).unwrap());

        // Write only to local
        let key = CacheKey::hash_data(b"local_only");
        let value = Value::tensor(vec![42.0], vec![1]);
        local.put(&key, &value).unwrap();

        let tiered = TieredCache::memory_and_local(memory, local);

        // Memory doesn't have it
        assert!(!tiered.tiers[0].1.exists(&key).unwrap());

        // Get should find it in local and promote to memory
        let result = tiered.get(&key).unwrap().unwrap();
        assert_eq!(result, value);

        // Now memory should have it
        assert!(tiered.tiers[0].1.exists(&key).unwrap());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn miss_returns_none() {
        let (cache, dir) = make_tiered();
        assert!(cache.get(&CacheKey::hash_data(b"nope")).unwrap().is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn remove_from_all_tiers() {
        let (cache, dir) = make_tiered();
        let key = CacheKey::hash_data(b"test");
        cache.put(&key, &Value::Empty).unwrap();
        cache.remove(&key).unwrap();

        assert!(!cache.tiers[0].1.exists(&key).unwrap());
        assert!(!cache.tiers[1].1.exists(&key).unwrap());

        let _ = fs::remove_dir_all(&dir);
    }
}
