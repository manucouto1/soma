use chrono::Utc;
use soma_core::cache::{CacheKey, CacheStore, EntryMeta, Origin};
use soma_core::error::Result;
use soma_core::value::Value;
use std::collections::HashMap;
use std::sync::Mutex;

/// In-memory cache store backed by a HashMap.
///
/// Thread-safe via Mutex. Suitable for single-process use.
/// For multi-process or distributed caching, use the RocksDB or S3 backends.
pub struct MemoryCache {
    store: Mutex<HashMap<CacheKey, CacheEntry>>,
    #[expect(dead_code)]
    max_bytes: usize,
}

struct CacheEntry {
    value: Value,
    meta: EntryMeta,
}

impl MemoryCache {
    /// Create a new memory cache with a maximum byte limit.
    pub fn new(max_bytes: usize) -> Self {
        Self {
            store: Mutex::new(HashMap::new()),
            max_bytes,
        }
    }

    /// Number of entries currently in the cache.
    pub fn len(&self) -> usize {
        self.store.lock().unwrap().len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear all entries.
    pub fn clear(&self) {
        self.store.lock().unwrap().clear();
    }
}

impl Default for MemoryCache {
    fn default() -> Self {
        Self::new(1024 * 1024 * 1024) // 1GB
    }
}

impl CacheStore for MemoryCache {
    fn get(&self, key: &CacheKey) -> Result<Option<Value>> {
        let mut store = self.store.lock().unwrap();
        if let Some(entry) = store.get_mut(key) {
            entry.meta.last_accessed = Utc::now();
            Ok(Some(entry.value.clone()))
        } else {
            Ok(None)
        }
    }

    fn put(&self, key: &CacheKey, value: &Value) -> Result<()> {
        let size = estimate_size(value);
        let now = Utc::now();

        let mut store = self.store.lock().unwrap();
        store.insert(
            key.clone(),
            CacheEntry {
                value: value.clone(),
                meta: EntryMeta {
                    key: key.clone(),
                    size_bytes: size as u64,
                    created_at: now,
                    last_accessed: now,
                    ttl: None,
                    origin: Origin::Computed {
                        node_id: String::new(),
                        run_id: String::new(),
                    },
                },
            },
        );
        Ok(())
    }

    fn exists(&self, key: &CacheKey) -> Result<bool> {
        Ok(self.store.lock().unwrap().contains_key(key))
    }

    fn remove(&self, key: &CacheKey) -> Result<()> {
        self.store.lock().unwrap().remove(key);
        Ok(())
    }

    fn metadata(&self, key: &CacheKey) -> Result<Option<EntryMeta>> {
        Ok(self
            .store
            .lock()
            .unwrap()
            .get(key)
            .map(|e| e.meta.clone()))
    }
}

fn estimate_size(value: &Value) -> usize {
    match value {
        Value::Tensor { values, shape } => {
            values.len() * std::mem::size_of::<f64>() + shape.len() * std::mem::size_of::<usize>()
        }
        Value::Json(v) => v.to_string().len(),
        Value::Bytes(b) => b.len(),
        Value::Empty => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn put_and_get() {
        let cache = MemoryCache::default();
        let key = CacheKey::hash_data(b"test");
        let value = Value::tensor(vec![1.0, 2.0, 3.0], vec![3]);

        cache.put(&key, &value).unwrap();
        let retrieved = cache.get(&key).unwrap().unwrap();
        assert_eq!(retrieved, value);
    }

    #[test]
    fn get_missing_returns_none() {
        let cache = MemoryCache::default();
        let key = CacheKey::hash_data(b"nonexistent");
        assert!(cache.get(&key).unwrap().is_none());
    }

    #[test]
    fn exists_check() {
        let cache = MemoryCache::default();
        let key = CacheKey::hash_data(b"test");
        assert!(!cache.exists(&key).unwrap());

        cache.put(&key, &Value::Empty).unwrap();
        assert!(cache.exists(&key).unwrap());
    }

    #[test]
    fn remove_entry() {
        let cache = MemoryCache::default();
        let key = CacheKey::hash_data(b"test");
        cache.put(&key, &Value::Empty).unwrap();
        assert_eq!(cache.len(), 1);

        cache.remove(&key).unwrap();
        assert_eq!(cache.len(), 0);
        assert!(!cache.exists(&key).unwrap());
    }

    #[test]
    fn metadata_available() {
        let cache = MemoryCache::default();
        let key = CacheKey::hash_data(b"test");
        let value = Value::tensor(vec![1.0; 100], vec![10, 10]);

        cache.put(&key, &value).unwrap();
        let meta = cache.metadata(&key).unwrap().unwrap();
        // 100 f64 values * 8 bytes + 2 shape elements * 8 bytes = 816
        assert_eq!(meta.size_bytes, 816);
    }

    #[test]
    fn clear_empties_cache() {
        let cache = MemoryCache::default();
        cache
            .put(&CacheKey::hash_data(b"a"), &Value::Empty)
            .unwrap();
        cache
            .put(&CacheKey::hash_data(b"b"), &Value::Empty)
            .unwrap();
        assert_eq!(cache.len(), 2);

        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn overwrite_existing_key() {
        let cache = MemoryCache::default();
        let key = CacheKey::hash_data(b"test");

        cache.put(&key, &Value::json(json!(1))).unwrap();
        cache.put(&key, &Value::json(json!(2))).unwrap();

        let val = cache.get(&key).unwrap().unwrap();
        assert_eq!(val, Value::json(json!(2)));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn multiple_keys() {
        let cache = MemoryCache::default();
        for i in 0..10 {
            let key = CacheKey::hash_data(format!("key_{i}").as_bytes());
            let val = Value::tensor(vec![i as f64], vec![1]);
            cache.put(&key, &val).unwrap();
        }
        assert_eq!(cache.len(), 10);

        let key5 = CacheKey::hash_data(b"key_5");
        let val = cache.get(&key5).unwrap().unwrap();
        let (data, _) = val.as_tensor().unwrap();
        assert_eq!(data, &[5.0]);
    }
}
