use chrono::Utc;
use soma_core::cache::{CacheKey, CacheStore, EntryMeta, Origin};
use soma_core::error::Result;
use soma_core::value::Value;
use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

/// In-memory LRU cache store.
///
/// Enforces a maximum byte limit. When the limit is exceeded,
/// the least recently accessed entries are evicted.
/// Thread-safe via Mutex.
pub struct MemoryCache {
    store: Mutex<LruStore>,
}

struct LruStore {
    entries: HashMap<CacheKey, CacheEntry>,
    /// Access order: most recent at back, least recent at front.
    access_order: VecDeque<CacheKey>,
    current_bytes: usize,
    max_bytes: usize,
}

struct CacheEntry {
    value: Value,
    meta: EntryMeta,
    size: usize,
}

impl LruStore {
    fn new(max_bytes: usize) -> Self {
        Self {
            entries: HashMap::new(),
            access_order: VecDeque::new(),
            current_bytes: 0,
            max_bytes,
        }
    }

    fn touch(&mut self, key: &CacheKey) {
        self.access_order.retain(|k| k != key);
        self.access_order.push_back(key.clone());
    }

    fn evict_until_fits(&mut self, needed: usize) {
        while self.current_bytes + needed > self.max_bytes && !self.access_order.is_empty() {
            if let Some(oldest_key) = self.access_order.pop_front()
                && let Some(entry) = self.entries.remove(&oldest_key)
            {
                self.current_bytes = self.current_bytes.saturating_sub(entry.size);
            }
        }
    }

    fn insert(&mut self, key: CacheKey, entry: CacheEntry) {
        let size = entry.size;

        // Remove old entry if exists
        if let Some(old) = self.entries.remove(&key) {
            self.current_bytes = self.current_bytes.saturating_sub(old.size);
            self.access_order.retain(|k| k != &key);
        }

        // Evict if needed
        self.evict_until_fits(size);

        self.current_bytes += size;
        self.access_order.push_back(key.clone());
        self.entries.insert(key, entry);
    }

    fn remove(&mut self, key: &CacheKey) {
        if let Some(entry) = self.entries.remove(key) {
            self.current_bytes = self.current_bytes.saturating_sub(entry.size);
            self.access_order.retain(|k| k != key);
        }
    }
}

impl MemoryCache {
    /// Create a new memory cache with a maximum byte limit.
    pub fn new(max_bytes: usize) -> Self {
        Self {
            store: Mutex::new(LruStore::new(max_bytes)),
        }
    }

    /// Number of entries currently in the cache.
    pub fn len(&self) -> usize {
        self.store
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entries
            .len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Current memory usage in bytes.
    pub fn current_bytes(&self) -> usize {
        self.store
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .current_bytes
    }

    /// Clear all entries.
    pub fn clear(&self) {
        let mut store = self.store.lock().unwrap_or_else(|e| e.into_inner());
        store.entries.clear();
        store.access_order.clear();
        store.current_bytes = 0;
    }
}

impl Default for MemoryCache {
    fn default() -> Self {
        Self::new(1024 * 1024 * 1024) // 1GB
    }
}

impl CacheStore for MemoryCache {
    fn get(&self, key: &CacheKey) -> Result<Option<Value>> {
        let mut store = self.store.lock().unwrap_or_else(|e| e.into_inner());
        if store.entries.contains_key(key) {
            store.touch(key);
            if let Some(entry) = store.entries.get_mut(key) {
                entry.meta.last_accessed = Utc::now();
                return Ok(Some(entry.value.clone()));
            }
        }
        Ok(None)
    }

    fn put(&self, key: &CacheKey, value: &Value) -> Result<()> {
        let size = estimate_size(value);
        let now = Utc::now();

        let mut store = self.store.lock().unwrap_or_else(|e| e.into_inner());
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
                size,
            },
        );
        Ok(())
    }

    fn exists(&self, key: &CacheKey) -> Result<bool> {
        Ok(self
            .store
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entries
            .contains_key(key))
    }

    fn remove(&self, key: &CacheKey) -> Result<()> {
        self.store
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(key);
        Ok(())
    }

    fn metadata(&self, key: &CacheKey) -> Result<Option<EntryMeta>> {
        Ok(self
            .store
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entries
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
        _ => 0,
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
        assert_eq!(cache.current_bytes(), 0);
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

    // ── LRU eviction tests ──

    #[test]
    fn lru_evicts_oldest_when_full() {
        // Cache with 100 bytes max
        let cache = MemoryCache::new(100);

        // Each tensor of 5 f64s = 5*8 + 1*8 = 48 bytes
        let k1 = CacheKey::hash_data(b"first");
        let k2 = CacheKey::hash_data(b"second");
        let k3 = CacheKey::hash_data(b"third");

        cache
            .put(&k1, &Value::tensor(vec![0.0; 5], vec![5]))
            .unwrap();
        cache
            .put(&k2, &Value::tensor(vec![0.0; 5], vec![5]))
            .unwrap();
        assert_eq!(cache.len(), 2);

        // Adding third should evict first (48+48+48=144 > 100)
        cache
            .put(&k3, &Value::tensor(vec![0.0; 5], vec![5]))
            .unwrap();

        assert!(!cache.exists(&k1).unwrap(), "k1 should be evicted");
        assert!(cache.exists(&k2).unwrap(), "k2 should remain");
        assert!(cache.exists(&k3).unwrap(), "k3 should remain");
    }

    #[test]
    fn lru_access_prevents_eviction() {
        let cache = MemoryCache::new(100);

        let k1 = CacheKey::hash_data(b"first");
        let k2 = CacheKey::hash_data(b"second");
        let k3 = CacheKey::hash_data(b"third");

        cache
            .put(&k1, &Value::tensor(vec![0.0; 5], vec![5]))
            .unwrap();
        cache
            .put(&k2, &Value::tensor(vec![0.0; 5], vec![5]))
            .unwrap();

        // Access k1, making k2 the least recently used
        cache.get(&k1).unwrap();

        // Adding k3 should evict k2 (LRU), not k1
        cache
            .put(&k3, &Value::tensor(vec![0.0; 5], vec![5]))
            .unwrap();

        assert!(cache.exists(&k1).unwrap(), "k1 was accessed, should remain");
        assert!(!cache.exists(&k2).unwrap(), "k2 was LRU, should be evicted");
        assert!(cache.exists(&k3).unwrap(), "k3 is new, should remain");
    }

    #[test]
    fn lru_tracks_byte_usage() {
        let cache = MemoryCache::new(1024);

        assert_eq!(cache.current_bytes(), 0);

        // 10 f64s = 80 bytes data + 8 bytes shape = 88
        cache
            .put(
                &CacheKey::hash_data(b"a"),
                &Value::tensor(vec![0.0; 10], vec![10]),
            )
            .unwrap();
        assert_eq!(cache.current_bytes(), 88);

        cache.remove(&CacheKey::hash_data(b"a")).unwrap();
        assert_eq!(cache.current_bytes(), 0);
    }

    #[test]
    fn lru_overwrite_updates_size() {
        let cache = MemoryCache::new(1024);

        let key = CacheKey::hash_data(b"key");
        cache
            .put(&key, &Value::tensor(vec![0.0; 10], vec![10]))
            .unwrap();
        let size1 = cache.current_bytes();

        // Replace with larger value
        cache
            .put(&key, &Value::tensor(vec![0.0; 20], vec![20]))
            .unwrap();
        let size2 = cache.current_bytes();

        assert!(size2 > size1, "larger value should use more bytes");
        assert_eq!(cache.len(), 1, "should still be one entry");
    }
}
