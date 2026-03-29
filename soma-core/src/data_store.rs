//! Data Store: abstraction for moving data between workers.
//!
//! Separates WHERE data lives from HOW it's processed.
//! Workers use DataRef to reference data without materializing it.

use crate::cache::CacheKey;
use crate::error::Result;
use crate::value::Value;
use serde::{Deserialize, Serialize};

/// A reference to data that may live in different places.
/// Workers exchange DataRefs instead of raw data.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum DataRef {
    /// Data in local filesystem
    Local { path: String },
    /// Data in S3-compatible object storage
    S3 {
        bucket: String,
        key: String,
        region: Option<String>,
    },
    /// Data in Soma cache (content-addressable)
    Cached { cache_key: CacheKey },
    /// Data available as a stream endpoint
    Stream {
        endpoint: String,
        format: StreamFormat,
    },
    /// Data materialized inline (small values only)
    Inline { value: Value },
}

/// Stream data format.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum StreamFormat {
    #[default]
    JsonLines,
    Csv,
    Arrow,
    Protobuf,
}

/// Storage configuration for an investigation/pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum StorageConfig {
    /// Local filesystem (NFS, mounted volume)
    #[serde(rename = "local")]
    Local { base_path: String },
    /// S3-compatible object storage
    #[serde(rename = "s3")]
    S3 {
        bucket: String,
        prefix: String,
        region: Option<String>,
        endpoint: Option<String>, // for MinIO etc.
    },
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self::Local {
            base_path: "/tmp/soma-data".to_string(),
        }
    }
}

/// The DataStore trait: put/get/stream data across workers.
///
/// Unlike CacheStore (which stores Values by CacheKey),
/// DataStore moves data between locations and supports streaming.
pub trait DataStore: Send + Sync {
    /// Store data and return a reference to it.
    fn put(&self, key: &CacheKey, data: &Value) -> Result<DataRef>;

    /// Retrieve data from a reference.
    fn get(&self, data_ref: &DataRef) -> Result<Value>;

    /// Check if data exists at a reference.
    fn exists(&self, data_ref: &DataRef) -> Result<bool>;

    /// Delete data at a reference.
    fn remove(&self, data_ref: &DataRef) -> Result<()>;

    /// Get the storage config.
    fn config(&self) -> &StorageConfig;
}

/// Local filesystem data store.
pub struct LocalDataStore {
    config: StorageConfig,
    base_path: std::path::PathBuf,
}

impl LocalDataStore {
    pub fn new(base_path: impl Into<std::path::PathBuf>) -> Self {
        let base = base_path.into();
        std::fs::create_dir_all(&base).ok();
        Self {
            config: StorageConfig::Local {
                base_path: base.to_string_lossy().to_string(),
            },
            base_path: base,
        }
    }
}

impl DataStore for LocalDataStore {
    fn put(&self, key: &CacheKey, data: &Value) -> Result<DataRef> {
        let path = self.base_path.join(key.to_hex());
        let bytes = serde_json::to_vec(data)
            .map_err(|e| crate::error::SomaError::DataStore(e.to_string()))?;
        std::fs::write(&path, &bytes)
            .map_err(|e| crate::error::SomaError::DataStore(e.to_string()))?;
        Ok(DataRef::Local {
            path: path.to_string_lossy().to_string(),
        })
    }

    fn get(&self, data_ref: &DataRef) -> Result<Value> {
        match data_ref {
            DataRef::Local { path } => {
                let bytes = std::fs::read(path)
                    .map_err(|e| crate::error::SomaError::DataStore(e.to_string()))?;
                serde_json::from_slice(&bytes)
                    .map_err(|e| crate::error::SomaError::DataStore(e.to_string()))
            }
            DataRef::Cached { cache_key } => {
                let path = self.base_path.join(cache_key.to_hex());
                let bytes = std::fs::read(&path)
                    .map_err(|e| crate::error::SomaError::DataStore(e.to_string()))?;
                serde_json::from_slice(&bytes)
                    .map_err(|e| crate::error::SomaError::DataStore(e.to_string()))
            }
            DataRef::Inline { value } => Ok(value.clone()),
            _ => Err(crate::error::SomaError::DataStore(
                "Cannot get non-local DataRef from LocalDataStore".into(),
            )),
        }
    }

    fn exists(&self, data_ref: &DataRef) -> Result<bool> {
        match data_ref {
            DataRef::Local { path } => Ok(std::path::Path::new(path).exists()),
            DataRef::Cached { cache_key } => {
                Ok(self.base_path.join(cache_key.to_hex()).exists())
            }
            DataRef::Inline { .. } => Ok(true),
            _ => Ok(false),
        }
    }

    fn remove(&self, data_ref: &DataRef) -> Result<()> {
        if let DataRef::Local { path } = data_ref {
            std::fs::remove_file(path).ok();
        }
        Ok(())
    }

    fn config(&self) -> &StorageConfig {
        &self.config
    }
}

/// Stream-aware cache for inference pipelines.
///
/// Key insight: during inference, the filter STATE is fixed (from training).
/// Only the DATA changes. So we cache:
/// 1. Filter states (from training) — keyed by config_hash + training_data_hash
/// 2. Chunk results — keyed by config_hash + state_hash + chunk_hash
///
/// This means: if the same chunk passes through the same filter with the
/// same trained state, the result is returned from cache instantly.
pub struct StreamCache {
    /// State cache: filter_id → (state_key, cached state)
    states: std::collections::HashMap<String, (CacheKey, Value)>,
    /// Chunk result cache: LRU of chunk results
    chunk_cache: std::collections::HashMap<CacheKey, Value>,
    /// Max cached chunks (LRU eviction)
    max_chunks: usize,
    /// Stats
    pub hits: u64,
    pub misses: u64,
}

impl StreamCache {
    pub fn new(max_chunks: usize) -> Self {
        Self {
            states: std::collections::HashMap::new(),
            chunk_cache: std::collections::HashMap::new(),
            max_chunks,
            hits: 0,
            misses: 0,
        }
    }

    /// Load a filter's trained state into the stream cache.
    pub fn load_state(&mut self, filter_id: &str, state_key: CacheKey, state: Value) {
        self.states.insert(filter_id.to_string(), (state_key, state));
    }

    /// Get a filter's cached state (for forward() during inference).
    pub fn get_state(&self, filter_id: &str) -> Option<&Value> {
        self.states.get(filter_id).map(|(_, v)| v)
    }

    /// Try to get a cached chunk result.
    /// chunk_key = hash(config_hash + state_hash + chunk_data_hash)
    pub fn get_chunk(&mut self, chunk_key: &CacheKey) -> Option<&Value> {
        if let Some(v) = self.chunk_cache.get(chunk_key) {
            self.hits += 1;
            Some(v)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Cache a chunk result.
    pub fn put_chunk(&mut self, chunk_key: CacheKey, value: Value) {
        if self.chunk_cache.len() >= self.max_chunks {
            // Simple eviction: remove first entry (not true LRU, but fast)
            if let Some(k) = self.chunk_cache.keys().next().cloned() {
                self.chunk_cache.remove(&k);
            }
        }
        self.chunk_cache.insert(chunk_key, value);
    }

    /// Cache hit rate.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_data_store_roundtrip() {
        let dir = std::env::temp_dir().join("soma-ds-test");
        let store = LocalDataStore::new(&dir);

        let key = CacheKey::hash_data(b"test_data");
        let value = Value::tensor(vec![1.0, 2.0, 3.0], vec![3]);

        let data_ref = store.put(&key, &value).unwrap();
        assert!(store.exists(&data_ref).unwrap());

        let retrieved = store.get(&data_ref).unwrap();
        let (data, _) = retrieved.as_tensor().unwrap();
        assert_eq!(data, &[1.0, 2.0, 3.0]);

        store.remove(&data_ref).unwrap();
        assert!(!store.exists(&data_ref).unwrap());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn inline_data_ref() {
        let dir = std::env::temp_dir().join("soma-ds-test-inline");
        let store = LocalDataStore::new(&dir);

        let data_ref = DataRef::Inline {
            value: Value::tensor(vec![42.0], vec![1]),
        };

        assert!(store.exists(&data_ref).unwrap());
        let v = store.get(&data_ref).unwrap();
        let (data, _) = v.as_tensor().unwrap();
        assert_eq!(data, &[42.0]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stream_cache_basics() {
        let mut cache = StreamCache::new(100);

        let state = Value::tensor(vec![0.0, 1.0], vec![2]);
        let state_key = CacheKey::hash_data(b"state_001");
        cache.load_state("normalize", state_key, state.clone());

        assert!(cache.get_state("normalize").is_some());
        assert!(cache.get_state("unknown").is_none());
    }

    #[test]
    fn stream_cache_chunks() {
        let mut cache = StreamCache::new(3);

        let k1 = CacheKey::hash_data(b"chunk_1");
        let k2 = CacheKey::hash_data(b"chunk_2");
        let k3 = CacheKey::hash_data(b"chunk_3");
        let k4 = CacheKey::hash_data(b"chunk_4");

        cache.put_chunk(k1.clone(), Value::tensor(vec![1.0], vec![1]));
        cache.put_chunk(k2.clone(), Value::tensor(vec![2.0], vec![1]));
        cache.put_chunk(k3.clone(), Value::tensor(vec![3.0], vec![1]));

        // All 3 should be cached
        assert!(cache.get_chunk(&k1).is_some());
        assert!(cache.get_chunk(&k2).is_some());
        assert!(cache.get_chunk(&k3).is_some());
        assert_eq!(cache.hits, 3);

        // Adding k4 should evict one (max_chunks = 3)
        cache.put_chunk(k4.clone(), Value::tensor(vec![4.0], vec![1]));
        assert!(cache.get_chunk(&k4).is_some());

        assert!(cache.hit_rate() > 0.0);
    }

    #[test]
    fn storage_config_serde() {
        let s3 = StorageConfig::S3 {
            bucket: "my-lab".into(),
            prefix: "experiments/".into(),
            region: Some("eu-west-1".into()),
            endpoint: None,
        };
        let json = serde_json::to_string(&s3).unwrap();
        assert!(json.contains("my-lab"));

        let local = StorageConfig::Local {
            base_path: "/data".into(),
        };
        let json = serde_json::to_string(&local).unwrap();
        assert!(json.contains("/data"));
    }

    #[test]
    fn data_ref_serde() {
        let refs = vec![
            DataRef::Local { path: "/tmp/x".into() },
            DataRef::S3 { bucket: "b".into(), key: "k".into(), region: None },
            DataRef::Cached { cache_key: CacheKey::hash_data(b"x") },
            DataRef::Inline { value: Value::Empty },
        ];
        for r in &refs {
            let json = serde_json::to_string(r).unwrap();
            let _: DataRef = serde_json::from_str(&json).unwrap();
        }
    }
}
