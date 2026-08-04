//! Data Store: abstraction for moving data between workers.
//!
//! Separates WHERE data lives from HOW it's processed.
//! Workers use DataRef to reference data without materializing it.

/// Maximum payload size for inline WebSocket transport.
/// Payloads above this threshold are uploaded via HTTP bulk or DataStore.
pub const INLINE_THRESHOLD_BYTES: usize = 10 * 1024 * 1024; // 10 MB

// The S3 and Zarr backends live in `somatize-store`. They each own a
// tokio runtime, and a contract crate must not hand one to everything
// that depends on it.

use crate::cache::CacheKey;
use crate::error::{Result, SomaError};
use crate::value::Value;
use serde::{Deserialize, Serialize};

/// Metadata about a stored value, queryable without loading data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreMeta {
    /// Total number of rows (`shape[0]` for tensors, 1 for scalar types).
    pub total_rows: usize,
    /// Remaining shape dimensions after the row axis (shape[1..] for tensors).
    pub shape_tail: Vec<usize>,
    /// Type tag: "tensor", "text", "json", "bytes", or "empty".
    pub dtype: String,
}

impl StoreMeta {
    /// Build metadata from an in-memory Value.
    pub fn from_value(value: &Value) -> Self {
        match value {
            Value::Tensor { shape, .. } => Self {
                total_rows: shape.first().copied().unwrap_or(0),
                shape_tail: shape.get(1..).unwrap_or_default().to_vec(),
                dtype: "tensor".into(),
            },
            Value::Text(_) => Self {
                total_rows: 1,
                shape_tail: vec![],
                dtype: "text".into(),
            },
            Value::Json(_) => Self {
                total_rows: 1,
                shape_tail: vec![],
                dtype: "json".into(),
            },
            Value::Bytes(b) | Value::Object(b) => Self {
                total_rows: b.len(),
                shape_tail: vec![],
                dtype: "bytes".into(),
            },
            Value::Empty => Self {
                total_rows: 0,
                shape_tail: vec![],
                dtype: "empty".into(),
            },
        }
    }
}

/// Slice rows `[start..start+len)` from a tensor value.
pub fn slice_tensor_rows(value: &Value, start: usize, len: usize) -> Result<Value> {
    match value {
        Value::Tensor { values, shape } => {
            if shape.is_empty() {
                return Err(SomaError::DataStore("cannot slice scalar tensor".into()));
            }
            let cols: usize = shape[1..].iter().product::<usize>().max(1);
            let row_start = start * cols;
            let row_end = (start + len) * cols;
            if row_end > values.len() {
                return Err(SomaError::DataStore(format!(
                    "row range {start}..{} out of bounds (total rows: {})",
                    start + len,
                    shape[0]
                )));
            }
            let mut new_shape = shape.clone();
            new_shape[0] = len;
            Ok(Value::tensor(
                values[row_start..row_end].to_vec(),
                new_shape,
            ))
        }
        _ => Err(SomaError::DataStore(
            "get_rows only works on Tensor values".into(),
        )),
    }
}

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
    /// Data stored as a Zarr v3 array in object storage (chunked tensors).
    Zarr {
        bucket: String,
        /// Root path of the Zarr array (contains zarr.json + chunk objects).
        array_path: String,
        region: Option<String>,
    },
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
        endpoint: Option<String>,
    },
    /// Zarr v3 chunked storage on S3-compatible backend.
    #[serde(rename = "zarr")]
    Zarr {
        bucket: String,
        prefix: String,
        region: Option<String>,
        endpoint: Option<String>,
        /// Rows per chunk (first dimension).
        chunk_rows: usize,
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

    /// Read a range of rows `[start..start+len)` from a tensor.
    /// Returns a `Value::Tensor` with `shape[0] == len`.
    /// Default impl downloads the full value and slices in memory.
    fn get_rows(&self, data_ref: &DataRef, start: usize, len: usize) -> Result<Value> {
        let value = self.get(data_ref)?;
        slice_tensor_rows(&value, start, len)
    }

    /// Get metadata about a stored value without reading the data.
    /// Default impl downloads the full value to extract metadata.
    fn meta(&self, data_ref: &DataRef) -> Result<StoreMeta> {
        let value = self.get(data_ref)?;
        Ok(StoreMeta::from_value(&value))
    }
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
            DataRef::Cached { cache_key } => Ok(self.base_path.join(cache_key.to_hex()).exists()),
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
        self.states
            .insert(filter_id.to_string(), (state_key, state));
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
            DataRef::Local {
                path: "/tmp/x".into(),
            },
            DataRef::S3 {
                bucket: "b".into(),
                key: "k".into(),
                region: None,
            },
            DataRef::Cached {
                cache_key: CacheKey::hash_data(b"x"),
            },
            DataRef::Inline {
                value: Value::Empty,
            },
            DataRef::Zarr {
                bucket: "b".into(),
                array_path: "data/abc".into(),
                region: None,
            },
        ];
        for r in &refs {
            let json = serde_json::to_string(r).unwrap();
            let _: DataRef = serde_json::from_str(&json).unwrap();
        }
    }

    #[test]
    fn slice_tensor_rows_basic() {
        // 4 rows × 3 cols
        let v = Value::tensor(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            vec![4, 3],
        );
        // Rows 1..3 → [[4,5,6], [7,8,9]]
        let sliced = slice_tensor_rows(&v, 1, 2).unwrap();
        let (data, shape) = sliced.as_tensor().unwrap();
        assert_eq!(shape, &[2, 3]);
        assert_eq!(data, &[4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn slice_tensor_rows_single() {
        let v = Value::tensor(vec![10.0, 20.0, 30.0], vec![3]);
        let sliced = slice_tensor_rows(&v, 1, 1).unwrap();
        let (data, shape) = sliced.as_tensor().unwrap();
        assert_eq!(shape, &[1]);
        assert_eq!(data, &[20.0]);
    }

    #[test]
    fn slice_tensor_rows_out_of_bounds() {
        let v = Value::tensor(vec![1.0, 2.0, 3.0], vec![3]);
        assert!(slice_tensor_rows(&v, 2, 5).is_err());
    }

    #[test]
    fn store_meta_from_tensor() {
        let v = Value::tensor(vec![0.0; 12], vec![4, 3]);
        let meta = StoreMeta::from_value(&v);
        assert_eq!(meta.total_rows, 4);
        assert_eq!(meta.shape_tail, vec![3]);
        assert_eq!(meta.dtype, "tensor");
    }

    #[test]
    fn store_meta_from_json() {
        let v = Value::json(serde_json::json!({"a": 1}));
        let meta = StoreMeta::from_value(&v);
        assert_eq!(meta.dtype, "json");
        assert_eq!(meta.total_rows, 1);
    }

    #[test]
    fn default_get_rows_on_local_store() {
        let dir = std::env::temp_dir().join("soma-ds-test-getrows");
        let store = LocalDataStore::new(&dir);

        let key = CacheKey::hash_data(b"rows_test");
        let value = Value::tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        let data_ref = store.put(&key, &value).unwrap();

        // Read rows 1..2 via default impl (full get + slice)
        let sliced = store.get_rows(&data_ref, 1, 2).unwrap();
        let (data, shape) = sliced.as_tensor().unwrap();
        assert_eq!(shape, &[2, 2]);
        assert_eq!(data, &[3.0, 4.0, 5.0, 6.0]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn default_meta_on_local_store() {
        let dir = std::env::temp_dir().join("soma-ds-test-meta");
        let store = LocalDataStore::new(&dir);

        let key = CacheKey::hash_data(b"meta_test");
        let value = Value::tensor(vec![0.0; 20], vec![5, 4]);
        let data_ref = store.put(&key, &value).unwrap();

        let meta = store.meta(&data_ref).unwrap();
        assert_eq!(meta.total_rows, 5);
        assert_eq!(meta.shape_tail, vec![4]);
        assert_eq!(meta.dtype, "tensor");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn zarr_storage_config_serde() {
        let zarr = StorageConfig::Zarr {
            bucket: "soma-research".into(),
            prefix: "data/".into(),
            region: None,
            endpoint: Some("s3.eu-central-003.backblazeb2.com".into()),
            chunk_rows: 1024,
        };
        let json = serde_json::to_string(&zarr).unwrap();
        assert!(json.contains("soma-research"));
        assert!(json.contains("1024"));
        let _: StorageConfig = serde_json::from_str(&json).unwrap();
    }
}
