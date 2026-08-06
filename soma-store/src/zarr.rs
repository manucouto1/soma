//! Zarr v3 DataStore implementation using `object_store`.
//!
//! Stores tensors as Zarr v3 arrays: a `zarr.json` metadata file plus
//! binary chunk objects (raw little-endian f64). Non-tensor values
//! (JSON, Bytes) fall back to plain JSON objects.
//!
//! Chunk layout:
//! ```text
//! {prefix}{hex}/
//!   zarr.json          ← shape, dtype, chunk_grid
//!   c/0                ← chunk 0 (1D) or c/0/0 (2D+)
//!   c/1                ← chunk 1
//!   ...
//! ```
//!
//! Requires the `zarr` feature flag:
//! ```toml
//! soma-core = { path = "../soma-core", features = ["zarr"] }
//! ```

use object_store::ObjectStore as ObjStore;
use object_store::aws::AmazonS3Builder;
use object_store::path::Path as ObjectPath;
use serde::{Deserialize, Serialize};
use somatize_core::cache::CacheKey;
use somatize_core::data::store::{DataRef, DataStore, StorageConfig, StoreMeta};
use somatize_core::data::value::Value;
use somatize_core::error::{Result, SomaError};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

const ELEMENT_SIZE: usize = 8; // f64 = 8 bytes
const ZSTD_LEVEL: i32 = 3; // default zstd compression level

/// Byte shuffle: transpose an array of N elements × 8 bytes.
/// Groups all byte-0s together, then byte-1s, etc.
/// This makes similar-magnitude floats highly compressible.
fn byte_shuffle(data: &[u8]) -> Vec<u8> {
    let n = data.len() / ELEMENT_SIZE;
    let mut shuffled = vec![0u8; data.len()];
    for i in 0..n {
        for b in 0..ELEMENT_SIZE {
            shuffled[b * n + i] = data[i * ELEMENT_SIZE + b];
        }
    }
    shuffled
}

/// Reverse byte shuffle: undo the transpose.
fn byte_unshuffle(data: &[u8]) -> Vec<u8> {
    let n = data.len() / ELEMENT_SIZE;
    let mut unshuffled = vec![0u8; data.len()];
    for i in 0..n {
        for b in 0..ELEMENT_SIZE {
            unshuffled[i * ELEMENT_SIZE + b] = data[b * n + i];
        }
    }
    unshuffled
}

/// Compress: byte shuffle → zstd.
fn compress_chunk(raw: &[u8]) -> Result<Vec<u8>> {
    let shuffled = byte_shuffle(raw);
    zstd::encode_all(shuffled.as_slice(), ZSTD_LEVEL)
        .map_err(|e| SomaError::DataStore(format!("zstd compress: {e}")))
}

/// Decompress: zstd → byte unshuffle.
fn decompress_chunk(compressed: &[u8]) -> Result<Vec<u8>> {
    let shuffled = zstd::decode_all(compressed)
        .map_err(|e| SomaError::DataStore(format!("zstd decompress: {e}")))?;
    Ok(byte_unshuffle(&shuffled))
}

/// Zarr v3 array metadata (stored as zarr.json).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ZarrMeta {
    zarr_format: u8,
    node_type: String,
    shape: Vec<usize>,
    data_type: String,
    chunk_grid: ChunkGrid,
    fill_value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChunkGrid {
    name: String,
    configuration: ChunkGridConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChunkGridConfig {
    chunk_shape: Vec<usize>,
}

impl ZarrMeta {
    fn new(shape: Vec<usize>, chunk_rows: usize) -> Self {
        let mut chunk_shape = shape.clone();
        if !chunk_shape.is_empty() {
            chunk_shape[0] = chunk_rows.min(shape[0]);
        }
        Self {
            zarr_format: 3,
            node_type: "array".into(),
            shape,
            data_type: "float64".into(),
            chunk_grid: ChunkGrid {
                name: "regular".into(),
                configuration: ChunkGridConfig { chunk_shape },
            },
            fill_value: 0.0,
        }
    }

    fn chunk_rows(&self) -> usize {
        self.chunk_grid
            .configuration
            .chunk_shape
            .first()
            .copied()
            .unwrap_or(1)
    }

    fn n_chunks(&self) -> usize {
        let total = self.shape.first().copied().unwrap_or(0);
        let cr = self.chunk_rows();
        if cr == 0 { 0 } else { total.div_ceil(cr) }
    }

    fn cols(&self) -> usize {
        self.shape
            .get(1..)
            .unwrap_or_default()
            .iter()
            .product::<usize>()
            .max(1)
    }
}

/// Default max local cache size: 512 MB.
const DEFAULT_MAX_CACHE_BYTES: u64 = 512 * 1024 * 1024;

/// LRU tracker for local chunk cache files.
/// Evicts least-recently-used entries when total size exceeds the limit.
struct ChunkLru {
    /// (path, size_bytes) in access order — most recent at back.
    entries: VecDeque<(PathBuf, u64)>,
    current_bytes: u64,
    max_bytes: u64,
}

impl ChunkLru {
    fn new(max_bytes: u64) -> Self {
        Self {
            entries: VecDeque::new(),
            current_bytes: 0,
            max_bytes,
        }
    }

    /// Record a cache write. Evicts old entries if over budget.
    fn record(&mut self, path: PathBuf, size: u64) {
        // Remove if already tracked (will re-add at back)
        if let Some(pos) = self.entries.iter().position(|(p, _)| p == &path) {
            let (_, old_size) = self.entries.remove(pos).unwrap();
            self.current_bytes = self.current_bytes.saturating_sub(old_size);
        }

        self.entries.push_back((path, size));
        self.current_bytes += size;

        // Evict from front until under budget
        while self.current_bytes > self.max_bytes && !self.entries.is_empty() {
            if let Some((evict_path, evict_size)) = self.entries.pop_front() {
                self.current_bytes = self.current_bytes.saturating_sub(evict_size);
                let _ = std::fs::remove_file(&evict_path);
            }
        }
    }

    /// Mark a path as recently accessed (move to back).
    fn touch(&mut self, path: &PathBuf) {
        if let Some(pos) = self.entries.iter().position(|(p, _)| p == path) {
            let entry = self.entries.remove(pos).unwrap();
            self.entries.push_back(entry);
        }
    }

    fn current_bytes(&self) -> u64 {
        self.current_bytes
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}

/// S3-compatible Zarr v3 data store.
pub struct ZarrStore {
    config: StorageConfig,
    store: Arc<dyn ObjStore>,
    prefix: String,
    chunk_rows: usize,
    local_cache: PathBuf,
    lru: Mutex<ChunkLru>,
    rt: tokio::runtime::Runtime,
}

impl ZarrStore {
    /// Create a new ZarrStore.
    pub fn new(
        bucket_name: impl Into<String>,
        prefix: impl Into<String>,
        endpoint: impl Into<String>,
        access_key: impl Into<String>,
        secret_key: impl Into<String>,
        local_cache: impl Into<PathBuf>,
        chunk_rows: usize,
    ) -> Result<Self> {
        let bucket_name = bucket_name.into();
        let prefix = prefix.into();
        let endpoint = endpoint.into();
        let cache_dir = local_cache.into();
        std::fs::create_dir_all(&cache_dir).ok();

        let store = AmazonS3Builder::new()
            .with_bucket_name(&bucket_name)
            .with_endpoint(format!("https://{endpoint}"))
            .with_access_key_id(access_key.into())
            .with_secret_access_key(secret_key.into())
            .with_region("")
            .with_virtual_hosted_style_request(false)
            .build()
            .map_err(|e| SomaError::DataStore(format!("object_store build: {e}")))?;

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| SomaError::DataStore(format!("tokio runtime: {e}")))?;

        Ok(Self {
            config: StorageConfig::Zarr {
                bucket: bucket_name,
                prefix: prefix.clone(),
                region: None,
                endpoint: Some(endpoint),
                chunk_rows,
            },
            store: Arc::new(store),
            prefix,
            chunk_rows,
            local_cache: cache_dir,
            lru: Mutex::new(ChunkLru::new(DEFAULT_MAX_CACHE_BYTES)),
            rt,
        })
    }

    /// Create from environment variables.
    pub fn from_env(
        prefix: impl Into<String>,
        local_cache: impl Into<PathBuf>,
        chunk_rows: usize,
    ) -> Result<Self> {
        let env = |name: &str| -> Result<String> {
            std::env::var(name).map_err(|_| SomaError::DataStore(format!("{name} not set")))
        };
        Self::new(
            env("BUCKET_NAME")?,
            prefix,
            env("BUCKET_ENDPOINT")?,
            env("BUCKET_KEY_ID")?,
            env("BUCKET_KEY_SECRET")?,
            local_cache,
            chunk_rows,
        )
    }

    /// Set the maximum local cache size in bytes.
    pub fn set_max_cache_bytes(&self, max_bytes: u64) {
        let mut lru = self.lru.lock().unwrap_or_else(|e| e.into_inner());
        lru.max_bytes = max_bytes;
    }

    /// Current local cache usage in bytes.
    pub fn cache_bytes(&self) -> u64 {
        self.lru
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .current_bytes()
    }

    /// Number of cached chunk files.
    pub fn cache_entries(&self) -> usize {
        self.lru.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Record a local cache write and evict if needed.
    fn cache_write(&self, path: PathBuf, data: &[u8]) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        std::fs::write(&path, data).ok();
        self.lru
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .record(path, data.len() as u64);
    }

    /// Mark a local cache file as recently accessed.
    fn cache_touch(&self, path: &PathBuf) {
        self.lru
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .touch(path);
    }

    /// Root path for a Zarr array.
    fn array_root(&self, key: &CacheKey) -> String {
        format!("{}{}", self.prefix, key.to_hex())
    }

    /// Path for a plain (non-chunked) object.
    fn object_path(&self, key: &CacheKey) -> String {
        format!("{}{}.json", self.prefix, key.to_hex())
    }

    /// Local cache path for a chunk file.
    fn local_chunk_path(&self, key: &CacheKey, chunk_idx: usize) -> PathBuf {
        self.local_cache
            .join(key.to_hex())
            .join(format!("c_{chunk_idx}"))
    }

    /// Local cache path for zarr metadata.
    fn local_meta_path(&self, key: &CacheKey) -> PathBuf {
        self.local_cache.join(key.to_hex()).join("zarr.json")
    }

    // ── S3 helpers ──

    fn s3_put(&self, path: &str, data: &[u8]) -> Result<()> {
        let obj_path = ObjectPath::from(path);
        let payload = object_store::PutPayload::from(data.to_vec());
        self.rt
            .block_on(self.store.put(&obj_path, payload))
            .map_err(|e| SomaError::DataStore(format!("PUT {path}: {e}")))?;
        Ok(())
    }

    fn s3_get(&self, path: &str) -> Result<Vec<u8>> {
        let obj_path = ObjectPath::from(path);
        let result = self
            .rt
            .block_on(self.store.get(&obj_path))
            .map_err(|e| SomaError::DataStore(format!("GET {path}: {e}")))?;
        let bytes = self
            .rt
            .block_on(result.bytes())
            .map_err(|e| SomaError::DataStore(format!("read bytes {path}: {e}")))?;
        Ok(bytes.to_vec())
    }

    fn s3_head(&self, path: &str) -> bool {
        let obj_path = ObjectPath::from(path);
        self.rt.block_on(self.store.head(&obj_path)).is_ok()
    }

    fn s3_delete(&self, path: &str) -> Result<()> {
        let obj_path = ObjectPath::from(path);
        self.rt
            .block_on(self.store.delete(&obj_path))
            .map_err(|e| SomaError::DataStore(format!("DELETE {path}: {e}")))?;
        Ok(())
    }

    // ── Tensor operations (Zarr v3) ──

    fn put_tensor(&self, key: &CacheKey, values: &[f64], shape: &[usize]) -> Result<DataRef> {
        let root = self.array_root(key);
        let meta = ZarrMeta::new(shape.to_vec(), self.chunk_rows);

        // Write zarr.json
        let meta_json = serde_json::to_vec_pretty(&meta)
            .map_err(|e| SomaError::DataStore(format!("meta serialize: {e}")))?;
        self.s3_put(&format!("{root}/zarr.json"), &meta_json)?;

        // Write chunks as raw little-endian f64 bytes
        let cols = meta.cols();
        let chunk_rows = meta.chunk_rows();
        let total_rows = shape.first().copied().unwrap_or(0);

        for chunk_idx in 0..meta.n_chunks() {
            let row_start = chunk_idx * chunk_rows;
            let row_end = (row_start + chunk_rows).min(total_rows);
            let elem_start = row_start * cols;
            let elem_end = row_end * cols;

            let raw_bytes: Vec<u8> = values[elem_start..elem_end]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();

            let compressed = compress_chunk(&raw_bytes)?;

            let chunk_path = format!("{root}/c/{chunk_idx}");
            self.s3_put(&chunk_path, &compressed)?;

            // Cache locally (compressed form — decompressed on read)
            let local = self.local_chunk_path(key, chunk_idx);
            self.cache_write(local, &compressed);
        }

        // Cache metadata locally
        let local_meta = self.local_meta_path(key);
        self.cache_write(local_meta, &meta_json);

        Ok(DataRef::Zarr {
            bucket: match &self.config {
                StorageConfig::Zarr { bucket, .. } => bucket.clone(),
                _ => String::new(),
            },
            array_path: root,
            region: None,
        })
    }

    fn read_meta(&self, key: &CacheKey, array_path: &str) -> Result<ZarrMeta> {
        let local = self.local_meta_path(key);
        let bytes = if local.exists() {
            self.cache_touch(&local);
            std::fs::read(&local).map_err(|e| SomaError::DataStore(e.to_string()))?
        } else {
            let bytes = self.s3_get(&format!("{array_path}/zarr.json"))?;
            self.cache_write(local, &bytes);
            bytes
        };
        serde_json::from_slice(&bytes)
            .map_err(|e| SomaError::DataStore(format!("meta deserialize: {e}")))
    }

    fn read_chunk(&self, key: &CacheKey, array_path: &str, chunk_idx: usize) -> Result<Vec<f64>> {
        let local = self.local_chunk_path(key, chunk_idx);
        let compressed = if local.exists() {
            self.cache_touch(&local);
            std::fs::read(&local).map_err(|e| SomaError::DataStore(e.to_string()))?
        } else {
            let chunk_path = format!("{array_path}/c/{chunk_idx}");
            let bytes = self.s3_get(&chunk_path)?;
            self.cache_write(local, &bytes);
            bytes
        };

        // Decompress (zstd → unshuffle) then decode f64
        let raw = decompress_chunk(&compressed)?;
        Ok(raw
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
            .collect())
    }

    fn get_tensor_rows_impl(
        &self,
        key: &CacheKey,
        array_path: &str,
        start: usize,
        len: usize,
    ) -> Result<Value> {
        let meta = self.read_meta(key, array_path)?;
        let chunk_rows = meta.chunk_rows();
        let cols = meta.cols();
        let total_rows = meta.shape.first().copied().unwrap_or(0);

        if start + len > total_rows {
            return Err(SomaError::DataStore(format!(
                "row range {start}..{} out of bounds (total: {total_rows})",
                start + len
            )));
        }

        let first_chunk = start / chunk_rows;
        let last_chunk = (start + len - 1) / chunk_rows;

        let mut result = Vec::with_capacity(len * cols);

        for ci in first_chunk..=last_chunk {
            let chunk_data = self.read_chunk(key, array_path, ci)?;
            let chunk_row_start = ci * chunk_rows;

            // Which rows of this chunk do we need?
            let local_start = if ci == first_chunk {
                start - chunk_row_start
            } else {
                0
            };
            let chunk_total_rows = chunk_data.len() / cols;
            let local_end = if ci == last_chunk {
                (start + len) - chunk_row_start
            } else {
                chunk_total_rows
            };

            let elem_start = local_start * cols;
            let elem_end = local_end * cols;
            result.extend_from_slice(&chunk_data[elem_start..elem_end]);
        }

        let mut new_shape = meta.shape.clone();
        new_shape[0] = len;
        Ok(Value::tensor(result, new_shape))
    }

    /// Extract CacheKey from array_path by parsing the hex suffix.
    fn key_from_path(&self, array_path: &str) -> CacheKey {
        let hex = array_path.strip_prefix(&self.prefix).unwrap_or(array_path);
        // Reconstruct key from hex (for local cache paths)
        CacheKey::hash_data(hex.as_bytes())
    }

    // ── Append ──

    /// Append rows to an existing Zarr array.
    ///
    /// Only rewrites the last (partial) chunk and adds new full chunks.
    /// Updates zarr.json with the new total shape.
    ///
    /// ```ignore
    /// let data_ref = store.put(&key, &initial_data)?;
    /// store.append(&data_ref, &Value::tensor(vec![1.0, 2.0], vec![1, 2]))?;
    /// ```
    pub fn append(&self, data_ref: &DataRef, new_rows: &Value) -> Result<()> {
        let DataRef::Zarr { array_path, .. } = data_ref else {
            return Err(SomaError::DataStore(
                "append only works on Zarr DataRefs".into(),
            ));
        };
        let Value::Tensor {
            values: new_values,
            shape: new_shape,
        } = new_rows
        else {
            return Err(SomaError::DataStore(
                "append only works on Tensor values".into(),
            ));
        };

        let key = self.key_from_path(array_path);
        let mut meta = self.read_meta(&key, array_path)?;
        let chunk_rows = meta.chunk_rows();
        let cols = meta.cols();

        // Validate shape compatibility
        let new_cols: usize = new_shape
            .get(1..)
            .unwrap_or_default()
            .iter()
            .product::<usize>()
            .max(1);
        if cols != new_cols {
            return Err(SomaError::DataStore(format!(
                "shape mismatch: array has {} cols, new data has {new_cols}",
                cols
            )));
        }

        let old_total = meta.shape[0];
        let append_rows = new_shape[0];
        let new_total = old_total + append_rows;

        // How many rows are in the last existing chunk?
        let last_chunk_rows = if old_total % chunk_rows == 0 && old_total > 0 {
            chunk_rows // last chunk is full
        } else {
            old_total % chunk_rows
        };

        let mut cursor = 0; // position in new_values

        // If last chunk is partial, read it, extend, rewrite
        if last_chunk_rows < chunk_rows && old_total > 0 {
            let last_chunk_idx = (old_total - 1) / chunk_rows;
            let mut chunk_data = self.read_chunk(&key, array_path, last_chunk_idx)?;

            let can_fill = chunk_rows - last_chunk_rows; // space left
            let take = can_fill.min(append_rows);
            let elem_take = take * cols;

            chunk_data.extend_from_slice(&new_values[..elem_take]);
            cursor = elem_take;

            // Rewrite this chunk
            let raw: Vec<u8> = chunk_data.iter().flat_map(|v| v.to_le_bytes()).collect();
            let compressed = compress_chunk(&raw)?;
            self.s3_put(&format!("{array_path}/c/{last_chunk_idx}"), &compressed)?;
            self.cache_write(self.local_chunk_path(&key, last_chunk_idx), &compressed);
        }

        // Write new full chunks from remaining data
        let remaining_elems = new_values.len() - cursor;
        let remaining_rows = remaining_elems / cols;
        let first_new_chunk =
            (old_total + (chunk_rows - last_chunk_rows).min(append_rows)).div_ceil(chunk_rows);

        let mut chunk_idx = first_new_chunk;
        let mut rows_written = 0;
        while rows_written < remaining_rows {
            let take = chunk_rows.min(remaining_rows - rows_written);
            let elem_start = cursor + rows_written * cols;
            let elem_end = elem_start + take * cols;

            let raw: Vec<u8> = new_values[elem_start..elem_end]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            let compressed = compress_chunk(&raw)?;
            self.s3_put(&format!("{array_path}/c/{chunk_idx}"), &compressed)?;
            self.cache_write(self.local_chunk_path(&key, chunk_idx), &compressed);

            rows_written += take;
            chunk_idx += 1;
        }

        // Update zarr.json with new shape
        meta.shape[0] = new_total;
        let meta_json = serde_json::to_vec_pretty(&meta)
            .map_err(|e| SomaError::DataStore(format!("meta serialize: {e}")))?;
        self.s3_put(&format!("{array_path}/zarr.json"), &meta_json)?;
        self.cache_write(self.local_meta_path(&key), &meta_json);

        Ok(())
    }

    // ── Plain object operations (JSON/Bytes fallback) ──

    fn put_object(&self, key: &CacheKey, value: &Value) -> Result<DataRef> {
        let bytes = serde_json::to_vec(value)
            .map_err(|e| SomaError::DataStore(format!("serialize: {e}")))?;
        let path = self.object_path(key);
        self.s3_put(&path, &bytes)?;
        Ok(DataRef::S3 {
            bucket: match &self.config {
                StorageConfig::Zarr { bucket, .. } => bucket.clone(),
                _ => String::new(),
            },
            key: path,
            region: None,
        })
    }

    fn get_object(&self, key: &str) -> Result<Value> {
        let bytes = self.s3_get(key)?;
        serde_json::from_slice(&bytes)
            .map_err(|e| SomaError::DataStore(format!("deserialize: {e}")))
    }
}

impl DataStore for ZarrStore {
    fn put(&self, key: &CacheKey, data: &Value) -> Result<DataRef> {
        match data {
            Value::Tensor { values, shape } => self.put_tensor(key, values, shape),
            _ => self.put_object(key, data),
        }
    }

    fn get(&self, data_ref: &DataRef) -> Result<Value> {
        match data_ref {
            DataRef::Zarr { array_path, .. } => {
                let key = self.key_from_path(array_path);
                let meta = self.read_meta(&key, array_path)?;
                let total_rows = meta.shape.first().copied().unwrap_or(0);
                self.get_tensor_rows_impl(&key, array_path, 0, total_rows)
            }
            DataRef::S3 { key, .. } => self.get_object(key),
            DataRef::Inline { value } => Ok(value.clone()),
            _ => Err(SomaError::DataStore(
                "unsupported DataRef for ZarrStore".into(),
            )),
        }
    }

    fn get_rows(&self, data_ref: &DataRef, start: usize, len: usize) -> Result<Value> {
        match data_ref {
            DataRef::Zarr { array_path, .. } => {
                let key = self.key_from_path(array_path);
                self.get_tensor_rows_impl(&key, array_path, start, len)
            }
            _ => {
                let value = self.get(data_ref)?;
                somatize_core::data::store::slice_tensor_rows(&value, start, len)
            }
        }
    }

    fn meta(&self, data_ref: &DataRef) -> Result<StoreMeta> {
        match data_ref {
            DataRef::Zarr { array_path, .. } => {
                let key = self.key_from_path(array_path);
                let meta = self.read_meta(&key, array_path)?;
                Ok(StoreMeta {
                    total_rows: meta.shape.first().copied().unwrap_or(0),
                    shape_tail: meta.shape.get(1..).unwrap_or_default().to_vec(),
                    dtype: "tensor".into(),
                })
            }
            _ => {
                let value = self.get(data_ref)?;
                Ok(StoreMeta::from_value(&value))
            }
        }
    }

    fn exists(&self, data_ref: &DataRef) -> Result<bool> {
        match data_ref {
            DataRef::Zarr { array_path, .. } => {
                Ok(self.s3_head(&format!("{array_path}/zarr.json")))
            }
            DataRef::S3 { key, .. } => Ok(self.s3_head(key)),
            DataRef::Inline { .. } => Ok(true),
            _ => Ok(false),
        }
    }

    fn remove(&self, data_ref: &DataRef) -> Result<()> {
        match data_ref {
            DataRef::Zarr { array_path, .. } => {
                // Read meta to know how many chunks to delete
                let key = self.key_from_path(array_path);
                if let Ok(meta) = self.read_meta(&key, array_path) {
                    for ci in 0..meta.n_chunks() {
                        let _ = self.s3_delete(&format!("{array_path}/c/{ci}"));
                    }
                }
                let _ = self.s3_delete(&format!("{array_path}/zarr.json"));
                // Clean local cache
                let local_dir = self
                    .local_cache
                    .join(array_path.strip_prefix(&self.prefix).unwrap_or(array_path));
                let _ = std::fs::remove_dir_all(&local_dir);
            }
            DataRef::S3 { key, .. } => {
                let _ = self.s3_delete(key);
            }
            _ => {}
        }
        Ok(())
    }

    fn config(&self) -> &StorageConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zarr_meta_structure() {
        let meta = ZarrMeta::new(vec![1000, 128], 256);
        assert_eq!(meta.shape, vec![1000, 128]);
        assert_eq!(meta.chunk_rows(), 256);
        assert_eq!(meta.n_chunks(), 4); // ceil(1000/256)
        assert_eq!(meta.cols(), 128);
    }

    #[test]
    fn zarr_meta_1d() {
        let meta = ZarrMeta::new(vec![50], 16);
        assert_eq!(meta.chunk_rows(), 16);
        assert_eq!(meta.n_chunks(), 4); // ceil(50/16)
        assert_eq!(meta.cols(), 1);
    }

    #[test]
    fn zarr_meta_small_tensor() {
        // chunk_rows > total rows → one chunk
        let meta = ZarrMeta::new(vec![10, 3], 1024);
        assert_eq!(meta.chunk_rows(), 10); // capped to total
        assert_eq!(meta.n_chunks(), 1);
    }

    #[test]
    fn zarr_meta_serde_roundtrip() {
        let meta = ZarrMeta::new(vec![500, 64], 100);
        let json = serde_json::to_string(&meta).unwrap();
        let parsed: ZarrMeta = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.shape, meta.shape);
        assert_eq!(parsed.chunk_rows(), meta.chunk_rows());
    }

    #[test]
    fn constructs_from_params() {
        let store = ZarrStore::new(
            "test-bucket",
            "data/",
            "s3.eu-central-003.backblazeb2.com",
            "fake_key",
            "fake_secret",
            std::env::temp_dir().join("soma-zarr-test-construct"),
            256,
        );
        assert!(store.is_ok());
    }

    #[test]
    fn array_root_generation() {
        let store = ZarrStore::new(
            "b",
            "prefix/",
            "localhost:9000",
            "k",
            "s",
            "/tmp/zarrtest",
            64,
        )
        .unwrap();
        let key = CacheKey::hash_data(b"test");
        let root = store.array_root(&key);
        assert!(root.starts_with("prefix/"));
        assert_eq!(root.len(), "prefix/".len() + 64);
    }

    #[test]
    fn byte_shuffle_roundtrip() {
        let values: Vec<f64> = (0..100).map(|i| i as f64 * 0.01).collect();
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let shuffled = byte_shuffle(&raw);
        assert_eq!(shuffled.len(), raw.len());
        assert_ne!(shuffled, raw); // shuffle changed the layout

        let unshuffled = byte_unshuffle(&shuffled);
        assert_eq!(unshuffled, raw); // roundtrip is lossless
    }

    #[test]
    fn compress_decompress_roundtrip() {
        let values: Vec<f64> = (0..1000).map(|i| (i as f64).sin()).collect();
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let compressed = compress_chunk(&raw).unwrap();
        let decompressed = decompress_chunk(&compressed).unwrap();

        assert_eq!(decompressed, raw);
    }

    #[test]
    fn compression_reduces_size_on_patterned_data() {
        // Lots of similar values → good compression
        let values: Vec<f64> = (0..1000).map(|i| 1.0 + (i as f64) * 0.0001).collect();
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let compressed = compress_chunk(&raw).unwrap();
        let ratio = raw.len() as f64 / compressed.len() as f64;

        println!(
            "Patterned data: {} bytes → {} bytes ({:.1}x)",
            raw.len(),
            compressed.len(),
            ratio
        );
        assert!(
            compressed.len() < raw.len(),
            "compression should reduce size on patterned data"
        );
    }

    #[test]
    fn compression_handles_zeros() {
        // All zeros → very compressible
        let raw = vec![0u8; 8000]; // 1000 f64 zeros
        let compressed = compress_chunk(&raw).unwrap();
        let ratio = raw.len() as f64 / compressed.len() as f64;

        println!(
            "Zeros: {} bytes → {} bytes ({:.1}x)",
            raw.len(),
            compressed.len(),
            ratio
        );
        assert!(ratio > 10.0, "zeros should compress at >10x");

        let decompressed = decompress_chunk(&compressed).unwrap();
        assert_eq!(decompressed, raw);
    }

    // ── LRU tests ──

    #[test]
    fn lru_tracks_entries() {
        let mut lru = ChunkLru::new(1000);
        lru.record(PathBuf::from("/tmp/a"), 100);
        lru.record(PathBuf::from("/tmp/b"), 200);

        assert_eq!(lru.len(), 2);
        assert_eq!(lru.current_bytes(), 300);
    }

    #[test]
    fn lru_evicts_when_over_budget() {
        let dir = std::env::temp_dir().join("soma-lru-test-evict");
        std::fs::create_dir_all(&dir).ok();

        let path_a = dir.join("a");
        let path_b = dir.join("b");
        let path_c = dir.join("c");

        // Create real files so eviction can delete them
        std::fs::write(&path_a, [0u8; 100]).unwrap();
        std::fs::write(&path_b, [0u8; 100]).unwrap();
        std::fs::write(&path_c, [0u8; 100]).unwrap();

        let mut lru = ChunkLru::new(250); // budget for ~2 entries
        lru.record(path_a.clone(), 100);
        lru.record(path_b.clone(), 100);
        assert_eq!(lru.len(), 2);
        assert!(path_a.exists());

        // Adding c pushes over budget → evicts a (oldest)
        lru.record(path_c.clone(), 100);
        assert_eq!(lru.current_bytes(), 200); // b + c
        assert!(!path_a.exists(), "a should be evicted");
        assert!(path_b.exists());
        assert!(path_c.exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn lru_touch_prevents_eviction() {
        let dir = std::env::temp_dir().join("soma-lru-test-touch");
        std::fs::create_dir_all(&dir).ok();

        let path_a = dir.join("a");
        let path_b = dir.join("b");
        let path_c = dir.join("c");

        std::fs::write(&path_a, [0u8; 100]).unwrap();
        std::fs::write(&path_b, [0u8; 100]).unwrap();
        std::fs::write(&path_c, [0u8; 100]).unwrap();

        let mut lru = ChunkLru::new(250);
        lru.record(path_a.clone(), 100);
        lru.record(path_b.clone(), 100);

        // Touch a → now b is the oldest
        lru.touch(&path_a);

        // Adding c should evict b (oldest after touch), not a
        lru.record(path_c.clone(), 100);
        assert!(path_a.exists(), "a was touched, should survive");
        assert!(!path_b.exists(), "b was oldest, should be evicted");
        assert!(path_c.exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn lru_duplicate_record_updates_size() {
        let mut lru = ChunkLru::new(1000);
        let path = PathBuf::from("/tmp/x");

        lru.record(path.clone(), 100);
        assert_eq!(lru.current_bytes(), 100);

        // Re-record with different size
        lru.record(path.clone(), 200);
        assert_eq!(lru.len(), 1);
        assert_eq!(lru.current_bytes(), 200);
    }
}
