//! S3-compatible DataStore implementation.
//!
//! Requires the `s3` feature flag:
//! ```toml
//! soma-core = { path = "../soma-core", features = ["s3"] }
//! ```
//!
//! Works with AWS S3, Backblaze B2, MinIO, and any S3-compatible service.
//! Uses the same `rust-s3` + `tokio-rustls-tls` setup as garras_video.
//! Async operations are bridged to sync via a dedicated tokio runtime.

use crate::cache::CacheKey;
use crate::error::{Result, SomaError};
use crate::store::{DataRef, DataStore, StorageConfig};
use crate::value::Value;
use s3::creds::Credentials;
use s3::{Bucket, Region};
use std::path::PathBuf;

/// S3-compatible data store with AWS Signature V4 authentication.
///
/// Stores values as JSON objects keyed by CacheKey hex.
/// Maintains a local filesystem cache for frequently accessed data.
pub struct S3DataStore {
    config: StorageConfig,
    bucket: Box<Bucket>,
    prefix: String,
    local_cache: PathBuf,
    rt: tokio::runtime::Runtime,
}

impl S3DataStore {
    /// Create a new S3 data store.
    ///
    /// For Backblaze B2:
    /// ```ignore
    /// S3DataStore::new(
    ///     "my-bucket",
    ///     "data/",
    ///     "s3.eu-central-003.backblazeb2.com",
    ///     "key_id",
    ///     "key_secret",
    ///     "/tmp/soma-s3-cache",
    /// )
    /// ```
    pub fn new(
        bucket_name: impl Into<String>,
        prefix: impl Into<String>,
        endpoint: impl Into<String>,
        access_key: impl Into<String>,
        secret_key: impl Into<String>,
        local_cache: impl Into<PathBuf>,
    ) -> Result<Self> {
        let bucket_name = bucket_name.into();
        let prefix = prefix.into();
        let endpoint = endpoint.into();
        let cache_dir = local_cache.into();
        std::fs::create_dir_all(&cache_dir).ok();

        let region = Region::Custom {
            region: String::new(),
            endpoint: format!("https://{endpoint}"),
        };

        let credentials = Credentials::new(
            Some(&access_key.into()),
            Some(&secret_key.into()),
            None,
            None,
            None,
        )
        .map_err(|e| SomaError::DataStore(format!("S3 credentials error: {e}")))?;

        let bucket = Bucket::new(&bucket_name, region, credentials)
            .map_err(|e| SomaError::DataStore(format!("S3 bucket error: {e}")))?
            .with_path_style();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| SomaError::DataStore(format!("tokio runtime error: {e}")))?;

        Ok(Self {
            config: StorageConfig::S3 {
                bucket: bucket_name,
                prefix: prefix.clone(),
                region: None,
                endpoint: Some(endpoint),
            },
            bucket,
            prefix,
            local_cache: cache_dir,
            rt,
        })
    }

    /// Create from environment variables:
    /// `BUCKET_NAME`, `BUCKET_ENDPOINT`, `BUCKET_KEY_ID`, `BUCKET_KEY_SECRET`.
    pub fn from_env(prefix: impl Into<String>, local_cache: impl Into<PathBuf>) -> Result<Self> {
        let bucket_name = std::env::var("BUCKET_NAME")
            .map_err(|_| SomaError::DataStore("BUCKET_NAME not set".into()))?;
        let endpoint = std::env::var("BUCKET_ENDPOINT")
            .map_err(|_| SomaError::DataStore("BUCKET_ENDPOINT not set".into()))?;
        let key_id = std::env::var("BUCKET_KEY_ID")
            .map_err(|_| SomaError::DataStore("BUCKET_KEY_ID not set".into()))?;
        let key_secret = std::env::var("BUCKET_KEY_SECRET")
            .map_err(|_| SomaError::DataStore("BUCKET_KEY_SECRET not set".into()))?;

        Self::new(bucket_name, prefix, endpoint, key_id, key_secret, local_cache)
    }

    fn s3_key(&self, cache_key: &CacheKey) -> String {
        format!("{}{}", self.prefix, cache_key.to_hex())
    }

    fn local_path(&self, cache_key: &CacheKey) -> PathBuf {
        self.local_cache.join(cache_key.to_hex())
    }
}

impl DataStore for S3DataStore {
    fn put(&self, key: &CacheKey, data: &Value) -> Result<DataRef> {
        let bytes = serde_json::to_vec(data)
            .map_err(|e| SomaError::DataStore(format!("serialize failed: {e}")))?;

        let s3_key = self.s3_key(key);

        self.rt
            .block_on(self.bucket.put_object(&s3_key, &bytes))
            .map_err(|e| SomaError::DataStore(format!("S3 PUT failed: {e}")))?;

        // Cache locally
        std::fs::write(self.local_path(key), &bytes).ok();

        Ok(DataRef::S3 {
            bucket: self.bucket.name().to_string(),
            key: s3_key,
            region: None,
        })
    }

    fn get(&self, data_ref: &DataRef) -> Result<Value> {
        match data_ref {
            DataRef::S3 { key, .. } => {
                // Try local cache first
                let hex = key.rsplit('/').next().unwrap_or(key);
                let local_path = self.local_cache.join(hex);
                if local_path.exists() {
                    let bytes =
                        std::fs::read(&local_path).map_err(|e| SomaError::DataStore(e.to_string()))?;
                    return serde_json::from_slice(&bytes)
                        .map_err(|e| SomaError::DataStore(e.to_string()));
                }

                let response = self
                    .rt
                    .block_on(self.bucket.get_object(key))
                    .map_err(|e| SomaError::DataStore(format!("S3 GET failed: {e}")))?;

                let bytes = response.bytes();
                std::fs::write(&local_path, bytes).ok();

                serde_json::from_slice(bytes)
                    .map_err(|e| SomaError::DataStore(format!("deserialize failed: {e}")))
            }
            DataRef::Cached { cache_key } => {
                let local_path = self.local_path(cache_key);
                if local_path.exists() {
                    let bytes =
                        std::fs::read(&local_path).map_err(|e| SomaError::DataStore(e.to_string()))?;
                    return serde_json::from_slice(&bytes)
                        .map_err(|e| SomaError::DataStore(e.to_string()));
                }
                let s3_key = self.s3_key(cache_key);
                let response = self
                    .rt
                    .block_on(self.bucket.get_object(&s3_key))
                    .map_err(|e| SomaError::DataStore(format!("S3 GET failed: {e}")))?;
                let bytes = response.bytes();
                std::fs::write(&local_path, bytes).ok();
                serde_json::from_slice(bytes).map_err(|e| SomaError::DataStore(e.to_string()))
            }
            DataRef::Inline { value } => Ok(value.clone()),
            DataRef::Local { path } => {
                let bytes = std::fs::read(path).map_err(|e| SomaError::DataStore(e.to_string()))?;
                serde_json::from_slice(&bytes).map_err(|e| SomaError::DataStore(e.to_string()))
            }
            _ => Err(SomaError::DataStore(
                "Unsupported DataRef type for S3DataStore".into(),
            )),
        }
    }

    fn exists(&self, data_ref: &DataRef) -> Result<bool> {
        match data_ref {
            DataRef::S3 { key, .. } => {
                let result = self.rt.block_on(self.bucket.head_object(key));
                Ok(result.is_ok())
            }
            DataRef::Cached { cache_key } => {
                if self.local_path(cache_key).exists() {
                    return Ok(true);
                }
                let s3_key = self.s3_key(cache_key);
                Ok(self.rt.block_on(self.bucket.head_object(&s3_key)).is_ok())
            }
            DataRef::Inline { .. } => Ok(true),
            DataRef::Local { path } => Ok(std::path::Path::new(path).exists()),
            _ => Ok(false),
        }
    }

    fn remove(&self, data_ref: &DataRef) -> Result<()> {
        match data_ref {
            DataRef::S3 { key, .. } => {
                self.rt
                    .block_on(self.bucket.delete_object(key))
                    .map_err(|e| SomaError::DataStore(format!("S3 DELETE failed: {e}")))?;
                let hex = key.rsplit('/').next().unwrap_or(key);
                let _ = std::fs::remove_file(self.local_cache.join(hex));
            }
            DataRef::Cached { cache_key } => {
                let _ = std::fs::remove_file(self.local_path(cache_key));
                let s3_key = self.s3_key(cache_key);
                let _ = self.rt.block_on(self.bucket.delete_object(&s3_key));
            }
            DataRef::Local { path } => {
                let _ = std::fs::remove_file(path);
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
    fn constructs_from_params() {
        let store = S3DataStore::new(
            "test-bucket",
            "data/",
            "s3.eu-central-003.backblazeb2.com",
            "fake_key_id",
            "fake_key_secret",
            std::env::temp_dir().join("soma-s3-test-construct"),
        );
        assert!(store.is_ok());
    }

    #[test]
    fn s3_key_generation() {
        let store = S3DataStore::new(
            "b",
            "prefix/",
            "localhost:9000",
            "key",
            "secret",
            "/tmp/s3test-keygen",
        )
        .unwrap();
        let key = CacheKey::hash_data(b"test");
        let s3_key = store.s3_key(&key);
        assert!(s3_key.starts_with("prefix/"));
        assert_eq!(s3_key.len(), "prefix/".len() + 64);
    }

    #[test]
    fn config_is_s3() {
        let store =
            S3DataStore::new("b", "p/", "localhost", "k", "s", "/tmp/s3test-cfg").unwrap();
        assert!(matches!(store.config(), StorageConfig::S3 { .. }));
    }
}
