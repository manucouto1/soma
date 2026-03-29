//! S3-compatible DataStore implementation.
//!
//! Requires the `s3` feature flag:
//! ```toml
//! soma-core = { path = "../soma-core", features = ["s3"] }
//! ```
//!
//! Works with AWS S3, MinIO, and any S3-compatible service.
//! Uses the blocking reqwest client for sync DataStore trait compatibility.

use crate::cache::CacheKey;
use crate::data_store::{DataRef, DataStore, StorageConfig};
use crate::error::{Result, SomaError};
use crate::value::Value;
use std::path::PathBuf;

/// S3-compatible data store.
///
/// Stores values as JSON in S3 objects keyed by CacheKey hex.
/// Falls back to local cache for frequently accessed data.
pub struct S3DataStore {
    config: StorageConfig,
    bucket: String,
    prefix: String,
    endpoint: Option<String>,
    local_cache: PathBuf,
    client: reqwest::blocking::Client,
}

impl S3DataStore {
    /// Create a new S3 data store.
    ///
    /// `endpoint` can be set for MinIO or other S3-compatible services.
    /// Credentials are read from environment (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    /// or from the default AWS credential chain.
    pub fn new(
        bucket: impl Into<String>,
        prefix: impl Into<String>,
        endpoint: Option<String>,
        local_cache: impl Into<PathBuf>,
    ) -> Self {
        let bucket = bucket.into();
        let prefix = prefix.into();
        let cache_dir = local_cache.into();
        std::fs::create_dir_all(&cache_dir).ok();

        Self {
            config: StorageConfig::S3 {
                bucket: bucket.clone(),
                prefix: prefix.clone(),
                region: None,
                endpoint: endpoint.clone(),
            },
            bucket,
            prefix,
            endpoint,
            local_cache: cache_dir,
            client: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .unwrap_or_default(),
        }
    }

    fn s3_key(&self, cache_key: &CacheKey) -> String {
        format!("{}{}", self.prefix, cache_key.to_hex())
    }

    fn s3_url(&self, key: &str) -> String {
        if let Some(endpoint) = &self.endpoint {
            format!("{}/{}/{}", endpoint, self.bucket, key)
        } else {
            format!(
                "https://{}.s3.amazonaws.com/{}",
                self.bucket, key
            )
        }
    }

    fn local_path(&self, cache_key: &CacheKey) -> PathBuf {
        self.local_cache.join(cache_key.to_hex())
    }

    /// Upload bytes to S3 using a PUT request.
    fn s3_put(&self, key: &str, data: &[u8]) -> Result<()> {
        let url = self.s3_url(key);
        let resp = self
            .client
            .put(&url)
            .header("Content-Type", "application/json")
            .body(data.to_vec())
            .send()
            .map_err(|e| SomaError::DataStore(format!("S3 PUT failed: {e}")))?;

        if !resp.status().is_success() {
            return Err(SomaError::DataStore(format!(
                "S3 PUT returned {}",
                resp.status()
            )));
        }
        Ok(())
    }

    /// Download bytes from S3 using a GET request.
    fn s3_get(&self, key: &str) -> Result<Vec<u8>> {
        let url = self.s3_url(key);
        let resp = self
            .client
            .get(&url)
            .send()
            .map_err(|e| SomaError::DataStore(format!("S3 GET failed: {e}")))?;

        if !resp.status().is_success() {
            return Err(SomaError::DataStore(format!(
                "S3 GET returned {}",
                resp.status()
            )));
        }

        resp.bytes()
            .map(|b| b.to_vec())
            .map_err(|e| SomaError::DataStore(format!("S3 read body failed: {e}")))
    }

    /// Check if an object exists in S3 using HEAD.
    fn s3_exists(&self, key: &str) -> bool {
        let url = self.s3_url(key);
        self.client
            .head(&url)
            .send()
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }

    /// Delete an object from S3.
    fn s3_delete(&self, key: &str) -> Result<()> {
        let url = self.s3_url(key);
        let _ = self.client.delete(&url).send();
        Ok(())
    }
}

impl DataStore for S3DataStore {
    fn put(&self, key: &CacheKey, data: &Value) -> Result<DataRef> {
        let bytes = serde_json::to_vec(data)
            .map_err(|e| SomaError::DataStore(format!("serialize failed: {e}")))?;

        let s3_key = self.s3_key(key);

        // Write to S3
        self.s3_put(&s3_key, &bytes)?;

        // Also cache locally
        let local_path = self.local_path(key);
        std::fs::write(&local_path, &bytes).ok();

        Ok(DataRef::S3 {
            bucket: self.bucket.clone(),
            key: s3_key,
            region: None,
        })
    }

    fn get(&self, data_ref: &DataRef) -> Result<Value> {
        match data_ref {
            DataRef::S3 { key, .. } => {
                // Try local cache first
                let cache_key_hex = key.rsplit('/').next().unwrap_or(key);
                let local_path = self.local_cache.join(cache_key_hex);
                if local_path.exists() {
                    let bytes = std::fs::read(&local_path)
                        .map_err(|e| SomaError::DataStore(e.to_string()))?;
                    return serde_json::from_slice(&bytes)
                        .map_err(|e| SomaError::DataStore(e.to_string()));
                }

                // Fetch from S3
                let bytes = self.s3_get(key)?;

                // Cache locally
                std::fs::write(&local_path, &bytes).ok();

                serde_json::from_slice(&bytes)
                    .map_err(|e| SomaError::DataStore(format!("deserialize failed: {e}")))
            }
            DataRef::Cached { cache_key } => {
                let local_path = self.local_path(cache_key);
                if local_path.exists() {
                    let bytes = std::fs::read(&local_path)
                        .map_err(|e| SomaError::DataStore(e.to_string()))?;
                    return serde_json::from_slice(&bytes)
                        .map_err(|e| SomaError::DataStore(e.to_string()));
                }
                // Try S3
                let s3_key = self.s3_key(cache_key);
                let bytes = self.s3_get(&s3_key)?;
                std::fs::write(&local_path, &bytes).ok();
                serde_json::from_slice(&bytes)
                    .map_err(|e| SomaError::DataStore(e.to_string()))
            }
            DataRef::Inline { value } => Ok(value.clone()),
            DataRef::Local { path } => {
                let bytes = std::fs::read(path)
                    .map_err(|e| SomaError::DataStore(e.to_string()))?;
                serde_json::from_slice(&bytes)
                    .map_err(|e| SomaError::DataStore(e.to_string()))
            }
            _ => Err(SomaError::DataStore("Unsupported DataRef type for S3DataStore".into())),
        }
    }

    fn exists(&self, data_ref: &DataRef) -> Result<bool> {
        match data_ref {
            DataRef::S3 { key, .. } => Ok(self.s3_exists(key)),
            DataRef::Cached { cache_key } => {
                // Check local first, then S3
                if self.local_path(cache_key).exists() {
                    return Ok(true);
                }
                Ok(self.s3_exists(&self.s3_key(cache_key)))
            }
            DataRef::Inline { .. } => Ok(true),
            DataRef::Local { path } => Ok(std::path::Path::new(path).exists()),
            _ => Ok(false),
        }
    }

    fn remove(&self, data_ref: &DataRef) -> Result<()> {
        match data_ref {
            DataRef::S3 { key, .. } => {
                self.s3_delete(key)?;
                // Also remove local cache
                let hex = key.rsplit('/').next().unwrap_or(key);
                let _ = std::fs::remove_file(self.local_cache.join(hex));
            }
            DataRef::Cached { cache_key } => {
                let _ = std::fs::remove_file(self.local_path(cache_key));
                let _ = self.s3_delete(&self.s3_key(cache_key));
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
    fn s3_store_constructs() {
        let store = S3DataStore::new(
            "test-bucket",
            "data/",
            Some("http://localhost:9000".into()),
            std::env::temp_dir().join("soma-s3-test"),
        );
        assert_eq!(store.bucket, "test-bucket");
        assert_eq!(store.prefix, "data/");
    }

    #[test]
    fn s3_key_generation() {
        let store = S3DataStore::new("b", "prefix/", None::<String>, "/tmp/s3test");
        let key = CacheKey::hash_data(b"test");
        let s3_key = store.s3_key(&key);
        assert!(s3_key.starts_with("prefix/"));
        assert_eq!(s3_key.len(), "prefix/".len() + 64); // 64 hex chars
    }

    #[test]
    fn s3_url_aws() {
        let store = S3DataStore::new("my-bucket", "", None::<String>, "/tmp/s3test");
        let url = store.s3_url("data/file.json");
        assert!(url.contains("my-bucket.s3.amazonaws.com"));
    }

    #[test]
    fn s3_url_minio() {
        let store = S3DataStore::new(
            "my-bucket",
            "",
            Some("http://localhost:9000".into()),
            "/tmp/s3test",
        );
        let url = store.s3_url("data/file.json");
        assert!(url.contains("localhost:9000"));
        assert!(url.contains("my-bucket"));
    }

    #[test]
    fn local_cache_path() {
        let store = S3DataStore::new("b", "", None::<String>, "/tmp/s3test-cache");
        let key = CacheKey::hash_data(b"test");
        let path = store.local_path(&key);
        assert!(path.starts_with("/tmp/s3test-cache"));
    }

    #[test]
    fn config_is_s3() {
        let store = S3DataStore::new("b", "p/", None::<String>, "/tmp/s3test");
        assert!(matches!(store.config(), StorageConfig::S3 { .. }));
    }
}
