//! Building a [`DataStore`] from the arguments Python passes.
//!
//! Shared by `Graph.set_data_store` and `Worker.set_data_store`, and the
//! sharing is the point: the two ends of a transfer have to agree, so the
//! second one existing at all is what makes the first one useful. A client
//! that uploads to a bucket needs a worker that can read the same bucket,
//! and until this was factored out only the client could be told about one.

use pyo3::prelude::*;
use std::sync::Arc;

use somatize_core::store::DataStore;

/// Construct a store from `(store_type, …)`, or explain what is missing.
///
/// The argument names match `Graph.set_data_store` exactly, because they
/// are the same arguments — a worker and its client are configured with
/// the same call.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_data_store(
    store_type: &str,
    path: Option<String>,
    bucket: Option<String>,
    prefix: Option<String>,
    endpoint: Option<String>,
    access_key: Option<String>,
    secret_key: Option<String>,
    cache_dir: Option<String>,
) -> PyResult<Arc<dyn DataStore>> {
    match store_type {
        "local" => {
            let p = path.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("local store requires 'path'")
            })?;
            Ok(Arc::new(somatize_core::store::LocalDataStore::new(p)))
        }
        "s3" => {
            let bucket = bucket.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("s3 store requires 'bucket'")
            })?;
            let prefix = prefix.unwrap_or_default();
            let endpoint = endpoint.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("s3 store requires 'endpoint'")
            })?;
            let ak = access_key
                .or_else(|| std::env::var("AWS_ACCESS_KEY_ID").ok())
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "s3 store requires 'access_key' or AWS_ACCESS_KEY_ID env var",
                    )
                })?;
            let sk = secret_key
                .or_else(|| std::env::var("AWS_SECRET_ACCESS_KEY").ok())
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "s3 store requires 'secret_key' or AWS_SECRET_ACCESS_KEY env var",
                    )
                })?;
            let cache = cache_dir.unwrap_or_else(|| {
                std::env::temp_dir()
                    .join(format!("soma-s3-cache-{bucket}"))
                    .to_string_lossy()
                    .to_string()
            });
            let store = somatize_store::S3DataStore::new(bucket, prefix, endpoint, ak, sk, cache)
                .map_err(crate::prelude::soma_err_to_py)?;
            Ok(Arc::new(store))
        }
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown store type: '{other}'. Available: local, s3"
        ))),
    }
}
