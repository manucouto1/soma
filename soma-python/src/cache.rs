//! The functions behind the `soma cache` CLI.

use crate::prelude::*;

// ── Cache management (backs the `soma cache` CLI) ──

fn resolve_cache_root(path: Option<String>) -> PyResult<std::path::PathBuf> {
    match path {
        Some(p) => Ok(p.into()),
        None => default_cache_dir().ok_or_else(|| {
            PyRuntimeError::new_err("no cache directory: set SOMA_CACHE_DIR or HOME")
        }),
    }
}

fn open_store(path: Option<String>) -> PyResult<FsActionStore> {
    let root = resolve_cache_root(path)?;
    FsActionStore::new(root).map_err(|e| PyRuntimeError::new_err(format!("cache open: {e}")))
}

/// Stats about the shared persistent cache.
#[pyfunction]
#[pyo3(signature = (path=None))]
pub(crate) fn cache_stats(py: Python<'_>, path: Option<String>) -> PyResult<PyObject> {
    let store = open_store(path)?;
    let actions = store
        .actions()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let cas_bytes = store
        .cas_bytes()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let pinned = store
        .pinned()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let total_compute_ms: u64 = actions.iter().map(|a| a.compute_ms).sum();
    let mut blob_hashes = std::collections::HashSet::new();
    let mut available = 0usize;
    for record in &actions {
        for hash in record.outputs.values() {
            if blob_hashes.insert(*hash)
                && somatize_core::cache::action::BlobStore::contains(&store, hash).unwrap_or(false)
            {
                available += 1;
            }
        }
    }

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("root", store.root().display().to_string())?;
    dict.set_item("actions", actions.len())?;
    dict.set_item("unique_outputs", blob_hashes.len())?;
    dict.set_item("blobs_available", available)?;
    dict.set_item("blobs_evicted", blob_hashes.len() - available)?;
    dict.set_item("cas_bytes", cas_bytes)?;
    dict.set_item("pinned", pinned.len())?;
    dict.set_item("saved_compute_ms", total_compute_ms)?;
    Ok(dict.into_any().unbind())
}

/// Cost-aware GC: evict blobs (records retained) down to `max_bytes`.
#[pyfunction]
#[pyo3(signature = (max_bytes, min_age_secs=3600, path=None))]
pub(crate) fn cache_gc(
    py: Python<'_>,
    max_bytes: u64,
    min_age_secs: u64,
    path: Option<String>,
) -> PyResult<PyObject> {
    let store = open_store(path)?;
    let policy = somatize_runtime::cache::gc::GcPolicy {
        max_bytes,
        min_age: std::time::Duration::from_secs(min_age_secs),
    };
    let report = somatize_runtime::cache::gc::collect(&store, &policy)
        .map_err(|e| PyRuntimeError::new_err(format!("gc: {e}")))?;
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("bytes_before", report.bytes_before)?;
    dict.set_item("bytes_after", report.bytes_after)?;
    dict.set_item("blobs_evicted", report.blobs_evicted)?;
    dict.set_item("blobs_kept", report.blobs_kept)?;
    dict.set_item("pinned_blobs", report.pinned_blobs)?;
    Ok(dict.into_any().unbind())
}

/// Pin an action as a GC root under a human-readable name.
#[pyfunction]
#[pyo3(signature = (name, key_hex, path=None))]
pub(crate) fn cache_pin(name: &str, key_hex: &str, path: Option<String>) -> PyResult<()> {
    if key_hex.len() != 64 || !key_hex.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "key must be a 64-char hex action key",
        ));
    }
    let mut digest = [0u8; 32];
    for (i, byte) in digest.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&key_hex[i * 2..i * 2 + 2], 16)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    }
    let store = open_store(path)?;
    store
        .pin(name, &somatize_core::cache::CacheKey(digest))
        .map_err(|e| PyRuntimeError::new_err(format!("pin: {e}")))
}

/// Verify every referenced blob against its content hash.
#[pyfunction]
#[pyo3(signature = (path=None))]
pub(crate) fn cache_verify(py: Python<'_>, path: Option<String>) -> PyResult<PyObject> {
    use somatize_core::cache::action::BlobStore;
    let store = open_store(path)?;
    let actions = store
        .actions()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let (mut ok, mut missing, mut corrupt) = (0usize, 0usize, 0usize);
    let mut seen = std::collections::HashSet::new();
    for record in &actions {
        for hash in record.outputs.values() {
            if !seen.insert(*hash) {
                continue;
            }
            // get_bytes verifies content and maps corrupt blobs to None,
            // so distinguish via raw existence.
            match (
                store.contains(hash).unwrap_or(false),
                store.get_bytes(hash).ok().flatten(),
            ) {
                (true, Some(_)) => ok += 1,
                (true, None) => corrupt += 1,
                (false, _) => missing += 1,
            }
        }
    }
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("ok", ok)?;
    dict.set_item("missing", missing)?;
    dict.set_item("corrupt", corrupt)?;
    Ok(dict.into_any().unbind())
}

/// Remove legacy Phase-1 (v1) cache entries: `<hh>/<hh>/<hex>.json`
/// shard trees at the store root. The v2 key namespace makes them
/// unreachable — they are pure dead weight.
#[pyfunction]
#[pyo3(signature = (path=None))]
pub(crate) fn cache_purge_v1(py: Python<'_>, path: Option<String>) -> PyResult<PyObject> {
    let root = resolve_cache_root(path)?;
    let mut removed_files = 0u64;
    let mut removed_bytes = 0u64;
    if root.exists() {
        for entry in std::fs::read_dir(&root).map_err(|e| PyRuntimeError::new_err(e.to_string()))? {
            let entry = entry.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let name = entry.file_name().to_string_lossy().to_string();
            // v1 shard dirs are exactly two lowercase hex chars.
            let is_v1_shard = name.len() == 2
                && name.chars().all(|c| c.is_ascii_hexdigit())
                && entry.path().is_dir();
            if is_v1_shard {
                fn walk(dir: &std::path::Path, files: &mut u64, bytes: &mut u64) {
                    if let Ok(entries) = std::fs::read_dir(dir) {
                        for e in entries.filter_map(|e| e.ok()) {
                            let p = e.path();
                            if p.is_dir() {
                                walk(&p, files, bytes);
                            } else {
                                *files += 1;
                                *bytes += e.metadata().map(|m| m.len()).unwrap_or(0);
                            }
                        }
                    }
                }
                walk(&entry.path(), &mut removed_files, &mut removed_bytes);
                std::fs::remove_dir_all(entry.path())
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            }
        }
    }
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("removed_files", removed_files)?;
    dict.set_item("removed_bytes", removed_bytes)?;
    Ok(dict.into_any().unbind())
}
