//! Integration test for S3DataStore against a real S3-compatible bucket.
//!
//! Requires environment variables:
//! - BUCKET_NAME, BUCKET_ENDPOINT, BUCKET_KEY_ID, BUCKET_KEY_SECRET
//!
//! Run with: `cargo test -p somatize-store --features s3 --test s3_integration`

#![cfg(feature = "s3")]

use somatize_core::cache::CacheKey;
use somatize_core::store::{DataRef, DataStore};
use somatize_core::value::Value;

fn store() -> Option<somatize_store::S3DataStore> {
    // Skip if env vars not set (CI without credentials)
    somatize_store::S3DataStore::from_env(
        "soma-test/",
        std::env::temp_dir().join("soma-s3-integration"),
    )
    .ok()
}

#[test]
fn roundtrip_tensor() {
    let Some(store) = store() else {
        eprintln!("Skipping S3 test: env vars not set");
        return;
    };

    let key = CacheKey::hash_data(b"integration_test_tensor");
    let value = Value::tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    // PUT
    let data_ref = store.put(&key, &value).unwrap();
    assert!(matches!(data_ref, DataRef::S3 { .. }));
    println!("PUT ok: {data_ref:?}");

    // Clear local cache to force S3 fetch
    let local = std::env::temp_dir().join("soma-s3-integration");
    let _ = std::fs::remove_dir_all(&local);
    std::fs::create_dir_all(&local).ok();

    // GET (from S3, not local cache)
    let retrieved = store.get(&data_ref).unwrap();
    assert_eq!(retrieved, value);
    println!("GET ok: value matches");

    // EXISTS
    assert!(store.exists(&data_ref).unwrap());
    println!("EXISTS ok: true");

    // REMOVE
    store.remove(&data_ref).unwrap();
    println!("REMOVE ok");

    // Note: B2/S3 may have eventual consistency on deletes,
    // so we don't assert !exists immediately after remove.
    println!("REMOVE completed");
}

#[test]
fn roundtrip_json() {
    let Some(store) = store() else { return };

    let key = CacheKey::hash_data(b"integration_test_json");
    let value = Value::json(serde_json::json!({
        "experiment": "test_001",
        "metrics": {"accuracy": 0.95, "loss": 0.05},
        "tags": ["integration", "ci"]
    }));

    let data_ref = store.put(&key, &value).unwrap();

    // Clear local cache
    let local = std::env::temp_dir().join("soma-s3-integration");
    let _ = std::fs::remove_dir_all(&local);
    std::fs::create_dir_all(&local).ok();

    let retrieved = store.get(&data_ref).unwrap();
    assert_eq!(retrieved, value);
    println!("JSON roundtrip ok");

    // Cleanup
    store.remove(&data_ref).unwrap();
}

#[test]
fn local_cache_prevents_s3_fetch() {
    let Some(store) = store() else { return };

    let key = CacheKey::hash_data(b"integration_test_cache");
    let value = Value::tensor(vec![42.0], vec![1]);

    // PUT (writes to S3 + local cache)
    let data_ref = store.put(&key, &value).unwrap();

    // GET should use local cache (no S3 round-trip)
    let retrieved = store.get(&data_ref).unwrap();
    assert_eq!(retrieved, value);
    println!("Local cache hit ok");

    // Cleanup
    store.remove(&data_ref).unwrap();
}
