//! Integration test for ZarrStore against a real S3-compatible bucket.
//!
//! Requires environment variables:
//! - BUCKET_NAME, BUCKET_ENDPOINT, BUCKET_KEY_ID, BUCKET_KEY_SECRET
//!
//! Run with: `cargo test -p soma-core --features zarr --test zarr_integration`

#![cfg(feature = "zarr")]

use somatize_core::cache::CacheKey;
use somatize_core::store::{DataRef, DataStore, ZarrStore};
use somatize_core::value::Value;

fn store() -> Option<ZarrStore> {
    ZarrStore::from_env(
        "soma-zarr-test/",
        std::env::temp_dir().join("soma-zarr-integration"),
        4, // small chunks for testing
    )
    .ok()
}

#[test]
fn roundtrip_tensor_2d() {
    let Some(store) = store() else {
        eprintln!("Skipping: env vars not set");
        return;
    };

    // 6 rows × 3 cols, chunk_rows=4 → 2 chunks (4 rows + 2 rows)
    let key = CacheKey::hash_data(b"zarr_test_2d");
    let data: Vec<f64> = (0..18).map(|i| i as f64).collect();
    let value = Value::tensor(data.clone(), vec![6, 3]);

    let data_ref = store.put(&key, &value).unwrap();
    assert!(matches!(data_ref, DataRef::Zarr { .. }));
    println!("PUT ok: {data_ref:?}");

    // Clear local cache to force S3 fetch
    let cache_dir = std::env::temp_dir().join("soma-zarr-integration");
    let _ = std::fs::remove_dir_all(&cache_dir);
    std::fs::create_dir_all(&cache_dir).ok();

    // Full GET
    let retrieved = store.get(&data_ref).unwrap();
    assert_eq!(retrieved, value);
    println!("Full GET ok");

    // META (no data download since we just cached it)
    let meta = store.meta(&data_ref).unwrap();
    assert_eq!(meta.total_rows, 6);
    assert_eq!(meta.shape_tail, vec![3]);
    assert_eq!(meta.dtype, "tensor");
    println!("META ok: {meta:?}");

    // Cleanup
    store.remove(&data_ref).unwrap();
    println!("REMOVE ok");
}

#[test]
fn partial_read_within_chunk() {
    let Some(store) = store() else { return };

    // 8 rows × 2 cols, chunk_rows=4 → 2 chunks
    let key = CacheKey::hash_data(b"zarr_test_partial");
    let data: Vec<f64> = (0..16).map(|i| i as f64).collect();
    let value = Value::tensor(data, vec![8, 2]);

    let data_ref = store.put(&key, &value).unwrap();

    // Clear local cache
    let cache_dir = std::env::temp_dir().join("soma-zarr-integration");
    let _ = std::fs::remove_dir_all(&cache_dir);
    std::fs::create_dir_all(&cache_dir).ok();

    // Read rows 1..3 (within chunk 0)
    let sliced = store.get_rows(&data_ref, 1, 2).unwrap();
    let (vals, shape) = sliced.as_tensor().unwrap();
    assert_eq!(shape, &[2, 2]);
    assert_eq!(vals, &[2.0, 3.0, 4.0, 5.0]);
    println!("Partial read within chunk ok");

    store.remove(&data_ref).unwrap();
}

#[test]
fn partial_read_across_chunks() {
    let Some(store) = store() else { return };

    // 8 rows × 2 cols, chunk_rows=4 → chunk0=[rows 0-3], chunk1=[rows 4-7]
    let key = CacheKey::hash_data(b"zarr_test_cross_chunk");
    let data: Vec<f64> = (0..16).map(|i| i as f64).collect();
    let value = Value::tensor(data, vec![8, 2]);

    let data_ref = store.put(&key, &value).unwrap();

    // Clear cache
    let cache_dir = std::env::temp_dir().join("soma-zarr-integration");
    let _ = std::fs::remove_dir_all(&cache_dir);
    std::fs::create_dir_all(&cache_dir).ok();

    // Read rows 3..6 — crosses chunk boundary (chunk0 row 3, chunk1 rows 4-5)
    let sliced = store.get_rows(&data_ref, 3, 3).unwrap();
    let (vals, shape) = sliced.as_tensor().unwrap();
    assert_eq!(shape, &[3, 2]);
    assert_eq!(vals, &[6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
    println!("Cross-chunk partial read ok");

    store.remove(&data_ref).unwrap();
}

#[test]
fn variable_batch_reads() {
    let Some(store) = store() else { return };

    // 12 rows × 1 col, chunk_rows=4 → 3 chunks
    let key = CacheKey::hash_data(b"zarr_test_variable");
    let data: Vec<f64> = (0..12).map(|i| i as f64).collect();
    let value = Value::tensor(data, vec![12]);

    let data_ref = store.put(&key, &value).unwrap();

    // Clear cache
    let cache_dir = std::env::temp_dir().join("soma-zarr-integration");
    let _ = std::fs::remove_dir_all(&cache_dir);
    std::fs::create_dir_all(&cache_dir).ok();

    // Simulate variable-size batch events
    // Event 1: 3 rows starting at 0
    let batch1 = store.get_rows(&data_ref, 0, 3).unwrap();
    let (v, s) = batch1.as_tensor().unwrap();
    assert_eq!(s, &[3]);
    assert_eq!(v, &[0.0, 1.0, 2.0]);

    // Event 2: 1 row starting at 3
    let batch2 = store.get_rows(&data_ref, 3, 1).unwrap();
    let (v, _) = batch2.as_tensor().unwrap();
    assert_eq!(v, &[3.0]);

    // Event 3: 7 rows starting at 4 (crosses chunk 1 and chunk 2)
    let batch3 = store.get_rows(&data_ref, 4, 7).unwrap();
    let (v, s) = batch3.as_tensor().unwrap();
    assert_eq!(s, &[7]);
    assert_eq!(v, &[4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

    // Event 4: 1 row at the end
    let batch4 = store.get_rows(&data_ref, 11, 1).unwrap();
    let (v, _) = batch4.as_tensor().unwrap();
    assert_eq!(v, &[11.0]);

    println!("Variable batch reads ok: 3 + 1 + 7 + 1 rows");

    store.remove(&data_ref).unwrap();
}

#[test]
fn json_fallback() {
    let Some(store) = store() else { return };

    let key = CacheKey::hash_data(b"zarr_test_json");
    let value = Value::json(serde_json::json!({"model": "svm", "accuracy": 0.95}));

    let data_ref = store.put(&key, &value).unwrap();
    // JSON goes through plain S3, not Zarr
    assert!(matches!(data_ref, DataRef::S3 { .. }));

    let retrieved = store.get(&data_ref).unwrap();
    assert_eq!(retrieved, value);
    println!("JSON fallback ok");

    store.remove(&data_ref).unwrap();
}

#[test]
fn local_cache_prevents_refetch() {
    let Some(store) = store() else { return };

    let key = CacheKey::hash_data(b"zarr_test_cache_hit");
    let data: Vec<f64> = (0..8).map(|i| i as f64).collect();
    let value = Value::tensor(data, vec![4, 2]);

    let data_ref = store.put(&key, &value).unwrap();

    // First read (populates local cache during PUT)
    let v1 = store.get_rows(&data_ref, 0, 2).unwrap();

    // Second read should hit local cache (no S3 round-trip)
    let v2 = store.get_rows(&data_ref, 0, 2).unwrap();
    assert_eq!(v1, v2);
    println!("Local cache hit ok");

    store.remove(&data_ref).unwrap();
}

#[test]
fn append_variable_batches() {
    let Some(store) = store() else { return };

    // Start with 3 rows × 2 cols, chunk_rows=4
    let key = CacheKey::hash_data(b"zarr_test_append");
    let initial = Value::tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
    let data_ref = store.put(&key, &initial).unwrap();

    // Append 1 row (fills partial chunk 0: was 3 rows, now 4 = full)
    store
        .append(&data_ref, &Value::tensor(vec![7.0, 8.0], vec![1, 2]))
        .unwrap();
    let meta = store.meta(&data_ref).unwrap();
    assert_eq!(meta.total_rows, 4);

    // Append 3 rows (creates new partial chunk 1)
    store
        .append(
            &data_ref,
            &Value::tensor(vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0], vec![3, 2]),
        )
        .unwrap();
    let meta = store.meta(&data_ref).unwrap();
    assert_eq!(meta.total_rows, 7);

    // Clear local cache to force S3 reads
    let cache_dir = std::env::temp_dir().join("soma-zarr-integration");
    let _ = std::fs::remove_dir_all(&cache_dir);
    std::fs::create_dir_all(&cache_dir).ok();

    // Read all 7 rows
    let full = store.get(&data_ref).unwrap();
    let (vals, shape) = full.as_tensor().unwrap();
    assert_eq!(shape, &[7, 2]);
    assert_eq!(
        vals,
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0
        ]
    );

    // Partial read across the append boundary
    let mid = store.get_rows(&data_ref, 2, 4).unwrap();
    let (vals, shape) = mid.as_tensor().unwrap();
    assert_eq!(shape, &[4, 2]);
    assert_eq!(vals, &[5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);

    println!("Append variable batches ok: 3 + 1 + 3 = 7 rows");
    store.remove(&data_ref).unwrap();
}

#[test]
fn append_creates_multiple_new_chunks() {
    let Some(store) = store() else { return };

    // 2 rows initial, chunk_rows=4
    let key = CacheKey::hash_data(b"zarr_test_append_big");
    let initial = Value::tensor(vec![1.0, 2.0], vec![2]);
    let data_ref = store.put(&key, &initial).unwrap();

    // Append 10 rows (fills chunk 0 partial + creates chunk 1 full + chunk 2 partial)
    let big: Vec<f64> = (3..13).map(|i| i as f64).collect();
    store
        .append(&data_ref, &Value::tensor(big, vec![10]))
        .unwrap();

    let meta = store.meta(&data_ref).unwrap();
    assert_eq!(meta.total_rows, 12);

    // Clear cache
    let cache_dir = std::env::temp_dir().join("soma-zarr-integration");
    let _ = std::fs::remove_dir_all(&cache_dir);
    std::fs::create_dir_all(&cache_dir).ok();

    let full = store.get(&data_ref).unwrap();
    let (vals, _) = full.as_tensor().unwrap();
    let expected: Vec<f64> = (1..13).map(|i| i as f64).collect();
    assert_eq!(vals, &expected);

    println!("Append big batch ok: 2 + 10 = 12 rows across 3 chunks");
    store.remove(&data_ref).unwrap();
}
