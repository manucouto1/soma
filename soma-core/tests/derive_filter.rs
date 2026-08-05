//! The `#[derive(SomaFilter)]` macro, exercised end to end.

use serde::{Deserialize, Serialize};
use somatize_core::SomaFilter;
use somatize_core::search::Searchable;
use std::collections::HashMap;

#[derive(SomaFilter, Serialize, Deserialize, Default)]
#[soma(kind = "Trainable", cacheable = true, differentiable = true)]
struct TestScaler {
    #[soma(search(low = 0.1, high = 10.0, scale = "log"))]
    scale: f64,

    #[soma(search(choices = ["standard", "robust", "minmax"]))]
    method: String,

    #[soma(search(low = 1, high = 100))]
    window: usize,

    // Not searchable, not hashed
    #[soma(skip_hash)]
    verbose: bool,
}

#[test]
fn config_hash_deterministic() {
    let f = TestScaler {
        scale: 2.0,
        method: "standard".into(),
        window: 10,
        verbose: false,
    };
    assert_eq!(f.config_hash(), f.config_hash());
}

#[test]
fn config_hash_changes_with_field() {
    let f1 = TestScaler {
        scale: 1.0,
        method: "standard".into(),
        window: 10,
        verbose: false,
    };
    let f2 = TestScaler {
        scale: 2.0,
        method: "standard".into(),
        window: 10,
        verbose: false,
    };
    assert_ne!(f1.config_hash(), f2.config_hash());
}

#[test]
fn config_hash_ignores_skip_hash_field() {
    let f1 = TestScaler {
        scale: 1.0,
        method: "standard".into(),
        window: 10,
        verbose: false,
    };
    let f2 = TestScaler {
        scale: 1.0,
        method: "standard".into(),
        window: 10,
        verbose: true, // different but skipped
    };
    assert_eq!(f1.config_hash(), f2.config_hash());
}

#[test]
fn search_space_has_correct_dimensions() {
    let space = TestScaler::search_space();
    assert_eq!(space.len(), 3); // scale, method, window (verbose not searchable)

    let names: Vec<&str> = space.active_dimensions().iter().map(|d| d.name()).collect();
    assert!(names.contains(&"scale"));
    assert!(names.contains(&"method"));
    assert!(names.contains(&"window"));
}

#[test]
fn from_sample_reconstructs() {
    let params = HashMap::from([
        ("scale".to_string(), serde_json::json!(3.5)),
        ("method".to_string(), serde_json::json!("robust")),
        ("window".to_string(), serde_json::json!(42)),
    ]);

    let scaler = TestScaler::from_sample(&params).unwrap();
    assert!((scaler.scale - 3.5).abs() < f64::EPSILON);
    assert_eq!(scaler.method, "robust");
    assert_eq!(scaler.window, 42);
}

#[test]
fn current_params_extracts() {
    let scaler = TestScaler {
        scale: 5.0,
        method: "minmax".into(),
        window: 25,
        verbose: true,
    };

    let params = scaler.current_params();
    assert_eq!(params["scale"], serde_json::json!(5.0));
    assert_eq!(params["method"], serde_json::json!("minmax"));
    assert_eq!(params["window"], serde_json::json!(25));
    // verbose not included (not searchable)
    assert!(!params.contains_key("verbose"));
}

#[test]
fn soma_meta_values() {
    let scaler = TestScaler::default();
    let meta = scaler.soma_meta();
    assert_eq!(meta.name, "TestScaler");
    assert_eq!(meta.kind, somatize_core::filter::FilterKind::Trainable);
    assert!(meta.cacheable);
    assert!(meta.differentiable);
    assert_eq!(
        meta.stream_mode,
        somatize_core::filter::StreamMode::FixedState
    );
}

// ── Test with auto bool search ──

#[derive(SomaFilter, Serialize, Deserialize, Default)]
#[soma(kind = "Opaque", differentiable = false)]
struct TestOpaque {
    #[soma(search)]
    enabled: bool,
}

#[test]
fn auto_bool_search() {
    let space = TestOpaque::search_space();
    assert_eq!(space.len(), 1);
    let dim = &space.active_dimensions()[0];
    assert_eq!(dim.name(), "enabled");
    // Should be Categorical with true/false
    if let somatize_core::search::SearchDimension::Categorical { choices, .. } = dim {
        assert_eq!(choices.len(), 2);
    } else {
        panic!("expected Categorical for bool, got {dim:?}");
    }
}

#[test]
fn opaque_meta() {
    let f = TestOpaque::default();
    let meta = f.soma_meta();
    assert_eq!(meta.kind, somatize_core::filter::FilterKind::Opaque);
    assert!(!meta.differentiable);
}

// ── Deterministic hashing of map-typed config fields ──

#[derive(SomaFilter, Serialize, Deserialize, Default)]
struct MapConfigFilter {
    params: std::collections::HashMap<String, f64>,
}

#[test]
fn hashmap_config_field_hashes_deterministically() {
    // The historical bug: serde_json of a HashMap follows random
    // iteration order, so the same config produced different hashes
    // across processes. Canonical CBOR sorts keys by encoded bytes.
    let mut reference = None;
    for _ in 0..100 {
        let mut params = std::collections::HashMap::new();
        params.insert("lr".to_string(), 0.001);
        params.insert("momentum".to_string(), 0.9);
        params.insert("weight_decay".to_string(), 1e-4);
        params.insert("epochs".to_string(), 100.0);
        let f = MapConfigFilter { params };
        let h = f.config_hash();
        match &reference {
            None => reference = Some(h),
            Some(r) => assert_eq!(r, &h),
        }
    }
}

// ── cache_version participates in the hash ──

#[derive(SomaFilter, Serialize, Deserialize, Default)]
#[soma(cache_version = "v1")]
struct VersionedFilter {
    alpha: f64,
}

#[test]
fn cache_version_changes_hash() {
    // Same field values; only the declared version differs. The hashes
    // must differ (the type names differ too, so compare against a
    // same-name baseline is impossible here — instead verify the
    // version string is load-bearing by checking against the unversioned
    // computation by hand).
    use somatize_core::cache::CacheKey;
    let f = VersionedFilter { alpha: 1.0 };
    let unversioned = CacheKey::from_parts(&[
        b"VersionedFilter",
        &somatize_core::canon::canonical_bytes(&1.0f64).unwrap(),
    ]);
    assert_ne!(f.config_hash(), unversioned, "cache_version must be hashed");
}

// ── Nondeterminism flag ──

#[derive(SomaFilter, Serialize, Deserialize, Default)]
#[soma(deterministic = false)]
struct RandomAugment {
    p: f64,
}

#[test]
fn deterministic_flag_flows_to_meta() {
    assert!(!RandomAugment::default().soma_meta().deterministic);
    assert!(TestOpaque::default().soma_meta().deterministic);
}

/// A filter is not differentiable unless it says so.
///
/// The derive defaulted to `true` while the Python bridge defaulted to
/// `false`, so the same concept had opposite meanings on the two sides of
/// the boundary — and `differentiable` is what makes the compiler group
/// nodes into a composite block for autograd. Opting in is the safe
/// direction: a filter that does not carry a module gains nothing from
/// being grouped, and one that does can say so.
#[test]
fn differentiable_defaults_to_false_on_both_sides_of_the_boundary() {
    #[derive(somatize_core::SomaFilter, serde::Serialize)]
    struct Plain {
        scale: f64,
    }

    let plain = Plain { scale: 1.0 };
    assert!(
        !plain.soma_meta().differentiable,
        "the derive must agree with soma-python's `_differentiable` default"
    );
}
