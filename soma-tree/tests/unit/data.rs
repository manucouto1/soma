//! Whose each value in the store is, and what is said about what is nobody's.
//!
//! The records are written **by hand and with soma's names**, for the same
//! reason as in `trials`: what has to be defended is that this reader
//! understands what is on somebody's disk, and calling the other library's
//! writer would have the two agree on any format at all, including one nobody
//! has stored.

use somatize_core::Key;
use somatize_store::{Local, Store, name_of};
use somatize_tree::data::{How, under};
use somatize_tree::snapshot::Snapshot;
use std::collections::HashMap;

fn somewhere() -> (tempfile::TempDir, Local) {
    let at = tempfile::tempdir().expect("a temporary directory");
    let kept = Local::at(at.path()).expect("a store inside");
    (at, kept)
}

/// A kept value, with what whoever produced it wrote beside it.
fn kept_value(kept: &Local, key: &str, node: &str, fingerprint: &str, env: &str) {
    let digest = kept.put(b"what it produced").expect("the blob");
    kept.bind(
        &name_of(&Key::new(key)),
        &digest,
        vec![
            ("node".into(), node.into()),
            ("fingerprint".into(), fingerprint.into()),
            ("input".into(), "sha256:the-input".into()),
            ("env".into(), env.into()),
        ],
    )
    .expect("it binds");
}

/// A probe's answer, with only the two fields that are read from inside.
fn taken(names: &[(&str, &str)], fingerprints: &[(&str, &str)]) -> Snapshot {
    let of = |pairs: &[(&str, &str)]| {
        pairs
            .iter()
            .map(|(node, told)| (node.to_string(), serde_json::json!(told)))
            .collect::<serde_json::Map<_, _>>()
    };
    Snapshot {
        commit: "aaaa".into(),
        built_from: "experiments.thing:build".into(),
        input: "sentinel".into(),
        environment: Default::default(),
        snapshot: serde_json::json!({
            "names": of(names),
            "fingerprints": of(fingerprints),
        }),
        architecture: serde_json::Value::Null,
        inside: serde_json::Value::Null,
        reaches: serde_json::Value::Null,
        declaring: None,
        code: Default::default(),
        mapped: Vec::new(),
        unneeded: Vec::new(),
    }
}

#[test]
fn a_value_belongs_to_the_version_that_will_ask_for_it() {
    // The strongest thing that can be said: not that something like it made
    // this, but that it is the value this version would ask for.
    let (_at, kept) = somewhere();
    kept_value(&kept, "sha256:one", "embed", "a1b2c3d4", "9f2c1a");
    let known = HashMap::from([("c1", taken(&[("embed", "sha256:one")], &[]))]);

    let said = under(&kept, &known).expect("it reads");

    assert_eq!(said.len(), 1);
    assert_eq!(said[0].of.get("c1"), Some(&How::Named));
}

#[test]
fn and_also_to_the_one_that_only_shares_its_code() {
    // The one that survives what the other does not. A key is computed against
    // the probing environment, so probing a three-month-old commit today gives
    // other keys; the fingerprint was written by whoever ran, and is still there.
    let (_at, kept) = somewhere();
    kept_value(&kept, "sha256:one", "embed", "a1b2c3d4", "9f2c1a");
    let known = HashMap::from([(
        "c1",
        taken(&[("embed", "sha256:other")], &[("embed", "a1b2c3d4")]),
    )]);

    let said = under(&kept, &known).expect("it reads");

    assert_eq!(said[0].of.get("c1"), Some(&How::Written));
}

#[test]
fn where_both_hold_the_key_wins() {
    // Saying the weaker thing when the stronger holds loses information.
    let (_at, kept) = somewhere();
    kept_value(&kept, "sha256:one", "embed", "a1b2c3d4", "9f2c1a");
    let known = HashMap::from([(
        "c1",
        taken(&[("embed", "sha256:one")], &[("embed", "a1b2c3d4")]),
    )]);

    let said = under(&kept, &known).expect("it reads");

    assert_eq!(said[0].of.get("c1"), Some(&How::Named));
}

#[test]
fn one_value_can_belong_to_several_versions_at_once() {
    // And not a tie to be broken: four commits in a row that do not touch
    // `embed` share its answer, which is exactly what a cache is for. Picking
    // one would be inventing an answer.
    let (_at, kept) = somewhere();
    kept_value(&kept, "sha256:one", "embed", "a1b2c3d4", "9f2c1a");
    let known = HashMap::from([
        ("c1", taken(&[("embed", "sha256:one")], &[])),
        ("c2", taken(&[("embed", "sha256:one")], &[])),
    ]);

    let said = under(&kept, &known).expect("it reads");

    assert_eq!(said[0].of.len(), 2, "{:?}", said[0].of);
}

#[test]
fn what_belongs_to_no_version_comes_out_anyway_saying_what_is_known() {
    // The case this exists not to keep quiet about. A mute hash stays in the
    // store for ever; one saying which node and which code made it is still a
    // true sentence even when it matches nothing that was looked at.
    let (_at, kept) = somewhere();
    kept_value(
        &kept,
        "sha256:one",
        "embed",
        "from-another-branch",
        "9f2c1a",
    );
    let known = HashMap::from([(
        "c1",
        taken(&[("embed", "sha256:other")], &[("embed", "a1b2c3d4")]),
    )]);

    let said = under(&kept, &known).expect("it reads");

    assert!(said[0].is_nobodys());
    assert_eq!(said[0].node.as_deref(), Some("embed"));
    assert_eq!(said[0].fingerprint.as_deref(), Some("from-another-branch"));
    assert_eq!(said[0].environment.as_deref(), Some("9f2c1a"));
}

#[test]
fn the_bookkeeping_of_whoever_looks_is_nobodys_data() {
    // Three writers share this store and only one leaves intermediates. The
    // journal, the probe cache and a reading of an environment are what
    // **explains** the attribution, not something to attribute — counting them
    // would show somebody their own notebook as if it were an intermediate
    // that might be spare.
    let (_at, kept) = somewhere();
    let digest = kept.put(b"whatever").expect("the blob");
    for name in [
        "exp/an-investigation/aaaa/trial/1/0",
        "snapshot:aaaa:sha256:recipe",
        "env/9f2c1a",
    ] {
        kept.bind(name, &digest, Vec::new()).expect("it binds");
    }
    kept_value(&kept, "sha256:one", "embed", "a1b2c3d4", "9f2c1a");

    let said = under(&kept, &HashMap::new()).expect("it reads");

    assert_eq!(
        said.len(),
        1,
        "{:?}",
        said.iter().map(|one| &one.name).collect::<Vec<_>>()
    );
    assert_eq!(said[0].node.as_deref(), Some("embed"));
}

#[test]
fn a_probe_from_before_this_existed_reads_the_same() {
    // A snapshot kept before the model published these fields is still a good
    // answer to everything else. Falling over for reading something nobody
    // asked it for back then would throw away an investigation's record for a
    // function added afterwards.
    let (_at, kept) = somewhere();
    kept_value(&kept, "sha256:one", "embed", "a1b2c3d4", "9f2c1a");
    let mut old = taken(&[], &[]);
    old.snapshot = serde_json::json!({"shape": {}});
    let known = HashMap::from([("c1", old)]);

    let said = under(&kept, &known).expect("it reads");

    assert!(
        said[0].is_nobodys(),
        "it is not known, and that is not falling over"
    );
}
