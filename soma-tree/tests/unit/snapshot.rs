//! What a snapshot says about itself, and about another one.

use somatize_tree::snapshot::Snapshot;

fn built_against(said: &[(&str, &str)]) -> Snapshot {
    Snapshot {
        commit: "aaaa".into(),
        built_from: "experiments.thing:build".into(),
        input: "sentinel".into(),
        environment: said
            .iter()
            .map(|(what, version)| (what.to_string(), version.to_string()))
            .collect(),
        snapshot: serde_json::json!({}),
        // What this test looks at is the environment, so the rest goes empty.
        // Not for convenience: a `Default::default()` for the whole struct
        // would let a new field into the test with nobody looking at it.
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
fn two_probes_from_different_environments_say_so() {
    // A checkout pins its own code and not the interpreter outside it, so the
    // same commit names its nodes identically against torch 2.3 and 2.6. That
    // is why the environment is not part of what a snapshot is remembered
    // under: a cached probe from months ago is meant to disagree out loud with
    // a fresh one.
    let before = built_against(&[("python", "3.13.13"), ("torch", "2.3.0")]);
    let after = built_against(&[("python", "3.13.13"), ("torch", "2.6.0")]);

    assert_eq!(
        before.drifted_from(&after),
        [("torch", "2.3.0".to_string(), "2.6.0".to_string())],
        "the interpreter held, and only torch is worth a line",
    );
}

#[test]
fn a_dependency_only_one_side_had_is_drift_and_not_a_crash() {
    let before = built_against(&[("torch", "2.3.0")]);
    let after = built_against(&[]);

    assert_eq!(
        before.drifted_from(&after),
        [("torch", "2.3.0".to_string(), "—".to_string())]
    );
}

#[test]
fn two_probes_from_one_sitting_have_nothing_to_say_about_it() {
    // The common case by far, and it has to be silent: both sides probed now
    // share an interpreter, so the axis cancels out of the comparison.
    let same = || built_against(&[("python", "3.13.13"), ("torch", "2.6.0")]);

    assert!(same().drifted_from(&same()).is_empty());
}

#[test]
fn what_the_model_answered_is_carried_and_never_reshaped() {
    // The opaque half. What a name is made of is soma's business, and a
    // reader here that understood its insides would be a second model with a
    // delay on it.
    let mut snapshot = built_against(&[]);
    snapshot.snapshot = serde_json::json!({"names": {"a": "sha256:…"}, "whatever": [1, 2]});

    let round = serde_json::to_string(&snapshot)
        .and_then(|said| serde_json::from_str::<Snapshot>(&said).map(|back| back.snapshot));

    assert_eq!(
        round.expect("it survives being written down"),
        snapshot.snapshot
    );
}
