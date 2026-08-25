//! Reading what the model answered.
//!
//! The model itself lives in `soma_next.foreseen` and is tested there. What is
//! checked here is the reading: which findings mean somebody typed something,
//! and which mean a name moved without anybody having.

use soma_tree::findings::Findings;

/// One step, as the probe hands it over.
fn found(said: &str) -> Findings {
    serde_json::from_str(said).expect("the probe writes JSON")
}

#[test]
fn nothing_said_is_nothing_to_say() {
    let step = found(r#"{"findings": {}, "declared": {}}"#);

    assert!(step.is_quiet());
    assert!(step.the_edit().is_empty());
}

#[test]
fn a_rewrite_that_moved_no_name_is_still_where_the_edit_is() {
    // The one that has to be right. A rewritten `forward` renames nothing, so
    // the model says `STALE` and nothing else — and a reader that only counted
    // `CHANGED` would answer "nobody edited anything" to the very edit this
    // exists to catch.
    let step = found(
        r#"{"findings": {"embed": ["STALE"], "head": ["SUSPECT"]},
                        "declared": {}}"#,
    );

    assert_eq!(step.the_edit(), ["embed"]);
    assert_eq!(step.saying("SUSPECT"), ["head"]);
    assert_eq!(
        step.not_comparable().len(),
        2,
        "the head reads what it left"
    );
}

#[test]
fn a_retrain_is_not_an_edit_and_neither_is_what_it_reaches() {
    // Weights are what a trial produces, not what makes another variant. The
    // numbers all move and none of them is answering a different question.
    let step = found(
        r#"{"findings": {"embed": ["RESETTLED"], "head": ["DOWNSTREAM"]},
                        "declared": {}}"#,
    );

    assert!(step.the_edit().is_empty());
    assert!(
        step.not_comparable().is_empty(),
        "nobody typed anything, so there is nothing that stopped comparing",
    );
}

#[test]
fn a_salt_is_not_an_edit_either() {
    let step = found(r#"{"findings": {"embed": ["SALTED"]}, "declared": {}}"#);

    assert!(step.the_edit().is_empty());
    assert!(step.not_comparable().is_empty());
}

#[test]
fn a_node_carries_every_finding_and_not_the_first() {
    // Recomputing and recomputing from a stale answer are two facts, and the
    // reassuring one is the one a single verdict would have kept.
    let step = found(r#"{"findings": {"head": ["CHANGED", "SUSPECT"]}, "declared": {}}"#);

    assert_eq!(step.the_edit(), ["head"]);
    assert_eq!(step.saying("SUSPECT"), ["head"]);
}

#[test]
fn what_somebody_actually_typed_comes_back_beside_the_finding() {
    // The fold turns a declaration into a digest so the model's own machinery
    // reaches it. This is the half that survives, so a report says what moved
    // rather than two hashes nobody can act on.
    let step = found(
        r#"{"findings": {"strict": ["STALE"]},
            "declared": {"strict": ["Classify(1.0)", "Classify(2.0)"]}}"#,
    );

    assert_eq!(
        step.declared.get("strict").map(|said| said.as_slice()),
        Some(["Classify(1.0)".to_string(), "Classify(2.0)".to_string()].as_slice()),
    );
}

#[test]
fn a_finding_nobody_here_knows_about_is_still_carried() {
    // The vocabulary is the model's and it is still growing. A reader that
    // refused what it had not been told about would break on the next commit
    // over there rather than on a mistake.
    let step = found(r#"{"findings": {"a": ["SOMETHING_NEW"]}, "declared": {}}"#);

    assert_eq!(step.saying("SOMETHING_NEW"), ["a"]);
    assert!(step.the_edit().is_empty());
}
