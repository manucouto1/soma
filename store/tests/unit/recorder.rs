//! What happened, written down: one record per `forward`.

use crate::tempdir;
use soma_next_core::{Fact, Host, NodeId, Watcher};
use soma_next_store::{Local, Recorder, Store};
use std::sync::Arc;
use std::time::Duration;

/// A store of its own, in a directory nobody else is using. Shared, because a
/// recorder holds on to one and the test looks in the same one.
fn store() -> (Arc<dyn Store>, tempdir::Dir) {
    let where_ = tempdir::Dir::new();
    let store: Arc<dyn Store> = Arc::new(Local::at(where_.path()).unwrap());
    (store, where_)
}

fn ran(who: &str) -> Fact {
    Fact::Ran {
        node: NodeId::from(who),
        took: Duration::from_millis(1),
        device: None,
    }
}

fn over(took: u64) -> Fact {
    Fact::Finished {
        took: Duration::from_millis(took),
    }
}

/// What a record says, as a map, so one field can be asserted at a time.
fn said(store: &dyn Store, name: &str) -> std::collections::HashMap<String, String> {
    store
        .resolve(name)
        .unwrap()
        .unwrap_or_else(|| panic!("`{name}` was not written"))
        .meta
        .into_iter()
        .collect()
}

/// The facts in a record's blob, as `(kind, node)` pairs.
fn detail(store: &dyn Store, name: &str) -> Vec<(String, String)> {
    let bound = store.resolve(name).unwrap().expect("a record");
    let bytes = store.get(&bound.digest).unwrap().expect("its blob");
    let facts: Vec<serde_json::Value> = serde_json::from_slice(&bytes).expect("readable JSON");
    facts
        .iter()
        .map(|fact| {
            (
                fact["fact"].as_str().unwrap_or_default().to_string(),
                fact["node"].as_str().unwrap_or_default().to_string(),
            )
        })
        .collect()
}

#[test]
fn nothing_is_written_until_the_forward_is_over() {
    // The grain, and it is the whole design: a record is a `forward`, so a node
    // running is not one. Otherwise five nodes over ten thousand steps would be
    // fifty thousand writes.
    let (store, _where) = store();
    let recorder = Recorder::over(store.clone());

    recorder.saw(&ran("a"));
    recorder.saw(&ran("b"));

    assert!(store.bound().unwrap().is_empty(), "nothing is over yet");

    recorder.saw(&over(9));

    assert_eq!(store.bound().unwrap().len(), 1);
}

#[test]
fn a_scan_says_how_it_went_without_reading_a_single_blob() {
    // The same split as a trial's record, for the same reason: "how is it
    // going" is asked constantly and the detail almost never.
    let (store, _where) = store();
    let recorder = Recorder::over(store.clone());

    recorder.saw(&ran("a"));
    recorder.saw(&ran("b"));
    recorder.saw(&over(9));

    let said = said(store.as_ref(), &format!("run/{}/0", recorder.run()));
    assert_eq!(said["run"], recorder.run());
    assert_eq!(said["forward"], "0");
    assert_eq!(said["state"], "ok");
    assert_eq!(said["nodes"], "2");
    assert_eq!(said["took_us"], "9000");
}

#[test]
fn the_detail_is_in_the_blob_in_the_order_it_arrived() {
    let (store, _where) = store();
    let recorder = Recorder::over(store.clone());

    recorder.saw(&ran("a"));
    recorder.saw(&Fact::Left {
        host: Host::new("worker1"),
        took: Duration::from_millis(4),
    });
    recorder.saw(&over(9));

    let detail = detail(store.as_ref(), &format!("run/{}/0", recorder.run()));
    let kinds: Vec<&str> = detail.iter().map(|(kind, _)| kind.as_str()).collect();
    assert_eq!(kinds, ["ran", "left", "finished"]);
    assert_eq!(detail[0].1, "a");
}

#[test]
fn each_forward_is_its_own_record_numbered_from_zero() {
    let (store, _where) = store();
    let recorder = Recorder::over(store.clone());

    for _ in 0..3 {
        recorder.saw(&ran("a"));
        recorder.saw(&over(1));
    }

    let mut names: Vec<String> = store.bound().unwrap().into_iter().map(|b| b.name).collect();
    names.sort();
    assert_eq!(
        names,
        [0, 1, 2]
            .map(|n| format!("run/{}/{n}", recorder.run()))
            .to_vec()
    );
}

#[test]
fn a_forward_that_broke_says_so_where_a_scan_can_see_it() {
    // Whoever is looking at a study of ten thousand steps wants the ones that
    // broke, and paying a fetch per step to find out would be the whole point
    // of the split thrown away.
    let (store, _where) = store();
    let recorder = Recorder::over(store.clone());

    recorder.saw(&ran("a"));
    recorder.saw(&Fact::Failed {
        node: NodeId::from("b"),
        why: "no".into(),
    });
    recorder.saw(&Fact::Broke {
        why: "`b` did not answer".into(),
    });

    let said = said(store.as_ref(), &format!("run/{}/0", recorder.run()));
    assert_eq!(said["state"], "broke");
    assert_eq!(said["nodes"], "1", "only the one that got to run");
}

#[test]
fn a_run_can_be_given_the_name_it_already_has() {
    // A training run is findable by the name it was known by; a `forward` in a
    // notebook has no reason to invent one, and gets one anyway.
    let (store, _where) = store();
    let recorder = Recorder::named(store.clone(), "tuesday");

    recorder.saw(&over(1));

    assert_eq!(recorder.run(), "tuesday");
    assert!(store.resolve("run/tuesday/0").unwrap().is_some());
}

// ── The other vocabulary, which arrives late ──

#[test]
fn what_level_two_says_lands_in_the_forward_it_belongs_to() {
    // A loss is computed **after** the `forward` that produced it has ended, so
    // a recorder that only knew how to open a new record would file every loss
    // one step late. It goes into the one that closed last, rewritten.
    let (store, _where) = store();
    let recorder = Recorder::over(store.clone());

    recorder.saw(&ran("body"));
    recorder.saw(&over(9));
    recorder.said("loss", vec![("value".into(), "0.25".into())]);

    let detail = detail(store.as_ref(), &format!("run/{}/0", recorder.run()));
    let kinds: Vec<&str> = detail.iter().map(|(kind, _)| kind.as_str()).collect();
    assert_eq!(kinds, ["ran", "finished", "loss"]);
    assert_eq!(
        store.bound().unwrap().len(),
        1,
        "it rewrote the record, it did not open another"
    );
}

#[test]
fn and_the_next_forward_still_starts_a_new_record() {
    // The rule has to hold both ways: level 2's door never opens a record and
    // level 1's always does once one has closed.
    let (store, _where) = store();
    let recorder = Recorder::over(store.clone());

    recorder.saw(&over(1));
    recorder.said("loss", vec![("value".into(), "0.25".into())]);
    recorder.saw(&ran("body"));
    recorder.saw(&over(1));

    assert_eq!(store.bound().unwrap().len(), 2);
    let second = said(store.as_ref(), &format!("run/{}/1", recorder.run()));
    assert_eq!(second["forward"], "1");
    assert_eq!(second["nodes"], "1");
}

#[test]
fn rewriting_a_record_says_the_same_thing_about_the_same_facts() {
    // A record that is rewritten is written by the same code from the same
    // facts, so the summary a scan reads cannot drift from the detail beside it.
    let (store, _where) = store();
    let recorder = Recorder::over(store.clone());

    recorder.saw(&ran("a"));
    recorder.saw(&over(9));
    for which in 0..3 {
        recorder.said("loss", vec![("value".into(), which.to_string())]);
    }

    let said = said(store.as_ref(), &format!("run/{}/0", recorder.run()));
    assert_eq!(said["nodes"], "1");
    assert_eq!(said["took_us"], "9000");
    assert_eq!(
        detail(store.as_ref(), &format!("run/{}/0", recorder.run())).len(),
        5
    );
}

// ── What is worth having in the record and not only in the blob ──

#[test]
fn what_was_asked_to_be_summarised_is_in_the_record_itself() {
    // The lesson CU18 already paid for: ten thousand losses read one blob at a
    // time is ten thousand round trips, and the number wanted from each of them
    // is one.
    let (store, _where) = store();
    let recorder = Recorder::over(store.clone()).summarising(["loss"]);

    recorder.saw(&ran("body"));
    recorder.saw(&over(9));
    recorder.said("loss", vec![("value".into(), "0.25".into())]);

    let said = said(store.as_ref(), &format!("run/{}/0", recorder.run()));
    assert_eq!(said["loss.value"], "0.25");
}

#[test]
fn what_was_not_asked_for_stays_in_the_blob() {
    let (store, _where) = store();
    let recorder = Recorder::over(store.clone());

    recorder.saw(&over(9));
    recorder.said("loss", vec![("value".into(), "0.25".into())]);

    let said = said(store.as_ref(), &format!("run/{}/0", recorder.run()));
    assert!(!said.contains_key("loss.value"));
    assert_eq!(
        detail(store.as_ref(), &format!("run/{}/0", recorder.run())).len(),
        2,
        "and it is still there, where the detail goes"
    );
}

#[test]
fn a_summarised_kind_said_twice_leaves_the_last_one_standing() {
    let (store, _where) = store();
    let recorder = Recorder::over(store.clone()).summarising(["loss"]);

    recorder.saw(&over(1));
    recorder.said("loss", vec![("value".into(), "1.0".into())]);
    recorder.said("loss", vec![("value".into(), "0.5".into())]);

    let said = said(store.as_ref(), &format!("run/{}/0", recorder.run()));
    assert_eq!(said["loss.value"], "0.5");
}
