//! One run, with the record turned the other way up.

use serde_json::json;
use soma_fabric_fleet::{ran, runs};
use soma_next_store::{Local, Store};
use tempfile::TempDir;

fn store() -> (TempDir, Local) {
    let dir = TempDir::new().expect("a temporary directory");
    let store = Local::at(dir.path()).expect("a store in it");
    (dir, store)
}

/// One `forward`'s record, written the way a run writes one.
fn record(store: &dyn Store, run: &str, which: usize, facts: &[serde_json::Value]) {
    let blob = serde_json::to_vec(&facts).expect("facts write");
    let digest = store.put(&blob).expect("a blob");
    store
        .bind(
            &format!("run/{run}/{which}"),
            &digest,
            vec![
                ("run".into(), run.into()),
                ("forward".into(), which.to_string()),
                ("state".into(), "ok".into()),
            ],
        )
        .expect("a record bound");
}

#[test]
fn what_a_machine_waited_is_the_trip_minus_what_ran_over_there() {
    // The column that only exists up here: neither half of the subtraction
    // belongs to a node — `left` is the client's fact and `ran` is the worker's.
    let (_dir, store) = store();
    record(
        &store,
        "3f8a",
        0,
        &[
            json!({ "fact": "left", "host": "gpu-box", "took_us": "402000" }),
            json!({ "fact": "ran", "host": "gpu-box", "node": "embed", "took_us": "96000" }),
        ],
    );

    let out = ran(&store, "3f8a", 40).expect("the run reads");
    let gpu = out.did.iter().find(|one| one.host == "gpu-box").unwrap();

    assert_eq!(gpu.slices, 1);
    assert_eq!(gpu.trip_us, 402_000);
    assert_eq!(gpu.took_us, 96_000);
    assert_eq!(gpu.waiting_us, 306_000);
}

#[test]
fn waiting_never_goes_below_zero() {
    // A `left` counted on one `forward` and the work it carried counted on
    // another would otherwise read as a machine that finished before it was
    // asked.
    let (_dir, store) = store();
    record(
        &store,
        "3f8a",
        0,
        &[json!({ "fact": "ran", "host": "w1", "node": "a", "took_us": "171000" })],
    );

    let out = ran(&store, "3f8a", 40).expect("the run reads");

    assert_eq!(out.did[0].waiting_us, 0);
}

#[test]
fn what_ran_with_no_host_on_it_ran_here() {
    // A row like any other, because it is one: a view that only showed other
    // people's machines would hide the one the graph was declared on.
    let (_dir, store) = store();
    record(
        &store,
        "3f8a",
        0,
        &[
            json!({ "fact": "ran", "node": "read", "took_us": "4000" }),
            json!({ "fact": "left", "host": "w1", "took_us": "184000" }),
        ],
    );

    let out = ran(&store, "3f8a", 40).expect("the run reads");

    assert_eq!(out.did[0].host, "aquí", "and it is the first row");
    assert_eq!(out.did[0].ran, 1);
    assert_eq!(out.did[0].slices, 0, "nothing crossed to this process");
}

#[test]
fn the_newest_reading_of_a_machine_wins() {
    // A reading is a snapshot: the question is what the machine is like now,
    // not what it averaged over the run.
    let (_dir, store) = store();
    record(
        &store,
        "3f8a",
        0,
        &[json!({ "fact": "machine", "host": "w1", "id": "node3-4127", "busy": "0.9" })],
    );
    record(
        &store,
        "3f8a",
        1,
        &[json!({ "fact": "machine", "host": "w1", "id": "node3-4127", "busy": "0.1" })],
    );

    let out = ran(&store, "3f8a", 40).expect("the run reads");

    assert_eq!(out.did[0].busy, Some(0.1));
    assert_eq!(out.did[0].id.as_deref(), Some("node3-4127"));
}

#[test]
fn each_node_that_ran_somewhere_is_named_once() {
    let (_dir, store) = store();
    for which in 0..3 {
        record(
            &store,
            "3f8a",
            which,
            &[
                json!({ "fact": "ran", "host": "w1", "node": "classify", "took_us": "1000" }),
                json!({ "fact": "ran", "host": "w1", "node": "embed", "took_us": "1000" }),
            ],
        );
    }

    let out = ran(&store, "3f8a", 40).expect("the run reads");

    assert_eq!(out.did[0].nodes, vec!["classify", "embed"]);
    assert_eq!(out.did[0].ran, 6, "and each running of it is counted");
}

#[test]
fn last_bounds_how_many_forwards_are_read() {
    // The join's whole price is a fetch per `forward`, and the question worth
    // asking of a fleet that is working now is the last few.
    let (_dir, store) = store();
    for which in 0..5 {
        record(
            &store,
            "3f8a",
            which,
            &[json!({ "fact": "ran", "host": "w1", "node": "a", "took_us": "1000" })],
        );
    }

    let out = ran(&store, "3f8a", 2).expect("the run reads");

    assert_eq!(out.forwards, 2);
    assert_eq!(out.did[0].ran, 2);
}

#[test]
fn one_run_does_not_see_another_s_work() {
    let (_dir, store) = store();
    record(
        &store,
        "mine",
        0,
        &[json!({ "fact": "ran", "host": "w1", "node": "a", "took_us": "1000" })],
    );
    record(
        &store,
        "theirs",
        0,
        &[json!({ "fact": "ran", "host": "w9", "node": "b", "took_us": "1000" })],
    );

    let out = ran(&store, "mine", 40).expect("the run reads");

    assert_eq!(out.did.len(), 1);
    assert_eq!(out.did[0].host, "w1");
}

#[test]
fn the_runs_in_a_store_are_a_scan_and_no_fetches() {
    let (_dir, store) = store();
    record(&store, "first", 0, &[json!({ "fact": "ran", "node": "a" })]);
    record(&store, "first", 1, &[json!({ "fact": "ran", "node": "a" })]);
    record(
        &store,
        "second",
        0,
        &[json!({ "fact": "ran", "node": "a" })],
    );

    let all = runs(&store).expect("the runs read");

    assert_eq!(all.len(), 2);
    let first = all.iter().find(|one| one.run == "first").unwrap();
    assert_eq!(first.forwards, 2);
    assert!(!first.broke);
}

#[test]
fn a_run_that_broke_says_so_in_the_scan() {
    let (_dir, store) = store();
    let digest = store.put(b"[]").expect("a blob");
    store
        .bind(
            "run/bad/0",
            &digest,
            vec![
                ("run".into(), "bad".into()),
                ("forward".into(), "0".into()),
                ("state".into(), "broke".into()),
            ],
        )
        .expect("bound");

    assert!(runs(&store).expect("the runs read")[0].broke);
}
