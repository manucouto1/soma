//! The join: what the graphs have been calling these machines.
//!
//! Every test here writes the record **the way a run writes it** — a JSON array
//! of flat facts, with `Elsewhere` already turned into a `host` field — because
//! what is being tested is a reader of somebody else's format, and a fixture
//! that agreed with the reader instead of with the writer would pass for ever
//! and mean nothing.

use somatize_fabric_fleet::{Fleet, Machine};
use somatize_store::{Local, Store};
use std::time::Duration;
use tempfile::TempDir;

/// A store nobody else is using.
fn store() -> (TempDir, Local) {
    let dir = TempDir::new().expect("a temporary directory");
    let store = Local::at(dir.path()).expect("a store in it");
    (dir, store)
}

/// A reading, filed the way the wire's idle reporting files one: the whole of
/// it in the record, and nothing in the blob.
fn reports(store: &dyn Store, id: &str) {
    let machine = Machine {
        up: Duration::from_secs(60),
        busy: Some(0.5),
        cores: Some(4),
        memory: Some(0.3),
        served: 7,
        id: id.into(),
    };
    let said = machine.said();
    let (kind, mut meta) = said.flattened();
    meta.insert(0, ("fact".into(), kind.to_string()));
    let digest = store.put(&[]).expect("an empty blob");
    store
        .bind(&somatize_fabric_wire::filed(id), &digest, meta)
        .expect("a reading filed");
}

/// One `forward`'s record, with these facts in its blob.
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
            ],
        )
        .expect("a record bound");
}

/// A reading that came down a wire: the client attributed it, so it carries the
/// graph's name, and the reading carries the machine's own.
fn came_down_a_wire(host: &str, id: &str) -> serde_json::Value {
    serde_json::json!({ "fact": "machine", "host": host, "id": id, "busy": "0.4213" })
}

fn named(store: &dyn Store) -> Vec<(Option<String>, String)> {
    Fleet::read(store, 90, 40)
        .expect("the fleet reads")
        .seen
        .into_iter()
        .map(|one| (one.named.map(|host| host.as_str().to_string()), one.id))
        .collect()
}

#[test]
fn a_reading_that_came_down_a_wire_is_where_the_two_names_meet() {
    let (_dir, store) = store();
    reports(&store, "node3-4127");
    record(&store, "3f8a", 0, &[came_down_a_wire("w1", "node3-4127")]);

    assert_eq!(
        named(&store),
        vec![(Some("w1".into()), "node3-4127".into())]
    );
}

#[test]
fn a_machine_nobody_has_talked_to_is_there_and_has_no_name() {
    // The whole reason the idle half exists: it is capacity, and a fleet view
    // that only knew about runs could not see it at all.
    let (_dir, store) = store();
    reports(&store, "node9-3312");

    assert_eq!(named(&store), vec![(None, "node9-3312".into())]);
}

#[test]
fn what_a_graph_called_it_most_recently_is_what_it_is_called() {
    // A name is not a fact about a machine, it is a fact about a run. The newest
    // is the only one that could still be true.
    let (_dir, store) = store();
    reports(&store, "node3-4127");
    record(&store, "old", 0, &[came_down_a_wire("w1", "node3-4127")]);
    record(&store, "new", 0, &[came_down_a_wire("gpu", "node3-4127")]);

    assert_eq!(
        named(&store),
        vec![(Some("gpu".into()), "node3-4127".into())]
    );
}

#[test]
fn a_fact_from_over_there_that_is_not_a_reading_names_nothing() {
    // Every fact from another machine carries a `host`; only a reading carries
    // what the machine calls itself. The half that cannot be guessed is the one
    // that only the reading has.
    let (_dir, store) = store();
    reports(&store, "node3-4127");
    record(
        &store,
        "3f8a",
        0,
        &[
            serde_json::json!({ "fact": "ran", "host": "w1", "node": "classify", "took_us": "171000" }),
        ],
    );

    assert_eq!(
        named(&store),
        vec![(None, "node3-4127".into())],
        "a host with no id beside it was taken for a join"
    );
}

#[test]
fn two_workers_on_one_box_keep_their_own_names() {
    // The case the pid is in the id for, and the one where matching hostnames
    // would have been wrong rather than merely unprincipled.
    let (_dir, store) = store();
    reports(&store, "node9-3312");
    reports(&store, "node9-3319");
    record(
        &store,
        "3f8a",
        0,
        &[
            came_down_a_wire("left", "node9-3312"),
            came_down_a_wire("right", "node9-3319"),
        ],
    );

    assert_eq!(
        named(&store),
        vec![
            (Some("left".into()), "node9-3312".into()),
            (Some("right".into()), "node9-3319".into()),
        ]
    );
}

#[test]
fn a_record_this_version_cannot_read_costs_that_record_and_not_the_fleet() {
    let (_dir, store) = store();
    reports(&store, "node3-4127");
    let digest = store.put(b"this was never a record").expect("bytes");
    store
        .bind("run/broken/0", &digest, vec![])
        .expect("bound anyway");
    record(&store, "3f8a", 0, &[came_down_a_wire("w1", "node3-4127")]);

    assert_eq!(
        named(&store),
        vec![(Some("w1".into()), "node3-4127".into())]
    );
}

#[test]
fn reading_no_records_at_all_still_answers_who_is_out_there() {
    // `records=0` is the cheap question — *what is there* — with the join's
    // whole price taken off. It has to answer, without names.
    let (_dir, store) = store();
    reports(&store, "node3-4127");
    record(&store, "3f8a", 0, &[came_down_a_wire("w1", "node3-4127")]);

    let fleet = Fleet::read(&store, 90, 0).expect("the fleet reads");

    assert_eq!(fleet.seen.len(), 1);
    assert_eq!(fleet.seen[0].named, None);
}
