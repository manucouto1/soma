//! Everything that is out there, once — and the rules the answer carries.

use somatize_fabric_fleet::{Fleet, Machine, Standing};
use somatize_store::{Local, Store};
use std::time::Duration;
use tempfile::TempDir;

fn store() -> (TempDir, Local) {
    let dir = TempDir::new().expect("a temporary directory");
    let store = Local::at(dir.path()).expect("a store in it");
    (dir, store)
}

/// A reading filed the way the wire files one.
fn reports(store: &dyn Store, machine: Machine) {
    let said = machine.said();
    let (kind, mut meta) = said.flattened();
    meta.insert(0, ("fact".into(), kind.to_string()));
    let digest = store.put(&[]).expect("an empty blob");
    store
        .bind(&somatize_fabric_wire::filed(&machine.id), &digest, meta)
        .expect("a reading filed");
}

fn a(id: &str) -> Machine {
    Machine {
        up: Duration::from_secs(60),
        busy: Some(0.5),
        cores: Some(4),
        memory: Some(0.3),
        served: 7,
        id: id.into(),
    }
}

#[test]
fn the_prefix_a_scan_filters_on_is_the_one_the_wire_files_under() {
    // The one thing this crate knows about somebody else's naming. Asked of
    // `filed` itself rather than written out twice, because two spellings of
    // one decision is how a scan silently starts finding nothing.
    let (_dir, store) = store();
    reports(&store, a("node3-4127"));

    let fleet = Fleet::read(&store, 90, 0).expect("the fleet reads");

    assert_eq!(
        somatize_fabric_wire::filed("node3-4127"),
        "machine/node3-4127"
    );
    assert_eq!(fleet.seen.len(), 1, "the scan found nothing it filed");
    assert_eq!(fleet.seen[0].id, "node3-4127");
}

#[test]
fn what_is_in_the_store_and_is_not_a_reading_is_not_a_machine() {
    // A store holds cached values, artifacts and records too. A fleet that
    // counted them would have as many machines as the cache has keys.
    let (_dir, store) = store();
    reports(&store, a("node3-4127"));
    let digest = store.put(b"a cached value").expect("bytes");
    store
        .bind("sha256:whatever", &digest, vec![])
        .expect("bound");
    store.bind("run/3f8a/0", &digest, vec![]).expect("bound");

    let fleet = Fleet::read(&store, 90, 40).expect("the fleet reads");

    assert_eq!(fleet.seen.len(), 1);
}

#[test]
fn a_whole_reading_comes_back_out_of_the_scan_with_no_fetch() {
    // The point of the wire writing everything into the record: what the panel
    // shows costs one scan. If a field had to come out of the blob this would
    // be `None` and nobody would notice until the machine mattered.
    let (_dir, store) = store();
    reports(
        &store,
        Machine {
            up: Duration::from_secs(3_600 * 24),
            busy: Some(0.4213),
            cores: Some(8),
            memory: Some(0.6187),
            served: 1_284,
            id: "node3-4127".into(),
        },
    );

    let one = &Fleet::read(&store, 90, 0).expect("the fleet reads").seen[0];

    assert_eq!(one.cores, Some(8));
    assert_eq!(one.served, 1_284);
    assert_eq!(one.up_s, 86_400);
    assert!((one.busy.unwrap() - 0.4213).abs() < 1e-9);
    assert!((one.memory.unwrap() - 0.6187).abs() < 1e-9);
}

#[test]
fn a_reading_is_that_machine_s_even_if_its_id_field_went_missing() {
    // The name it is filed under is the one to believe: it is what the store
    // sorted it by and what a second reading would overwrite.
    let (_dir, store) = store();
    let nameless = Machine {
        id: String::new(),
        ..a("ignored")
    };
    let digest = store.put(&[]).expect("bytes");
    let said = nameless.said();
    let (kind, mut meta) = said.flattened();
    meta.insert(0, ("fact".into(), kind.to_string()));
    store
        .bind("machine/node7-991", &digest, meta)
        .expect("filed by hand");

    let one = &Fleet::read(&store, 90, 0).expect("the fleet reads").seen[0];

    assert_eq!(one.id, "node7-991");
}

#[test]
fn the_order_is_a_name_and_never_a_number_that_moves() {
    // A list that sorted by load would move the row somebody is reading every
    // time a machine got busier. The load is shown; it is not the order.
    let (_dir, store) = store();
    reports(
        &store,
        Machine {
            busy: Some(0.01),
            ..a("node9-3312")
        },
    );
    reports(
        &store,
        Machine {
            busy: Some(0.99),
            ..a("node3-4127")
        },
    );
    reports(
        &store,
        Machine {
            busy: Some(0.50),
            ..a("node7-991")
        },
    );

    let ids: Vec<String> = Fleet::read(&store, 90, 0)
        .expect("the fleet reads")
        .seen
        .into_iter()
        .map(|one| one.id)
        .collect();

    assert_eq!(ids, ["node3-4127", "node7-991", "node9-3312"]);
}

#[test]
fn a_machine_that_has_just_written_is_not_quiet_and_says_how_long_ago() {
    let (_dir, store) = store();
    reports(&store, a("node3-4127"));

    let one = &Fleet::read(&store, 90, 0).expect("the fleet reads").seen[0];

    assert_eq!(one.standing, Standing::Loose);
    assert!(one.silent_for < 5, "it wrote just now: {}", one.silent_for);
    assert!(one.wrote > 0, "the store stamps every write");
}

#[test]
fn the_rules_travel_with_the_answer() {
    // *Quiet* is not a fact in the store, it is a bound somebody chose, and a
    // screen that showed it without saying which bound would be presenting an
    // opinion as a reading.
    let (_dir, store) = store();
    reports(&store, a("node3-4127"));

    let fleet = Fleet::read(&store, 120, 7).expect("the fleet reads");

    assert_eq!(fleet.quiet_after_s, 120);
    assert_eq!(fleet.read_records, 7);
    let json = serde_json::to_value(&fleet).expect("a fleet writes");
    assert_eq!(json["quiet_after_s"], 120);
}

#[test]
fn a_store_with_nothing_in_it_is_an_empty_fleet_and_not_a_failure() {
    let (_dir, store) = store();

    let fleet = Fleet::read(&store, 90, 40).expect("an empty store still reads");

    assert!(fleet.seen.is_empty());
}

// ── And the one that stands up a real worker ──
//
// Everything above builds a reading the way the wire builds one, which is the
// right fixture and still a fixture: it agrees with `said()` because both halves
// are written here. This one has a **real worker** report into a real store on a
// thread of this process, and reads what it actually wrote. It is the only test
// in this file that would fail the day the two crates stopped agreeing about
// what a reading is, which is the failure the format has one owner to avoid.

#[test]
fn a_worker_reporting_for_real_turns_up_in_the_fleet() {
    use somatize_core::Catalog;
    use somatize_fabric_wire::Serving;
    use std::sync::mpsc::channel;

    let dir = TempDir::new().expect("a temporary directory");
    let where_ = dir.path().to_path_buf();
    let (opened, up) = channel();
    // Detached on purpose: `listen` does not return, and a worker outliving the
    // test binary is the arrangement the wire's own suite already uses.
    std::thread::Builder::new()
        .name("a-worker-reporting".into())
        .spawn(move || {
            let store = Local::at(&where_).expect("a store");
            let catalog = Catalog::new();
            let _ = Serving::own(&catalog)
                .store(&store)
                .reporting(Duration::from_millis(20))
                .listen_at("127.0.0.1:0", |_| {
                    let _ = opened.send(());
                });
        })
        .expect("a thread for it");
    up.recv().expect("the worker never came up");

    let store = Local::at(dir.path()).expect("the same store");
    // It writes on a clock, so the first reading is not there the instant the
    // socket is. Polled rather than slept through, so a fast machine is fast.
    let seen = (0..100)
        .find_map(|_| {
            let fleet = Fleet::read(&store, 90, 0).expect("the fleet reads");
            match fleet.seen.into_iter().next() {
                Some(one) => Some(one),
                None => {
                    std::thread::sleep(Duration::from_millis(20));
                    None
                }
            }
        })
        .expect("a worker reporting every 20 ms wrote nothing in two seconds");

    assert!(!seen.id.is_empty(), "it filed itself under nothing");
    assert!(
        seen.id.contains('-'),
        "a machine is a hostname and a pid: {}",
        seen.id
    );
    assert_eq!(seen.standing, Standing::Loose, "nobody has talked to it");
    assert!(
        seen.silent_for < 5,
        "it is writing now: {}",
        seen.silent_for
    );
    assert_eq!(seen.served, 0, "it has run nothing");
}
