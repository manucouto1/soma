//! The fixture, and the two things that have to be true of it.
//!
//! It is written through the store's own public types at the store's own
//! documented layout, and the only way to know that stayed true is to read every
//! one of them back through the trait. If a layout moves, these fail loudly
//! rather than seeding a store nobody can read.

use somatize_fabric_fleet::{Fleet, Listing, Standing, ran, runs, seed};
use somatize_store::{Local, Store};
use tempfile::TempDir;

fn sown() -> (TempDir, Local) {
    let dir = TempDir::new().expect("a temporary directory");
    seed::sow(dir.path(), Some(&dir.path().join("listing.toml"))).expect("it sows");
    let store = Local::at(dir.path()).expect("a store where it sowed");
    (dir, store)
}

#[test]
fn everything_it_writes_reads_back_through_the_store() {
    let (_dir, store) = sown();

    let bound = store.bound().expect("the store scans");

    assert!(
        bound
            .iter()
            .filter(|one| one.name.starts_with("machine/"))
            .count()
            >= 5,
        "the readings did not land where a scan looks: {:?}",
        bound.iter().map(|one| &one.name).collect::<Vec<_>>()
    );
    assert!(bound.iter().any(|one| one.name.starts_with("run/")));
}

#[test]
fn the_fixture_shows_every_state_there_is() {
    // A fixture that only showed the ordinary case would hide the half worth
    // looking at, which is the half these screens exist for.
    let (_dir, store) = sown();

    let fleet = Fleet::read(&store, 90, 40).expect("the fleet reads");
    let standing = |what: Standing| fleet.seen.iter().filter(|one| one.standing == what).count();

    assert!(standing(Standing::Joined) >= 1, "nothing has a name");
    assert!(standing(Standing::Loose) >= 1, "nothing is free");
    assert!(standing(Standing::Quiet) >= 1, "nothing has stopped");
}

#[test]
fn the_one_that_stopped_keeps_the_name_it_had() {
    // What somebody needs to see about a machine that stopped is **which**
    // machine stopped.
    let (_dir, store) = sown();

    let fleet = Fleet::read(&store, 90, 40).expect("the fleet reads");
    let stopped = fleet
        .seen
        .iter()
        .find(|one| one.standing == Standing::Quiet)
        .expect("nothing had stopped");

    assert_eq!(stopped.named.as_ref().map(|host| host.as_str()), Some("w2"));
}

#[test]
fn one_of_them_measured_nothing_and_says_so() {
    // A kernel that keeps no load average is not a machine that is idle, and a
    // fixture without one would let a screen get that wrong unnoticed.
    let (_dir, store) = sown();

    let fleet = Fleet::read(&store, 90, 40).expect("the fleet reads");
    let bare = fleet
        .seen
        .iter()
        .find(|one| one.busy.is_none())
        .expect("every machine measured itself, so nothing tests the absent case");

    assert_eq!(bare.cores, None);
    assert!(bare.up_s > 0, "it still knows how long it has been up");
}

#[test]
fn two_of_them_are_on_one_box() {
    // The case the pid is in the id for, and the one where matching hostnames
    // would have been wrong rather than merely unprincipled.
    let (_dir, store) = sown();

    let fleet = Fleet::read(&store, 90, 40).expect("the fleet reads");
    let same_box = fleet
        .seen
        .iter()
        .filter(|one| one.id.starts_with("node9-"))
        .count();

    assert_eq!(same_box, 2);
}

#[test]
fn the_run_it_writes_has_a_machine_that_waits_more_than_it_runs() {
    // The row the whole third screen exists for: not busy, waiting.
    let (_dir, store) = sown();
    let which = runs(&store).expect("the runs read").pop().expect("a run");

    let out = ran(&store, &which.run, 40).expect("the run reads");
    let gpu = out
        .did
        .iter()
        .find(|one| one.host == "gpu-box")
        .expect("no gpu-box");

    assert!(gpu.waiting_us > gpu.took_us, "{gpu:?}");
    assert!(gpu.busy.unwrap() < 0.1, "and the machine is not busy");
}

#[test]
fn the_listing_it_writes_has_two_names_for_one_wire() {
    let (dir, _store) = sown();

    let paper = Listing::read(&dir.path().join("listing.toml")).expect("a listing");
    let wires = paper.wires(&Default::default()).expect("grouped");

    assert!(
        wires.wires.iter().any(|one| one.names.len() == 2),
        "nothing in the fixture shares a wire, so the rule that matters most is untested by eye"
    );
    assert!(
        wires.wires.iter().any(|one| !one.shared),
        "and nothing is a command"
    );
}
