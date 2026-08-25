//! One machine, and how a name and a machine stand towards each other.
//!
//! The rule lives here rather than in the whole read because it is a rule about
//! a **pair** and not about a store: what it needs is a name, a reading and a
//! bound, and all three fit in three lines.

use soma_fabric_fleet::{Host, Machine, Seen, Standing};
use std::time::Duration;

fn reading(id: &str) -> Machine {
    Machine {
        up: Duration::from_secs(3_600),
        busy: Some(0.42),
        cores: Some(8),
        memory: Some(0.62),
        served: 12,
        id: id.into(),
    }
}

#[test]
fn a_name_and_a_machine_that_found_each_other_are_joined() {
    let seen = Seen::of(reading("node3-4127"), 1_000, 4, Some(Host::new("w1")), 90);

    assert_eq!(seen.standing, Standing::Joined);
    assert_eq!(seen.named, Some(Host::new("w1")));
}

#[test]
fn writing_with_nobody_s_name_on_it_is_a_row_and_not_a_gap() {
    // The machine nobody is using says it is there. A view derived from a run
    // cannot have this row at all, which is why the idle reporting exists.
    let seen = Seen::of(reading("node9-3312"), 1_000, 4, None, 90);

    assert_eq!(seen.standing, Standing::Loose);
    assert_eq!(seen.named, None, "it is unnamed, not badly named");
}

#[test]
fn quiet_wins_over_both_and_keeps_whatever_name_it_had() {
    // What somebody needs to see about `w1` gone quiet is that it is `w1`.
    let seen = Seen::of(
        reading("node4-8810"),
        1_000,
        2_400,
        Some(Host::new("w2")),
        90,
    );

    assert_eq!(seen.standing, Standing::Quiet);
    assert_eq!(seen.named, Some(Host::new("w2")));
}

#[test]
fn a_machine_nobody_named_can_go_quiet_too() {
    let seen = Seen::of(reading("node9-3312"), 1_000, 2_400, None, 90);

    assert_eq!(seen.standing, Standing::Quiet);
}

#[test]
fn the_bound_is_a_bound_and_not_a_threshold_on_anything_measured() {
    // A machine at 0.99 busy is not quiet, and one at 0.0 is not either. The
    // only thing the standing is about is whether it is still writing — no
    // reading of a kernel takes part in it.
    let flat_out = Machine {
        busy: Some(0.99),
        memory: Some(0.98),
        ..reading("node7-991")
    };

    assert_eq!(
        Seen::of(flat_out, 1_000, 4, Some(Host::new("gpu-box")), 90).standing,
        Standing::Joined
    );
}

#[test]
fn what_nobody_measured_stays_nobody_measured_on_the_way_out() {
    // The wire's rule, carried through the view: a kernel that keeps no load
    // average is not a machine that is idle, and JSON has a word for that which
    // is not zero.
    let bare = Machine {
        up: Duration::from_secs(60),
        served: 3,
        id: "laptop-91".into(),
        ..Machine::default()
    };

    let seen = Seen::of(bare, 1_000, 1, None, 90);
    let json = serde_json::to_value(&seen).expect("a view of a machine will write");

    assert!(json["busy"].is_null(), "{}", json);
    assert!(json["cores"].is_null(), "{}", json);
    assert_eq!(json["served"], 3);
    assert_eq!(json["standing"], "loose");
    assert_eq!(json["named"], serde_json::Value::Null);
}
