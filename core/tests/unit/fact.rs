//! The vocabulary of level 1: what a fact says, and how it is written down.

use somatize_core::{Device, Fact, Host, Key, NodeId};
use std::time::Duration;

/// The fields of a flattened fact, as a map, for asserting on one at a time.
fn said(fact: &Fact) -> (String, std::collections::HashMap<String, String>) {
    let (kind, fields) = fact.flattened();
    (kind.to_string(), fields.into_iter().collect())
}

#[test]
fn a_node_that_ran_says_which_one_how_long_and_where() {
    let (kind, said) = said(&Fact::Ran {
        node: NodeId::from("embed"),
        began: Duration::ZERO,
        took: Duration::from_millis(12),
        device: Some("cuda:0".parse::<Device>().unwrap()),
    });

    assert_eq!(kind, "ran");
    assert_eq!(said["node"], "embed");
    assert_eq!(said["took_us"], "12000");
    assert_eq!(said["device"], "cuda:0");
}

#[test]
fn a_node_nobody_placed_says_nothing_about_a_device() {
    // Absent and not `"none"`: the record is read by whoever draws it, and a
    // string that looks like a device is worse than no column at all.
    let (_, said) = said(&Fact::Ran {
        node: NodeId::from("clean"),
        began: Duration::ZERO,
        took: Duration::from_micros(3),
        device: None,
    });

    assert!(!said.contains_key("device"));
}

#[test]
fn a_duration_is_whole_microseconds_and_not_a_float() {
    // Text somebody reads with `cat` and something else parses, and neither
    // should have to decide how many decimals were written.
    let (_, said) = said(&Fact::Finished {
        took: Duration::from_nanos(1_500),
    });

    assert_eq!(said["took_us"], "1");
}

#[test]
fn what_happened_elsewhere_comes_out_as_a_host_field_and_not_as_a_tree() {
    // The nesting exists so that nothing which travelled has to be rewritten;
    // it is not meant to reach the record, where a reader wants columns.
    let there = Fact::Elsewhere {
        host: Host::new("worker1"),
        saw: Box::new(Fact::Ran {
            node: NodeId::from("tokenize"),
            began: Duration::ZERO,
            took: Duration::from_millis(5),
            device: None,
        }),
    };

    let (kind, said) = said(&there);

    assert_eq!(kind, "ran", "it is still a node that ran");
    assert_eq!(said["node"], "tokenize");
    assert_eq!(said["host"], "worker1");
}

#[test]
fn a_fact_that_crossed_two_machines_keeps_the_route_in_order() {
    // A slice that carries on to a third host. The nearest host is written last,
    // which is the order somebody reading the route wants.
    let twice = Fact::Elsewhere {
        host: Host::new("first"),
        saw: Box::new(Fact::Elsewhere {
            host: Host::new("second"),
            saw: Box::new(Fact::Finished {
                took: Duration::from_millis(1),
            }),
        }),
    };

    let (kind, fields) = twice.flattened();
    let hosts: Vec<&str> = fields
        .iter()
        .filter(|(name, _)| name == "host")
        .map(|(_, what)| what.as_str())
        .collect();

    assert_eq!(kind, "finished");
    assert_eq!(hosts, ["second", "first"]);
}

#[test]
fn the_two_that_end_a_run_say_so_and_the_others_do_not() {
    // Asked by whoever writes records, so a writer never has to know the
    // vocabulary: one record ends where one of these arrives.
    assert!(
        Fact::Finished {
            took: Duration::ZERO
        }
        .ends_a_run()
    );
    assert!(Fact::Broke { why: "gone".into() }.ends_a_run());
    assert!(
        !Fact::Failed {
            node: NodeId::from("a"),
            why: "boom".into(),
        }
        .ends_a_run(),
        "a node failing is not the run ending: the run says so itself, after"
    );
    assert!(
        !Fact::Kept {
            node: NodeId::from("a"),
            key: Key::new("k"),
        }
        .ends_a_run()
    );
}
