//! What a machine says about itself, and what it says when it cannot measure.

use soma_next_core::Fact;
use soma_next_transport::Machine;
use std::time::Duration;

fn said(machine: &Machine) -> (String, std::collections::HashMap<String, String>) {
    let fact = machine.said();
    let (kind, pairs) = fact.flattened();
    (kind.to_string(), pairs.into_iter().collect())
}

#[test]
fn a_reading_crosses_as_a_flat_fact_and_not_as_a_variant_of_its_own() {
    // The core does not learn what a load average is, the same way it never
    // learned what a loss is. What crosses is `(kind, pairs)` inside
    // `Fact::Said`, which is the shape everything is written down as anyway.
    let one = Machine {
        up: Duration::from_micros(1234),
        busy: Some(0.25),
        cores: Some(8),
        memory: Some(0.5),
        served: 3,
    };

    let fact = one.said();

    assert!(matches!(&fact, Fact::Said { kind, .. } if kind == "machine"));
    let (kind, pairs) = said(&one);
    assert_eq!(kind, "machine");
    assert_eq!(pairs["up_us"], "1234");
    assert_eq!(pairs["served"], "3");
    assert_eq!(pairs["cores"], "8");
}

#[test]
fn what_nobody_measured_is_absent_and_not_zero() {
    // The rule `Seen` spends its docstring on, kept here too: a kernel that
    // does not say what its load is has not said its load is nothing.
    let (_, pairs) = said(&Machine {
        up: Duration::from_micros(5),
        served: 1,
        ..Machine::default()
    });

    assert!(!pairs.contains_key("busy"), "{pairs:?}");
    assert!(!pairs.contains_key("memory"), "{pairs:?}");
    assert!(!pairs.contains_key("cores"), "{pairs:?}");
    assert_eq!(pairs["up_us"], "5");
}

#[test]
fn a_reading_of_this_machine_says_how_long_it_has_been_up_wherever_it_runs() {
    // Uptime and how much it has served need no kernel, so they are there on
    // any platform; the rest is `/proc` and says nothing where there is none.
    let one = Machine::here(Duration::from_secs(7), 42);

    assert_eq!(one.up, Duration::from_secs(7));
    assert_eq!(one.served, 42);
    if let Some(busy) = one.busy {
        assert!(busy >= 0.0, "a run queue is not negative");
    }
    if let Some(memory) = one.memory {
        assert!(
            (0.0..=1.0).contains(&memory),
            "a fraction of memory is a fraction"
        );
    }
}

#[test]
fn nothing_in_it_is_a_judgement() {
    // No bound, no flag, no word for bad. Whether 0.9 busy is trouble is an
    // opinion at a threshold, and those live in `health/` against a record that
    // has already been written.
    let (_, pairs) = said(&Machine {
        up: Duration::from_secs(1),
        busy: Some(9.0),
        cores: Some(1),
        memory: Some(0.99),
        served: 1,
    });

    assert_eq!(pairs["busy"], "9.0000");
    assert!(
        !pairs
            .keys()
            .any(|one| one.contains("healthy") || one.contains("flag"))
    );
}
