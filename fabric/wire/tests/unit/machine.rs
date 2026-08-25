//! What a machine says about itself, and what it says when it cannot measure.

use somatize_core::Fact;
use somatize_fabric_wire::Machine;
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
        id: "box-7".into(),
    };

    let fact = one.said();

    assert!(matches!(&fact, Fact::Said { kind, .. } if kind == "machine"));
    let (kind, pairs) = said(&one);
    assert_eq!(kind, "machine");
    assert_eq!(pairs["up_us"], "1234");
    assert_eq!(pairs["served"], "3");
    assert_eq!(pairs["cores"], "8");
    assert_eq!(
        pairs["id"], "box-7",
        "what the machine calls itself, not what the graph does"
    );
}

#[test]
fn a_machine_is_filed_under_what_it_calls_itself() {
    // `w1` is the client's word for it and there is no client on the idle
    // path — so a reading written to a store has to be filed under something
    // the worker can know on its own. Whoever reads joins the two by seeing
    // the same `id` on a reading that did come down a wire.
    use somatize_fabric_wire::filed;

    assert_eq!(filed("box-7"), "machine/box-7");
    assert_ne!(
        Machine::here(Duration::ZERO, 0).id,
        "",
        "a machine always knows something to call itself"
    );
}

#[test]
fn two_workers_on_one_box_are_two_machines() {
    // Filing under the hostname alone would have the second quietly
    // overwriting the first, and a fleet of two would read as a fleet of one.
    let one = Machine::here(Duration::ZERO, 0).id;

    assert!(one.ends_with(&format!("-{}", std::process::id())), "{one}");
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
        id: "box-7".into(),
    });

    assert_eq!(pairs["busy"], "9.0000");
    assert!(
        !pairs
            .keys()
            .any(|one| one.contains("healthy") || one.contains("flag"))
    );
}

// ── And back again, because somebody has to read one ──

#[test]
fn a_reading_written_down_comes_back_the_same_reading() {
    let said = Machine {
        up: Duration::from_secs(3_600),
        busy: Some(0.4213),
        cores: Some(8),
        memory: Some(0.6187),
        served: 1_284,
        id: "node3-4127".into(),
    };

    let (_, pairs) = said.said().flattened();

    // Not `assert_eq!` on the whole of it: `said` writes four decimals, so the
    // fifth is gone on purpose and a round trip that demanded it back would be
    // asking the format to be something it deliberately is not.
    let back = Machine::read(&pairs);
    assert_eq!(back.up, said.up);
    assert_eq!(back.cores, said.cores);
    assert_eq!(back.served, said.served);
    assert_eq!(back.id, said.id);
    assert!(
        (back.busy.unwrap() - 0.4213).abs() < 1e-9,
        "{:?}",
        back.busy
    );
    assert!(
        (back.memory.unwrap() - 0.6187).abs() < 1e-9,
        "{:?}",
        back.memory
    );
}

#[test]
fn what_nobody_measured_reads_back_as_nobody_measured() {
    // The rule the other half writes by, from this side: a kernel with no
    // `/proc` is not a machine that is idle, and a reader has to be able to
    // tell the two apart.
    let bare = Machine {
        up: Duration::from_secs(60),
        served: 3,
        id: "laptop-91".into(),
        ..Machine::default()
    };

    let (_, pairs) = bare.said().flattened();
    let back = Machine::read(&pairs);

    assert_eq!(
        back.busy, None,
        "a load average nobody kept read as a number"
    );
    assert_eq!(back.memory, None);
    assert_eq!(back.cores, None);
    assert_eq!(back.served, 3, "and what it did count is still there");
}

#[test]
fn a_field_this_version_cannot_read_costs_that_field_and_not_the_reading() {
    // What another version would have written. Refusing all of it would throw
    // away the uptime for the sake of the load average, and the uptime is the
    // half that says whether the machine is even there.
    let pairs = vec![
        ("up_us".to_string(), "60000000".to_string()),
        ("served".to_string(), "7".to_string()),
        ("id".to_string(), "node9-3312".to_string()),
        ("busy".to_string(), "0.42 (one minute)".to_string()),
    ];

    let back = Machine::read(&pairs);

    assert_eq!(
        back.busy, None,
        "it should read as nobody said, not as 0.42"
    );
    assert_eq!(back.up, Duration::from_secs(60));
    assert_eq!(back.served, 7);
    assert_eq!(back.id, "node9-3312");
}

#[test]
fn a_reading_from_a_version_that_says_more_than_this_one_is_still_a_reading() {
    // The forward half of the same promise: the day `said` grows a field, a
    // binary that predates it goes on reading everything it knew about.
    let pairs = vec![
        ("up_us".to_string(), "1000000".to_string()),
        ("served".to_string(), "1".to_string()),
        ("id".to_string(), "node7-991".to_string()),
        ("gpus".to_string(), "4".to_string()),
        ("accepts".to_string(), "pickle project".to_string()),
    ];

    let back = Machine::read(&pairs);

    assert_eq!(back.id, "node7-991");
    assert_eq!(back.served, 1);
}
