//! What the two sides say, put into bytes and taken out again.
//!
//! With no processes in the way: if the round trip does not give the same thing,
//! you do not need a worker to find out. What does need testing here and nowhere
//! else is what **does not** cross.

use somatize_core::{Device, Host, Key, Keys, Memory, NodeId, Outcome, Placement, Plan, Value};
use somatize_fabric_wire::{Answer, Label, MessageError, Request};

fn work(input: Value) -> Request {
    Request::Work {
        plan: Plan::Empty,
        input,
        known: Vec::new(),
        keys: Vec::new(),
        placement: Placement::new(),
        memory: Memory::new(),
    }
}

/// The message in bytes, or why it could not be. A helper so the tests read the
/// same as before the operations moved onto the types.
fn bytes_of(request: &Request) -> Result<Vec<u8>, MessageError> {
    request.to_bytes()
}

/// The same for the other side's.
fn bytes_of_answer(answer: &Answer) -> Result<Vec<u8>, MessageError> {
    answer.to_bytes()
}

fn round_trip(request: Request) {
    let bytes = bytes_of(&request).unwrap();
    assert_eq!(Request::from_bytes(&bytes).unwrap(), request);
}

#[test]
fn the_values_go_and_come_back_equal() {
    for value in [
        Value::Null,
        Value::number(-0.5),
        Value::text("hello, ñandú"),
        Value::Bytes(std::sync::Arc::new(vec![0, 255, 7])),
        Value::list(vec![Value::number(1.0), Value::text("two")]),
        Value::map(vec![
            ("a".to_string(), Value::number(1.0)),
            ("b".to_string(), Value::Null),
        ]),
    ] {
        round_trip(work(value));
    }
}

#[test]
fn a_nested_value_keeps_its_shape() {
    round_trip(work(Value::list(vec![Value::map(vec![(
        "inside".to_string(),
        Value::list(vec![Value::number(1.0)]),
    )])])));
}

#[test]
fn a_map_keeps_its_order() {
    // It is the reason `Value::Map` is a list of pairs and not a `HashMap`, and
    // this is where not being one would be paid for: the two sides of the wire
    // are two processes, and a `HashMap` iterates differently in each.
    let order = vec![
        ("z".to_string(), Value::number(1.0)),
        ("a".to_string(), Value::number(2.0)),
        ("m".to_string(), Value::number(3.0)),
    ];
    let bytes = bytes_of(&work(Value::map(order.clone()))).unwrap();
    let Request::Work { input, .. } = Request::from_bytes(&bytes).unwrap() else {
        panic!("not a work message")
    };

    let Value::Map(pairs) = &input else {
        panic!("not a map")
    };
    assert_eq!(
        pairs.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>(),
        ["z", "a", "m"]
    );
}

#[test]
fn the_plans_go_and_come_back_equal() {
    round_trip(Request::Work {
        plan: Plan::Sequence(vec![
            Plan::Execute {
                node: "a".into(),
                from: Vec::new(),
            },
            Plan::Wave(vec![
                Plan::Execute {
                    node: "b".into(),
                    from: vec!["a".into()],
                },
                Plan::Remote {
                    host: Host::new("other"),
                    inner: Box::new(Plan::Execute {
                        node: "c".into(),
                        from: vec!["a".into(), "b".into()],
                    }),
                },
            ]),
            Plan::Empty,
        ]),
        input: Value::Null,
        known: Vec::new(),
        keys: Vec::new(),
        placement: Placement::new(),
        memory: Memory::new(),
    });
}

#[test]
fn what_was_already_produced_travels_with_its_id() {
    round_trip(Request::Work {
        plan: Plan::Empty,
        input: Value::Null,
        known: vec![
            (NodeId::from("a"), Value::number(1.0)),
            (NodeId::from("b"), Value::text("two")),
        ],
        keys: Vec::new(),
        placement: Placement::new(),
        memory: Memory::new(),
    });
}

#[test]
fn the_placement_crosses_the_wire() {
    let mut placement = Placement::new();
    placement.place("a", Device::Cuda(3));

    let bytes = bytes_of(&Request::Work {
        plan: Plan::Execute {
            node: "a".into(),
            from: Vec::new(),
        },
        input: Value::Null,
        known: Vec::new(),
        keys: Vec::new(),
        placement,
        memory: Memory::new(),
    })
    .unwrap();

    let Request::Work { placement, .. } = Request::from_bytes(&bytes).unwrap() else {
        panic!("not a work message")
    };
    assert_eq!(placement.of(&"a".into()), Some(&Device::Cuda(3)));
}

#[test]
fn only_the_placement_of_the_nodes_that_travel_is_sent() {
    // Sending the whole `Placement` would put on the wire where nodes that do
    // not even exist there run.
    let mut placement = Placement::new();
    placement.place("a", Device::Cpu);
    placement.place("stays", Device::Cuda(0));

    let bytes = bytes_of(&Request::Work {
        plan: Plan::Execute {
            node: "a".into(),
            from: Vec::new(),
        },
        input: Value::Null,
        known: Vec::new(),
        keys: Vec::new(),
        placement,
        memory: Memory::new(),
    })
    .unwrap();

    let Request::Work { placement, .. } = Request::from_bytes(&bytes).unwrap() else {
        panic!("not a work message")
    };
    assert_eq!(placement.of(&"a".into()), Some(&Device::Cpu));
    assert_eq!(placement.of(&"stays".into()), None);
    assert_eq!(placement.len(), 1);
}

#[test]
fn the_names_of_what_it_reads_cross_the_wire() {
    // Without them the slice over there can name nothing it produces, and the
    // cache stops at the process boundary.
    let bytes = bytes_of(&Request::Work {
        plan: Plan::Empty,
        input: Value::Null,
        known: vec![(NodeId::from("a"), Value::number(1.0))],
        keys: vec![(NodeId::from("a"), Keys::One(Key::new("sha256:abc")))],
        placement: Placement::new(),
        memory: Memory::new(),
    })
    .unwrap();

    let Request::Work { keys, .. } = Request::from_bytes(&bytes).unwrap() else {
        panic!("not a work message")
    };
    assert_eq!(
        keys,
        [(NodeId::from("a"), Keys::One(Key::new("sha256:abc")))]
    );
}

#[test]
fn only_what_is_remembered_of_the_nodes_that_travel_is_sent() {
    // The same rule as the placement, and for the same reason: what is
    // remembered of a node that does not exist over there is nobody's business
    // over there.
    let mut memory = Memory::new();
    memory.identify("a", "Encoder");
    memory.freeze("a", Some("sha256:weights".into()));
    memory.cache("a", Some("a100-fp16".into()));
    memory.written_as("a", "v1");
    memory.identify("stays", "Head");
    memory.cache("stays", None);

    let bytes = bytes_of(&Request::Work {
        plan: Plan::Execute {
            node: "a".into(),
            from: Vec::new(),
        },
        input: Value::Null,
        known: Vec::new(),
        keys: Vec::new(),
        placement: Placement::new(),
        memory,
    })
    .unwrap();

    let Request::Work { memory, .. } = Request::from_bytes(&bytes).unwrap() else {
        panic!("not a work message")
    };
    assert_eq!(memory.identity_of(&"a".into()), Some("Encoder"));
    assert_eq!(memory.state_of(&"a".into()), Some("sha256:weights"));
    assert_eq!(memory.salt_of(&"a".into()), Some("a100-fp16"));
    assert_eq!(memory.fingerprint_of(&"a".into()), Some("v1"));
    assert!(!memory.is_cached(&"stays".into()));
    assert_eq!(memory.len(), 1);
}

#[test]
fn the_host_does_not_travel_as_a_placement() {
    // What goes over the wire is the device. The host has already been used: it
    // is what decided this slice would travel, and over there it means nothing.
    let mut placement = Placement::new();
    placement.place_at("a", Host::new("worker1"));

    let bytes = bytes_of(&Request::Work {
        plan: Plan::Execute {
            node: "a".into(),
            from: Vec::new(),
        },
        input: Value::Null,
        known: Vec::new(),
        keys: Vec::new(),
        placement,
        memory: Memory::new(),
    })
    .unwrap();

    let Request::Work { placement, .. } = Request::from_bytes(&bytes).unwrap() else {
        panic!("not a work message")
    };
    assert!(placement.is_local());
}

#[test]
fn a_greeting_without_an_artifact_goes_and_comes_back() {
    round_trip(Request::Hello {
        runtime: "rust".into(),
        offering: None,
    });
}

#[test]
fn a_greeting_announces_the_artifact_without_carrying_it() {
    // What makes the `have`/`want` cheap: the name is forty bytes and the
    // catalog can be megabytes.
    let hello = Request::Hello {
        runtime: "cpython-3.13/cloudpickle-3.1".into(),
        offering: Some(Label {
            kind: "pickle".into(),
            id: "sha256:abc".into(),
        }),
    };
    let bytes = bytes_of(&hello).unwrap();

    assert!(
        bytes.len() < 80,
        "the greeting weighs {} bytes",
        bytes.len()
    );
    round_trip(hello);
}

#[test]
fn the_artifact_travels_in_its_own_message() {
    round_trip(Request::Provision {
        bytes: vec![0, 1, 2, 255],
    });
}

#[test]
fn an_artifacts_bytes_are_not_looked_at() {
    // The asymmetry with `Opaque`: an artifact is a pile of bytes opaque by
    // design, and here they pass through as they are. Garbage included.
    round_trip(Request::Provision {
        bytes: vec![0xff; 1000],
    });
}

#[test]
fn the_answers_go_and_come_back_equal() {
    for answer in [
        Answer::Ready,
        Answer::Send,
        Answer::Refused("I do not like you".into()),
        Answer::Failed("node `broken` failed".into()),
        Answer::Done(Outcome {
            last: Value::number(42.0),
            produced: vec![
                (NodeId::from("a"), Value::number(1.0)),
                (NodeId::from("b"), Value::number(42.0)),
            ],
            keys: vec![(NodeId::from("a"), Keys::One(Key::new("sha256:abc")))],
        }),
    ] {
        let bytes = bytes_of_answer(&answer).unwrap();
        assert_eq!(Answer::from_bytes(&bytes).unwrap(), answer);
    }
}

#[test]
fn an_opaque_does_not_fit_on_the_wire() {
    let broken = bytes_of(&work(Value::opaque(7u32))).unwrap_err();

    assert_eq!(broken, MessageError::Opaque);
    assert!(
        broken.to_string().contains("only exists"),
        "the message has to say why, not just no: {broken}"
    );
}

#[test]
fn an_opaque_hidden_inside_a_list_does_not_either() {
    // The one that would slip through if the check were on the top-level value's
    // type and not on each one.
    assert_eq!(
        bytes_of(&work(Value::list(vec![
            Value::number(1.0),
            Value::opaque(7u32)
        ])))
        .unwrap_err(),
        MessageError::Opaque
    );
}

#[test]
fn an_opaque_inside_what_was_already_produced_does_not_either() {
    assert_eq!(
        bytes_of(&Request::Work {
            plan: Plan::Empty,
            input: Value::Null,
            known: vec![(NodeId::from("a"), Value::opaque(7u32))],
            keys: Vec::new(),
            placement: Placement::new(),
            memory: Memory::new(),
        })
        .unwrap_err(),
        MessageError::Opaque
    );
}

#[test]
fn an_opaque_does_not_come_back_either() {
    // The other direction: what was produced there and cannot come back.
    assert_eq!(
        bytes_of_answer(&Answer::Done(Outcome {
            last: Value::opaque(7u32),
            produced: Vec::new(),
            keys: Vec::new(),
        }))
        .unwrap_err(),
        MessageError::Opaque
    );
}

#[test]
fn half_the_bytes_are_not_read_halfway() {
    let bytes = bytes_of(&work(Value::text("something fairly long"))).unwrap();

    assert!(matches!(
        Request::from_bytes(&bytes[..bytes.len() - 3]).unwrap_err(),
        MessageError::Malformed(_)
    ));
}

#[test]
fn leftover_bytes_are_as_suspicious_as_missing_ones() {
    // No format checks this for you — measured on all three that were
    // considered — so the codec checks it, and the count is in the message.
    let mut bytes = bytes_of(&work(Value::Null)).unwrap();
    bytes.push(0);

    let said = Request::from_bytes(&bytes).unwrap_err().to_string();
    assert!(said.contains("1 bytes left over"), "{said}");
}

#[test]
fn bytes_that_are_not_a_message_are_reported() {
    assert!(matches!(
        Request::from_bytes(&[99, 99, 99]).unwrap_err(),
        MessageError::Malformed(_)
    ));
    assert!(matches!(
        Answer::from_bytes(&[99, 99, 99]).unwrap_err(),
        MessageError::Malformed(_)
    ));
}

/// A value the way it was written down before bytes were asked for by name.
///
/// The same variants in the same order, so the same encoding — except `Bytes`,
/// which here has no `serialize_bytes` behind it and so goes out as serde's
/// default for a slice: **a sequence, one element per byte**. That is what a
/// store filled last week has in it.
#[derive(serde::Serialize)]
enum Older<'a> {
    #[allow(dead_code)]
    Null,
    #[allow(dead_code)]
    Number(f64),
    #[allow(dead_code)]
    Text(&'a str),
    Bytes(&'a [u8]),
    #[allow(dead_code)]
    List(Vec<Older<'a>>),
    #[allow(dead_code)]
    Map(Vec<(&'a str, Older<'a>)>),
}

#[test]
fn bytes_written_as_a_list_of_numbers_are_still_read_as_bytes() {
    // A store outlives every binary that wrote into it, so the day the encoding
    // gets narrower is the day reading has to take both.
    let old = rmp_serde::to_vec(&Older::Bytes(b"a tensor, once")).expect("it writes");

    let read: Value = rmp_serde::from_slice(&old).expect("and it still reads");

    assert_eq!(
        read,
        Value::Bytes(std::sync::Arc::new(b"a tensor, once".to_vec()))
    );
}

#[test]
fn and_what_is_written_now_is_the_size_of_the_data_and_not_twice_it() {
    // The size is the visible half. The invisible half — an element of work per
    // byte, at each end — is the one that cost the time, and it does not fit in
    // an assert. Bytes over 127 on purpose: those are the ones msgpack spends
    // two bytes on as numbers, and a tensor's bytes are uniformly spread.
    const RAW: usize = 1024;
    let value = Value::Bytes(std::sync::Arc::new(vec![200u8; RAW]));

    let now = rmp_serde::to_vec(&value).expect("it writes");
    let then = rmp_serde::to_vec(&Older::Bytes(&[200u8; RAW])).expect("it writes");

    assert!(
        now.len() < RAW + 32,
        "now: {} bytes for {RAW} of data",
        now.len()
    );
    assert!(
        then.len() > 2 * RAW,
        "then: {} bytes for {RAW} of data",
        then.len()
    );
}

#[test]
fn what_maps_still_maps_on_the_other_side() {
    // The projection is written out one fact at a time, so a new one that is not
    // added to it does not fail: it stops being true over there, and a node goes
    // on answering the same thing while its cache quietly loses the grain it was
    // asked for. Nothing catches that but a test that crosses.
    let mut memory = Memory::new();
    memory.identify("embed", "Embed");
    memory.map("embed");
    memory.map("elsewhere");

    let bytes = bytes_of(&Request::Work {
        plan: Plan::Execute {
            node: "embed".into(),
            from: Vec::new(),
        },
        input: Value::Null,
        known: Vec::new(),
        keys: Vec::new(),
        placement: Placement::new(),
        memory,
    })
    .unwrap();

    let Request::Work { memory, .. } = Request::from_bytes(&bytes).unwrap() else {
        panic!("not a work message")
    };
    assert!(memory.is_mapped(&NodeId::from("embed")));
    assert!(
        !memory.is_mapped(&NodeId::from("elsewhere")),
        "what is remembered of a node that is not in this plan does not travel"
    );
}
