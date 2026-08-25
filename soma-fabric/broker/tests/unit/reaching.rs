//! One host, reached through a broker, standing in the engine's hole.
//!
//! Two kinds of test here and both are needed. The **accounting** ones use a
//! desk that writes down what it was asked, and pin the rules that have no
//! visible effect until something goes wrong: that a session is greeted once
//! across four hosts, that a rendezvous nobody took is not released. The
//! **end-to-end** one stands up a real worker on a thread of this process and
//! carries a real slice to it, because everything else in this file would pass
//! just as happily if the wire were never opened at all.

use somatize_core::{
    Catalog, Ctx, Executor, Host as CoreHost, Node, NodeError, Plan, Transport, Value, compile,
    distribute, node,
};
use somatize_fabric_broker::{
    Ask, Embedded, Endpoint, Host, Needs, PROTOCOL, Path, Reaching, Reply, Session, SlotId,
};
use somatize_fabric_wire::Serving;
use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex};

// ── Doubles ──

struct Add(f64);

impl Node for Add {
    fn forward(&self, input: &Value, _ctx: &Ctx<'_>) -> Result<Value, NodeError> {
        match input {
            Value::Number(x) => Ok(Value::number(x + self.0)),
            other => Err(NodeError::new(format!(
                "Add needs a number, it was given {}",
                other.type_name()
            ))),
        }
    }
}

/// A broker that writes down what it was asked before answering it. The only
/// way to see rules whose whole point is that nothing happens.
fn noting(listing: Vec<(Host, Path)>) -> (Arc<Session>, Arc<Mutex<Vec<Ask>>>) {
    let heard = Arc::new(Mutex::new(Vec::new()));
    let noted = Arc::clone(&heard);
    let broker = Embedded::served_by(move |ask| {
        noted.lock().unwrap().push(ask.clone());
        match &ask {
            Ask::Hello { protocol, .. } => Reply::to_greeting(*protocol),
            Ask::Reach { host, .. } => match listing.iter().find(|(known, _)| known == host) {
                Some((_, path)) => Reply::Met {
                    path: path.clone(),
                    good_for: None,
                },
                None => Reply::Unreachable(format!("`{host}` is not listed")),
            },
            Ask::Done { .. } => Reply::Welcome { protocol: PROTOCOL },
        }
    });
    (Arc::new(Session::with(Arc::new(broker))), heard)
}

fn nowhere() -> Path {
    // A port nothing is on. Reaching it is meant to fail; what matters is *when*.
    Path::Direct {
        endpoint: Endpoint::Address("127.0.0.1:1".into()),
    }
}

/// Carries one node, placed on `host`, through `transport`.
fn run_through(host: &str, transport: &dyn Transport, input: f64) -> Result<Value, String> {
    // The client's `x` adds one. If the answer is anything else, it ran there.
    let (graph, catalog, mut placement, memory) = node("x", Add(1.0)).somatize().unwrap();
    placement.place_at("x", CoreHost::new(host));
    let plan = distribute(&compile(&graph, &catalog).unwrap(), &placement);
    assert!(
        matches!(plan, Plan::Remote { .. }),
        "the point of this test is a slice that leaves"
    );

    Executor::new(&catalog)
        .placed(&placement)
        .remembering(&memory)
        .reaching(CoreHost::new(host), transport)
        .run(&plan, Value::number(input))
        .map_err(|why| why.to_string())
}

// ── The rendezvous waits until somebody needs it ──

#[test]
fn building_a_handle_asks_the_broker_nothing() {
    // A graph names hosts a run may never reach: a branch not taken is a worker
    // not needed. So nothing is asked until something is dispatched.
    let (session, heard) = noting(vec![(Host::new("w1"), nowhere())]);
    let _reaching = Reaching::new(Arc::clone(&session), Host::new("w1"));

    assert!(
        heard.lock().unwrap().is_empty(),
        "it asked before anybody needed it: {:?}",
        heard.lock().unwrap()
    );
}

#[test]
fn an_unreachable_host_fails_when_it_is_needed_and_not_when_it_is_named() {
    // The behaviour change this buys, and it is deliberate: before a broker,
    // `Worker::at("bad:7000")` failed in the constructor.
    let (session, _heard) = noting(vec![]);
    let reaching = Reaching::new(Arc::clone(&session), Host::new("w1"));

    let why = run_through("w1", &reaching, 0.0).unwrap_err();
    assert!(why.contains("w1"), "it has to name the host: {why}");
    assert!(why.contains("not listed"), "{why}");
}

#[test]
fn a_host_whose_address_nobody_is_on_names_the_host_and_the_address() {
    let (session, _heard) = noting(vec![(Host::new("w1"), nowhere())]);
    let reaching = Reaching::new(Arc::clone(&session), Host::new("w1"));

    let why = run_through("w1", &reaching, 0.0).unwrap_err();
    assert!(why.contains("w1") && why.contains("127.0.0.1:1"), "{why}");
}

// ── The session is opened once, for all the hosts ──

#[test]
fn four_hosts_share_one_greeting() {
    // A greeting belongs to the conversation, not to a host. Four of them would
    // be asking the same question four times.
    let listing: Vec<(Host, Path)> = ["w1", "w2", "w3", "w4"]
        .iter()
        .map(|name| (Host::new(*name), nowhere()))
        .collect();
    let (session, heard) = noting(listing);

    for name in ["w1", "w2", "w3", "w4"] {
        let reaching = Reaching::new(Arc::clone(&session), Host::new(name));
        // Each fails to connect, which is fine: the greeting and the rendezvous
        // have already happened by then.
        let _ = run_through(name, &reaching, 0.0);
    }

    let heard = heard.lock().unwrap();
    let hellos = heard
        .iter()
        .filter(|ask| matches!(ask, Ask::Hello { .. }))
        .count();
    let reaches = heard
        .iter()
        .filter(|ask| matches!(ask, Ask::Reach { .. }))
        .count();
    assert_eq!(hellos, 1, "one session, not four: {heard:?}");
    assert_eq!(reaches, 4, "one rendezvous each: {heard:?}");
}

// ── What is not taken is not released ──

#[test]
fn a_rendezvous_nobody_took_is_not_let_go_of() {
    let (session, heard) = noting(vec![(Host::new("w1"), nowhere())]);
    drop(Reaching::new(Arc::clone(&session), Host::new("w1")));

    assert!(
        !heard
            .lock()
            .unwrap()
            .iter()
            .any(|ask| matches!(ask, Ask::Done { .. })),
        "it released something it never held"
    );
}

// ── The three paths this client cannot take yet ──

#[test]
fn the_paths_the_negotiation_has_not_arrived_for_are_refused_by_name() {
    // They are in the message from the first version on purpose. Until the
    // negotiation picks one, saying so beats failing further down as a
    // connection that never happened.
    for path in [
        Path::InProcess { slot: SlotId(0) },
        Path::Mount {
            dir: "/mnt/cluster/scratch".into(),
        },
        Path::Relayed {
            session: somatize_fabric_broker::SessionId("s-1".into()),
        },
    ] {
        let (session, _heard) = noting(vec![(Host::new("w1"), path.clone())]);
        let reaching = Reaching::new(Arc::clone(&session), Host::new("w1"));

        let why = run_through("w1", &reaching, 0.0).unwrap_err();
        assert!(
            why.contains("negotiation"),
            "for {path} it should say what is missing: {why}"
        );
    }
}

// ── End to end, over a real wire ──

#[test]
fn a_slice_reaches_a_real_worker_through_the_broker() {
    // Everything above this line would pass if the wire were never opened.
    // Here one is: a worker standing on a port of this machine, found through
    // the broker, sent a slice, and its answer comes back.
    let (opened, address) = channel();
    std::thread::Builder::new()
        .name("a-worker".into())
        .spawn(move || {
            let mut catalog = Catalog::new();
            // Adds five, where the client's `x` adds one. If the answer is 5,
            // it ran over there.
            catalog.insert("x", Arc::new(Add(5.0)));
            let _ = Serving::own(&catalog).listen_at("127.0.0.1:0", |addr| {
                let _ = opened.send(addr);
            });
        })
        .unwrap();
    let address = address.recv().expect("the worker never came up");

    let session = Arc::new(Session::with(Arc::new(Embedded::open([(
        Host::new("w1"),
        Path::Direct {
            endpoint: Endpoint::Address(address.to_string()),
        },
    )]))));
    let reaching = Reaching::new(Arc::clone(&session), Host::new("w1"));

    assert_eq!(
        run_through("w1", &reaching, 0.0).unwrap(),
        Value::number(5.0),
        "it ran with the client's catalog instead of the worker's"
    );
}

#[test]
fn the_second_slice_reuses_the_wire_and_does_not_ask_again() {
    let (opened, address) = channel();
    std::thread::Builder::new()
        .name("a-worker".into())
        .spawn(move || {
            let mut catalog = Catalog::new();
            catalog.insert("x", Arc::new(Add(5.0)));
            let _ = Serving::own(&catalog).listen_at("127.0.0.1:0", |addr| {
                let _ = opened.send(addr);
            });
        })
        .unwrap();
    let address = address.recv().expect("the worker never came up");

    let (session, heard) = noting(vec![(
        Host::new("w1"),
        Path::Direct {
            endpoint: Endpoint::Address(address.to_string()),
        },
    )]);
    let reaching = Reaching::new(Arc::clone(&session), Host::new("w1"));

    assert_eq!(
        run_through("w1", &reaching, 0.0).unwrap(),
        Value::number(5.0)
    );
    assert_eq!(
        run_through("w1", &reaching, 1.0).unwrap(),
        Value::number(6.0)
    );

    let heard = heard.lock().unwrap();
    assert_eq!(
        heard
            .iter()
            .filter(|ask| matches!(ask, Ask::Reach { .. }))
            .count(),
        1,
        "the broker was asked twice for a wire it had already given: {heard:?}"
    );
}

#[test]
fn letting_the_handle_go_lets_the_rendezvous_go() {
    let (opened, address) = channel();
    std::thread::Builder::new()
        .name("a-worker".into())
        .spawn(move || {
            let mut catalog = Catalog::new();
            catalog.insert("x", Arc::new(Add(5.0)));
            let _ = Serving::own(&catalog).listen_at("127.0.0.1:0", |addr| {
                let _ = opened.send(addr);
            });
        })
        .unwrap();
    let address = address.recv().expect("the worker never came up");

    let (session, heard) = noting(vec![(
        Host::new("w1"),
        Path::Direct {
            endpoint: Endpoint::Address(address.to_string()),
        },
    )]);
    let reaching = Reaching::new(Arc::clone(&session), Host::new("w1"));
    run_through("w1", &reaching, 0.0).unwrap();
    drop(reaching);

    // `done` is sent and not waited for, so asking now would be racing the
    // desk. One errand queue served by one thread means FIFO: any question that
    // comes back is a `Done` that has already been through. A barrier, not a
    // sleep.
    let _ = session.find(&Host::new("a name nobody listed"));

    assert!(
        heard
            .lock()
            .unwrap()
            .iter()
            .any(|ask| matches!(ask, Ask::Done { host } if host.as_str() == "w1")),
        "a rendezvous that was taken has to be let go of without the client \
         remembering to"
    );
}

#[test]
fn needs_is_empty_and_travels_anyway() {
    // The named hole: it goes on the wire from the first version so that the day
    // it fills, `Reach` does not change shape.
    let (session, heard) = noting(vec![(Host::new("w1"), nowhere())]);
    let reaching = Reaching::new(Arc::clone(&session), Host::new("w1"));
    let _ = run_through("w1", &reaching, 0.0);

    assert!(
        heard
            .lock()
            .unwrap()
            .iter()
            .any(|ask| matches!(ask, Ask::Reach { needs, .. } if *needs == Needs::default())),
        "the rendezvous did not carry a `Needs`"
    );
}

// ── Two names for one place, and one name run twice ──

/// A worker standing on a port of this machine, and the address it took.
fn a_worker_standing() -> String {
    let (opened, address) = channel();
    std::thread::Builder::new()
        .name("a-worker".into())
        .spawn(move || {
            let mut catalog = Catalog::new();
            catalog.insert("x", Arc::new(Add(5.0)));
            let _ = Serving::own(&catalog).listen_at("127.0.0.1:0", |addr| {
                let _ = opened.send(addr);
            });
        })
        .unwrap();
    address
        .recv()
        .expect("the worker never came up")
        .to_string()
}

#[test]
fn two_hosts_at_one_address_share_one_wire() {
    // The rule that matters most and shows least. A worker has *one* catalog,
    // so a process named twice and provisioned twice keeps only the second
    // half — and takes every activation over there with it.
    let addr = a_worker_standing();
    let at_it = Path::Direct {
        endpoint: Endpoint::Address(addr),
    };
    let (session, heard) = noting(vec![
        (Host::new("left"), at_it.clone()),
        (Host::new("right"), at_it),
    ]);

    let left = session.wire(&Host::new("left"), None).unwrap();
    let right = session.wire(&Host::new("right"), None).unwrap();

    assert!(
        Arc::ptr_eq(&left, &right),
        "two names for one place opened two wires, so one catalog would have \
         replaced the other"
    );
    // Two rendezvous, because they are two names; one wire, because they are
    // one place.
    assert_eq!(
        heard
            .lock()
            .unwrap()
            .iter()
            .filter(|ask| matches!(ask, Ask::Reach { .. }))
            .count(),
        2
    );
}

#[test]
fn two_hosts_with_the_same_command_are_two_processes() {
    // The other half, and today's suite already requires it: two hosts built
    // from the identical `argv` have to be two workers. A command is a thing to
    // run, and running it twice gives two of them.
    // `cat` stands there while its input is open and leaves when it closes,
    // which is exactly what the wire's `Drop` does to a child. Anything that
    // outlives its stdin makes this test wait for it.
    let argv = vec!["cat".to_string()];
    let same = Path::Direct {
        endpoint: Endpoint::Command(argv),
    };
    let (session, _heard) = noting(vec![
        (Host::new("one"), same.clone()),
        (Host::new("two"), same),
    ]);

    let one = session.wire(&Host::new("one"), None).unwrap();
    let two = session.wire(&Host::new("two"), None).unwrap();

    assert!(
        !Arc::ptr_eq(&one, &two),
        "the same command was run once and shared, where it has to be run twice"
    );
}

#[test]
fn a_host_is_asked_about_once_however_many_times_it_is_wanted() {
    // Deciding what to pack resolves every host before the run; the run then
    // resolves them again as it reaches them. That has to be one question.
    let (session, heard) = noting(vec![(Host::new("w1"), nowhere())]);

    let first = session.find(&Host::new("w1")).unwrap();
    let again = session.find(&Host::new("w1")).unwrap();
    let reaching = Reaching::new(Arc::clone(&session), Host::new("w1"));
    let _ = run_through("w1", &reaching, 0.0);

    assert_eq!(first, again);
    assert_eq!(
        heard
            .lock()
            .unwrap()
            .iter()
            .filter(|ask| matches!(ask, Ask::Reach { .. }))
            .count(),
        1,
        "the same host was asked about more than once in one session"
    );
}
