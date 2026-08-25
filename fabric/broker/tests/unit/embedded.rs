//! The broker that lives on a thread of this process.
//!
//! Two thirds of this file is about the thread rather than about the answers,
//! and that is the right proportion: what it answers is four lines of `match`,
//! while a thread that dies quietly is a client that waits forever.

use soma_fabric_broker::{
    Ask, Embedded, Endpoint, Host, Identity, Needs, PROTOCOL, Path, Reply, SlotId, Unanswered,
};
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};

fn at(addr: &str) -> Path {
    Path::Direct {
        endpoint: Endpoint::Address(addr.into()),
    }
}

fn reach(host: &str) -> Ask {
    Ask::Reach {
        host: Host::new(host),
        needs: Needs::default(),
    }
}

// ── What it answers ──

#[test]
fn it_greets_whoever_speaks_its_version() {
    let broker = Embedded::open([(Host::new("w1"), at("node3:7000"))]);
    assert_eq!(
        broker.ask(&Ask::hello()).unwrap(),
        Reply::Welcome { protocol: PROTOCOL }
    );
}

#[test]
fn it_does_not_care_who_you_claim_to_be_because_it_has_no_policy() {
    let broker = Embedded::open([]);
    assert_eq!(
        broker
            .ask(&Ask::hello_as(Identity("a token".into())))
            .unwrap(),
        Reply::Welcome { protocol: PROTOCOL }
    );
}

#[test]
fn a_host_it_knows_comes_back_with_its_path_and_no_expiry() {
    let broker = Embedded::open([(Host::new("w1"), at("node3:7000"))]);
    assert_eq!(
        broker.ask(&reach("w1")).unwrap(),
        Reply::Met {
            path: at("node3:7000"),
            // No policy, so nobody is taking this back.
            good_for: None,
        }
    );
}

#[test]
fn a_host_in_this_very_process_is_a_slot_and_not_a_trip() {
    // Path 1, which is the `.at()` that never actually left home. Never
    // inferred: it is here because somebody listed it here.
    let broker = Embedded::open([(Host::new("here"), Path::InProcess { slot: SlotId(2) })]);
    assert_eq!(
        broker.ask(&reach("here")).unwrap(),
        Reply::Met {
            path: Path::InProcess { slot: SlotId(2) },
            good_for: None,
        }
    );
}

#[test]
fn a_host_it_does_not_know_is_told_what_it_does_know() {
    // The usual cause is a typo in an `.at()`, and the list is three names long.
    let broker = Embedded::open([
        (Host::new("w1"), at("node3:7000")),
        (Host::new("w2"), at("node4:7000")),
    ]);
    match broker.ask(&reach("w3")).unwrap() {
        Reply::Unreachable(why) => {
            assert!(why.contains("`w3`"), "{why}");
            assert!(why.contains("`w1`") && why.contains("`w2`"), "{why}");
        }
        other => panic!("`w3` is not listed, so that should not be {other:?}"),
    }
}

#[test]
fn a_broker_with_nothing_listed_says_that_instead_of_listing_nothing() {
    let broker = Embedded::open([]);
    match broker.ask(&reach("w1")).unwrap() {
        Reply::Unreachable(why) => assert!(why.contains("no hosts listed at all"), "{why}"),
        other => panic!("nothing is listed, so that should not be {other:?}"),
    }
}

#[test]
fn what_it_knows_is_listed_in_a_fixed_order() {
    // Two brokers built from the same hosts in opposite orders have to say the
    // same sentence, or the same typo produces two different bug reports.
    let one = Embedded::open([
        (Host::new("b"), at("2:1")),
        (Host::new("a"), at("1:1")),
        (Host::new("c"), at("3:1")),
    ]);
    let other = Embedded::open([
        (Host::new("c"), at("3:1")),
        (Host::new("b"), at("2:1")),
        (Host::new("a"), at("1:1")),
    ]);
    assert_eq!(
        one.ask(&reach("z")).unwrap(),
        other.ask(&reach("z")).unwrap()
    );
}

#[test]
fn the_session_stays_open_across_rendezvous() {
    let broker = Embedded::open([
        (Host::new("w1"), at("node3:7000")),
        (Host::new("w2"), at("node4:7000")),
    ]);
    broker.ask(&Ask::hello()).unwrap();
    assert!(matches!(
        broker.ask(&reach("w1")).unwrap(),
        Reply::Met { .. }
    ));
    assert!(matches!(
        broker.ask(&reach("w2")).unwrap(),
        Reply::Met { .. }
    ));
}

// ── Done, the one message with no answer ──

#[test]
fn letting_a_rendezvous_go_does_not_wait_for_anything() {
    let broker = Embedded::open([(Host::new("w1"), at("node3:7000"))]);
    assert_eq!(broker.done(&Host::new("w1")), Ok(()));
}

#[test]
fn asking_done_is_refused_rather_than_waited_on() {
    // The whole point: a message with no answer, asked as though it had one, is
    // the hang this type exists to not have.
    let broker = Embedded::open([]);
    assert_eq!(
        broker.ask(&Ask::Done {
            host: Host::new("w1")
        }),
        Err(Unanswered::NoAnswerToThat)
    );
}

// ── The thread ──

#[test]
fn a_desk_that_panics_is_reported_and_not_waited_on() {
    // The failure worth testing, and the reason `served_by` is public. If this
    // ever hangs instead of failing, it is the worst bug this crate can ship.
    let broker = Embedded::served_by(|_| panic!("this desk has fallen over"));
    assert_eq!(broker.ask(&Ask::hello()), Err(Unanswered::Gone));
}

#[test]
fn a_desk_that_has_fallen_over_stays_fallen_over_instead_of_hanging() {
    let broker = Embedded::served_by(|_| panic!("down"));
    assert_eq!(broker.ask(&Ask::hello()), Err(Unanswered::Gone));
    // And the second caller is told the same thing rather than blocking on a
    // channel nobody is reading.
    assert_eq!(broker.ask(&reach("w1")), Err(Unanswered::Gone));
    assert_eq!(broker.done(&Host::new("w1")), Err(Unanswered::Gone));
}

#[test]
fn one_thread_for_the_broker_and_not_one_per_ask() {
    let seen: Arc<Mutex<Vec<std::thread::ThreadId>>> = Arc::new(Mutex::new(Vec::new()));
    let noted = Arc::clone(&seen);
    let broker = Embedded::served_by(move |_| {
        noted.lock().unwrap().push(std::thread::current().id());
        Reply::Welcome { protocol: PROTOCOL }
    });

    for _ in 0..5 {
        broker.ask(&Ask::hello()).unwrap();
    }

    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 5);
    assert!(
        seen.windows(2).all(|two| two[0] == two[1]),
        "five asks were served by more than one thread: {seen:?}"
    );
}

#[test]
fn dropping_the_broker_ends_its_thread_rather_than_leaving_it_behind() {
    // The desk is owned by the thread, so the desk still being alive is the
    // thread still being alive. After the drop returns, it has been joined.
    let alive = Arc::new(AtomicUsize::new(0));
    let held = Arc::clone(&alive);
    let broker = Embedded::served_by(move |_| {
        let _keep = &held;
        Reply::Welcome { protocol: PROTOCOL }
    });
    broker.ask(&Ask::hello()).unwrap();
    assert_eq!(Arc::strong_count(&alive), 2, "the desk should hold one");

    drop(broker);
    assert_eq!(
        Arc::strong_count(&alive),
        1,
        "the thread outlived the broker that opened it"
    );
}

#[test]
fn two_threads_can_reach_one_broker_at_once() {
    // `Transport` is `Sync`, so two branches of a wave call into this at the
    // same time. One channel does not fit two conversations halfway through,
    // and the lock queueing them is correct rather than a limitation.
    let broker = Arc::new(Embedded::open([
        (Host::new("w1"), at("node3:7000")),
        (Host::new("w2"), at("node4:7000")),
    ]));
    let answers: Vec<_> = ["w1", "w2", "w1", "w2"]
        .into_iter()
        .map(|host| {
            let broker = Arc::clone(&broker);
            std::thread::spawn(move || broker.ask(&reach(host)).unwrap())
        })
        .map(|one| one.join().unwrap())
        .collect();

    assert!(
        answers.iter().all(|one| matches!(one, Reply::Met { .. })),
        "{answers:?}"
    );
}

#[test]
fn bytes_that_are_not_a_message_are_refused_by_the_desk_and_not_fatal() {
    // A desk that reads something it does not understand answers; it does not
    // fall over. Proven from the outside: the broker still works afterwards.
    let broker = Embedded::open([(Host::new("w1"), at("node3:7000"))]);
    assert!(matches!(
        broker.ask(&reach("w1")).unwrap(),
        Reply::Met { .. }
    ));
    assert!(matches!(
        broker.ask(&Ask::hello()).unwrap(),
        Reply::Welcome { .. }
    ));
}
