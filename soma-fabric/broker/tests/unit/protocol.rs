//! What the two sides say, put into bytes and taken out again.
//!
//! No broker in the way: if the round trip does not give back the same thing,
//! you do not need one to find out. What needs testing here and nowhere else is
//! the **greeting**, because it is the only message that has to survive meeting
//! a binary that disagrees with this one.

use serde::Serialize;
use somatize_fabric_broker::{Ask, Endpoint, Host, Identity, Needs, PROTOCOL, Path, Reply, SlotId};
use std::time::Duration;

fn round_trip(ask: Ask) {
    let bytes = ask.to_bytes().unwrap();
    assert_eq!(Ask::from_bytes(&bytes).unwrap(), ask);
}

fn round_trip_reply(reply: Reply) {
    let bytes = reply.to_bytes().unwrap();
    assert_eq!(Reply::from_bytes(&bytes).unwrap(), reply);
}

// ── The six messages ──

#[test]
fn every_thing_the_client_says_goes_and_comes_back_equal() {
    for ask in [
        Ask::hello(),
        Ask::hello_as(Identity("a token, opaque here".into())),
        Ask::Reach {
            host: Host::new("gpu-a"),
            needs: Needs::default(),
        },
        Ask::Done {
            host: Host::new("gpu-a"),
        },
    ] {
        round_trip(ask);
    }
}

#[test]
fn every_thing_the_broker_says_goes_and_comes_back_equal() {
    for reply in [
        Reply::Welcome { protocol: PROTOCOL },
        Reply::Refused("no".into()),
        Reply::Met {
            path: Path::InProcess { slot: SlotId(7) },
            good_for: None,
        },
        Reply::Met {
            path: Path::Direct {
                endpoint: Endpoint::Address("node3:7000".into()),
            },
            good_for: Some(Duration::from_secs(300)),
        },
        Reply::Unreachable("nobody registered `w1`".into()),
    ] {
        round_trip_reply(reply);
    }
}

#[test]
fn a_host_keeps_its_name_across_the_wire() {
    let bytes = Ask::Reach {
        host: Host::new("worker-ñ-2"),
        needs: Needs::default(),
    }
    .to_bytes()
    .unwrap();
    match Ask::from_bytes(&bytes).unwrap() {
        Ask::Reach { host, .. } => assert_eq!(host.as_str(), "worker-ñ-2"),
        other => panic!("that was a Reach, not {other:?}"),
    }
}

// ── The greeting, which is the only reason any of this is versioned ──

/// What another version's client would put on the wire: the same first message
/// with a number this binary has never heard of.
#[derive(Serialize)]
enum StrangersAsk {
    #[allow(dead_code)]
    Hello { protocol: u16, who: Option<String> },
}

/// And what a version that grew the greeting would put there.
#[derive(Serialize)]
enum GrownAsk {
    #[allow(dead_code)]
    Hello {
        protocol: u16,
        who: Option<String>,
        and_one_more: bool,
    },
}

#[test]
fn a_greeting_from_a_version_we_do_not_speak_is_still_readable() {
    let theirs = rmp_serde::to_vec(&StrangersAsk::Hello {
        protocol: 999,
        who: None,
    })
    .unwrap();

    // The whole point of the version being the first field of the first
    // message: we can read enough of a stranger's greeting to say no properly.
    match Ask::from_bytes(&theirs).unwrap() {
        Ask::Hello { protocol, .. } => assert_eq!(protocol, 999),
        other => panic!("that was a Hello, not {other:?}"),
    }
}

#[test]
fn a_version_we_do_not_speak_is_refused_naming_both_numbers() {
    match Reply::to_greeting(999) {
        Reply::Refused(why) => {
            assert!(why.contains("999"), "the client's number is missing: {why}");
            assert!(
                why.contains(&PROTOCOL.to_string()),
                "our own number is missing: {why}"
            );
        }
        other => panic!("999 is not a version we speak, so that should not be {other:?}"),
    }
}

#[test]
fn our_own_version_is_welcomed_with_our_own_number() {
    assert_eq!(
        Reply::to_greeting(PROTOCOL),
        Reply::Welcome { protocol: PROTOCOL }
    );
}

#[test]
fn a_greeting_that_grew_a_field_cannot_be_read_which_is_why_it_must_not_grow() {
    let theirs = rmp_serde::to_vec(&GrownAsk::Hello {
        protocol: 2,
        who: None,
        and_one_more: true,
    })
    .unwrap();

    // Positional encoding means a greeting with one more field in it is not a
    // greeting any more, and the version cannot be got out of it to explain
    // why. So `Hello` is the one message that may never gain a field: a new
    // version adds messages, or changes the others, and leaves this one alone.
    assert!(
        Ask::from_bytes(&theirs).is_err(),
        "if this ever passes, version negotiation has stopped working and \
         nobody will find out until two binaries meet"
    );
}

// ── Bytes that are not messages ──

#[test]
fn leftovers_are_as_suspicious_as_missing_bytes() {
    let mut bytes = Ask::hello().to_bytes().unwrap();
    bytes.push(0);
    let why = Ask::from_bytes(&bytes).unwrap_err();
    assert!(
        why.message().contains("left over"),
        "it should say what it found: {why}"
    );
}

#[test]
fn a_truncated_message_is_refused() {
    let bytes = Ask::Reach {
        host: Host::new("gpu-a"),
        needs: Needs::default(),
    }
    .to_bytes()
    .unwrap();
    assert!(Ask::from_bytes(&bytes[..bytes.len() - 1]).is_err());
}

#[test]
fn bytes_that_were_never_a_message_are_refused_and_say_so() {
    let why = Reply::from_bytes(b"not a message at all").unwrap_err();
    assert!(
        why.to_string().contains("not the bytes that were written"),
        "{why}"
    );
}

#[test]
fn an_answer_is_not_a_question() {
    // The two enums have variants at the same indices, so this is worth pinning:
    // reading one as the other has to fail rather than quietly give variant 0.
    let welcome = Reply::Welcome { protocol: PROTOCOL }.to_bytes().unwrap();
    assert!(Ask::from_bytes(&welcome).is_err());
}
