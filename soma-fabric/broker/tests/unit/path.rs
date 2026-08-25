//! The four ways a pair of endpoints ends up talking.
//!
//! Only two of them can be answered today. All four are tested, because the
//! reason they are all in the message is that the ones nobody answers yet must
//! not change shape when somebody does.

use somatize_fabric_broker::{Endpoint, Path, Reply, SessionId, SlotId};
use std::path::PathBuf;

fn round_trip(path: Path) {
    let reply = Reply::Met {
        path: path.clone(),
        good_for: None,
    };
    let bytes = reply.to_bytes().unwrap();
    assert_eq!(Reply::from_bytes(&bytes).unwrap(), reply);
}

#[test]
fn all_four_paths_cross_including_the_one_that_transfers_nothing() {
    for path in [
        Path::InProcess { slot: SlotId(0) },
        Path::Mount {
            dir: PathBuf::from("/mnt/cluster/scratch"),
        },
        Path::Direct {
            endpoint: Endpoint::Address("node3:7000".into()),
        },
        Path::Relayed {
            session: SessionId("s-abc".into()),
        },
    ] {
        round_trip(path);
    }
}

#[test]
fn a_pipe_and_a_socket_are_the_same_path() {
    // Not two variants: what varies is how the stream is obtained, which is the
    // whole reason `Direct` carries an endpoint and not an address.
    for endpoint in [
        Endpoint::Address("node3:7000".into()),
        Endpoint::Command(vec!["python".into(), "-m".into(), "somatize.worker".into()]),
    ] {
        round_trip(Path::Direct { endpoint });
    }
}

#[test]
fn a_command_keeps_its_arguments_in_order() {
    let argv = vec!["srun".to_string(), "-n1".to_string(), "worker".to_string()];
    let bytes = Reply::Met {
        path: Path::Direct {
            endpoint: Endpoint::Command(argv.clone()),
        },
        good_for: None,
    }
    .to_bytes()
    .unwrap();
    match Reply::from_bytes(&bytes).unwrap() {
        Reply::Met {
            path: Path::Direct {
                endpoint: Endpoint::Command(back),
            },
            ..
        } => assert_eq!(back, argv),
        other => panic!("that was a command, not {other:?}"),
    }
}

#[test]
fn how_long_it_is_good_for_crosses_as_a_duration() {
    // A duration and not an instant, so that what is read here does not depend
    // on two machines agreeing about what time it is.
    let reply = Reply::Met {
        path: Path::InProcess { slot: SlotId(3) },
        good_for: Some(std::time::Duration::from_millis(1500)),
    };
    let bytes = reply.to_bytes().unwrap();
    assert_eq!(Reply::from_bytes(&bytes).unwrap(), reply);
}

#[test]
fn a_path_says_itself_the_way_a_reader_would() {
    // These end up in the record and in error messages both.
    assert_eq!(
        Path::Direct {
            endpoint: Endpoint::Address("node3:7000".into())
        }
        .to_string(),
        "straight to node3:7000"
    );
    assert_eq!(
        Path::InProcess { slot: SlotId(4) }.to_string(),
        "in this process, slot 4"
    );
    assert_eq!(
        Path::Direct {
            endpoint: Endpoint::Command(vec![
                "python".into(),
                "-m".into(),
                "somatize.worker".into()
            ])
        }
        .to_string(),
        "straight to python -m somatize.worker"
    );
}
