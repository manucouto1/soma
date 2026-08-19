//! The name of the process where a node runs.
//!
//! There is little to check and that is on purpose: a host is a name, not an
//! address, so there is nothing to validate here. What does get pinned down is
//! exactly that — that it does not validate — because it is what separates it
//! from `Device`, and the temptation to give it a grammar will come along by
//! itself.

use soma_next_core::{Device, Host};
use std::str::FromStr;

#[test]
fn a_host_is_its_name() {
    let host = Host::new("worker1");
    assert_eq!(host.as_str(), "worker1");
    assert_eq!(host.to_string(), "worker1");
}

#[test]
fn it_is_written_however_it_comes() {
    assert_eq!(Host::from("worker1"), Host::new("worker1"));
    assert_eq!(Host::from("worker1".to_string()), Host::new("worker1"));
}

#[test]
fn two_different_names_are_two_hosts() {
    assert_ne!(Host::new("worker1"), Host::new("worker2"));
}

#[test]
fn a_host_has_no_grammar_and_a_device_does() {
    // The asymmetry, written down so it shows if anyone deletes it. `Device` is
    // a closed set we decide, so a typo fails **at declaration time**. Hosts are
    // named by the user: there is no list to close, and `afternoon-worker` is no
    // less valid than `worker1`.
    assert!(Device::from_str("cude:0").is_err());
    assert_eq!(Host::new("afternoon-worker").as_str(), "afternoon-worker");
}
