//! The name of the process where a node runs. There is little to check and that
//! is the point: a host is a name, not an address.

use somatize_core::{Device, Host};
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
    // The asymmetry, written down so it shows if anyone deletes it: `Device` is
    // a closed set, so a typo fails at declaration time. Hosts are the user's.
    assert!(Device::from_str("cude:0").is_err());
    assert_eq!(Host::new("afternoon-worker").as_str(), "afternoon-worker");
}
