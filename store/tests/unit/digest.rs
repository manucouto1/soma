//! What identifies some bytes.

use soma_next_store::Digest;

#[test]
fn the_same_bytes_give_the_same_digest() {
    assert_eq!(Digest::of(b"hello"), Digest::of(b"hello"));
    assert_ne!(Digest::of(b"hello"), Digest::of(b"hellO"));
}

#[test]
fn it_says_which_algorithm_it_is() {
    // The prefix is what allows another one the day it is needed without every
    // stored name becoming ambiguous.
    let digest = Digest::of(b"hello").to_string();

    assert!(digest.starts_with("sha256:"), "{digest}");
    assert_eq!(digest.len(), "sha256:".len() + 64);
}

#[test]
fn it_is_the_sha256_everyone_else_computes() {
    // Not our own hash: `sha256sum`, `hashlib`, and the `sha256:` ids the Python
    // client already writes have to agree with this.
    assert_eq!(
        Digest::of(b"abc").to_string(),
        "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

#[test]
fn one_someone_else_computed_is_read_as_it_is() {
    assert_eq!(Digest::parse("sha256:abc").as_str(), "sha256:abc");
}
