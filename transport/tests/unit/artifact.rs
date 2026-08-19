//! What an empty worker is provisioned with.
//!
//! There is little to check because this crate does not interpret the bytes.
//! What does get pinned down is the split between the cheap half — the name —
//! and the heavy one, which is what makes the `have`/`want` worth having.

use soma_next_transport::{Artifact, Label};

#[test]
fn an_artifact_is_a_kind_an_identity_and_some_bytes() {
    let a = Artifact::new("pickle", "sha256:abc", vec![1, 2, 3]);

    assert_eq!(a.kind, "pickle");
    assert_eq!(a.id, "sha256:abc");
    assert_eq!(a.bytes, vec![1, 2, 3]);
}

#[test]
fn the_label_is_the_name_without_the_weight() {
    // The cheap half of the greeting: asking "do you have it?" must not put a
    // single byte of the catalog on the wire.
    let a = Artifact::new("pickle", "sha256:abc", vec![0xff; 10_000]);

    assert_eq!(
        a.label(),
        Label {
            kind: "pickle".into(),
            id: "sha256:abc".into()
        }
    );
}

#[test]
fn two_artifacts_with_the_same_name_and_different_bytes_share_a_label() {
    // The consequence of the id being set by whoever produces the artifact:
    // here we only compare strings, and saying two are the same one is their
    // call, not ours.
    let a = Artifact::new("pickle", "v1", vec![1]);
    let b = Artifact::new("pickle", "v1", vec![2]);

    assert_eq!(a.label(), b.label());
    assert_ne!(a, b);
}
