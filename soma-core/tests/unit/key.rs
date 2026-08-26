//! What a node's output is called. Thin on purpose: the core carries a key, it
//! does not compute one, and it stays exactly the text it was given.

use somatize_core::Key;
use std::collections::HashMap;

#[test]
fn a_key_is_the_text_it_was_given() {
    let key = Key::new("sha256:abc");
    assert_eq!(key.as_str(), "sha256:abc");
    assert_eq!(key.to_string(), "sha256:abc");
}

#[test]
fn two_recipes_that_hash_the_same_are_the_same_key() {
    assert_eq!(Key::new("sha256:abc"), Key::new("sha256:abc"));
    assert_ne!(Key::new("sha256:abc"), Key::new("sha256:abd"));
}

#[test]
fn it_can_be_looked_things_up_by() {
    // The engine keeps a table of them beside what each node produced, so it has
    // to be a map key and not only comparable.
    let mut kept = HashMap::new();
    kept.insert(Key::new("sha256:abc"), 1);
    assert_eq!(kept.get(&Key::new("sha256:abc")), Some(&1));
    assert_eq!(kept.get(&Key::new("sha256:abd")), None);
}
