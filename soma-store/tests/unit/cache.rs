//! The engine's keeper, against a store that is a real directory.
//!
//! What is checked here is what the core cannot check on its own, because the
//! core cannot hash: that two recipes never share a name, that what goes in
//! comes out the same, and that the failures are the ones with a name.

use crate::tempdir;
use somatize_core::{Keeper, Key, Value};
use somatize_store::{Cache, Local, Store};

/// A cache over a store nobody else is using.
fn cache() -> (Local, tempdir::Dir) {
    let where_ = tempdir::Dir::new();
    let store = Local::at(where_.path()).unwrap();
    (store, where_)
}

fn value(x: f64) -> Value {
    Value::number(x)
}

// ── Naming ──

#[test]
fn the_pieces_of_a_recipe_cannot_run_into_each_other() {
    // The one failure a cache must not have: two recipes under one name. Run
    // together, both of these are the string `abc`.
    let (store, _where) = cache();
    let cache = Cache::over(&store);

    assert_ne!(cache.combine(&["ab", "c"]), cache.combine(&["a", "bc"]));
}

#[test]
fn the_same_recipe_is_the_same_name_every_time() {
    // And it has to hold across processes, which is why it is a hash of the
    // recipe and not a counter.
    let (store, _where) = cache();
    let cache = Cache::over(&store);

    assert_eq!(
        cache.combine(&["Encoder", "sha256:weights", ""]),
        cache.combine(&["Encoder", "sha256:weights", ""])
    );
}

#[test]
fn only_a_root_is_named_by_its_content() {
    let (store, _where) = cache();
    let cache = Cache::over(&store);

    assert_eq!(cache.key_of(&value(1.0)), cache.key_of(&value(1.0)));
    assert_ne!(cache.key_of(&value(1.0)), cache.key_of(&value(2.0)));
    assert!(cache.key_of(&value(1.0)).is_some());
}

#[test]
fn what_only_exists_in_this_process_has_no_name() {
    // Not a failure: it means nothing under it can be named either, and the run
    // goes on without a cache.
    let (store, _where) = cache();
    let cache = Cache::over(&store);

    assert_eq!(cache.key_of(&Value::opaque(7u32)), None);
    assert_eq!(
        cache.key_of(&Value::map(vec![("x".to_string(), Value::opaque(7u32))])),
        None
    );
}

// ── Keeping and recalling ──

#[test]
fn what_is_kept_comes_back_the_same() {
    let (store, _where) = cache();
    let cache = Cache::over(&store);
    let key = cache.combine(&["Encoder", "", ""]);

    cache
        .keep(&key, &Value::text("an embedding"), &[("node", "encoder")])
        .unwrap();

    let kept = cache.recall(&[&key]).unwrap().pop().unwrap().unwrap();
    assert_eq!(kept.value, Value::text("an embedding"));
    assert_eq!(kept.meta, [("node".to_string(), "encoder".to_string())]);
}

#[test]
fn a_map_of_numbers_survives_the_round_trip() {
    // What actually crosses: not a scalar but whatever a node returned.
    let (store, _where) = cache();
    let cache = Cache::over(&store);
    let key = cache.combine(&["Encoder", "", ""]);
    let output = Value::map(vec![
        ("left".to_string(), value(1.0)),
        ("right".to_string(), Value::text("two")),
    ]);

    cache.keep(&key, &output, &[]).unwrap();

    assert_eq!(
        cache.recall(&[&key]).unwrap()[0].as_ref().unwrap().value,
        output
    );
}

#[test]
fn a_name_nobody_kept_is_a_miss_and_not_a_failure() {
    let (store, _where) = cache();
    let cache = Cache::over(&store);

    assert_eq!(cache.recall(&[&Key::new("sha256:never")]).unwrap(), [None]);
}

#[test]
fn a_batch_answers_in_the_order_it_was_asked_holes_included() {
    // The reason `recall` is batched from the first day: against a store on the
    // far end of a network, one question per key is one round trip per key.
    let (store, _where) = cache();
    let cache = Cache::over(&store);
    let (first, third) = (
        cache.combine(&["first", "", ""]),
        cache.combine(&["third", "", ""]),
    );
    let missing = Key::new("sha256:never");

    cache.keep(&first, &value(1.0), &[]).unwrap();
    cache.keep(&third, &value(3.0), &[]).unwrap();

    let answers = cache.recall(&[&first, &missing, &third]).unwrap();
    assert_eq!(
        answers
            .iter()
            .map(|kept| kept.as_ref().map(|kept| kept.value.clone()))
            .collect::<Vec<_>>(),
        [Some(value(1.0)), None, Some(value(3.0))]
    );
}

#[test]
fn keeping_the_same_name_again_replaces_what_it_points_at() {
    // A name is the question and the answer can be refreshed, which is what
    // `.overwrite()` will be doing.
    let (store, _where) = cache();
    let cache = Cache::over(&store);
    let key = cache.combine(&["Encoder", "", ""]);

    cache.keep(&key, &value(1.0), &[]).unwrap();
    cache.keep(&key, &value(2.0), &[]).unwrap();

    assert_eq!(
        cache.recall(&[&key]).unwrap()[0].as_ref().unwrap().value,
        value(2.0)
    );
}

#[test]
fn what_cannot_leave_this_process_cannot_be_kept_and_says_so() {
    // The frontier stays visible, and it moves in the next slice from "the
    // variant" to "the variant with nobody to turn it into bytes".
    let (store, _where) = cache();
    let cache = Cache::over(&store);
    let key = cache.combine(&["Encoder", "", ""]);

    let why = cache
        .keep(&key, &Value::opaque(7u32), &[])
        .unwrap_err()
        .to_string();

    assert!(why.contains("opaque"), "{why}");
    assert!(why.contains("process"), "{why}");
}

// ── Living in the same store as everything else ──

#[test]
fn a_kept_value_can_be_found_by_looking_at_what_is_there() {
    // A store you cannot explore is one nobody trusts. The record is JSON and
    // says what it is, who produced it and when.
    let (store, _where) = cache();
    let cache = Cache::over(&store);
    let key = cache.combine(&["Encoder", "", ""]);

    cache
        .keep(
            &key,
            &value(1.0),
            &[("node", "encoder"), ("fingerprint", "v1")],
        )
        .unwrap();

    let bound = store.bound().unwrap();
    assert_eq!(bound.len(), 1);
    assert!(bound[0].name.starts_with("value:"), "{}", bound[0].name);
    assert!(bound[0].name.contains(key.as_str()));
    assert_eq!(
        bound[0].meta,
        [
            ("node".to_string(), "encoder".to_string()),
            ("fingerprint".to_string(), "v1".to_string())
        ]
    );
}

#[test]
fn a_value_and_an_artifact_do_not_collide_in_the_same_directory() {
    // Both live in one store on purpose. What keeps them apart is the namespace
    // of the name, not two directories.
    let (store, _where) = cache();
    let cache = Cache::over(&store);
    let key = cache.combine(&["Encoder", "", ""]);

    cache.keep(&key, &value(1.0), &[]).unwrap();
    let digest = store.put(b"a pickled catalog").unwrap();
    store
        .bind(&format!("artifact:pickle:{key}"), &digest, Vec::new())
        .unwrap();

    assert_eq!(
        cache.recall(&[&key]).unwrap()[0].as_ref().unwrap().value,
        value(1.0)
    );
    assert_eq!(store.bound().unwrap().len(), 2);
}

#[test]
fn the_same_bytes_under_two_names_are_stored_once() {
    // The blobs are content-addressed underneath: two nodes that produce the
    // same thing cost one copy of it.
    let (store, _where) = cache();
    let cache = Cache::over(&store);

    for who in ["left", "right"] {
        cache
            .keep(&cache.combine(&[who, "", ""]), &value(1.0), &[])
            .unwrap();
    }

    let bound = store.bound().unwrap();
    assert_eq!(bound.len(), 2, "two names");
    assert_eq!(bound[0].digest, bound[1].digest, "and one blob");
}
