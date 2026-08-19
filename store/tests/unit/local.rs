//! A store that is a directory.
//!
//! Everything here happens in a real one: what is being tested is precisely
//! that two processes could share it, and that cannot be tested against a map
//! in memory.

use soma_next_store::{Digest, Local, Store, StoreError};

/// A store of its own, in a directory nobody else is using.
fn store() -> (Local, tempdir::Dir) {
    let where_ = tempdir::Dir::new();
    let store = Local::at(where_.path()).unwrap();
    (store, where_)
}

/// A temporary directory, without a dependency for it.
mod tempdir {
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};

    static COUNT: AtomicUsize = AtomicUsize::new(0);

    pub struct Dir(PathBuf);

    impl Dir {
        pub fn new() -> Self {
            let at = std::env::temp_dir().join(format!(
                "soma-store-{}-{}",
                std::process::id(),
                COUNT.fetch_add(1, Ordering::SeqCst)
            ));
            std::fs::create_dir_all(&at).unwrap();
            Self(at)
        }

        pub fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

#[test]
fn what_goes_in_comes_out() {
    let (store, _where) = store();
    let digest = store.put(b"some weights").unwrap();

    assert_eq!(store.get(&digest).unwrap(), Some(b"some weights".to_vec()));
}

#[test]
fn what_was_never_put_is_not_there_and_is_not_an_error() {
    let (store, _where) = store();

    assert_eq!(store.get(&Digest::of(b"never stored")).unwrap(), None);
    assert_eq!(store.resolve("never bound").unwrap(), None);
}

#[test]
fn the_same_bytes_stored_twice_are_stored_once() {
    // What content addressing is for. Observable from outside: the same digest,
    // and putting it again does not fail.
    let (store, _where) = store();

    let once = store.put(b"the same").unwrap();
    let twice = store.put(b"the same").unwrap();

    assert_eq!(once, twice);
    assert_eq!(store.get(&once).unwrap(), Some(b"the same".to_vec()));
}

#[test]
fn a_name_points_at_bytes_and_carries_what_was_said_about_it() {
    let (store, _where) = store();
    let digest = store.put(b"the embeddings").unwrap();

    store
        .bind(
            "recipe:abc",
            &digest,
            vec![("code".into(), "Embed(a1b2c3d4)".into())],
        )
        .unwrap();

    let bound = store.resolve("recipe:abc").unwrap().unwrap();
    assert_eq!(bound.digest, digest);
    assert_eq!(bound.name, "recipe:abc");
    assert_eq!(
        bound.meta,
        vec![("code".to_string(), "Embed(a1b2c3d4)".to_string())]
    );
    assert!(
        bound.when > 0,
        "a store you cannot sort by time is one you cannot explore"
    );
}

#[test]
fn a_name_is_a_question_and_its_answer_can_be_refreshed() {
    // What `.overwrite()` will do: the recipe is the same, what it produced is
    // not, and the name has to end up pointing at the new one.
    let (store, _where) = store();
    let old = store.put(b"before").unwrap();
    let new = store.put(b"after").unwrap();

    store.bind("recipe:abc", &old, Vec::new()).unwrap();
    store.bind("recipe:abc", &new, Vec::new()).unwrap();

    assert_eq!(store.resolve("recipe:abc").unwrap().unwrap().digest, new);
    assert_eq!(
        store.get(&old).unwrap(),
        Some(b"before".to_vec()),
        "the old bytes are still there: a blob is written once and never changes"
    );
}

#[test]
fn any_name_can_be_bound_however_it_is_written() {
    // A cache key is hex, an artifact's id is whatever its producer chose, and
    // no filesystem takes every string a caller can invent.
    let (store, _where) = store();
    let digest = store.put(b"x").unwrap();

    for name in [
        "sha256:abc",
        "a/b/c",
        "with spaces and ñ",
        &"long".repeat(200),
        "..",
    ] {
        store.bind(name, &digest, Vec::new()).unwrap();
        assert_eq!(
            store.resolve(name).unwrap().unwrap().name,
            name,
            "`{name}` did not come back"
        );
    }
}

#[test]
fn asking_for_many_answers_in_the_order_they_were_asked() {
    // The shape an item-by-item cache needs: it asks thousands at once and reads
    // the answers by position.
    let (store, _where) = store();
    let digest = store.put(b"x").unwrap();
    store.bind("second", &digest, Vec::new()).unwrap();

    let answers = store.resolve_many(&["first", "second", "third"]).unwrap();

    assert_eq!(answers.len(), 3);
    assert!(answers[0].is_none());
    assert_eq!(answers[1].as_ref().unwrap().name, "second");
    assert!(answers[2].is_none());
}

#[test]
fn the_bytes_can_be_asked_for_many_at_a_time_too() {
    let (store, _where) = store();
    let there = store.put(b"here").unwrap();
    let missing = Digest::of(b"not here");

    assert_eq!(
        store.get_many(&[&there, &missing]).unwrap(),
        vec![Some(b"here".to_vec()), None]
    );
}

#[test]
fn what_is_bound_can_be_looked_at() {
    // The requirement that started all this: knowing what you have stored.
    let (store, _where) = store();
    let digest = store.put(b"x").unwrap();
    for name in ["b", "a", "c"] {
        store
            .bind(name, &digest, vec![("kind".into(), "weights".into())])
            .unwrap();
    }

    let bound = store.bound().unwrap();

    assert_eq!(bound.len(), 3);
    let names: Vec<&str> = bound.iter().map(|b| b.name.as_str()).collect();
    assert_eq!(
        names,
        ["a", "b", "c"],
        "two runs of this have to see the same"
    );
    assert!(bound.iter().all(|b| b.meta[0].1 == "weights"));
}

#[test]
fn an_empty_store_has_nothing_bound_and_does_not_complain() {
    let (store, _where) = store();

    assert_eq!(store.bound().unwrap(), Vec::new());
}

#[test]
fn two_stores_on_the_same_directory_see_each_other() {
    // The whole point of the format: a shared folder, no server, no lock. Two
    // handles here stand for two processes.
    let where_ = tempdir::Dir::new();
    let mine = Local::at(where_.path()).unwrap();
    let theirs = Local::at(where_.path()).unwrap();

    let digest = mine.put(b"what I computed").unwrap();
    mine.bind("recipe:abc", &digest, Vec::new()).unwrap();

    let found = theirs.resolve("recipe:abc").unwrap().unwrap();
    assert_eq!(
        theirs.get(&found.digest).unwrap(),
        Some(b"what I computed".to_vec())
    );
}

#[test]
fn the_records_are_readable_with_cat() {
    // A requirement before it was a format: what is stored has to be
    // inspectable without this library.
    let where_ = tempdir::Dir::new();
    let store = Local::at(where_.path()).unwrap();
    let digest = store.put(b"x").unwrap();
    store
        .bind("recipe:abc", &digest, vec![("run".into(), "17".into())])
        .unwrap();

    let written = std::fs::read_dir(where_.path().join("names"))
        .unwrap()
        .flat_map(|head| std::fs::read_dir(head.unwrap().path()).unwrap())
        .map(|record| std::fs::read_to_string(record.unwrap().path()).unwrap())
        .collect::<String>();

    assert!(written.contains("recipe:abc"), "{written}");
    assert!(written.contains("\"run\""), "{written}");
    assert!(written.contains(digest.as_str()), "{written}");
}

#[test]
fn a_record_that_is_not_one_is_reported_as_such() {
    let where_ = tempdir::Dir::new();
    let store = Local::at(where_.path()).unwrap();
    let digest = store.put(b"x").unwrap();
    store.bind("recipe:abc", &digest, Vec::new()).unwrap();

    // Somebody wrote something else where a record was.
    let head = std::fs::read_dir(where_.path().join("names"))
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let record = std::fs::read_dir(&head)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    std::fs::write(&record, b"not a record").unwrap();

    assert!(matches!(
        store.resolve("recipe:abc").unwrap_err(),
        StoreError::Corrupt(_)
    ));
}

#[test]
fn the_blobs_are_spread_out_and_not_piled_in_one_directory() {
    // The digests all start with `sha256:`, so splitting the whole string would
    // put every blob in the same `sh/` and the split would buy nothing. What
    // names the directory is the hash.
    let (store, where_) = store();
    for n in 0..32u8 {
        store.put(&[n]).unwrap();
    }

    let directories: Vec<_> = std::fs::read_dir(where_.path().join("blobs"))
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect();

    assert!(
        directories.len() > 1,
        "32 blobs ended up in {} directory: {directories:?}",
        directories.len()
    );
}

#[test]
fn a_blobs_file_still_says_which_algorithm_it_is() {
    // Only the directory drops the `sha256:`; the file keeps the whole digest,
    // so the same hex under two algorithms could never be one file.
    let (store, where_) = store();
    let digest = store.put(b"hello").unwrap();

    let found = std::fs::read_dir(where_.path().join("blobs"))
        .unwrap()
        .flat_map(|head| std::fs::read_dir(head.unwrap().path()).unwrap())
        .map(|file| file.unwrap().file_name().to_string_lossy().into_owned())
        .next()
        .expect("something was just put in there");

    assert!(found.starts_with("sha256_"), "{found}");
    assert!(
        digest.as_str().ends_with(&found["sha256_".len()..]),
        "{found}"
    );
}

#[test]
fn many_threads_writing_at_once_do_not_tread_on_each_other() {
    // Every write lands somewhere else and is renamed into place, and two of
    // them picking the same landing spot would give one the other's bytes. It
    // exercises the property rather than forcing it: what makes it hold is that
    // the spot comes from a counter, and a counter cannot be read twice.
    let (store, _where) = store();
    let payloads: Vec<Vec<u8>> = (0..32u8).map(|n| vec![n; 4096]).collect();

    let digests: Vec<Digest> = std::thread::scope(|scope| {
        let running: Vec<_> = payloads
            .iter()
            .map(|bytes| scope.spawn(|| store.put(bytes).unwrap()))
            .collect();
        running
            .into_iter()
            .map(|thread| thread.join().expect("no writer panicked"))
            .collect()
    });

    for (digest, bytes) in digests.iter().zip(&payloads) {
        assert_eq!(store.get(digest).unwrap().as_ref(), Some(bytes));
    }
}
