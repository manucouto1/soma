//! A store that is a directory.
//!
//! Everything here happens in a real one: what is being tested is precisely
//! that two processes could share it, and that cannot be tested against a map
//! in memory.

use crate::tempdir;
use somatize_store::{Digest, Local, Meta, Store, StoreError};

/// A store of its own, in a directory nobody else is using.
fn store() -> (Local, tempdir::Dir) {
    let where_ = tempdir::Dir::new();
    let store = Local::at(where_.path()).unwrap();
    (store, where_)
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

// ── Claiming, which is how work gets handed out ──

#[test]
fn a_name_nobody_has_can_be_claimed() {
    let (store, _dir) = store();
    let mine = store.put(b"me").unwrap();

    assert!(store.claim("round/0/client/2", &mine, Meta::new()).unwrap());
    assert_eq!(
        store.resolve("round/0/client/2").unwrap().unwrap().digest,
        mine
    );
}

#[test]
fn and_a_name_somebody_has_cannot() {
    let (store, _dir) = store();
    let first = store.put(b"me").unwrap();
    let second = store.put(b"somebody else").unwrap();

    assert!(store.claim("the/work", &first, Meta::new()).unwrap());

    assert!(!store.claim("the/work", &second, Meta::new()).unwrap());
    assert_eq!(
        store.resolve("the/work").unwrap().unwrap().digest,
        first,
        "the second one overwrote the first, which is what `bind` does and this must not"
    );
}

#[test]
fn what_bind_replaces_claim_refuses() {
    // The two are next to each other on purpose and they are not the same
    // question: a name whose answer can be refreshed, and a name that is a piece
    // of work somebody took.
    let (store, _dir) = store();
    let first = store.put(b"one").unwrap();
    let second = store.put(b"other").unwrap();

    store.bind("latest", &first, Meta::new()).unwrap();
    store.bind("latest", &second, Meta::new()).unwrap();
    assert_eq!(store.resolve("latest").unwrap().unwrap().digest, second);

    store.claim("taken", &first, Meta::new()).unwrap();
    store.claim("taken", &second, Meta::new()).unwrap();
    assert_eq!(store.resolve("taken").unwrap().unwrap().digest, first);
}

#[test]
fn a_claim_carries_what_was_said_beside_it_like_any_other_record() {
    let (store, _dir) = store();
    let mine = store.put(b"me").unwrap();

    store
        .claim("work", &mine, vec![("who".into(), "node3".into())])
        .unwrap();

    let found = store.resolve("work").unwrap().unwrap();
    assert_eq!(found.meta, vec![("who".to_string(), "node3".to_string())]);
    assert!(found.when > 0);
}

#[test]
fn eight_at_once_and_exactly_one_of_them_wins() {
    // The whole point, and the only way to check it is to really race: eight
    // threads on one name, and seven of them have to be told no. `resolve` and
    // then `bind` passes every test above and loses this one.
    let (store, dir) = store();
    let racers = 8;
    let start = std::sync::Barrier::new(racers);

    let won: usize = std::thread::scope(|threads| {
        let handles: Vec<_> = (0..racers)
            .map(|which| {
                let (store, start) = (&store, &start);
                threads.spawn(move || {
                    let mine = store.put(format!("racer {which}").as_bytes()).unwrap();
                    start.wait();
                    store
                        .claim("one/piece/of/work", &mine, Meta::new())
                        .unwrap()
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .filter(|won| *won)
            .count()
    });

    assert_eq!(
        won, 1,
        "eight racers and {won} of them were told they had it"
    );
    let _ = dir;
}

#[test]
fn and_whoever_was_told_they_won_is_the_one_written_down() {
    // Not enough that one wins: the record has to be **that** one's, or the
    // winner does the work and somebody else's name is on it.
    let (store, _dir) = store();
    let racers = 8;
    let start = std::sync::Barrier::new(racers);

    let winner: Vec<usize> = std::thread::scope(|threads| {
        let handles: Vec<_> = (0..racers)
            .map(|which| {
                let (store, start) = (&store, &start);
                threads.spawn(move || {
                    let mine = store.put(format!("racer {which}").as_bytes()).unwrap();
                    start.wait();
                    match store.claim("work", &mine, Meta::new()).unwrap() {
                        true => Some(which),
                        false => None,
                    }
                })
            })
            .collect();
        handles
            .into_iter()
            .filter_map(|h| h.join().unwrap())
            .collect()
    });

    let [which] = winner[..] else {
        panic!("expected exactly one winner, got {winner:?}")
    };
    let digest = store.resolve("work").unwrap().unwrap().digest;
    assert_eq!(
        store.get(&digest).unwrap().unwrap(),
        format!("racer {which}").as_bytes()
    );
}

#[test]
fn a_claim_leaves_nothing_behind_in_the_temporaries() {
    // It writes the record somewhere else and links it into place, and the
    // somewhere else has to go: a store that grows a file per claim is one that
    // fills a network folder over a weekend.
    let (store, dir) = store();
    let mine = store.put(b"me").unwrap();

    for round in 0..20 {
        store
            .claim(&format!("round/{round}"), &mine, Meta::new())
            .unwrap();
        store
            .claim(&format!("round/{round}"), &mine, Meta::new())
            .unwrap();
    }

    let left: Vec<_> = std::fs::read_dir(dir.path().join("tmp"))
        .unwrap()
        .map(|each| each.unwrap().file_name())
        .collect();
    assert!(left.is_empty(), "left behind: {left:?}");
}
