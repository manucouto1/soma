//! What a `Store` promises, checked on every implementor there is.
//!
//! This file is the reason a second implementation is worth having: every
//! assertion is written against `&dyn Store` and run against each — a directory
//! always, a bucket when there is one to talk to. What a trait's contract
//! actually **is** only becomes visible when something other than the first
//! implementor has to keep it; until then "the contract" and "whatever `Local`
//! does" are the same sentence and nobody can tell them apart.
//!
//! It has no counterpart in `src/`, the same way `study`'s `invariants` has
//! none: what it covers is the trait, not a file.
//!
//! The bucket half is opt-in, because it needs something to talk to:
//!
//! ```text
//! docker compose -f store/tests/docker/compose.yaml up -d
//! SOMA_S3=http://127.0.0.1:9000 cargo test -p somatize-store --features s3
//! ```
//!
//! Without the feature, or without `SOMA_S3`, that half says so on `stderr` and
//! the directory half runs exactly as before. **Not gated as a whole**: a
//! contract that only runs when an optional feature is on is a contract nobody
//! checks.

use crate::tempdir::Dir;
use somatize_store::{Digest, Local, Store, StoreError};
use std::env;

/// The bucket under test, or `None` when there is none to reach.
#[cfg(feature = "s3")]
fn bucket() -> Option<somatize_store::Bucket> {
    use somatize_store::{Bucket, Credentials, UrlStyle};

    let endpoint = env::var("SOMA_S3").ok()?;
    let key = env::var("SOMA_S3_KEY").unwrap_or_else(|_| "somanext".into());
    let secret = env::var("SOMA_S3_SECRET").unwrap_or_else(|_| "somanextsecret".into());
    let name = env::var("SOMA_S3_BUCKET").unwrap_or_else(|_| "soma".into());
    let region = env::var("SOMA_S3_REGION").unwrap_or_else(|_| "us-east-1".into());
    match Bucket::at(
        &endpoint,
        &name,
        &region,
        UrlStyle::Path,
        Credentials::new(key, secret),
    ) {
        Ok(bucket) => Some(bucket),
        // Not a skip: `SOMA_S3` being set is somebody saying there is a bucket.
        Err(e) => panic!("`SOMA_S3` is set and the bucket would not open: {e}"),
    }
}

/// Runs one assertion against every store there is.
fn on_every_store(called: &str, contract: impl Fn(&dyn Store, &str)) {
    let scratch = Dir::new();
    let directory = Local::at(scratch.path()).expect("a directory store");
    contract(&directory, called);

    #[cfg(feature = "s3")]
    match bucket() {
        Some(bucket) => contract(&bucket, called),
        None => eprintln!("no `SOMA_S3`: `{called}` only ran against a directory"),
    }
    #[cfg(not(feature = "s3"))]
    let _ = env::var("SOMA_S3");
}

/// Names of this run's own, so two runs over one bucket do not collide — a
/// bucket, unlike a scratch directory, is not thrown away between them.
fn mine(called: &str, what: &str) -> String {
    format!("test/{}/{called}/{what}", std::process::id())
}

#[test]
fn the_same_bytes_are_the_same_digest_however_often_they_are_written() {
    on_every_store("twice", |store, _| {
        let once = store.put(b"the same bytes").expect("a first write");
        let again = store.put(b"the same bytes").expect("a second write");
        assert_eq!(once, again);
        assert_eq!(
            store.get(&once).expect("a read").as_deref(),
            Some(&b"the same bytes"[..])
        );
    });
}

#[test]
fn what_was_never_put_is_not_there_and_is_not_an_error() {
    on_every_store("absent", |store, _| {
        let nobodys = Digest::of(b"nobody ever wrote this");
        assert_eq!(store.get(&nobodys).expect("a read"), None);
        assert_eq!(store.resolve("no/such/name").expect("a resolve"), None);
    });
}

#[test]
fn a_name_points_at_bytes_and_carries_what_was_said_about_it() {
    on_every_store("bind", |store, called| {
        let name = mine(called, "one");
        let digest = store.put(b"pointed at").expect("a write");
        let meta = vec![("who".to_string(), "a test".to_string())];
        store.bind(&name, &digest, meta.clone()).expect("a bind");

        let found = store.resolve(&name).expect("a resolve").expect("bound");
        assert_eq!(found.name, name);
        assert_eq!(found.digest, digest);
        assert_eq!(found.meta, meta);
        assert!(found.when > 0, "a record is stamped when it is written");
    });
}

#[test]
fn binding_the_same_name_again_replaces_it() {
    on_every_store("rebind", |store, called| {
        let name = mine(called, "moving");
        let first = store.put(b"before").expect("a write");
        let second = store.put(b"after").expect("a write");
        store.bind(&name, &first, Vec::new()).expect("a bind");
        store.bind(&name, &second, Vec::new()).expect("a rebind");

        let found = store.resolve(&name).expect("a resolve").expect("bound");
        assert_eq!(
            found.digest, second,
            "a name is a question, and the answer moves"
        );
    });
}

#[test]
fn a_claimed_name_cannot_be_claimed_twice_and_the_first_one_keeps_it() {
    on_every_store("claim", |store, called| {
        let name = mine(called, "contested");
        let first = store.put(b"the winner").expect("a write");
        let second = store.put(b"the loser").expect("a write");

        assert!(
            store
                .claim(&name, &first, vec![("who".into(), "first".into())])
                .expect("a claim"),
            "a free name is taken"
        );
        assert!(
            !store
                .claim(&name, &second, vec![("who".into(), "second".into())])
                .expect("a second claim"),
            "a taken name is not taken again"
        );

        let found = store.resolve(&name).expect("a resolve").expect("bound");
        assert_eq!(found.digest, first, "whoever claimed it keeps it");
        assert_eq!(found.meta, vec![("who".to_string(), "first".to_string())]);
    });
}

#[test]
fn a_claim_does_not_overwrite_what_bind_put_there() {
    on_every_store("claim-after-bind", |store, called| {
        let name = mine(called, "already");
        let there = store.put(b"already here").expect("a write");
        let other = store.put(b"trying").expect("a write");
        store.bind(&name, &there, Vec::new()).expect("a bind");

        assert!(
            !store.claim(&name, &other, Vec::new()).expect("a claim"),
            "a name bound by hand is still taken"
        );
        let found = store.resolve(&name).expect("a resolve").expect("bound");
        assert_eq!(found.digest, there);
    });
}

#[test]
fn many_are_answered_in_the_order_they_were_asked_with_gaps_where_there_is_nothing() {
    on_every_store("many", |store, called| {
        let (one, two) = (mine(called, "a"), mine(called, "b"));
        let missing = mine(called, "never");
        let first = store.put(b"first").expect("a write");
        let second = store.put(b"second").expect("a write");
        store.bind(&one, &first, Vec::new()).expect("a bind");
        store.bind(&two, &second, Vec::new()).expect("a bind");

        let asked = [one.as_str(), missing.as_str(), two.as_str()];
        let found = store.resolve_many(&asked).expect("a batch resolve");
        assert_eq!(found.len(), 3);
        assert_eq!(found[0].as_ref().map(|b| &b.digest), Some(&first));
        assert!(found[1].is_none(), "the gap stays where it was asked");
        assert_eq!(found[2].as_ref().map(|b| &b.digest), Some(&second));

        let nobodys = Digest::of(b"nobody wrote this either");
        let bytes = store
            .get_many(&[&first, &nobodys, &second])
            .expect("a batch get");
        assert_eq!(bytes[0].as_deref(), Some(&b"first"[..]));
        assert!(bytes[1].is_none());
        assert_eq!(bytes[2].as_deref(), Some(&b"second"[..]));
    });
}

#[test]
fn a_scan_finds_what_was_bound_in_a_settled_order() {
    on_every_store("scan", |store, called| {
        let names: Vec<String> = ["x", "y", "z"].iter().map(|s| mine(called, s)).collect();
        let digest = store.put(b"scanned").expect("a write");
        for name in &names {
            store.bind(name, &digest, Vec::new()).expect("a bind");
        }

        let all = store.bound().expect("a scan");
        for name in &names {
            assert!(
                all.iter().any(|bound| &bound.name == name),
                "`{name}` was bound and the scan did not find it"
            );
        }

        // Only this test's own names, and that is the point rather than a
        // convenience: a directory here is this test's alone, but a bucket is
        // shared with everything else running, so between two scans somebody
        // legitimately binds something. What has to be settled is the order of
        // what *this* test wrote, not that the store stood still.
        let ours = |scan: &[somatize_store::Bound]| -> Vec<String> {
            scan.iter()
                .filter(|bound| names.contains(&bound.name))
                .map(|bound| bound.name.clone())
                .collect()
        };
        let again = store.bound().expect("a second scan");
        assert_eq!(
            ours(&all),
            ours(&again),
            "two scans of the same store answer in the same order"
        );
        assert_eq!(ours(&all).len(), names.len());
    });
}

#[test]
fn a_record_that_is_not_a_record_is_corrupt_and_not_missing() {
    // The directory alone: putting rubbish where a record goes needs to reach
    // behind the store, and only one of the two lets a test do that honestly.
    let scratch = Dir::new();
    let store = Local::at(scratch.path()).expect("a directory store");
    let digest = store.put(b"fine").expect("a write");
    store.bind("readable", &digest, Vec::new()).expect("a bind");

    let record = std::fs::read_dir(scratch.path().join("names"))
        .expect("names/")
        .filter_map(Result::ok)
        .flat_map(|head| std::fs::read_dir(head.path()).expect("a shard"))
        .filter_map(Result::ok)
        .next()
        .expect("one record");
    std::fs::write(record.path(), b"not json").expect("a write");

    assert!(matches!(
        store.resolve("readable"),
        Err(StoreError::Corrupt(_))
    ));
}

/// An endpoint that says yes to everything, which is the failure this store
/// exists to refuse. Answers `200` to every request; a handful of connections is
/// plenty for two probes and a delete.
#[cfg(feature = "s3")]
fn always_yes() -> String {
    use std::io::{Read, Write};
    use std::net::TcpListener;

    let listening = TcpListener::bind("127.0.0.1:0").expect("a port");
    let at = format!("http://{}", listening.local_addr().expect("an address"));
    std::thread::spawn(move || {
        for stream in listening.incoming().take(8) {
            let Ok(mut stream) = stream else { continue };
            let mut seen = [0u8; 4096];
            let _ = stream.read(&mut seen);
            let _ = stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
        }
    });
    at
}

#[test]
#[cfg(feature = "s3")]
fn an_endpoint_that_ignores_the_condition_is_refused_and_not_used() {
    use somatize_store::{Bucket, Credentials, UrlStyle};

    let refused = Bucket::at(
        &always_yes(),
        "soma",
        "us-east-1",
        UrlStyle::Path,
        Credentials::new("who", "cares"),
    );
    let Err(StoreError::Io(why)) = refused else {
        panic!("a store that cannot hand work out was handed over anyway");
    };
    assert!(
        why.contains("took the same name twice"),
        "the refusal has to name the reason, and it said: {why}"
    );
}
