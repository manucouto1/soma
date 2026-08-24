//! A store that keeps count of what it was asked to do.
//!
//! Shared, because the questions this crate exists to answer are questions
//! about **what was not read**: a version that costs no bytes, a dataset that
//! is not opened twice. Neither can be asserted without counting.

use soma_next_store::{Bound, Digest, Local, Meta, Store, StoreError};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct Counting {
    inner: Local,
    resolves: AtomicUsize,
    fetched: Mutex<Vec<String>>,
}

impl Counting {
    pub fn over(inner: Local) -> Self {
        Self {
            inner,
            resolves: AtomicUsize::new(0),
            fetched: Mutex::new(Vec::new()),
        }
    }

    /// How many lookups, and how many fetches.
    pub fn seen(&self) -> (usize, usize) {
        (
            self.resolves.load(Ordering::SeqCst),
            self.fetched.lock().unwrap().len(),
        )
    }

    /// Whether those exact bytes were asked for.
    pub fn fetched(&self, digest: &Digest) -> bool {
        self.fetched
            .lock()
            .unwrap()
            .iter()
            .any(|seen| seen == digest.as_str())
    }

    pub fn forget(&self) {
        self.resolves.store(0, Ordering::SeqCst);
        self.fetched.lock().unwrap().clear();
    }
}

impl Store for Counting {
    fn put(&self, bytes: &[u8]) -> Result<Digest, StoreError> {
        self.inner.put(bytes)
    }

    fn get(&self, digest: &Digest) -> Result<Option<Vec<u8>>, StoreError> {
        self.fetched
            .lock()
            .unwrap()
            .push(digest.as_str().to_string());
        self.inner.get(digest)
    }

    fn bind(&self, name: &str, digest: &Digest, meta: Meta) -> Result<(), StoreError> {
        self.inner.bind(name, digest, meta)
    }

    fn claim(&self, name: &str, digest: &Digest, meta: Meta) -> Result<bool, StoreError> {
        self.inner.claim(name, digest, meta)
    }

    fn resolve(&self, name: &str) -> Result<Option<Bound>, StoreError> {
        self.resolves.fetch_add(1, Ordering::SeqCst);
        self.inner.resolve(name)
    }

    fn bound(&self) -> Result<Vec<Bound>, StoreError> {
        self.inner.bound()
    }
}
