//! A store that keeps count of what it was asked to do.
//!
//! Here because two of this crate's promises are about **what is not read**: a
//! name is resolved and never scanned for, and a version's trials come back in
//! one walk. Neither can be asserted by looking at the answer — a correct
//! answer arrived at expensively looks exactly like a cheap one.

use somatize_store::{Bound, Digest, Local, Meta, Store, StoreError};
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct Counting {
    inner: Local,
    resolves: AtomicUsize,
    scans: AtomicUsize,
    fetches: AtomicUsize,
}

impl Counting {
    pub fn over(inner: Local) -> Self {
        Self {
            inner,
            resolves: AtomicUsize::new(0),
            scans: AtomicUsize::new(0),
            fetches: AtomicUsize::new(0),
        }
    }

    /// Lookups, walks of everything, and fetches of bytes.
    pub fn seen(&self) -> (usize, usize, usize) {
        (
            self.resolves.load(Ordering::SeqCst),
            self.scans.load(Ordering::SeqCst),
            self.fetches.load(Ordering::SeqCst),
        )
    }

    pub fn forget(&self) {
        self.resolves.store(0, Ordering::SeqCst);
        self.scans.store(0, Ordering::SeqCst);
        self.fetches.store(0, Ordering::SeqCst);
    }
}

impl Store for Counting {
    fn put(&self, bytes: &[u8]) -> Result<Digest, StoreError> {
        self.inner.put(bytes)
    }

    fn get(&self, digest: &Digest) -> Result<Option<Vec<u8>>, StoreError> {
        self.fetches.fetch_add(1, Ordering::SeqCst);
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
        self.scans.fetch_add(1, Ordering::SeqCst);
        self.inner.bound()
    }
}
