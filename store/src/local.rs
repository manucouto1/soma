//! A store that is a directory. The one that works today.
//!
//! No dependencies beyond hashing and a text format, because a shared folder is
//! what there already is — a network mount, a scratch directory, `/tmp` in a
//! test. S3 arrives the day there is a MinIO to point at, as another
//! implementation of the same trait and in another crate: HTTP has no business
//! here.
//!
//! ```text
//! <root>/blobs/ab/sha256_abc…    the bytes, named by their content
//! <root>/names/de/sha256_def…    one JSON record per name
//! <root>/tmp/…                   where a write lands before its rename
//! ```
//!
//! The two directory characters come from the **hash**, not from the front of
//! the digest: `sha256:` is the same in all of them, and a single directory with
//! every blob in it is what this split exists to avoid.
//!
//! A record's file is named by the digest **of the name**, not by the name: a
//! cache key is hex and an artifact's id is whatever its producer chose, and no
//! filesystem takes every string a caller can invent. The name itself is inside
//! the record, so `grep` still finds it.

use crate::{Bound, Digest, Meta, Store, StoreError};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// So no two writes of this process pick the same landing spot. A counter and
/// not a clock: the pid separates processes, but two threads can read the same
/// nanosecond, and then one of them lands on the other's bytes.
static LANDINGS: AtomicU64 = AtomicU64::new(0);

/// A store kept in a directory.
pub struct Local {
    root: PathBuf,
}

impl Local {
    /// The store in this directory, creating it if it is not there.
    pub fn at(root: impl Into<PathBuf>) -> Result<Self, StoreError> {
        let root = root.into();
        for each in ["blobs", "names", "tmp"] {
            fs::create_dir_all(root.join(each)).map_err(io_error)?;
        }
        Ok(Self { root })
    }

    fn blob(&self, digest: &Digest) -> PathBuf {
        let (head, rest) = digest.path();
        self.root.join("blobs").join(head).join(rest)
    }

    fn record(&self, name: &str) -> PathBuf {
        let (head, rest) = Digest::of(name.as_bytes()).path();
        self.root.join("names").join(head).join(rest)
    }

    /// Writes it somewhere else and moves it into place, which is what makes a
    /// half-written file impossible to read: a rename either happened or did
    /// not.
    fn land(&self, at: &Path, bytes: &[u8]) -> Result<(), StoreError> {
        if let Some(directory) = at.parent() {
            fs::create_dir_all(directory).map_err(io_error)?;
        }
        let landing = self.root.join("tmp").join(format!(
            "{}-{}",
            std::process::id(),
            LANDINGS.fetch_add(1, Ordering::Relaxed)
        ));
        fs::write(&landing, bytes).map_err(io_error)?;
        fs::rename(&landing, at).map_err(io_error)
    }
}

impl Store for Local {
    fn put(&self, bytes: &[u8]) -> Result<Digest, StoreError> {
        let digest = Digest::of(bytes);
        let at = self.blob(&digest);
        // Already there is already done: the content is the name.
        if !at.exists() {
            self.land(&at, bytes)?;
        }
        Ok(digest)
    }

    fn get(&self, digest: &Digest) -> Result<Option<Vec<u8>>, StoreError> {
        match fs::read(self.blob(digest)) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(io_error(e)),
        }
    }

    fn bind(&self, name: &str, digest: &Digest, meta: Meta) -> Result<(), StoreError> {
        let bound = Bound {
            name: name.to_string(),
            digest: digest.clone(),
            meta,
            when: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|since| since.as_secs())
                .unwrap_or(0),
        };
        let written = serde_json::to_vec_pretty(&bound)
            .map_err(|e| StoreError::Corrupt(format!("that record cannot be written: {e}")))?;
        self.land(&self.record(name), &written)
    }

    fn resolve(&self, name: &str) -> Result<Option<Bound>, StoreError> {
        match fs::read(self.record(name)) {
            Ok(bytes) => read_record(&bytes).map(Some),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(io_error(e)),
        }
    }

    fn bound(&self) -> Result<Vec<Bound>, StoreError> {
        let mut all = Vec::new();
        let names = self.root.join("names");
        for head in read_dir(&names)? {
            for record in read_dir(&head)? {
                all.push(read_record(&fs::read(record).map_err(io_error)?)?);
            }
        }
        // By time, and by name within the same second, so two runs of this see
        // the same thing.
        all.sort_by(|a, b| (a.when, &a.name).cmp(&(b.when, &b.name)));
        Ok(all)
    }
}

/// What is inside a directory, or nothing if it is not there yet.
fn read_dir(at: &Path) -> Result<Vec<PathBuf>, StoreError> {
    match fs::read_dir(at) {
        Ok(entries) => entries
            .map(|entry| entry.map(|entry| entry.path()).map_err(io_error))
            .collect(),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(e) => Err(io_error(e)),
    }
}

fn read_record(bytes: &[u8]) -> Result<Bound, StoreError> {
    serde_json::from_slice(bytes)
        .map_err(|e| StoreError::Corrupt(format!("that record cannot be read: {e}")))
}

fn io_error(e: io::Error) -> StoreError {
    StoreError::Io(e.to_string())
}
