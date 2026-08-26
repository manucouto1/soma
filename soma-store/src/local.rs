//! A store that is a directory. The one that works today, with no dependencies
//! beyond hashing and a text format, because a shared folder is what there
//! already is. [`Bucket`](crate::Bucket) is the other one and lays its bytes out
//! exactly like this, so one can be copied onto the other.
//!
//! ```text
//! <root>/blobs/ab/sha256_abc…    the bytes, named by their content
//! <root>/names/de/sha256_def…    one JSON record per name
//! <root>/tmp/…                   where a write lands before its rename
//! ```
//!
//! The two directory characters come from the **hash** and not the front of the
//! digest, which is the same in all of them. A record's file is named by the
//! digest **of the name**: no filesystem takes every string a caller can invent,
//! and the name itself is inside the record, so `grep` still finds it.

use crate::store::{read_record, record};
use crate::{Bound, Digest, Meta, Store, StoreError};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

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
        let landing = self.landing();
        fs::write(&landing, bytes).map_err(io_error)?;
        fs::rename(&landing, at).map_err(io_error)
    }

    /// A path nobody else is writing to: this process, and a number that only
    /// goes up. Inside the store's own `tmp`, so that landing it is a move
    /// within one filesystem and never a copy.
    fn landing(&self) -> PathBuf {
        self.root.join("tmp").join(format!(
            "{}-{}",
            std::process::id(),
            LANDINGS.fetch_add(1, Ordering::Relaxed)
        ))
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
        self.land(&self.record(name), &record(name, digest, meta)?)
    }

    fn claim(&self, name: &str, digest: &Digest, meta: Meta) -> Result<bool, StoreError> {
        let at = self.record(name);
        if let Some(directory) = at.parent() {
            fs::create_dir_all(directory).map_err(io_error)?;
        }
        let written = record(name, digest, meta)?;
        let landing = self.landing();
        fs::write(&landing, &written).map_err(io_error)?;
        // **`link` and not `rename`**, which is the whole difference: a rename
        // replaces what is there and would hand the same work to everybody, and
        // `link` fails when the name is taken. It is also the one that has
        // always been trusted over NFS, where `O_EXCL` has not.
        let taken = match fs::hard_link(&landing, &at) {
            Ok(()) => true,
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => false,
            Err(e) => {
                let _ = fs::remove_file(&landing);
                return Err(io_error(e));
            }
        };
        // The temporary is the second name for the same bytes, and one name is
        // enough. Failing to tidy up is not failing to claim.
        let _ = fs::remove_file(&landing);
        Ok(taken)
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

fn io_error(e: io::Error) -> StoreError {
    StoreError::Io(e.to_string())
}
