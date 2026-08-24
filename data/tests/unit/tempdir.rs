//! A temporary directory, without a dependency for it.
//!
//! Shared by whoever needs a store of their own: everything about a store is
//! tested against a real directory, because what is being tested is precisely
//! that two processes could share it.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

static COUNT: AtomicUsize = AtomicUsize::new(0);

pub struct Dir(PathBuf);

impl Dir {
    pub fn new() -> Self {
        let at = std::env::temp_dir().join(format!(
            "soma-data-{}-{}",
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
