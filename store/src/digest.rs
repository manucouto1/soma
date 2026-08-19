//! What identifies some bytes: their content.

use sha2::{Digest as _, Sha256};
use std::fmt;

/// The identity of some bytes, written as `sha256:` and hex.
///
/// The prefix is not decoration: it is what allows another algorithm the day one
/// is needed without every stored name becoming ambiguous.
#[derive(
    Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct Digest(String);

impl Digest {
    /// What identifies these bytes.
    pub fn of(bytes: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        Self(format!("sha256:{:x}", hasher.finalize()))
    }

    /// A digest someone else computed — the id of an artifact, a key read back
    /// from a record.
    pub fn parse(text: impl Into<String>) -> Self {
        Self(text.into())
    }

    /// As text.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// How it is split into a directory and a file, so no directory ends up with
    /// a million entries in it.
    ///
    /// The directory comes from the **hash** and not from the whole string: with
    /// the `sha256:` in front, every digest starts the same way and everything
    /// would land in a single `sh/`, which is the one directory this split
    /// exists to avoid. The file keeps the whole thing, so the same hex under
    /// two algorithms is still two files.
    pub(crate) fn path(&self) -> (String, String) {
        let hash = flatten(self.0.rsplit(':').next().unwrap_or(&self.0));
        let head = hash.get(..2).unwrap_or(hash.as_str()).to_string();
        (head, flatten(&self.0))
    }
}

impl fmt::Display for Digest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Whatever a caller invented, as something a filesystem takes.
fn flatten(text: &str) -> String {
    text.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}
