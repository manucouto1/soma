//! Who hashes and who keeps. The hole.
//!
//! The same shape as the rest, and the reason is the same: **the core provides
//! the hole; whoever knows what goes in it is a library.** Here it is doubly
//! true — hashing is `sha256` and keeping is a directory or a bucket, and the
//! core has no dependencies at all.
//!
//! | hole | who fills it | what they know that the core does not |
//! |---|---|---|
//! | [`Node`](crate::Node) | the user | what a node does |
//! | [`Transport`](crate::Transport) | a library | what a wire is |
//! | [`Codec`](crate::Codec) | a library | how to write down what lives in one process |
//! | [`Watcher`](crate::Watcher) | whoever executes | what to do with a fact |
//! | `Keeper` | a library | what a hash is, and where bytes live |

use crate::{Key, Value};
use std::fmt;

/// Hashes recipes and keeps what they name.
pub trait Keeper: Send + Sync {
    /// The key of a value **by its content**, which only a root needs: from
    /// there down, keys come from keys. `None` if the value cannot leave this
    /// process, which is not a failure — nothing below it is cached either.
    fn key_of(&self, value: &Value) -> Option<Key>;

    /// One key out of the ingredients of a recipe, in the order given. **The
    /// parts have to stay apart**: run together, `["ab", "c"]` and
    /// `["a", "bc"]` would name the same thing.
    fn combine(&self, parts: &[&str]) -> Key;

    /// What is kept under each of these, in the order they were asked. In batch
    /// form from the first day: against a remote store, one question per item is
    /// one round trip per item.
    fn recall(&self, keys: &[&Key]) -> Result<Vec<Option<Kept>>, KeeperError>;

    /// Whether each of these is kept, **without reading any of it**.
    ///
    /// A key is knowable before anything runs, so the engine can ask which
    /// answers it already has and then not execute what only fed one of them.
    /// The default is honest and expensive — it reads them; whoever can answer
    /// by name alone should say so, or asking early costs what it saves.
    fn present(&self, keys: &[&Key]) -> Result<Vec<bool>, KeeperError> {
        Ok(self
            .recall(keys)?
            .into_iter()
            .map(|kept| kept.is_some())
            .collect())
    }

    /// Keeps this, with what should be remembered beside it — the fingerprint
    /// of the code that produced it, above all, which is **not** in the key and
    /// is what a hit gets compared against.
    fn keep(&self, key: &Key, value: &Value, meta: &[(&str, &str)]) -> Result<(), KeeperError>;
}

/// Something that was kept, on the way back: the value, and what was said
/// beside it. The metadata comes back because the fingerprint of the code is
/// not in the key — it is written next to the value and compared on a hit.
#[derive(Debug, Clone, PartialEq)]
pub struct Kept {
    /// What was kept.
    pub value: Value,
    /// What was said beside it, in the order it was said.
    pub meta: Vec<(String, String)>,
}

/// Why something could not be kept, or found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeeperError(String);

impl KeeperError {
    /// A failure described by a message.
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }

    /// The message.
    pub fn message(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for KeeperError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for KeeperError {}
