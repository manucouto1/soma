//! Who hashes and who keeps. The hole.
//!
//! The fourth of the same shape, and the reason is the same as the other three:
//! **the core provides the hole; whoever knows what goes in it is a library.**
//! Here it is doubly true — hashing is `sha256` and keeping is a directory or a
//! bucket, and the core has no dependencies at all.
//!
//! | hole | who fills it | what they know that the core does not |
//! |---|---|---|
//! | [`Node`](crate::Node) | the user | what a step does |
//! | [`Driver`](crate::Driver) | whoever executes | what a request means |
//! | [`Transport`](crate::Transport) | a library | what a wire is |
//! | `Keeper` | a library | what a hash is, and where bytes live |
//!
//! Only one implementor today, which brushes against the rule that a trait needs
//! two. It stands for the same reason [`Transport`](crate::Transport) does: the
//! implementation **has to** be in another crate, or the core eats `sha2` and
//! `serde`.

use crate::{Key, Value};
use std::fmt;

/// Hashes recipes and keeps what they name.
pub trait Keeper: Send + Sync {
    /// The key of a value **by its content**, which only a root needs: from
    /// there down, keys come from keys.
    ///
    /// `None` if the value cannot leave this process — a
    /// [`Value::Opaque`](crate::Value::Opaque) at any depth — and that is not a
    /// failure. It means nothing below it can be keyed, so nothing below it is
    /// cached, and the run goes on.
    fn key_of(&self, value: &Value) -> Option<Key>;

    /// One key out of the ingredients of a recipe, in the order given.
    ///
    /// **The parts have to stay apart**: run together, `["ab", "c"]` and
    /// `["a", "bc"]` would name the same thing, and two different recipes
    /// sharing a key is the one failure a cache must not have.
    fn combine(&self, parts: &[&str]) -> Key;

    /// What is kept under each of these, in the order they were asked.
    ///
    /// In the trait in **batch form** from the first day, and not for symmetry:
    /// against a store on the far end of a network, one question per item is
    /// one round trip per item.
    fn recall(&self, keys: &[&Key]) -> Result<Vec<Option<Kept>>, KeeperError>;

    /// Keeps this, with what should be remembered beside it — the fingerprint
    /// of the code that produced it, above all, which is **not** in the key and
    /// is what a hit gets compared against.
    fn keep(&self, key: &Key, value: &Value, meta: &[(&str, &str)]) -> Result<(), KeeperError>;
}

/// Something that was kept, on the way back: the value, and what was said
/// beside it when it was written.
///
/// The metadata comes back because of one line of the design — the fingerprint
/// of the code is **not** in the key, it is written next to the value and
/// compared on a hit. Without it here, a cache that quietly went cold could not
/// be told from one that is working.
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
