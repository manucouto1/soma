//! What names what a node produces, before it produces it.
//!
//! A key is a Merkle hash **over the recipe**, not over the data: the identity
//! of the node, the digest of the state it is settled at, and the keys of its
//! predecessors. Only a root hashes content. That is the whole point — the key
//! is known *before* anything runs, so changing the classifier does not touch
//! the key of the embeddings underneath it.
//!
//! The core does not compute one. Hashing needs an algorithm and the core has no
//! dependencies, so a `Key` arrives from the [`Keeper`](crate::Keeper) and this
//! is only the shape it arrives in — the same division of labour as
//! [`Host`](crate::Host), which is a name and not an address.
//!
//! # Why there is no `Keys` yet
//!
//! Caching item by item wants a key **per item** and not per node, and the plan
//! for it is written. It is not here because nothing produces one today, and a
//! variant nobody can construct is worse than a variant that arrives late: the
//! day it does, every `match` on it stops compiling and someone decides, which
//! is exactly what [`Plan`](crate::Plan) has no `#[non_exhaustive]` for.

use std::fmt;

/// What a node's output is called, wherever it is kept.
///
/// Text and not bytes because it is a name: it ends up in a store's index, in a
/// log line and in an error message, and hex that cannot be read aloud helps
/// nobody.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(transparent)
)]
pub struct Key(String);

impl Key {
    /// A key somebody else computed. Only a [`Keeper`](crate::Keeper) should be
    /// calling this: two keys made by different recipes have to be different,
    /// and nothing here can check that.
    pub fn new(text: impl Into<String>) -> Self {
        Self(text.into())
    }

    /// As text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
