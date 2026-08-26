//! What names what a node produces, before it produces it.
//!
//! A key is a Merkle hash **over the recipe** and not over the data: the
//! identity of the node, what it was built with, the digest of the state it is
//! settled at, and the keys of its predecessors. Only a root hashes content, so
//! the key is known before anything runs and changing the classifier does not
//! touch the key of the embeddings underneath it.
//!
//! The core computes none of them: hashing needs an algorithm and the core has
//! no dependencies, so a `Key` arrives from the [`Keeper`](crate::Keeper) and
//! this is only the shape it arrives in.

use std::fmt;

/// What a node's output is called, wherever it is kept. Text and not bytes
/// because it is a name: it ends up in an index, a log line and an error.
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

/// What a node's output is called: one name, or one per item.
///
/// The second is what a [`.mapped()`](crate::Memory::map) node produces: with
/// one name per node, adding a document to a list of a thousand misses all
/// thousand; with one per item, the thousand hit and the new one runs.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Keys {
    /// The output is one thing and has one name.
    One(Key),
    /// The output is a list, and each item is named on its own — in order, and
    /// as long as the list.
    PerItem(Vec<Key>),
}

impl Keys {
    /// The one name, if there is one. `None` for a list of them: what a single
    /// name over many is made of is
    /// [`Keeper::combine`](crate::Keeper::combine)'s to decide.
    pub fn one(&self) -> Option<&Key> {
        match self {
            Self::One(key) => Some(key),
            Self::PerItem(_) => None,
        }
    }

    /// Every name in it, in order: one, or as many as there are items.
    pub fn each(&self) -> &[Key] {
        match self {
            Self::One(key) => std::slice::from_ref(key),
            Self::PerItem(keys) => keys,
        }
    }
}

impl fmt::Display for Keys {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::One(key) => key.fmt(f),
            Self::PerItem(keys) => write!(f, "{} items", keys.len()),
        }
    }
}
