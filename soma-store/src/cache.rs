//! The engine's [`Keeper`], filled in by a [`Store`].
//!
//! The hole the core left, plugged with the two things the core cannot have: an
//! algorithm to hash with, and somewhere for bytes to live. From here the engine
//! gets a name for what a node is about to produce, and an answer to whether
//! that name has been seen before.
//!
//! # What a name is made of
//!
//! ```text
//! key(root) = sha256( the input, in bytes )
//! key(node) = sha256( identity | state | salt | the keys above it )
//! ```
//!
//! The pieces of a recipe are **framed by their length** before being hashed,
//! and that is not a detail: run together, `["ab", "c"]` and `["a", "bc"]` would
//! be the same string, and two different recipes under one name is the one
//! failure a cache must not have.
//!
//! # Two namespaces in one store
//!
//! A cached value is bound under `value:<key>` where an artifact is bound under
//! `artifact:<kind>:<id>`. The same directory holds both — which is the point of
//! a store that is a directory — and an artifact whose id happens to read like a
//! key still cannot be mistaken for one.
//!
//! # MessagePack again, and it is not the same decision
//!
//! The bytes of a value are written the same way the worker's protocol writes
//! them, and the few lines that do it are written twice on purpose. What crosses
//! a wire and what sits in a store for a year are the same shape today and have
//! no reason to stay so: the wire's two ends are the same binary from the same
//! `cargo build`, and a store outlives every binary that ever wrote into it.
//! Sharing the code would have made that one decision instead of two.
//!
//! What does **not** travel is the same, though, and for the same reason: a
//! [`Value::Opaque`] points into the process that made it. Here it fails with
//! the type in front of you, and `somatize.torch` is what turns a tensor into
//! bytes before it ever gets this far.

use crate::{Digest, Meta, Store, StoreError};
use somatize_core::{Keeper, KeeperError, Kept, Key, Value};

/// Names what a graph produces, and keeps it in a store.
pub struct Cache<'a> {
    store: &'a dyn Store,
}

impl<'a> Cache<'a> {
    /// The cache kept in this store.
    pub fn over(store: &'a dyn Store) -> Self {
        Self { store }
    }
}

impl Keeper for Cache<'_> {
    fn key_of(&self, value: &Value) -> Option<Key> {
        match value.travels() {
            true => bytes_of(value).ok().map(|bytes| key(Digest::of(&bytes))),
            false => None,
        }
    }

    fn combine(&self, parts: &[&str]) -> Key {
        let mut recipe = Vec::new();
        for part in parts {
            // The length first, so the pieces cannot run into each other.
            recipe.extend_from_slice(&(part.len() as u64).to_le_bytes());
            recipe.extend_from_slice(part.as_bytes());
        }
        key(Digest::of(&recipe))
    }

    /// One scan and **no fetches**, which is the whole reason the engine asks
    /// this instead of reading: what it wants to know is whether it can skip
    /// the node underneath, and the bytes of the answer are somebody else's
    /// business — often nobody's.
    fn present(&self, keys: &[&Key]) -> Result<Vec<bool>, KeeperError> {
        let names: Vec<String> = keys.iter().map(|key| name_of(key)).collect();
        let asked: Vec<&str> = names.iter().map(String::as_str).collect();
        Ok(self
            .store
            .resolve_many(&asked)
            .map_err(failed)?
            .into_iter()
            .map(|bound| bound.is_some())
            .collect())
    }

    fn recall(&self, keys: &[&Key]) -> Result<Vec<Option<Kept>>, KeeperError> {
        let names: Vec<String> = keys.iter().map(|key| name_of(key)).collect();
        let asked: Vec<&str> = names.iter().map(String::as_str).collect();
        let bound = self.store.resolve_many(&asked).map_err(failed)?;

        // Two round trips and not two per key: the names first, then the bytes
        // of every one that answered. It is the whole reason both questions are
        // batched in the trait.
        let wanted: Vec<&Digest> = bound.iter().flatten().map(|bound| &bound.digest).collect();
        let mut bytes = self.store.get_many(&wanted).map_err(failed)?.into_iter();

        bound
            .into_iter()
            .map(|bound| {
                let Some(bound) = bound else { return Ok(None) };
                // A name that answers and bytes that are gone is a miss, not a
                // failure: a store can be swept, and what it says it has is a
                // record, not a promise.
                let Some(Some(bytes)) = bytes.next() else {
                    return Ok(None);
                };
                Ok(Some(Kept {
                    value: value_of(&bytes)?,
                    meta: bound.meta,
                }))
            })
            .collect()
    }

    fn keep(&self, key: &Key, value: &Value, meta: &[(&str, &str)]) -> Result<(), KeeperError> {
        let bytes = bytes_of(value)?;
        let digest = self.store.put(&bytes).map_err(failed)?;
        let meta: Meta = meta
            .iter()
            .map(|(what, said)| (what.to_string(), said.to_string()))
            .collect();
        self.store
            .bind(&name_of(key), &digest, meta)
            .map_err(failed)
    }
}

/// A digest, read as the name of what a recipe produces.
fn key(digest: Digest) -> Key {
    Key::new(digest.to_string())
}

/// Where a value is bound, which is not where an artifact is.
///
/// **Public, and that is a decision**, the same one the engine's metadata
/// constants got and for the same reason. A key is what a recipe is called; the
/// name is where a value under that key is bound, and the two are not the same
/// string. Anybody reading a store back — *which of these hashes is the answer
/// this version would ask for?* — needs to get from one to the other, and the
/// alternative is every reader carrying its own `format!` of this, which is two
/// places saying the same thing with no way to tell which governs the day they
/// disagree.
pub fn name_of(key: &Key) -> String {
    format!("value:{key}")
}

/// A value in bytes, refusing what only exists in this process.
pub fn bytes_of(value: &Value) -> Result<Vec<u8>, KeeperError> {
    if !value.travels() {
        return Err(KeeperError::new(
            "an opaque value cannot be kept: what it carries only exists in this process, \
             and a store outlives it. Whoever knows how to turn it into bytes has to do so \
             before it gets here",
        ));
    }
    rmp_serde::to_vec(value)
        .map_err(|e| KeeperError::new(format!("that value could not be written down: {e}")))
}

/// And back. **Nothing may be left over**: leftovers are as suspicious as
/// missing bytes, and no format checks that for you.
pub fn value_of(bytes: &[u8]) -> Result<Value, KeeperError> {
    let mut rest = bytes;
    let value: Value = rmp_serde::from_read(&mut rest)
        .map_err(|e| KeeperError::new(format!("what is kept there cannot be read: {e}")))?;
    match rest.len() {
        0 => Ok(value),
        left => Err(KeeperError::new(format!(
            "what is kept there has {left} bytes too many at the end"
        ))),
    }
}

fn failed(e: StoreError) -> KeeperError {
    KeeperError::new(e.to_string())
}
