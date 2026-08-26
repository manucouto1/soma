//! A keeper with a codec in front of it.

use crate::{Codec, Keeper, KeeperError, Kept, Key, Value};

/// Whatever a [`Codec`] can write down, kept — by a [`Keeper`] that never finds
/// out any of it was ever anything but bytes.
///
/// A store and a wire ask an opaque value the same question, so the pair
/// `(keeper, codec)` is wired up once here rather than once per tenant. What is
/// decided here is that the directions are **not** symmetrical: failing to
/// *name* a value costs the name and the run goes on, while failing to keep it
/// or to read it back is the keeper's error.
pub struct Packing<'a> {
    inner: &'a dyn Keeper,
    codec: &'a dyn Codec,
}

impl<'a> Packing<'a> {
    /// That keeper, with that codec in front of it.
    pub fn over(inner: &'a dyn Keeper, codec: &'a dyn Codec) -> Self {
        Self { inner, codec }
    }
}

impl Keeper for Packing<'_> {
    fn key_of(&self, value: &Value) -> Option<Key> {
        self.inner.key_of(&self.codec.packed(value).ok()?)
    }

    fn combine(&self, parts: &[&str]) -> Key {
        self.inner.combine(parts)
    }

    /// Straight through: whether something is kept is a question about names,
    /// and a codec has nothing to say about a name.
    fn present(&self, keys: &[&Key]) -> Result<Vec<bool>, KeeperError> {
        self.inner.present(keys)
    }

    fn recall(&self, keys: &[&Key]) -> Result<Vec<Option<Kept>>, KeeperError> {
        self.inner
            .recall(keys)?
            .into_iter()
            .map(|kept| match kept {
                None => Ok(None),
                Some(kept) => Ok(Some(Kept {
                    value: self
                        .codec
                        .unpacked(&kept.value)
                        .map_err(|e| KeeperError::new(e.to_string()))?,
                    meta: kept.meta,
                })),
            })
            .collect()
    }

    fn keep(&self, key: &Key, value: &Value, meta: &[(&str, &str)]) -> Result<(), KeeperError> {
        let written = self
            .codec
            .packed(value)
            .map_err(|e| KeeperError::new(e.to_string()))?;
        self.inner.keep(key, &written, meta)
    }
}
