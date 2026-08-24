//! A keeper with a codec in front of it.

use crate::{Codec, Keeper, KeeperError, Kept, Key, Value};

/// Whatever a [`Codec`] can write down, kept — by a [`Keeper`] that never finds
/// out any of it was ever anything but bytes.
///
/// # Why the two holes meet here and not in each library
///
/// A store and a wire ask the same question of an opaque value — *what does this
/// weigh in bytes* — and it has one answer. So the pair `(keeper, codec)` is
/// wired up once, here, rather than once per thing that has a codec: the Python
/// side hands its registry of `dump`/`load` pairs, `data/` hands Arrow IPC, and
/// neither writes this again. Two copies of it would be two chances to disagree
/// about when a failure is a miss and when it is a stop.
///
/// Which is decided here, and the two directions are **not** symmetrical:
///
/// | | when the codec cannot | why |
/// |---|---|---|
/// | naming a value | no name, and the run goes on | a cache is an optimization; one that can kill a run at hour three is not one |
/// | keeping it | the keeper's error | somebody asked for it to be kept, and it was not |
/// | reading it back | the keeper's error | bytes in a store that nobody can read are worse news than a miss |
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
