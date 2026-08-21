//! `k` folds over the samples, each held out in turn.

use super::dealing::{assemble, checked, deal, ordering};
use super::{Fold, Partition, PartitionError};
use crate::Samples;
use std::fmt;

/// The plain cut: the samples in `k` parts, each one held out while the rest
/// train.
///
/// `LeaveOneOut` is this with `k = n`, and a holdout of one part in `k` is its
/// fold 0. Neither earns a scheme of its own — a scheme that is a parameter is
/// a name you have to remember for nothing.
///
/// ```
/// use soma_next_study::{KFold, Samples};
///
/// let folds = KFold { k: 5, shuffle: Some(0) }.folds(&Samples::of(100))?;
/// assert_eq!(folds.len(), 5);
/// # Ok::<(), soma_next_study::PartitionError>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KFold {
    /// How many folds.
    pub k: usize,
    /// The seed, or `None` for the order they came in.
    ///
    /// The seed both switches shuffling on and makes it repeatable, so
    /// "shuffled but not reproducible" is a state that cannot be written down.
    pub shuffle: Option<u64>,
}

impl KFold {
    /// The folds, in order. It uses neither the classes nor the groups.
    pub fn folds(&self, samples: &Samples) -> Result<Vec<Fold>, PartitionError> {
        let n = samples.n();
        checked(self.k, n)?;
        Ok(assemble(n, deal(&ordering(n, self.shuffle), self.k)))
    }
}

impl fmt::Display for KFold {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        super::scheme(f, "kfold", self.k, self.shuffle)
    }
}

impl From<KFold> for Partition {
    fn from(cut: KFold) -> Self {
        Self::KFold(cut)
    }
}
