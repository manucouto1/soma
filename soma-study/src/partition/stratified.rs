//! `k` folds where every class keeps the share it has in the whole.

use super::dealing::{assemble, checked, deal, grouped_by, ordering};
use super::{Fold, Partition, PartitionError};
use crate::Samples;
use std::fmt;

/// Stratifying is not a different algorithm: it is a [`KFold`](crate::KFold)
/// applied **inside each class**, the folds concatenated. That is why there is
/// one scheme here and not sklearn's `KFold` / `StratifiedKFold` pair.
///
/// Needs [`by_class`](Samples::by_class).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Stratified {
    /// How many folds.
    pub k: usize,
    /// The seed, or `None` for the order they came in.
    pub shuffle: Option<u64>,
}

impl Stratified {
    /// The folds, in order.
    ///
    /// A class with fewer members than folds cannot be in all of them, and that
    /// is an **error**: sklearn warns and carries on, which leaves a result you
    /// cannot tell from a good one.
    pub fn folds(&self, samples: &Samples) -> Result<Vec<Fold>, PartitionError> {
        let n = samples.n();
        let strata = samples.strata().ok_or(PartitionError::NeedsClasses)?;
        checked(self.k, n)?;

        let mut tests = vec![Vec::new(); self.k];
        for (class, members) in grouped_by(strata, &ordering(n, self.shuffle)) {
            if members.len() < self.k {
                return Err(PartitionError::ClassTooSmall {
                    class,
                    count: members.len(),
                    k: self.k,
                });
            }
            for (fold, share) in deal(&members, self.k).into_iter().enumerate() {
                tests[fold].extend(share);
            }
        }
        Ok(assemble(n, tests))
    }
}

impl fmt::Display for Stratified {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        super::scheme(f, "stratified", self.k, self.shuffle)
    }
}

impl From<Stratified> for Partition {
    fn from(cut: Stratified) -> Self {
        Self::Stratified(cut)
    }
}
