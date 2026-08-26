//! `k` folds where all the samples of a group land on the same side.

use super::dealing::{assemble, checked, grouped_by, heaviest_first, in_order};
use super::{Fold, Partition, PartitionError};
use crate::Samples;
use std::fmt;

/// Grouping is a k-fold **over the groups**, with the samples following theirs.
/// Needs [`in_groups`](Samples::in_groups) and takes no seed: it places the
/// biggest groups first into whichever fold is emptiest, which is what keeps the
/// folds comparable when the groups are not.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Grouped {
    /// How many folds.
    pub k: usize,
}

impl Grouped {
    /// The folds, in order. Two samples with the same group never end up on
    /// opposite sides.
    pub fn folds(&self, samples: &Samples) -> Result<Vec<Fold>, PartitionError> {
        let n = samples.n();
        let groups = samples.groups().ok_or(PartitionError::NeedsGroups)?;
        checked(self.k, n)?;

        let mut tests = vec![Vec::new(); self.k];
        let mut carried = vec![0usize; self.k];
        for (_, indices) in heaviest_first(grouped_by(groups, &in_order(n)), self.k)? {
            let fold = emptiest(&carried);
            carried[fold] += indices.len();
            tests[fold].extend(indices);
        }
        Ok(assemble(n, tests))
    }
}

/// The fold carrying the fewest samples, the first of them on a tie.
fn emptiest(carried: &[usize]) -> usize {
    let mut best = 0;
    for (fold, &load) in carried.iter().enumerate() {
        if load < carried[best] {
            best = fold;
        }
    }
    best
}

impl fmt::Display for Grouped {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "grouped:{}", self.k)
    }
}

impl From<Grouped> for Partition {
    fn from(cut: Grouped) -> Self {
        Self::Grouped(cut)
    }
}
