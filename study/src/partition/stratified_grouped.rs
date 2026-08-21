//! Groups whole, and among the ways of doing that the one that leaves the
//! classes most even.

use super::dealing::{assemble, checked, grouped_by, heaviest_first, in_order};
use super::{Fold, Partition, PartitionError};
use crate::Samples;
use std::collections::BTreeSet;
use std::fmt;

/// The two constraints at once, which is where they stop composing cleanly:
/// with the groups kept whole, exact strata are usually **not reachable at
/// all**. So this is greedy and approximate, and says so — sklearn's
/// `StratifiedGroupKFold` is greedy for the same reason, because the exact
/// problem is a bin packing.
///
/// Needs both [`by_class`](Samples::by_class) and
/// [`in_groups`](Samples::in_groups).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StratifiedGrouped {
    /// How many folds.
    pub k: usize,
}

impl StratifiedGrouped {
    /// The folds, in order.
    pub fn folds(&self, samples: &Samples) -> Result<Vec<Fold>, PartitionError> {
        let n = samples.n();
        let strata = samples.strata().ok_or(PartitionError::NeedsClasses)?;
        let groups = samples.groups().ok_or(PartitionError::NeedsGroups)?;
        checked(self.k, n)?;
        let classes = census(strata, self.k)?;

        let mut tests = vec![Vec::new(); self.k];
        let mut carried = vec![vec![0usize; classes.len()]; self.k];
        for (_, indices) in heaviest_first(grouped_by(groups, &in_order(n)), self.k)? {
            let mine = tally(&indices, strata, &classes);
            let fold = evenest(&carried, &mine, classes.len());
            for (class, count) in mine.iter().enumerate() {
                carried[fold][class] += count;
            }
            tests[fold].extend(indices);
        }
        Ok(assemble(n, tests))
    }
}

/// The classes present, and the check that each reaches every fold.
fn census(strata: &[u32], k: usize) -> Result<Vec<u32>, PartitionError> {
    let classes: Vec<u32> = strata
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    for &class in &classes {
        let count = strata.iter().filter(|&&c| c == class).count();
        if count < k {
            return Err(PartitionError::ClassTooSmall { class, count, k });
        }
    }
    Ok(classes)
}

/// How many of each class this group carries.
fn tally(indices: &[usize], strata: &[u32], classes: &[u32]) -> Vec<usize> {
    let mut counts = vec![0usize; classes.len()];
    for &index in indices {
        if let Some(class) = classes.iter().position(|&c| c == strata[index]) {
            counts[class] += 1;
        }
    }
    counts
}

/// The fold where putting this group leaves the classes most evenly spread.
/// Decided once per group and never revisited.
fn evenest(carried: &[Vec<usize>], mine: &[usize], classes: usize) -> usize {
    let mut best = 0;
    let mut best_spread = f64::MAX;
    for fold in 0..carried.len() {
        let mut spread = 0.0;
        for class in 0..classes {
            let shares: Vec<f64> = carried
                .iter()
                .enumerate()
                .map(|(other, counts)| {
                    let extra = if other == fold { mine[class] } else { 0 };
                    (counts[class] + extra) as f64
                })
                .collect();
            spread += deviation(&shares);
        }
        if spread < best_spread {
            best_spread = spread;
            best = fold;
        }
    }
    best
}

/// How far from equal a spread is. Standard deviation, unnormalised: only its
/// ordering is used.
fn deviation(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64).sqrt()
}

impl fmt::Display for StratifiedGrouped {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "stratified-grouped:{}", self.k)
    }
}

impl From<StratifiedGrouped> for Partition {
    fn from(cut: StratifiedGrouped) -> Self {
        Self::StratifiedGrouped(cut)
    }
}
