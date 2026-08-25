//! `k` growing prefixes, so nothing is ever trained on its own future.

use super::{Fold, Partition, PartitionError};
use crate::Samples;
use std::fmt;

/// The one scheme that is deliberately **not** a partition: the first block has
/// nothing before it to learn from, so it only ever trains and is never held
/// out. Every other scheme here holds out each sample exactly once.
///
/// `gap` drops that many samples between the two sides, which is what purged
/// and embargoed cross-validation are — a parameter, not a scheme. It uses
/// neither the classes nor the groups, and unlike the rest `k = 1` is
/// meaningful: a plain holdout of the tail.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TimeSeries {
    /// How many splits.
    pub k: usize,
    /// How many samples to drop between training and test.
    pub gap: usize,
}

impl TimeSeries {
    /// The folds, oldest first.
    pub fn folds(&self, samples: &Samples) -> Result<Vec<Fold>, PartitionError> {
        let n = samples.n();
        if self.k == 0 {
            return Err(PartitionError::TooFewFolds { k: self.k });
        }
        // One more part than splits: the first one only ever trains.
        let size = n / (self.k + 1);
        if size == 0 {
            return Err(PartitionError::MoreFoldsThanSamples { k: self.k, n });
        }
        (0..self.k)
            .map(|i| {
                let start = n - (self.k - i) * size;
                if start <= self.gap {
                    return Err(PartitionError::GapTooLarge {
                        gap: self.gap,
                        k: self.k,
                    });
                }
                Ok(Fold {
                    train: (0..start - self.gap).collect(),
                    test: (start..start + size).collect(),
                })
            })
            .collect()
    }
}

impl fmt::Display for TimeSeries {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.gap {
            0 => write!(f, "timeseries:{}", self.k),
            gap => write!(f, "timeseries:{}:gap:{gap}", self.k),
        }
    }
}

impl From<TimeSeries> for Partition {
    fn from(cut: TimeSeries) -> Self {
        Self::TimeSeries(cut)
    }
}
