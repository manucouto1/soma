//! Which of the schemes a cut is, when that is decided by data rather than in
//! the source.
//!
//! Each scheme is a type of its own with its own `folds`, in its own file. When
//! you know which one you want, say it and the enum is not in the way:
//!
//! ```
//! use soma_next_study::{KFold, Samples};
//!
//! let folds = KFold { k: 5, shuffle: None }.folds(&Samples::of(100))?;
//! # Ok::<(), soma_next_study::PartitionError>(())
//! ```
//!
//! [`Partition`] is for the other case, which is the common one at this level:
//! the scheme arrives from a Python call, a configuration file or the record of
//! a trial. There, the type is not known when this compiles.
//!
//! # Why the family is an enum and each scheme a struct
//!
//! The dispatch is static either way — `Self::KFold(cut) => cut.folds(samples)`
//! resolves to a concrete function with no vtable. What the enum adds is
//! everything that happens at the edges:
//!
//! - **The name is structural, not a convention.** A cut is part of a cache key
//!   (CU13), and with a trait the name would be supplied by the implementor:
//!   two that collide, or one that changes between versions, gives the wrong
//!   fold back **in silence**. Here it is derived, and there is a test that
//!   two cuts which differ are written differently.
//! - **It comes back from a record.** To deserialize you must name the type
//!   when it compiles, and the type is inside the JSON. Without the enum that
//!   is a `match` on strings — the same `match`, minus the compiler checking
//!   it is complete — written once per consumer instead of once here.
//! - **A new scheme stops compiling in three places, and they are listed.**
//!   With a trait it compiles, and what you forgot is the registration.
//!
//! What is deliberately **not** here is an `Explicit { folds }` escape hatch for
//! a scheme nobody has written yet: it costs three lines the day someone needs
//! it, and the indices hash themselves.

mod dealing;
mod grouped;
mod kfold;
mod stratified;
mod stratified_grouped;
mod time_series;

pub use grouped::Grouped;
pub use kfold::KFold;
pub use stratified::Stratified;
pub use stratified_grouped::StratifiedGrouped;
pub use time_series::TimeSeries;

use crate::Samples;
use std::fmt;

/// One cut: who trains and who is held out. Both sides in ascending order —
/// shuffling decides **who** is in each fold, never the order they are listed
/// in, so a fold reads the same whatever produced it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fold {
    /// The indices to train on.
    pub train: Vec<usize>,
    /// The indices held out.
    pub test: Vec<usize>,
}

/// Whichever of the schemes a cut is.
///
/// ```
/// use soma_next_study::{Partition, Samples, Stratified};
///
/// let cut: Partition = Stratified { k: 5, shuffle: None }.into();
/// assert_eq!(cut.to_string(), "stratified:5");
/// # Ok::<(), soma_next_study::PartitionError>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Partition {
    /// [`KFold`]: the plain cut.
    KFold(KFold),
    /// [`Stratified`]: every class keeps its share.
    Stratified(Stratified),
    /// [`Grouped`]: a group never splits.
    Grouped(Grouped),
    /// [`StratifiedGrouped`]: both, as far as both can be had.
    StratifiedGrouped(StratifiedGrouped),
    /// [`TimeSeries`]: growing prefixes.
    TimeSeries(TimeSeries),
}

impl Partition {
    /// The folds, in order — whichever scheme this is.
    ///
    /// Everything that cannot be honoured is an error **here**, before a single
    /// index comes out: too few folds, more folds than samples, a class or a
    /// group that cannot reach every fold. None of it is a warning, because a
    /// silently degraded cut is a result you cannot tell from a good one.
    pub fn folds(&self, samples: &Samples) -> Result<Vec<Fold>, PartitionError> {
        match self {
            Self::KFold(cut) => cut.folds(samples),
            Self::Stratified(cut) => cut.folds(samples),
            Self::Grouped(cut) => cut.folds(samples),
            Self::StratifiedGrouped(cut) => cut.folds(samples),
            Self::TimeSeries(cut) => cut.folds(samples),
        }
    }

    /// How many folds it produces, without producing them.
    pub fn k(&self) -> usize {
        match self {
            Self::KFold(cut) => cut.k,
            Self::Stratified(cut) => cut.k,
            Self::Grouped(cut) => cut.k,
            Self::StratifiedGrouped(cut) => cut.k,
            Self::TimeSeries(cut) => cut.k,
        }
    }
}

impl fmt::Display for Partition {
    /// As text, which is the form that goes into a cache key and into the record
    /// of a trial. Written by the scheme itself, because the name belongs with
    /// the thing it names.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KFold(cut) => cut.fmt(f),
            Self::Stratified(cut) => cut.fmt(f),
            Self::Grouped(cut) => cut.fmt(f),
            Self::StratifiedGrouped(cut) => cut.fmt(f),
            Self::TimeSeries(cut) => cut.fmt(f),
        }
    }
}

/// `name:k`, and the seed only when there is one. Shared by the two schemes
/// that take one, so their text cannot drift apart.
pub(super) fn scheme(
    f: &mut fmt::Formatter<'_>,
    name: &str,
    k: usize,
    shuffle: Option<u64>,
) -> fmt::Result {
    match shuffle {
        None => write!(f, "{name}:{k}"),
        Some(seed) => write!(f, "{name}:{k}:shuffled:{seed}"),
    }
}

/// Why the samples cannot be cut that way.
///
/// One type for the five schemes, and not one each: they are the ways a **cut**
/// fails, and which scheme was asked for is already in the message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PartitionError {
    /// Fewer than two folds.
    TooFewFolds {
        /// What was asked for.
        k: usize,
    },
    /// More folds than there are samples to put in them.
    MoreFoldsThanSamples {
        /// What was asked for.
        k: usize,
        /// How many samples there are.
        n: usize,
    },
    /// Stratifying was asked for and no class was said.
    NeedsClasses,
    /// Grouping was asked for and no group was said.
    NeedsGroups,
    /// A class with fewer members than folds cannot be in every fold.
    ClassTooSmall {
        /// Which one.
        class: u32,
        /// How many it has.
        count: usize,
        /// How many folds.
        k: usize,
    },
    /// Fewer groups than folds: one fold would get nothing.
    TooFewGroups {
        /// How many distinct groups there are.
        groups: usize,
        /// How many folds.
        k: usize,
    },
    /// The gap eats the whole of the first fold's training set.
    GapTooLarge {
        /// What was asked for.
        gap: usize,
        /// How many splits.
        k: usize,
    },
}

impl fmt::Display for PartitionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewFolds { k } => write!(
                f,
                "{k} folds is not a cut: with one, everything is held out and nothing trains"
            ),
            Self::MoreFoldsThanSamples { k, n } => write!(
                f,
                "{k} folds over {n} samples leaves folds with nothing in them"
            ),
            Self::NeedsClasses => f.write_str(
                "stratifying needs the class of each sample: `Samples::of(n).by_class(…)`",
            ),
            Self::NeedsGroups => f.write_str(
                "grouping needs the group of each sample: `Samples::of(n).in_groups(…)`",
            ),
            Self::ClassTooSmall { class, count, k } => write!(
                f,
                "class `{class}` has {count} samples and there are {k} folds: it cannot be in \
                 all of them. Either fewer folds, or do not stratify by it"
            ),
            Self::TooFewGroups { groups, k } => write!(
                f,
                "{groups} groups over {k} folds: a group does not split, so one fold gets nothing"
            ),
            Self::GapTooLarge { gap, k } => write!(
                f,
                "a gap of {gap} leaves the first of {k} splits with nothing to train on"
            ),
        }
    }
}

impl std::error::Error for PartitionError {}
