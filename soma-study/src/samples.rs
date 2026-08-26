//! What is known about the samples being cut, which is never the samples.
//!
//! A [`Partition`](crate::Partition) is given how many there are and, when the
//! cut needs it, one `u32` per sample for its class or its group. The labels are
//! a tensor and the tensor stays in Python: what crosses is the class **as a
//! number**. That is the whole reason cross-validation can be Rust without the
//! core ever learning what a dataset is.

use std::fmt;

/// How many samples there are, and what is known about each. Built up, because
/// most cuts need only the count:
///
/// ```
/// use somatize_study::Samples;
///
/// let plain = Samples::of(100);
/// let labelled = Samples::of(6).by_class(vec![0, 0, 1, 1, 0, 1])?;
/// let both = Samples::of(6)
///     .by_class(vec![0, 0, 1, 1, 0, 1])?
///     .in_groups(vec![7, 7, 8, 8, 9, 9])?;
/// # Ok::<(), somatize_study::SamplesError>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Samples {
    n: usize,
    strata: Option<Vec<u32>>,
    groups: Option<Vec<u32>>,
}

impl Samples {
    /// `n` samples, with nothing known about any of them.
    pub fn of(n: usize) -> Self {
        Self {
            n,
            strata: None,
            groups: None,
        }
    }

    /// The class of each sample, in the same order. What
    /// [`Stratified`](crate::Partition::Stratified) honours.
    ///
    /// The values are opaque: they are compared, never ordered or counted on to
    /// start at zero. `0`/`1` and `31337`/`4` cut the same.
    pub fn by_class(mut self, strata: Vec<u32>) -> Result<Self, SamplesError> {
        self.check("classes", strata.len())?;
        self.strata = Some(strata);
        Ok(self)
    }

    /// The group of each sample, in the same order. What
    /// [`Grouped`](crate::Partition::Grouped) keeps whole: two samples with the
    /// same group never land on opposite sides of a fold.
    pub fn in_groups(mut self, groups: Vec<u32>) -> Result<Self, SamplesError> {
        self.check("groups", groups.len())?;
        self.groups = Some(groups);
        Ok(self)
    }

    /// How many there are.
    pub fn n(&self) -> usize {
        self.n
    }

    /// The class of each, if it was said.
    pub fn strata(&self) -> Option<&[u32]> {
        self.strata.as_deref()
    }

    /// The group of each, if it was said.
    pub fn groups(&self) -> Option<&[u32]> {
        self.groups.as_deref()
    }

    /// One key per sample, or the mismatch that says which one is short.
    fn check(&self, what: &'static str, given: usize) -> Result<(), SamplesError> {
        if given == self.n {
            Ok(())
        } else {
            Err(SamplesError::Mismatch {
                what,
                given,
                n: self.n,
            })
        }
    }
}

/// Why that is not one key per sample.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SamplesError {
    /// As many keys as samples, and there were not.
    Mismatch {
        /// `classes` or `groups`.
        what: &'static str,
        /// How many arrived.
        given: usize,
        /// How many there are.
        n: usize,
    },
}

impl fmt::Display for SamplesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mismatch { what, given, n } => write!(
                f,
                "{given} {what} for {n} samples: there has to be exactly one per sample, \
                 in the same order"
            ),
        }
    }
}

impl std::error::Error for SamplesError {}
