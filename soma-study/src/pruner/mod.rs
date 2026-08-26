//! Whether a trial that is going badly is worth another epoch.
//!
//! Each scheme is a type of its own with its own `verdict`, and [`Pruner`] is the
//! family for when the scheme arrives as data.
//!
//! | scheme | judged against | needs other trials |
//! |---|---|---|
//! | [`Percentile`] | **the others** at the same step | yes |
//! | [`Threshold`] | **a constant** you already know is hopeless | no |
//! | [`Patience`] | **itself**: it has stopped improving | no |
//!
//! `Median` is not a fourth: it is [`Percentile`] with `p = 50`. Successive
//! halving and Hyperband are deliberately not here — they are not verdicts on a
//! trial but a way of handing budget out across the whole population, which is
//! the shape of the loop, and the loop belongs to whoever writes it.
//!
//! A pruner does not stop anything. It answers, and **the loop stops calling**:
//!
//! ```python
//! for epoch in range(50):
//!     reported.append(trainer.fit(data, epochs=1).loss)
//!     if why := pruner.verdict(reported, finished):
//!         break
//! ```
//!
//! So this adds **zero lines to level 2**. A trainer that had to be told to stop
//! would be a callback crossing the boundary.

mod judging;
mod patience;
mod percentile;
mod threshold;

pub use patience::Patience;
pub use percentile::Percentile;
pub use threshold::Threshold;

use std::fmt;

/// What a pruner answers.
#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    /// Worth another epoch.
    Continue,
    /// Not, and why.
    Prune(Reason),
}

impl Verdict {
    /// Whether this trial is to be dropped.
    pub fn is_prune(&self) -> bool {
        matches!(self, Self::Prune(_))
    }

    /// Why, or `None` if it is to carry on.
    pub fn reason(&self) -> Option<&Reason> {
        match self {
            Self::Continue => None,
            Self::Prune(why) => Some(why),
        }
    }
}

/// Why a trial is not worth another epoch. Structured and not a string, because
/// *how many were pruned, and for which of the three reasons* is the question
/// you ask of a search that pruned too much.
#[derive(Debug, Clone, PartialEq)]
pub enum Reason {
    /// It reported something that is not a number. Every scheme prunes this.
    NotANumber {
        /// Which report.
        at: usize,
    },
    /// Worse at this step than the bar the others set.
    Worse {
        /// The bar it did not clear.
        than: f64,
        /// Which report.
        at: usize,
    },
    /// Outside the bounds that were declared hopeless.
    OutOfBounds {
        /// What it reported.
        value: f64,
        /// The bound it crossed.
        bound: f64,
    },
    /// It has not improved on its own best for long enough.
    NotImproving {
        /// The report its best is from.
        since: usize,
        /// How many without an improvement were allowed.
        steps: usize,
    },
}

impl fmt::Display for Reason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotANumber { at } => {
                write!(f, "report {at} is not a number: this one diverged")
            }
            Self::Worse { than, at } => write!(
                f,
                "at report {at} it is behind the bar the finished trials set, {than}"
            ),
            Self::OutOfBounds { value, bound } => {
                write!(
                    f,
                    "{value} is past the bound {bound} that was called hopeless"
                )
            }
            Self::NotImproving { since, steps } => write!(
                f,
                "its best is still the one from report {since}, and {steps} without an \
                 improvement was the allowance"
            ),
        }
    }
}

/// Whichever of the schemes a pruner is.
#[derive(Debug, Clone, PartialEq)]
pub enum Pruner {
    /// [`Percentile`]: behind the others.
    Percentile(Percentile),
    /// [`Threshold`]: past a bound.
    Threshold(Threshold),
    /// [`Patience`]: going nowhere.
    Patience(Patience),
}

impl Pruner {
    /// Continue, or the reason not to. `mine` is what this trial has reported so
    /// far, in order; `others` the same for those that finished. A step is **the
    /// n-th report**, so trials have to report on the same schedule — true of
    /// every pruner that compares across trials, optuna's included.
    pub fn verdict(&self, mine: &[f64], others: &[Vec<f64>]) -> Verdict {
        match self {
            Self::Percentile(rule) => rule.verdict(mine, others),
            Self::Threshold(rule) => rule.verdict(mine, others),
            Self::Patience(rule) => rule.verdict(mine, others),
        }
    }
}

impl fmt::Display for Pruner {
    /// As text, which is the form that goes into the record of a run. Written by
    /// the scheme itself, because the name belongs with the thing it names.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Percentile(rule) => rule.fmt(f),
            Self::Threshold(rule) => rule.fmt(f),
            Self::Patience(rule) => rule.fmt(f),
        }
    }
}
