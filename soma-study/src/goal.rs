//! Which way is better.
//!
//! A loss goes down and an accuracy goes up, and nothing in a number says which,
//! so everything at this level that compares two results has to be told. It
//! lives on the piece that compares rather than being passed to every call, so a
//! pruner without a direction is a state that cannot be written down.

use std::fmt;
use std::str::FromStr;

/// Which way is better.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Goal {
    /// Smaller is better: a loss, an error, a runtime.
    Minimize,
    /// Larger is better: an accuracy, an F1, a reward.
    Maximize,
}

impl Goal {
    /// Whether `one` is better than `than`. Strictly: equal is not better, so a
    /// trial that ties is never pruned for tying.
    pub fn better(&self, one: f64, than: f64) -> bool {
        match self {
            Self::Minimize => one < than,
            Self::Maximize => one > than,
        }
    }

    /// The best of them, or `None` if there are none. Values that are not
    /// numbers are skipped: they are not comparable to anything.
    pub fn best_of(&self, values: &[f64]) -> Option<f64> {
        values
            .iter()
            .copied()
            .filter(|v| !v.is_nan())
            .reduce(|best, v| if self.better(v, best) { v } else { best })
    }
}

impl FromStr for Goal {
    type Err = GoalError;

    /// `min` or `max`, the way it is written down. A typo is caught where it
    /// was typed rather than becoming a search that optimised backwards.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "min" => Ok(Self::Minimize),
            "max" => Ok(Self::Maximize),
            _ => Err(GoalError::Unknown(s.to_string())),
        }
    }
}

/// Why that does not say which way is better.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GoalError {
    /// We do not know that direction.
    Unknown(String),
}

impl fmt::Display for GoalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown(said) => write!(
                f,
                "`{said}` does not say which way is better: write `min` for a loss                  or `max` for an accuracy"
            ),
        }
    }
}

impl std::error::Error for GoalError {}

impl fmt::Display for Goal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Minimize => f.write_str("min"),
            Self::Maximize => f.write_str("max"),
        }
    }
}
