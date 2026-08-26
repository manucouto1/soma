//! Judged against a constant.

use super::judging::{latest, not_a_number};
use super::{Pruner, Reason, Verdict};
use std::fmt;

/// Prune what leaves the bounds you already know are hopeless. The only scheme
/// that needs **no other trial**, so it works on the very first one — where a
/// diverged configuration costs most and the other two have nothing to compare
/// against. With neither bound it still prunes what is not a number, which is
/// what [`diverged`](Threshold::diverged) is.
#[derive(Debug, Clone, PartialEq)]
pub struct Threshold {
    /// Below this is hopeless. `None` for no floor.
    pub lower: Option<f64>,
    /// Above this is hopeless. `None` for no ceiling.
    pub upper: Option<f64>,
}

impl Threshold {
    /// Only what blew up: no bounds, so nothing goes but a value that is not a
    /// number.
    pub fn diverged() -> Self {
        Self {
            lower: None,
            upper: None,
        }
    }

    /// Continue, or the reason not to. It ignores the other trials on purpose.
    pub fn verdict(&self, mine: &[f64], _others: &[Vec<f64>]) -> Verdict {
        let Some((at, value)) = latest(mine) else {
            return Verdict::Continue;
        };
        if let Some(why) = not_a_number(at, value) {
            return Verdict::Prune(why);
        }
        if let Some(bound) = self.lower.filter(|&bound| value < bound) {
            return Verdict::Prune(Reason::OutOfBounds { value, bound });
        }
        if let Some(bound) = self.upper.filter(|&bound| value > bound) {
            return Verdict::Prune(Reason::OutOfBounds { value, bound });
        }
        Verdict::Continue
    }
}

impl fmt::Display for Threshold {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "threshold:lower:{}:upper:{}",
            said(self.lower),
            said(self.upper)
        )
    }
}

/// A bound, or that there is none. Written out either way so the form cannot be
/// read two ways.
fn said(bound: Option<f64>) -> String {
    match bound {
        None => "none".to_string(),
        Some(bound) => bound.to_string(),
    }
}

impl From<Threshold> for Pruner {
    fn from(rule: Threshold) -> Self {
        Self::Threshold(rule)
    }
}
