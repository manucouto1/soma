//! Judged against itself.

use super::judging::{latest, not_a_number};
use super::{Pruner, Reason, Verdict};
use crate::Goal;
use std::fmt;
use std::num::NonZeroUsize;

/// Prune what has stopped improving on its own best. Early stopping, and the
/// third thing a verdict can be measured against: the others, a constant,
/// **itself**. Unlike both of the others it can prune a run that is doing
/// perfectly well in the field and simply is not going anywhere.
#[derive(Debug, Clone, PartialEq)]
pub struct Patience {
    /// How many reports without an improvement before it goes.
    ///
    /// Non-zero because zero patience would prune every trial at its first
    /// report, improvement or not. Made impossible rather than validated.
    pub steps: NonZeroUsize,
    /// How much counts as an improvement. `0.0` means any at all, which makes
    /// noise look like progress.
    pub min_delta: f64,
    /// Which way is better.
    pub goal: Goal,
}

impl Patience {
    /// Continue, or the reason not to. It ignores the other trials on purpose.
    pub fn verdict(&self, mine: &[f64], _others: &[Vec<f64>]) -> Verdict {
        let Some((at, value)) = latest(mine) else {
            return Verdict::Continue;
        };
        if let Some(why) = not_a_number(at, value) {
            return Verdict::Prune(why);
        }

        let mut best = f64::NAN;
        let mut since = 0;
        for (step, &value) in mine.iter().enumerate() {
            // The first report there is has nothing to beat, so it counts.
            if best.is_nan() || self.goal.better(value, best + self.moved()) {
                best = value;
                since = step;
            }
        }
        if at - since >= self.steps.get() {
            Verdict::Prune(Reason::NotImproving {
                since,
                steps: self.steps.get(),
            })
        } else {
            Verdict::Continue
        }
    }

    /// How far the best has to move to count, in the direction that is better.
    fn moved(&self) -> f64 {
        match self.goal {
            Goal::Minimize => -self.min_delta,
            Goal::Maximize => self.min_delta,
        }
    }
}

impl fmt::Display for Patience {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "patience:{}:delta:{}:{}",
            self.steps, self.min_delta, self.goal
        )
    }
}

impl From<Patience> for Pruner {
    fn from(rule: Patience) -> Self {
        Self::Patience(rule)
    }
}
