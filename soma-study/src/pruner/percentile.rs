//! Judged against the trials that already finished.

use super::judging::{latest, not_a_number, quantile};
use super::{Pruner, Reason, Verdict};
use crate::Goal;
use std::fmt;

/// Prune what is doing worse at this step than `p` percent of the others.
///
/// **The median pruner is `p = 50`**, which is why there is no `Median` scheme
/// of its own — the original had both, and one of them was the other with a
/// number filled in. [`median`](Percentile::median) is a constructor.
///
/// What is compared is each trial's **best so far**, not its latest value: a
/// single bad epoch is noise, and a run that already touched a good number has
/// shown it can.
#[derive(Debug, Clone, PartialEq)]
pub struct Percentile {
    /// Between `0` and `100`, and it is **the share that is kept** — so a
    /// smaller one prunes more, which is the way round optuna reads it too.
    ///
    /// At `50` the better half survives. At `100` only what is behind *every*
    /// finished trial goes. At `0` everything but the best of them goes.
    pub p: f64,
    /// Which way is better.
    pub goal: Goal,
    /// No verdict before this many reports, however bad it looks. What buys a
    /// slow starter the epochs it needs.
    pub warmup: usize,
    /// No verdict until this many other trials have reached this step. Without
    /// it the first trial to finish becomes the bar for everybody.
    pub startup: usize,
}

impl Percentile {
    /// The median: `p = 50`.
    pub fn median(goal: Goal, warmup: usize, startup: usize) -> Self {
        Self {
            p: 50.0,
            goal,
            warmup,
            startup,
        }
    }

    /// Continue, or the reason not to.
    pub fn verdict(&self, mine: &[f64], others: &[Vec<f64>]) -> Verdict {
        let Some((at, value)) = latest(mine) else {
            return Verdict::Continue;
        };
        if let Some(why) = not_a_number(at, value) {
            return Verdict::Prune(why);
        }
        if mine.len() <= self.warmup {
            return Verdict::Continue;
        }

        // Only the ones that got this far: a trial that stopped at step 2 says
        // nothing about what is good at step 7.
        let mut bar: Vec<f64> = others
            .iter()
            .filter(|curve| curve.len() > at)
            .filter_map(|curve| self.goal.best_of(&curve[..=at]))
            .collect();
        if bar.len() < self.startup.max(1) {
            return Verdict::Continue;
        }
        bar.sort_by(|one, other| one.total_cmp(other));

        // From the good end, whichever end that is: at `p = 50` both readings
        // are the median, and the two directions stay symmetric away from it.
        let bar = quantile(
            &bar,
            match self.goal {
                Goal::Minimize => self.p / 100.0,
                Goal::Maximize => 1.0 - self.p / 100.0,
            },
        );

        match self.goal.best_of(mine) {
            Some(best) if self.goal.better(bar, best) => {
                Verdict::Prune(Reason::Worse { than: bar, at })
            }
            _ => Verdict::Continue,
        }
    }
}

impl fmt::Display for Percentile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "percentile:{}:{}:warmup:{}:startup:{}",
            self.p, self.goal, self.warmup, self.startup
        )
    }
}

impl From<Percentile> for Pruner {
    fn from(rule: Percentile) -> Self {
        Self::Percentile(rule)
    }
}
