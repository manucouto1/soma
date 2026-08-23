//! Whether what happened is healthy. **An opinion, and it says so.**
//!
//! The third of the three things CU19 split observability into. The first is
//! the declaration drawn, the second is the record of what happened, and this
//! is a judgement **about** that record — with thresholds somebody chose, which
//! is exactly what makes it not a fact.
//!
//! The invariant that keeps the split real, and it is a test rather than an
//! aspiration:
//!
//! > **A diagnosis has to be reproducible from the stored record, without
//! > training again.**
//!
//! Which is why this crate has no dependencies and touches nothing: numbers in,
//! [`Flag`]s out. The statistics are measured where torch is, cross as numbers,
//! and are judged here. Change a threshold and the same record answers again —
//! and an alarm you cannot re-ask is an alarm you cannot argue with.
//!
//! # Two questions, and they are not the same one
//!
//! [`verdict`] asks whether a network is **learning**: gradients, activations,
//! channels, the update. [`leaning`] asks whether it is learning **what you
//! think** — which no amount of looking at a gradient will ever say, and which
//! cost a real research project months to find out the hard way.
//!
//! # What it is not allowed to do
//!
//! It does not measure. It does not decide **when** to look, or how often, or
//! at what grain: that is the caller's, and a verdict that quietly needed a
//! particular cadence would be a threshold hiding in a schedule.
//!
//! # Where the numbers come from
//!
//! | family | what it reads | who measured it |
//! |---|---|---|
//! | gradients | the L2 norm of a node's parameter gradients | a torch hook |
//! | activations | how much of the output is zero, how much is saturated | a torch hook |
//! | channels | per-channel means over a window | a torch hook |
//! | representation | effective rank, CKA between declared groups | a torch hook |
//! | the update | the stable rank of `W_t - W_{t-d}` | a torch hook |
//!
//! # The one rule about reducing a window
//!
//! **`Dead` and `Saturated` read the maximum and not the mean.** A layer that
//! dies one step in four is dead, and the average is exactly what hides it. It
//! is the original soma's finding, it is written down in its own source, and it
//! is the kind of thing that is obvious only once somebody has been bitten.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod flag;
mod leaning;
mod seen;
mod thresholds;

pub use flag::Flag;
pub use leaning::{Contribution, leaning, shares};
pub use seen::Seen;
pub use thresholds::Thresholds;

/// Everything wrong with what was seen, in the order it is worth reading.
///
/// Empty means nothing tripped, which is not the same as healthy: a metric
/// nobody measured cannot flag, and [`Seen`] says which those were.
///
/// The order is deliberate — what stops a run first comes first. A `NaN` makes
/// every number below it meaningless, so it is read before them.
pub fn verdict(seen: &Seen, thresholds: &Thresholds) -> Vec<Flag> {
    let mut flags = Vec::new();
    if seen.nan {
        flags.push(Flag::Nan);
    }
    if seen.inf {
        flags.push(Flag::Inf);
    }
    if let Some(norm) = seen.grad_norm {
        if norm < thresholds.grad_low {
            flags.push(Flag::Vanishing);
        } else if norm > thresholds.grad_high {
            flags.push(Flag::Exploding);
        }
    }
    // The maximum over the window, never the mean.
    if seen.zero_frac_max.is_some_and(|f| f > thresholds.dead_frac) {
        flags.push(Flag::Dead);
    }
    if seen
        .sat_frac_max
        .is_some_and(|f| f > thresholds.saturated_frac)
    {
        flags.push(Flag::Saturated);
    }
    if let Some(ratio) = seen.update_ratio {
        if ratio < thresholds.update_low {
            flags.push(Flag::Stalled);
        } else if ratio > thresholds.update_high {
            flags.push(Flag::Overstepping);
        }
    }
    if seen.dead_channels > 0 {
        flags.push(Flag::DeadChannels(seen.dead_channels));
    }
    if seen.ignored_channels > 0 {
        flags.push(Flag::IgnoredChannels(seen.ignored_channels));
    }
    if seen
        .group_cka
        .is_some_and(|cka| cka > thresholds.leakage_cka)
    {
        flags.push(Flag::Leakage);
    }
    if narrowing(seen, thresholds) {
        flags.push(Flag::Narrowing);
    }
    if losing_plasticity(seen, thresholds) {
        flags.push(Flag::LosingPlasticity);
    }
    flags
}

/// Whether the update has collapsed into a few directions **relative to what
/// this run was doing before**.
///
/// Huang et al. (2026) monitor the spectrum of `dW = W_t - W_{t-d}` and find it
/// collapses thousands of steps before the loss does — *"loss, gradient norms
/// and weight norms are the most delayed indicators"*. Their certificate is the
/// deviation from a **healthy baseline run**, which nobody watching one
/// training run has, so this compares the update against its own recent median
/// instead.
///
/// That substitution was measured and it does not hold: see
/// [`Thresholds::narrowing_of_usual`], which is `0.0` by default and therefore
/// never fires. The metric is still recorded and drawn — a collapse is visible
/// to somebody looking at the curve, and saying that is a weaker claim than an
/// alarm, which is the point.
///
/// Silent, too, until there is a history to compare against: a run that has not
/// established what it usually does cannot be said to have departed from it.
fn narrowing(seen: &Seen, thresholds: &Thresholds) -> bool {
    if thresholds.narrowing_of_usual <= 0.0 {
        return false;
    }
    let (Some(rank), Some(usual)) = (seen.update_rank, seen.update_rank_usual) else {
        return false;
    };
    usual > 0.0 && rank / usual < thresholds.narrowing_of_usual
}

/// Whether the network is losing its ability to learn anything new.
///
/// **A conjunction, and that is the point.** Dohare et al. (2024) tie plasticity
/// loss to three things together — parameter norms rising, units going dormant,
/// and the rank of the representation falling — and any one of them alone is
/// ordinary. A network whose weights grow is training; one with some dormant
/// units is a ReLU network. All three at once, and pointing the same way, is
/// the finding.
fn losing_plasticity(seen: &Seen, thresholds: &Thresholds) -> bool {
    let (Some(weights), Some(rank), Some(dormant)) = (
        seen.param_norm_slope,
        seen.eff_rank_slope,
        seen.dormancy_frac,
    ) else {
        return false;
    };
    weights > thresholds.plasticity_growth
        && rank < -thresholds.plasticity_growth
        && dormant > thresholds.dormant_frac
}
