//! Whether what happened is healthy. **An opinion, and it says so.**
//!
//! The third of the three things observability splits into: the declaration
//! drawn, the record of what happened, and a judgement **about** that record —
//! with thresholds somebody chose, which is what makes it not a fact.
//!
//! > A diagnosis has to be reproducible from the stored record, without training
//! > again.
//!
//! Which is why this crate has no dependencies and touches nothing: numbers in,
//! [`Flag`]s out. The statistics are measured where torch is, cross as numbers,
//! and are judged here — so changing a threshold costs a scan, and an alarm you
//! cannot re-ask is one you cannot argue with.
//!
//! [`verdict`] asks whether a network is **learning**; [`leaning`] asks whether
//! it is learning **what you think**, which no gradient will ever say.
//!
//! It does not measure, and it does not decide when to look or how often: a
//! verdict that quietly needed a particular cadence would be a threshold hiding
//! in a schedule.
//!
//! **`Dead` and `Saturated` read the maximum and not the mean.** A layer that
//! dies one step in four is dead, and the average is what hides it. It is the
//! original's finding, and obvious only once somebody has been bitten.

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

/// Everything wrong with what was seen, in the order it is worth reading. Empty
/// means nothing tripped, which is not the same as healthy — a metric nobody
/// measured cannot flag, and [`Seen`] says which those were. What stops a run
/// first comes first: a `NaN` makes every number below it meaningless.
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
    // One-sided: growing is the half that was measured to separate. See
    // `Thresholds::gain_drift`, and `health/tests/normalisation.py` for why
    // there is no bound underneath it.
    if seen
        .signal_gain
        .is_some_and(|gain| gain > thresholds.gain_drift)
    {
        flags.push(Flag::MissingNormalisation);
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
/// collapses thousands of steps before the loss does. Their certificate is the
/// deviation from a healthy baseline run, which nobody watching one run has, so
/// this compares against its own recent median instead — a substitution that was
/// measured and does **not** hold. Hence
/// [`Thresholds::narrowing_of_usual`] is `0.0` and this never fires: the metric
/// is recorded and drawn, which is a weaker claim than an alarm.
fn narrowing(seen: &Seen, thresholds: &Thresholds) -> bool {
    if thresholds.narrowing_of_usual <= 0.0 {
        return false;
    }
    let (Some(rank), Some(usual)) = (seen.update_rank, seen.update_rank_usual) else {
        return false;
    };
    usual > 0.0 && rank / usual < thresholds.narrowing_of_usual
}

/// Whether the network is losing its ability to learn anything new. A
/// conjunction, and that is the point: Dohare et al. (2024) tie plasticity loss
/// to parameter norms rising, units going dormant and the rank falling **at
/// once**. Any one alone is a network that is training.
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
