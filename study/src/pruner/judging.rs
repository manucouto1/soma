//! The mechanics every scheme shares: which report is being judged, the one
//! value that is not comparable to anything, and reading a quantile.

use super::Reason;

/// The report being judged — its index and its value — or `None` when nothing
/// has been reported yet. Nothing is ever pruned before its first report.
pub(super) fn latest(mine: &[f64]) -> Option<(usize, f64)> {
    mine.last().map(|&value| (mine.len() - 1, value))
}

/// A value that is not a number is not comparable to anything, and it is
/// exactly what a run that diverged reports. **Every scheme asks this first**,
/// including the ones that would otherwise never prune this early: a `NaN` loss
/// does not recover, and the epochs spent finding that out are the cheapest
/// ones a pruner can save.
pub(super) fn not_a_number(at: usize, value: f64) -> Option<Reason> {
    value.is_nan().then_some(Reason::NotANumber { at })
}

/// The `q`-th quantile of already-sorted values, `q` in `0.0..=1.0`,
/// interpolating between the two that straddle it — so the median of four
/// values is between the middle two and not one of them.
pub(super) fn quantile(sorted: &[f64], q: f64) -> f64 {
    let last = sorted.len() - 1;
    let place = q.clamp(0.0, 1.0) * last as f64;
    let below = place.floor() as usize;
    let above = place.ceil() as usize;
    if below == above {
        sorted[below]
    } else {
        sorted[below] + (sorted[above] - sorted[below]) * (place - below as f64)
    }
}
