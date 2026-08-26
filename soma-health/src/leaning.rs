//! What a model is **leaning on** — not the same question as whether it is
//! healthy. A network can pass every check in [`verdict`](crate::verdict) and be
//! learning the wrong thing.
//!
//! It comes from a real project: symptom channels for detecting a mental-health
//! condition, months spent on the architecture, and the predictive signal was in
//! the **self-disclosure** and not in the presence of symptoms. No amount of
//! looking at gradients was ever going to say so. What says so is cheap: take
//! one input away and score it again.
//!
//! A contribution is the score with an input **shuffled** minus the score with
//! it intact, as a share of what all of them are worth. Shuffled and not zeroed,
//! because a zero is a value — often an unusually informative one.
//!
//! It is a **ranking** and not an attribution. Two inputs carrying the same
//! signal both look unimportant, because removing either leaves the other, and
//! that is a true thing about the data rather than a flaw in the method.

use crate::{Flag, Thresholds};

/// What one input turned out to be worth.
#[derive(Debug, Clone, PartialEq)]
pub struct Contribution {
    /// What it is called — the key of the input, or the node that reads it.
    pub name: String,
    /// How much worse the score gets without it, as a share of the total drop
    /// across every input. Between `0.0` and `1.0` when the drops are positive;
    /// a **negative** one is real and means the model does better without that
    /// input, which is worth seeing rather than clamping away.
    pub share: f64,
    /// And the raw difference, in whatever the score was measured in.
    pub drop: f64,
}

/// What is wrong with what a model is leaning on. Empty is not a clean bill:
/// with one input there is nothing to compare.
///
/// The two findings are opposite ends of one worry. An input that costs nothing
/// to remove is one the model is not using; an input that carries everything is
/// a model with one leg.
pub fn leaning(shares: &[Contribution], thresholds: &Thresholds) -> Vec<Flag> {
    if shares.len() < 2 {
        return Vec::new();
    }
    let mut flags = Vec::new();
    for one in shares {
        if one.share < thresholds.ignored_input {
            flags.push(Flag::IgnoredInput(one.name.clone()));
        }
    }
    if let Some(most) = shares.iter().max_by(|a, b| {
        a.share
            .partial_cmp(&b.share)
            .unwrap_or(std::cmp::Ordering::Equal)
    }) && most.share > thresholds.sole_reliance
    {
        flags.push(Flag::SoleReliance(most.name.clone()));
    }
    flags
}

/// Turns raw drops into shares, which is the only arithmetic here. Divided by
/// the **total** and not the largest, so they add up to one. When nothing
/// matters, every share is zero rather than a division by nothing, and
/// `IGNORED_INPUT` fires on all of them — which is right.
pub fn shares(drops: &[(String, f64)]) -> Vec<Contribution> {
    let total: f64 = drops.iter().map(|(_, drop)| drop.max(0.0)).sum();
    drops
        .iter()
        .map(|(name, drop)| Contribution {
            name: name.clone(),
            share: if total > 0.0 { drop / total } else { 0.0 },
            drop: *drop,
        })
        .collect()
}
