//! What a model is **leaning on** — which is not the same question as whether
//! it is healthy.
//!
//! A network can pass every check in [`verdict`](crate::verdict) and be learning
//! the wrong thing. The gradients arrive, the units are alive, the loss goes
//! down, and the signal it found is not the one anybody meant to give it.
//!
//! # Where this came from
//!
//! From a real research project, and it is the reason this exists at all:
//! symptom channels for detecting a mental-health condition, where
//! interpretability and performance could be had one at a time and never
//! together. Months went into diagnosing the architecture. The problem was in
//! the data — the predictive signal was in the **self-disclosure** and not in
//! the presence of symptoms — and no amount of looking at gradients was ever
//! going to say so.
//!
//! What says so is cheap: take one input away and score it again. If the
//! channel you built the whole study around costs nothing to remove, it is not
//! what the model is using.
//!
//! # What a contribution is
//!
//! The score with an input **shuffled** minus the score with it intact, as a
//! share of what all of them together are worth. Shuffled rather than zeroed,
//! because a zero is a value — often an unusually informative one — and a
//! network that has never seen one will fall over for the wrong reason.
//!
//! It is a **ranking** and not an attribution: it says this input orders the
//! results and not that it is worth so many points. Two inputs that carry the
//! same signal both look unimportant, because removing either leaves the other,
//! and that is a true thing about the data rather than a flaw in the method.

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

/// What is wrong with what a model is leaning on.
///
/// Empty is not a clean bill: with one input there is nothing to compare, and
/// with none there is nothing to say.
///
/// The two findings are opposite ends of the same worry. An input that costs
/// nothing to remove is one the model is not using — which, when it is the
/// input the research was about, is the whole answer. An input that carries
/// everything is a model with one leg, and the day that channel is missing or
/// shifts, nothing else takes over.
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

/// Turns raw drops into shares, which is the only arithmetic here.
///
/// Divided by the **total** and not by the largest, so the numbers add up to
/// one and can be read as *how much of what matters is this*. When nothing
/// matters — every drop zero or negative — every share is zero rather than a
/// division by nothing, and `IGNORED_INPUT` fires on all of them, which is
/// exactly right: a model that loses nothing whatever you take away is not
/// using its inputs.
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
