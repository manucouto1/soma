//! What can be wrong. The vocabulary of a diagnosis.

use std::fmt;

/// One thing that looks wrong with a node.
///
/// An enum because the set is closed and named: a diagnosis is only useful if
/// two runs of it say the same word for the same thing, and a free-text
/// complaint is a thing nobody can filter on. When one is added the compiler
/// finds every `match` — including whoever draws them.
///
/// Every variant here is an **opinion at a threshold**, and none of them is a
/// fact. The facts are in the record; these are what somebody thinks of them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Flag {
    /// A number stopped being a number. Nothing below it means anything.
    Nan,
    /// Or stopped being finite.
    Inf,
    /// The parameter gradients are so small this node is not being trained.
    ///
    /// The classic depth pathology: with unit-gain init through a saturating
    /// non-linearity the backpropagated signal shrinks geometrically, so it is
    /// the **early** layers that go quiet while the last one still learns.
    /// Vanishing is a profile over depth, not a property of a network.
    Vanishing,
    /// The parameter gradients are so large the next step will not be a step.
    Exploding,
    /// Most of what this node outputs is zero, on at least one step.
    ///
    /// Read off the **maximum** over the window: a layer that dies one step in
    /// four is dead, and the mean is exactly what hides it.
    Dead,
    /// Most of what it outputs is pinned at the far end of its non-linearity,
    /// where the derivative is nothing. Also read off the maximum.
    Saturated,
    /// It is moving, but by so little relative to its own weights that it will
    /// not arrive.
    ///
    /// The ratio of update to weight, which practice puts near `1e-3`. It is
    /// the cheapest signal there is and the original measured it without ever
    /// saying anything about it.
    Stalled,
    /// It is moving so much relative to its own weights that each step throws
    /// away where it was.
    Overstepping,
    /// How many channels are dead — output near zero across the window.
    ///
    /// Separate from [`Flag::Dead`] because a layer can be perfectly alive with
    /// a quarter of its width doing nothing, and that is a width problem rather
    /// than a layer problem.
    DeadChannels(usize),
    /// How many channels are **alive and never asked for**: they compute
    /// something and no gradient ever comes back for it.
    ///
    /// Gradient starvation, and the distinction from a dormant channel matters
    /// — a dormant one is not computing anything to be ignored.
    IgnoredChannels(usize),
    /// Two groups of channels the architecture means to keep apart are carrying
    /// the same information, by linear CKA.
    Leakage,
    /// The update has collapsed into a few directions compared with what this
    /// run was doing before.
    ///
    /// The earliest warning there is: it moves thousands of steps before the
    /// loss does, because by the time a loss spikes the damage is already in
    /// the weights.
    Narrowing,
    /// The weights keep growing, the representation keeps narrowing and the
    /// units keep going quiet — **all three at once**, which is what tells a
    /// network that has stopped being able to learn from one that is training.
    LosingPlasticity,
}

impl Flag {
    /// The word this flag is written down as, which is what a record keeps and
    /// what somebody greps for.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Nan => "NAN",
            Self::Inf => "INF",
            Self::Vanishing => "VANISHING",
            Self::Exploding => "EXPLODING",
            Self::Dead => "DEAD",
            Self::Saturated => "SATURATED",
            Self::Stalled => "STALLED",
            Self::Overstepping => "OVERSTEPPING",
            Self::DeadChannels(_) => "DEAD_CHANNELS",
            Self::IgnoredChannels(_) => "IGNORED_CHANNELS",
            Self::Leakage => "LEAKAGE",
            Self::Narrowing => "NARROWING",
            Self::LosingPlasticity => "LOSING_PLASTICITY",
        }
    }

    /// What to do about it, in one line.
    ///
    /// Part of the flag and not of whoever draws it: the thresholds and the
    /// advice are the same opinion, and splitting them is how a dashboard ends
    /// up telling somebody something this crate never said.
    pub fn about(&self) -> &'static str {
        match self {
            Self::Nan => "a number stopped being one; every metric below this is meaningless",
            Self::Inf => "something overflowed; look at the step before this one",
            Self::Vanishing => {
                "this node is barely being trained — look at the depth profile, not at this node \
                 alone: it is the early layers that go quiet first"
            }
            Self::Exploding => "the next step will not be a step; clip, or lower the rate",
            Self::Dead => {
                "most of the output is zero on at least one step; the non-linearity or \
                           the init is cutting everything off"
            }
            Self::Saturated => "most of the output is pinned where the derivative is nothing",
            Self::Stalled => {
                "the update is tiny next to the weights; the rate is too low for \
                              this node, or nothing is reaching it"
            }
            Self::Overstepping => "each step throws away where it was; the rate is too high",
            Self::DeadChannels(_) => {
                "part of the width is doing nothing — a width problem, not a \
                                      layer problem"
            }
            Self::IgnoredChannels(_) => {
                "these channels compute something nobody asks for; the \
                                         gradient never comes back for them"
            }
            Self::Leakage => "two groups meant to stay apart carry the same information",
            Self::Narrowing => {
                "the update has collapsed into a few directions; this moves long \
                                before the loss does"
            }
            Self::LosingPlasticity => {
                "weights growing, rank falling, units going quiet — it is \
                                       losing the ability to learn anything new"
            }
        }
    }
}

impl fmt::Display for Flag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeadChannels(n) | Self::IgnoredChannels(n) => write!(f, "{}({n})", self.name()),
            _ => f.write_str(self.name()),
        }
    }
}
