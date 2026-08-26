//! What can be wrong. The vocabulary of a diagnosis.

use std::fmt;

/// One thing that looks wrong with a node.
///
/// An enum because the set is closed and named: a diagnosis is only useful if
/// two runs of it say the same word for the same thing. Every variant is an
/// **opinion at a threshold** and none of them is a fact — the facts are in the
/// record; these are what somebody thinks of them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Flag {
    /// A number stopped being a number. Nothing below it means anything.
    Nan,
    /// Or stopped being finite.
    Inf,
    /// The parameter gradients are so small this node is not being trained. The
    /// classic depth pathology, and a **profile over depth** rather than a
    /// property of a network: the early layers go quiet while the last learns.
    Vanishing,
    /// The parameter gradients are so large the next step will not be a step.
    Exploding,
    /// The signal has grown over a stretch nobody is normalising.
    ///
    /// A conjunction, and both halves are load-bearing: the structural half is
    /// baked into the measurement, which counts the gain from the last
    /// normalisation upstream. A badly-initialised stack *with* a norm layer
    /// drifts under 3x and trains.
    ///
    /// One-sided, and measured rather than assumed: it says nothing about a
    /// signal that shrank, because Adam is scale-invariant per parameter. See
    /// `health/tests/normalisation.py`.
    MissingNormalisation,
    /// Most of what this node outputs is zero, on at least one step.
    ///
    /// Read off the **maximum** over the window: a layer that dies one step in
    /// four is dead, and the mean is exactly what hides it.
    Dead,
    /// Most of what it outputs is pinned at the far end of its non-linearity,
    /// where the derivative is nothing. Also read off the maximum.
    Saturated,
    /// It is moving, but by so little relative to its own weights that it will
    /// not arrive. The ratio of update to weight, which practice puts near
    /// `1e-3` — the cheapest signal there is.
    Stalled,
    /// It is moving so much relative to its own weights that each step throws
    /// away where it was.
    Overstepping,
    /// How many channels are dead — output near zero across the window. Separate
    /// from [`Flag::Dead`]: a layer can be alive with a quarter of its width
    /// doing nothing, which is a width problem and not a layer problem.
    DeadChannels(usize),
    /// How many channels are **alive and never asked for**: they compute
    /// something and no gradient comes back. Gradient starvation, and a dormant
    /// channel is not the same thing — it is computing nothing to be ignored.
    IgnoredChannels(usize),
    /// Two groups of channels the architecture means to keep apart are carrying
    /// the same information, by linear CKA.
    Leakage,
    /// The update has collapsed into a few directions compared with what this
    /// run was doing before. The earliest warning there is: it moves thousands
    /// of steps before the loss does.
    Narrowing,
    /// An input the model is not using: taking it away costs nothing. A network
    /// with a perfectly healthy gradient can be ignoring an input all afternoon
    /// without a single other flag firing.
    IgnoredInput(String),
    /// One input carries everything, and nothing else would take over.
    ///
    /// Not a failure and not always wrong: sometimes one channel really is the
    /// signal. It is worth knowing before the day that channel is missing.
    SoleReliance(String),
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
            Self::IgnoredInput(_) => "IGNORED_INPUT",
            Self::SoleReliance(_) => "SOLE_RELIANCE",
            Self::MissingNormalisation => "MISSING_NORMALISATION",
            Self::Leakage => "LEAKAGE",
            Self::Narrowing => "NARROWING",
            Self::LosingPlasticity => "LOSING_PLASTICITY",
        }
    }

    /// Which family of trouble this is. A closed set, so a figure can give each
    /// family a colour instead of painting everything one red. By **what to do
    /// about them** and not by what was measured: `VANISHING` and `EXPLODING`
    /// are both answered by looking at depth and initialisation.
    pub fn family(&self) -> &'static str {
        match self {
            Self::Nan | Self::Inf => "numeric",
            Self::Vanishing | Self::Exploding | Self::MissingNormalisation => "signal",
            Self::Dead | Self::Saturated => "activation",
            Self::Stalled | Self::Overstepping => "step",
            Self::DeadChannels(_)
            | Self::IgnoredChannels(_)
            | Self::Leakage
            | Self::Narrowing
            | Self::LosingPlasticity => "capacity",
            Self::IgnoredInput(_) | Self::SoleReliance(_) => "data",
        }
    }

    /// What to do about it, in one line. Part of the flag and not of whoever
    /// draws it: the thresholds and the advice are the same opinion.
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
            Self::IgnoredInput(_) => {
                "the model is not using this input: taking it away costs nothing. If this is \
                 the channel the work is about, nothing in the network is the problem"
            }
            Self::SoleReliance(_) => {
                "one input carries everything and nothing else would take over if it went"
            }
            Self::MissingNormalisation => {
                "the signal grows over a stretch with nothing normalising it; the first \
                 step will be taken on numbers this size"
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
            Self::IgnoredInput(what) | Self::SoleReliance(what) => {
                write!(f, "{}({what})", self.name())
            }
            _ => f.write_str(self.name()),
        }
    }
}
