//! Level 3: what is above one training run.
//!
//! The graph is a network — the scale of one `forward`. The `Trainer` is a
//! training run — the scale of an afternoon. This is the level above, and it
//! has no type: N training runs are a `for`, exactly as a federated round is.
//! What lives here are the **pieces that `for` asks for**, and they all have
//! the same shape:
//!
//! > indices and keys in, indices out. **Never a tensor.**
//!
//! That is what lets all of it be Rust while the loop stays in Python. The one
//! step of the loop that cannot move here is "train", because training is
//! torch; a loop written in Rust would have to call back out for it, and a
//! trait that calls back out is not an abstraction, it is the loop leaking.
//! The original measured this: its `TrialExecutor` has exactly one implementor
//! and it is a closure wrapper.
//!
//! # Cutting the samples
//!
//! | piece | what it is |
//! |---|---|
//! | [`Samples`] | what is known about them: how many, and their class and group |
//! | [`KFold`] | the plain cut: `k` parts, each held out in turn |
//! | [`Stratified`] | a k-fold **inside each class**, so every class keeps its share |
//! | [`Grouped`] | a k-fold **over the groups**, so a group never splits |
//! | [`StratifiedGrouped`] | both, as far as both can be had at once |
//! | [`TimeSeries`] | growing prefixes, so nothing trains on its own future |
//! | [`Partition`] | whichever of them, when that is decided by data |
//! | [`Fold`] | one cut: which indices train, which are held out |
//!
//! # Deciding where to look next
//!
//! | piece | looks at |
//! |---|---|
//! | [`Space`] | not a scheme: the named knobs and what each may be |
//! | [`Grid`] | **the space's shape** — and it is the one that runs out |
//! | [`Random`] | **nothing**, which over a space where few knobs matter beats a grid |
//! | [`Halton`] | **nothing** either, but it covers every prefix instead of the whole |
//! | [`Sobol`] | the same, without Halton's seam and with a table's ceiling |
//! | [`Tpe`] | **what already happened**: imitate the good, avoid the bad |
//! | [`Sampler`] | whichever of them, when that is decided by data |
//! | [`Point`] | one configuration, which is also a trial's **name** |
//!
//! `ask` is a function of the **index**, not of what was asked before, so a
//! machine that claimed trial 7 out of a shared folder derives the same point
//! without replaying six. [`Tpe`] is the exception and says so.
//!
//! Three of them look at nothing and are still three schemes: [`Random`] is
//! uniform in expectation, so two machines *can* draw neighbouring points and it
//! is merely unlikely; the other two are uniform for **every prefix**, so there
//! is no arrangement of the claimed indices that puts two trials on top of each
//! other. That is what a study handed out of a folder wants.
//!
//! # Deciding a trial is not worth another epoch
//!
//! | piece | judged against |
//! |---|---|
//! | [`Percentile`] | **the others** at the same step — the median pruner is `p = 50` |
//! | [`Threshold`] | **a constant** already known to be hopeless |
//! | [`Patience`] | **itself**: it has stopped improving |
//! | [`Pruner`] | whichever of them, when that is decided by data |
//! | [`Goal`] | which way is better, without which none of them can decide |
//!
//! A pruner does not stop anything: it answers [`Verdict`] and the loop stops
//! calling the trainer. `Trainer.step` was already the primitive, so none of
//! this added a line to level 2.
//!
//! Five cutting schemes and not sklearn's fifteen, because stratifying and grouping are
//! not different algorithms and the rest are parameters: `LeaveOneOut` is
//! `KFold { k: n }`, a holdout of one part in `k` is fold 0 of a k-fold, and
//! purged and embargoed cross-validation are `TimeSeries { gap }`.
//!
//! It is not called `Split`: `soma_next.torch.Split` is already split learning,
//! and two alike names for two unrelated things is how a framework stops being
//! readable.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod goal;
mod partition;
mod point;
mod pruner;
mod sampler;
mod samples;
mod space;

pub use goal::{Goal, GoalError};
pub use partition::{
    Fold, Grouped, KFold, Partition, PartitionError, Stratified, StratifiedGrouped, TimeSeries,
};
pub use point::{Point, Setting};
pub use pruner::{Patience, Percentile, Pruner, Reason, Threshold, Verdict};
pub use sampler::{Grid, Halton, KNOBS, Random, Sampler, Sobol, Tpe};
pub use samples::{Samples, SamplesError};
pub use space::{Dimension, ReadError, Space, SpaceError};
