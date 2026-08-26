//! Level 3: what is above one training run.
//!
//! The graph is a network — one `forward`. The `Trainer` is a training run — an
//! afternoon. This is the level above, and it **has no type**: N training runs
//! are a `for`. What lives here are the pieces that `for` asks for, and they all
//! have one shape:
//!
//! > indices and keys in, indices out. **Never a tensor.**
//!
//! That is what lets all of it be Rust while the loop stays in Python. The step
//! that cannot move here is *train*, because training is torch — and a trait
//! that calls back out for it is not an abstraction, it is the loop leaking. The
//! original measured this: its `TrialExecutor` has one implementor, a closure.
//!
//! | cutting the samples | |
//! |---|---|
//! | [`Samples`] | how many, and their class and group |
//! | [`KFold`] | `k` parts, each held out in turn |
//! | [`Stratified`] | a k-fold **inside each class** |
//! | [`Grouped`] | a k-fold **over the groups**, so a group never splits |
//! | [`StratifiedGrouped`] | both, as far as both can be had at once |
//! | [`TimeSeries`] | growing prefixes, so nothing trains on its own future |
//! | [`Partition`], [`Fold`] | the family, and one cut |
//!
//! | where to look next | looks at |
//! |---|---|
//! | [`Space`], [`Point`] | the knobs, and one configuration — also a trial's name |
//! | [`Grid`] | **the space's shape**, and the one that runs out |
//! | [`Random`] | **nothing**; over a space where few knobs matter it beats a grid |
//! | [`Halton`], [`Sobol`] | nothing either, but uniform for **every prefix** |
//! | [`Tpe`] | **what already happened** |
//! | [`Sampler`] | the family |
//!
//! `ask` is a function of the **index** and not of what was asked before, so a
//! machine that claimed trial 7 out of a shared folder derives the same point
//! without replaying six. [`Tpe`] is the exception and says so. Uniform for
//! every prefix is what stops two machines proposing neighbours — [`Random`] is
//! uniform only in expectation, so it merely makes it unlikely.
//!
//! | when to give up | judged against |
//! |---|---|
//! | [`Percentile`] | **the others** at the same step; the median pruner is `p = 50` |
//! | [`Threshold`] | **a constant** already known to be hopeless |
//! | [`Patience`] | **itself**: it has stopped improving |
//! | [`Pruner`], [`Goal`] | the family, and which way is better |
//!
//! A pruner stops nothing: it answers [`Verdict`] and the loop stops calling the
//! trainer, so none of this added a line to level 2.
//!
//! Five cutting schemes and not sklearn's fifteen, because stratifying and
//! grouping are not different algorithms and the rest are parameters:
//! `LeaveOneOut` is `KFold { k: n }` and purged cross-validation is
//! `TimeSeries { gap }`. Not called `Split`, because `somatize.torch.Split` is
//! already split learning.

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
