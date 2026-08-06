//! Hyperparameter search: the half that actually runs.
//!
//! `soma-core`'s `optimizer` says what a search space and a study *are*.
//! This one walks them. [`sampler`] proposes the next point — grid, random,
//! or TPE over the trials so far — and [`pruner`] decides whether a running
//! trial is worth finishing, by comparing its reported metric against
//! completed trials at the same step.
//!
//! [`study`] is the loop that puts those together, and it is where the
//! distinction between a pruned trial and a failed one lives:
//! `TrialOutcome` separates control flow from error, so a pruner stopping a
//! trial early is not an exception anywhere.
//!
//! [`pbt`] is the other shape of the same idea — population-based training
//! evolves a population across generations instead of sampling points from
//! a fixed space. It is an executor with callbacks rather than a
//! `TrainingStrategy`, because a member needs its own hyperparameters
//! applied to the graph and a worker is sent a plan, not a way to build one.
//!
//! [`study_io`] is where a study meets the disk.

pub mod pbt;
pub mod pruner;
pub mod sampler;
pub mod study;
pub mod study_io;
