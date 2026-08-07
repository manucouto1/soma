//! Hyperparameter search, as Python types.
//!
//! [`study`] is `Study` and `Trial` — a campaign over a search space, with
//! `trial.report()` feeding the pruner and `SomaPruned` carrying the
//! stop as an exception rather than a return value. [`pbt`] is `Pbt`,
//! population-based training, which is an executor with callbacks rather
//! than a `TrainingStrategy`: a member needs its own hyperparameters applied
//! to the graph, and a worker is sent a plan, not a way to build one.
//!
//! Both parse search dimensions through the same function, so what `search()`
//! means is defined once.

pub(crate) mod pbt;
pub(crate) mod study;
