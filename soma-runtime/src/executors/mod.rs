//! Executors — high-level execution loops that USE a Runner.
//!
//! Each Executor defines WHAT to do (study loop, PBT evolution, streaming).
//! The Runner decides WHERE to execute (local or remote).
//!
//! - [`StudyRunner`] — hyperparameter optimization loop
//! - [`PbtRunner`] — population-based training (evolutionary)
//! - [`StreamExecutor`] — chunk-based streaming through fitted filters

pub mod pbt;
pub mod stream;
pub mod study;

pub use pbt::{FnPbtExecutor, PbtConfig, PbtExecutor, PbtRunner, PopulationMember};
pub use stream::{FittedFilter, StreamExecutor, materialize_buffer};
pub use study::{FnTrialExecutor, StudyRunner, TrialContext, TrialExecutor, TrialOutcome};
