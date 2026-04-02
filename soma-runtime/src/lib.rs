//! Execution engine for Soma pipelines.
//!
//! Provides the runtime components that execute compiled plans:
//! - [`Pipeline`] — sequential filter chain with fit/predict
//! - [`executor`] — walks [`ExecutionPlan`] trees (sequence, parallel, cached, remote)
//! - [`cache`] — LRU memory cache, local disk cache, tiered cache
//! - [`sampler`] — hyperparameter samplers (Grid, Random, Bayesian/TPE)
//! - [`pruner`] — early stopping strategies (Median, Percentile)
//! - [`stream`] — chunk-based streaming executor
//! - [`StudyRunner`] — orchestrates optimization studies

pub mod cache;
pub mod event_bus;
pub mod executor;
pub mod filter_library;
pub mod graph_session;
pub mod pipeline;
pub mod pruner;
pub mod sampler;
pub mod stream;
pub mod study_runner;

pub use cache::{LocalCache, MemoryCache, TieredCache};
pub use event_bus::EventBus;
pub use executor::{Context, FilterStore, GraphInfo, RemoteExecutor, execute};
pub use filter_library::FilterLibrary;
pub use graph_session::{graph_fit, graph_predict, graph_run};
pub use pipeline::Pipeline;
pub use pruner::{MedianPruner, PercentilePruner, Pruner};
pub use sampler::{BayesianSampler, GridSampler, RandomSampler, Sampler};
pub use stream::{FittedFilter, StreamExecutor};
pub use study_runner::{FnTrialExecutor, StudyRunner, TrialExecutor, TrialOutcome};
