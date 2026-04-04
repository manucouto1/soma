//! Execution engine for Soma computational graphs.
//!
//! Provides the runtime components that execute compiled plans:
//! - [`runner`] — trait-based execution: LocalRunner, StreamExecutor, StudyRunner, PbtRunner
//! - [`executor`] — walks [`ExecutionPlan`] trees (sequence, parallel, cached, remote)
//! - [`GraphSession`] — the primary orchestrator: Graph + filters → compile → execute
//! - [`cache`] — LRU memory cache, local disk cache, tiered cache
//! - [`sampler`] — hyperparameter samplers (Grid, Random, Bayesian/TPE)
//! - [`pruner`] — early stopping strategies (Median, Percentile)

pub mod cache;
pub mod event_bus;
pub mod executor;
pub mod filter_library;
pub mod graph_session;
pub mod pruner;
pub mod runner;
pub mod sampler;

pub use cache::{LocalCache, MemoryCache, TieredCache};
pub use event_bus::EventBus;
pub use executor::{Context, GraphInfo, RemoteExecutor, execute, resolve_input};
pub use filter_library::FilterLibrary;
pub use graph_session::{GraphSession, graph_fit, graph_predict, graph_run};
pub use pruner::{MedianPruner, PercentilePruner, Pruner};
pub use runner::{
    FittedFilter, FnPbtExecutor, FnTrialExecutor, LocalRunner, PbtConfig, PbtExecutor, PbtRunner,
    PopulationMember, Runner, StreamExecutor, StudyRunner, TrialExecutor, TrialOutcome,
};
pub use sampler::{BayesianSampler, GridSampler, RandomSampler, Sampler};
