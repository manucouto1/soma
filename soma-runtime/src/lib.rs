//! Execution engine for Soma computational graphs.
//!
//! Provides the runtime components that execute compiled plans:
//! - [`runner`] — trait-based execution: LocalRunner, StreamExecutor, StudyRunner, PbtRunner
//! - [`executor`] — walks [`ExecutionPlan`] trees (sequence, parallel, cached, remote)
//! - [`GraphSession`] — the primary orchestrator: Graph + filters → compile → execute
//! - [`cache`] — LRU memory cache, local disk cache, tiered cache
//! - [`sampler`] — hyperparameter samplers (Grid, Random, Bayesian/TPE)
//! - [`pruner`] — early stopping strategies (Median, Percentile)
//! - [`tracking`] — local run directories: JSONL event sink, LocalTracker

pub mod cache;
pub mod event_bus;
pub mod executor;
pub mod executors;
pub mod filter_library;
pub mod forward;
pub mod graph_session;
pub mod pruner;
pub mod runner;
pub mod sampler;
pub mod tracking;

pub use cache::{LocalCache, MemoryCache, TieredCache};
pub use event_bus::EventBus;
pub use executor::{Context, GraphInfo, execute, resolve_input};
pub use executors::{
    FittedFilter, FnPbtExecutor, FnTrialExecutor, PbtConfig, PbtExecutor, PbtRunner,
    PopulationMember, StreamExecutor, StudyRunner, TrialExecutor, TrialOutcome,
};
pub use filter_library::FilterLibrary;
pub use forward::{Batched, ForwardStrategy, Standard, Stream};
pub use graph_session::{GraphSession, graph_fit, graph_predict, graph_run};
pub use pruner::{MedianPruner, PercentilePruner, Pruner};
pub use runner::{LocalRunner, RemoteRunner, Runner, Transport};
pub use sampler::{BayesianSampler, GridSampler, RandomSampler, Sampler};
pub use tracking::{JsonlEventSink, LocalTracker, collect_git_info, load_manifest, load_status};
