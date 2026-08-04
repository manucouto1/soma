//! Execution engine for Soma computational graphs.
//!
//! Two decisions shape this crate. [`NodeCatalog`] is THE registry — every
//! node, filter or step, plus the trained states; the compiler and the
//! executor read the same value, which is what keeps "what compiles" and
//! "what runs" from drifting apart. And `run_node` (in [`executor`]) is the
//! one execution site: input resolution, the output cache, panic
//! containment and the start/complete/fail events happen once, for both
//! kinds — whether a node is cacheable is *data* on its
//! [`NodeMeta`](somatize_core::node::NodeMeta), not a branch on its kind.
//!
//! The pieces:
//! - [`runner`] — trait-based execution: LocalRunner, StudyRunner, PbtRunner
//! - [`executor`] — walks `ExecutionPlan` trees (sequence, parallel, step, loop, remote)
//! - [`GraphSession`] — the primary orchestrator: Graph + catalog → compile → execute;
//!   give it an [`EffectDriver`] via `with_driver`
//!   and it drives steps too
//! - [`effects`] — [`EffectDriver`] performs what steps
//!   ask for; [`EffectJournal`] records once and
//!   replays on resume; [`GraphHandler`](effects::GraphHandler) runs a
//!   pipeline on an agent's behalf
//! - [`cache`] — LRU memory cache, local disk cache, tiered cache
//! - [`sampler`] — hyperparameter samplers (Grid, Random, Bayesian/TPE)
//! - [`pruner`] — early stopping strategies (Median, Percentile)
//! - [`tracking`] — local run directories: JSONL event sink, LocalTracker

pub mod cache;
pub mod effects;
pub mod event_bus;
pub mod executor;
pub mod executors;
pub mod forward;
pub mod graph_session;
pub mod node_catalog;
pub mod pruner;
pub mod runner;
pub mod sampler;
pub mod strategy;
pub mod study_io;
pub mod tracking;

pub use cache::{LocalCache, MemoryCache, TieredCache};
pub use effects::{EffectDriver, EffectHandler, EffectJournal, EffectSite, NodeOutcome};
pub use event_bus::EventBus;
pub use executor::{Context, GraphInfo, execute};
pub use executors::{
    FnPbtExecutor, FnTrialExecutor, PbtConfig, PbtExecutor, PbtRunner, PopulationMember,
    StudyRunner, TrialContext, TrialExecutor, TrialOutcome,
};
pub use forward::{Batched, ForwardStrategy, Standard, Stream};
pub use graph_session::{GraphSession, graph_fit, graph_predict, graph_run};
pub use node_catalog::{NodeCatalog, NodeImpl};
pub use pruner::{MedianPruner, PercentilePruner, Pruner};
pub use runner::{LocalRunner, RemoteRunner, Runner, Transport};
pub use sampler::{BayesianSampler, GridSampler, RandomSampler, Sampler};
pub use study_io::StudyIo;
pub use tracking::{
    JsonlEventSink, LocalTracker, RunInfo, RunReader, collect_git_info, list_runs, load_manifest,
    load_status,
};
