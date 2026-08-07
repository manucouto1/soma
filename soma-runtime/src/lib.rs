// The crate is fully documented and clippy runs with -D warnings in CI,
// so this makes "public API without docs" a build error from here on.
#![warn(missing_docs)]

//! Execution engine for Soma computational graphs.
//!
//! Two decisions shape this crate. [`NodeCatalog`] is THE registry — every
//! node, filter or step, plus the trained states; the compiler and the
//! executor read the same value, which is what keeps "what compiles" and
//! "what runs" from drifting apart. And `run_node` (in
//! [`execution::executor`]) is the one execution site: input resolution,
//! the output cache, panic containment and the start/complete/fail events
//! happen once, for both kinds — whether a node is cacheable is *data* on
//! its [`NodeMeta`](somatize_core::graph::node::NodeMeta), not a branch on
//! its kind.
//!
//! Six domains, named as they are across the workspace. `soma-core` says
//! what each of these *is*; this crate is where they run.
//!
//! - [`execution`] — the stack that turns a plan into results:
//!   [`GraphSession`] on top, [`Runner`] deciding where, `execute` walking
//!   the tree, `run_node` at every leaf, and the stream driver composing
//!   the same three primitives per chunk
//! - [`agentic`] — [`EffectDriver`] performs what steps ask for;
//!   [`EffectJournal`] records once and replays on resume;
//!   [`GraphHandler`](agentic::GraphHandler) runs a pipeline on an agent's
//!   behalf
//! - [`optimizer`] — samplers (Grid, Random, TPE), pruners, the study loop,
//!   and population-based training
//! - [`cache`] — LRU memory cache, local disk cache, tiered cache
//! - [`tracking`] — the event bus, local run directories, the JSONL sink,
//!   and [`RunReader`], which aggregates a run back into chart-ready data
//! - [`distributed`] — running a `TrainingStrategy` across workers

pub mod agentic;
pub mod cache;
pub mod distributed;
pub mod execution;
pub mod fsutil;
pub mod optimizer;
pub mod tracking;

// ── The convenience surface ─────────────────────────────────────────
//
// Every type below is also reachable at its own path
// (`execution::executor::Context`). These are the names used often enough
// that the domain prefix is noise at the call site.

pub use agentic::{EffectDriver, EffectHandler, EffectJournal, EffectSite, NodeOutcome};
pub use cache::{LocalCache, MemoryCache, TieredCache};
pub use execution::executor::{Context, GraphInfo, execute};
pub use execution::forward::{Batched, ForwardStrategy, Standard, Stream};
pub use execution::graph_session::{GraphSession, graph_fit, graph_predict, graph_run};
pub use execution::node_catalog::{NodeCatalog, NodeImpl};
pub use execution::runner::{LocalRunner, Runner, Transport};
pub use execution::stream::{StreamOutput, StreamRun, materialize_buffer};
pub use optimizer::pbt::{FnPbtExecutor, PbtConfig, PbtExecutor, PbtRunner, PopulationMember};
pub use optimizer::pruner::{MedianPruner, PercentilePruner, Pruner};
pub use optimizer::sampler::{BayesianSampler, GridSampler, RandomSampler, Sampler};
pub use optimizer::study::{
    FnTrialExecutor, StudyRunner, TrialContext, TrialExecutor, TrialOutcome,
};
pub use optimizer::study_io::StudyIo;
pub use tracking::event_bus::EventBus;
pub use tracking::{
    JsonlEventSink, LocalTracker, RunInfo, RunReader, collect_git_info, list_runs, load_manifest,
    load_status,
};
