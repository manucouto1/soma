// The crate is fully documented and clippy runs with -D warnings in CI,
// so this makes "public API without docs" a build error from here on.
#![warn(missing_docs)]

//! Core types and traits for the Soma computational graph runtime.
//!
//! A graph holds two kinds of node, and this crate defines both sides of
//! that split. A [`Filter`] *computes* — deterministic, content-cacheable,
//! `fit`/`forward`. A [`Step`] *decides* — it polls, asks the
//! runtime to perform [`Effect`]s (a model, a tool, another
//! graph), and returns a [`Transition`]. One
//! [`NodeMeta`](graph::node::NodeMeta) describes either kind; one
//! [`NodeOutcome`](graph::node::NodeOutcome) says how either kind finished.
//! Everything downstream reads the metadata, never the kind.
//!
//! The contracts every other crate depends on:
//! - [`Filter`] — the computation unit (fit/forward)
//! - [`Step`] / [`Effect`] — the decision unit
//!   and what it asks for; effects are journaled, filters are cached
//! - [`Value`] — typed data flowing between nodes (Tensor, JSON, Bytes)
//! - [`Graph`] — DAG of nodes and edges, computational and effectful alike
//! - [`CacheKey`] / [`CacheStore`] — content-addressable caching
//! - [`DataStore`] — abstraction for moving data between workers
//! - [`Schema`] — dtype + shape for compile-time validation, on every edge
//! - [`Event`] — runtime lifecycle events

pub mod agentic;
pub mod cache;
pub mod data;
pub mod distributed;
pub mod error;
pub mod graph;
pub mod optimizer;
pub mod tracking;
pub mod util;
pub mod viz;

// ── The convenience surface ─────────────────────────────────────────
//
// Every type below is also reachable at its own path (`data::value::Value`).
// These are the names used often enough that the domain prefix is noise at
// the call site, and they are what `use somatize_core::…` is expected to
// find.

pub use agentic::effect::{
    Effect, EffectResult, GraphEffectMode, JoinPolicy, LlmRequest, LlmResponse, NodeSpec,
    StopReason, SuspendReason, ToolSpec, Usage,
};
pub use agentic::message::{ContentBlock, Message, Messages, Role};
pub use cache::{CacheKey, CacheStore, CacheTier, EntryMeta, Origin};
pub use data::schema::{DataType, Dimension, Schema};
pub use data::state::{MemoryStateStore, StateStore};
pub use data::store::{
    DataRef, DataStore, LocalDataStore, StorageConfig, StoreMeta, slice_tensor_rows,
};
pub use data::value::Value;
pub use data::virtual_value::{ValueStatus, VirtualValue};
pub use distributed::{
    ClientSelection, CommunicationProtocol, ExploitStrategy, ExploreStrategy, FederatedAggregation,
    GradientAggregation, Partition, TrainingStrategy,
};
pub use error::{Result, SomaError};
pub use graph::control::{LoopCondition, LoopSignal, read_arm_selector, read_loop_signal};
pub use graph::filter::{Distribution, Filter, FilterKind, FilterMeta, RemoteTarget, StreamMode};
pub use graph::step::{Step, StepCtx, StepMeta, Transition};
pub use graph::{Edge, EdgeKind, Graph, Node, NodeId};
pub use optimizer::search::{Scale, SearchDimension, SearchSpace, Searchable};
pub use optimizer::study::{
    CompositeObjective, Direction, Objective, PruningStrategy, Scalarizer, SearchStrategy, Study,
    Trial, TrialState,
};
pub use tracking::event::{Event, MetricRecord, PlanSummary, RunId, StudyId, TrialId};
pub use tracking::fingerprint::{
    ArchitectureFingerprint, EdgeRef, pipeline_summary, structural_similarity,
};
pub use tracking::summary::{
    FlagCount, NodeCost, RunConclusion, RunOutcome, RunSummary, TrialSummary, human_duration,
};
pub use tracking::{
    EventEnvelope, EventSink, GitInfo, GraphSummaryInfo, RUN_SCHEMA_VERSION, RunKind, RunManifest,
    RunState, RunStatus, Tracker,
};
pub use viz::{GraphOverlay, NodeOverlay, NodeStatus};

// Re-export derive macro
pub use somatize_macros::{SomaFilter, SomaStep};
