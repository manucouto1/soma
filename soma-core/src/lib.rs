//! Core types and traits for the Soma computational graph runtime.
//!
//! A graph holds two kinds of node, and this crate defines both sides of
//! that split. A [`Filter`] *computes* — deterministic, content-cacheable,
//! `fit`/`forward`. A [`Step`] *decides* — it polls, asks the
//! runtime to perform [`Effect`]s (a model, a tool, another
//! graph), and returns a [`Transition`]. One
//! [`NodeMeta`](node::NodeMeta) describes either kind; one
//! [`NodeOutcome`](node::NodeOutcome) says how either kind finished.
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

pub mod action;
pub mod any;
pub mod cache;
pub mod canon;
pub mod codec;
pub mod control;
pub mod effect;
pub mod error;
pub mod event;
pub mod filter;
pub mod fingerprint;
pub mod graph;
pub mod keys;
pub mod message;
pub mod node;
pub mod schema;
pub mod search;
pub mod state;
pub mod step;
pub mod store;
pub mod strategy;
pub mod study;
pub mod summary;
pub mod svg;
pub mod tool;
pub mod tracking;
pub mod util;
pub mod value;
pub mod virtual_value;
pub mod viz;

// Re-export core types for convenience.
pub use cache::{CacheKey, CacheStore, CacheTier, EntryMeta, Origin};
pub use control::{LoopCondition, LoopSignal, read_arm_selector, read_loop_signal};
pub use effect::{
    Effect, EffectResult, GraphEffectMode, JoinPolicy, LlmRequest, LlmResponse, NodeSpec,
    StopReason, SuspendReason, ToolSpec, Usage,
};
pub use error::{Result, SomaError};
pub use event::{Event, MetricRecord, PlanSummary, RunId, StudyId, TrialId};
pub use filter::{Distribution, Filter, FilterKind, FilterMeta, RemoteTarget, StreamMode};
pub use fingerprint::{ArchitectureFingerprint, EdgeRef, pipeline_summary, structural_similarity};
pub use graph::{Edge, EdgeKind, Graph, Node, NodeId};
pub use message::{ContentBlock, Message, Messages, Role};
pub use schema::{DataType, Dimension, Schema};
pub use search::{Scale, SearchDimension, SearchSpace, Searchable};
pub use state::{MemoryStateStore, StateStore};
pub use step::{Step, StepCtx, StepMeta, Transition};
pub use store::{
    DataRef, DataStore, LocalDataStore, StorageConfig, StoreMeta, StreamCache, StreamFormat,
    slice_tensor_rows,
};
pub use strategy::{
    ClientSelection, CommunicationProtocol, ExploitStrategy, ExploreStrategy, FederatedAggregation,
    GradientAggregation, Partition, TrainingStrategy,
};
pub use study::{
    CompositeObjective, Direction, Objective, PruningStrategy, Scalarizer, SearchStrategy, Study,
    Trial, TrialState,
};
pub use summary::{
    FlagCount, NodeCost, RunConclusion, RunOutcome, RunSummary, TrialSummary, human_duration,
};
pub use tracking::{
    EventEnvelope, EventSink, GitInfo, GraphSummaryInfo, RUN_SCHEMA_VERSION, RunKind, RunManifest,
    RunState, RunStatus, Tracker,
};
pub use value::Value;
pub use virtual_value::{ValueStatus, VirtualValue};
pub use viz::{GraphOverlay, NodeOverlay, NodeStatus};

// Re-export derive macro
pub use somatize_macros::{SomaFilter, SomaStep};
