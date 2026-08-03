//! # Soma
//!
//! Computational graph runtime for research pipelines, agent orchestration,
//! and data virtualization.
//!
//! This crate re-exports all Soma subcrates for convenience:
//!
//! ```toml
//! # Everything
//! [dependencies]
//! somatize = "0.4"
//!
//! # With S3 + Zarr storage
//! [dependencies]
//! somatize = { version = "0.4", features = ["s3", "zarr"] }
//!
//! # Or pick individual crates
//! [dependencies]
//! somatize-core = "0.4"
//! somatize-runtime = "0.4"
//! ```
//!
//! ## Quick start
//!
//! ```ignore
//! use soma::prelude::*;
//!
//! let graph = linear_pipeline(vec![
//!     Node::new("scaler", "Scaler", "Scaler"),
//!     Node::new("model", "Model", "Model"),
//! ]);
//!
//! let mut lib = NodeCatalog::new();
//! lib.register("scaler", Box::new(MyScaler));
//! lib.register("model", Box::new(MyModel));
//!
//! let mut session = GraphSession::new(graph, lib);
//! session.fit(&train_data, None)?;
//! let output = session.forward(&test_data)?;
//! ```

/// Core types and traits: Filter, Value, Graph, CacheKey, Schema, Event.
pub use somatize_core as core;

/// Graph → ExecutionPlan compiler, Scheduler.
pub use somatize_compiler as compiler;

/// Execution engine: GraphSession, executor, caches, samplers, StudyRunner, PbtRunner.
pub use somatize_runtime as runtime;

/// Knowledge base and experiment tracking.
pub use somatize_memory as memory;

/// Distributed worker: Worker, Coordinator, Protocol, EnvManager.
pub use somatize_worker as worker;

/// Autonomous research agent.
pub use somatize_agent as agent;

/// Provider-agnostic LLM access: the OpenAI-compatible client, the
/// provider catalog, `ReactStep`, `JudgeStep`, the toolbox.
///
/// Absent until now, which meant the facade could not reach the agentic
/// surface it advertises — a Rust caller had to depend on `somatize-llm`
/// directly while `soma::` pretended the workspace ended at `agent`.
pub use somatize_llm as llm;

/// Worker registry and placement for a cluster of workers.
///
/// A workspace member the facade never reached, so a Rust caller wiring up
/// a cluster had to depend on `somatize-coordinator` directly while
/// `soma::` behaved as though the workspace ended at `worker`.
pub use somatize_coordinator as coordinator;

/// Derive macros (#[derive(SomaFilter)]).
pub use somatize_macros as macros;

/// Prelude — import the most commonly used types.
pub mod prelude {
    // Core types
    pub use somatize_core::cache::{CacheKey, CacheStore};
    pub use somatize_core::error::{Result, SomaError};
    pub use somatize_core::event::Event;
    pub use somatize_core::filter::{Filter, FilterKind, FilterMeta};
    pub use somatize_core::graph::{Edge, Graph, Node, NodeKind, linear_pipeline};
    pub use somatize_core::schema::Schema;
    pub use somatize_core::strategy::TrainingStrategy;
    pub use somatize_core::value::Value;
    pub use somatize_core::virtual_value::VirtualValue;

    // The effectful half. A `Step` sits beside a `Filter` and both are
    // nodes in the same graph — so both belong in the same prelude. These
    // were missing, which meant the one import line the docs recommend
    // reached only the computational half of the model.
    pub use somatize_core::effect::{Effect, EffectResult};
    pub use somatize_core::node::{NodeMeta, NodeOutcome};
    pub use somatize_core::step::{Step, StepCtx, StepMeta, Transition};

    // Compiler
    pub use somatize_compiler::{CompileMode, ExecutionPlan};

    // Runtime
    pub use somatize_runtime::cache::MemoryCache;
    pub use somatize_runtime::graph_session::GraphSession;
    pub use somatize_runtime::node_catalog::NodeCatalog;
    pub use somatize_runtime::{EventBus, execute};

    // Derive macros
    pub use somatize_macros::{SomaFilter, SomaStep};
}
