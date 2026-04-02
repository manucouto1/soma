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
//! soma = "0.2"
//!
//! # With S3 + Zarr storage
//! [dependencies]
//! soma = { version = "0.2", features = ["s3", "zarr"] }
//!
//! # Or pick individual crates
//! [dependencies]
//! soma-core = "0.2"
//! soma-runtime = "0.2"
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
//! let mut lib = FilterLibrary::new();
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

/// Derive macros (#[derive(SomaFilter)]).
pub use somatize_macros;

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

    // Compiler
    pub use somatize_compiler::{CompileMode, ExecutionPlan};

    // Runtime
    pub use somatize_runtime::cache::MemoryCache;
    pub use somatize_runtime::filter_library::FilterLibrary;
    pub use somatize_runtime::graph_session::GraphSession;
    pub use somatize_runtime::{EventBus, execute};

    // Derive macro
    pub use somatize_macros::SomaFilter;
}
