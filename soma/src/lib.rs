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
pub use soma_core as core;

/// Graph → ExecutionPlan compiler, Scheduler.
pub use soma_compiler as compiler;

/// Execution engine: GraphSession, executor, caches, samplers, StudyRunner, PbtRunner.
pub use soma_runtime as runtime;

/// Knowledge base and experiment tracking.
pub use soma_memory as memory;

/// Distributed worker: Worker, Coordinator, Protocol, EnvManager.
pub use soma_worker as worker;

/// Autonomous research agent.
pub use soma_agent as agent;

/// Derive macros (#[derive(SomaFilter)]).
pub use soma_macros;

/// Prelude — import the most commonly used types.
pub mod prelude {
    // Core types
    pub use soma_core::cache::{CacheKey, CacheStore};
    pub use soma_core::error::{Result, SomaError};
    pub use soma_core::event::Event;
    pub use soma_core::filter::{Filter, FilterKind, FilterMeta};
    pub use soma_core::graph::{Edge, Graph, Node, NodeKind, linear_pipeline};
    pub use soma_core::schema::Schema;
    pub use soma_core::strategy::TrainingStrategy;
    pub use soma_core::value::Value;
    pub use soma_core::virtual_value::VirtualValue;

    // Compiler
    pub use soma_compiler::{CompileMode, ExecutionPlan};

    // Runtime
    pub use soma_runtime::cache::MemoryCache;
    pub use soma_runtime::filter_library::FilterLibrary;
    pub use soma_runtime::graph_session::GraphSession;
    pub use soma_runtime::{EventBus, execute};

    // Derive macro
    pub use soma_macros::SomaFilter;
}
