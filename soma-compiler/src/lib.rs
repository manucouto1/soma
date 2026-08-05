// The crate is fully documented and clippy runs with -D warnings in CI,
// so this makes "public API without docs" a build error from here on.
#![warn(missing_docs)]

//! Graph compiler for Soma pipelines.
//!
//! Transforms a `Graph` of filter nodes into an [`ExecutionPlan`]:
//! - Topological ordering and parallelism detection
//! - Cache resolution (content-addressable, cascade invalidation)
//! - Schema validation between connected filters
//! - Distribution wrapping for remote execution
//! - the `Scheduler` assigns plan nodes to workers

pub mod compiler;
pub mod plan;
pub mod scheduler;

pub use compiler::{
    CompileMode, CompileResult, Compiler, Diagnostic, DiagnosticLevel, NodeRegistry,
    SimpleNodeRegistry, compile, compile_stream,
};
pub use plan::ExecutionPlan;
pub use scheduler::{
    Assignment, DataTransfer, DistributionPlan, Phase, PlanPhase, WorkerInfo, schedule,
};
