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
    CompileMode, CompileResult, Compiler, Diagnostic, DiagnosticLevel, FilterRegistry,
    SimpleFilterRegistry, compile, compile_stream,
};
pub use plan::ExecutionPlan;
pub use scheduler::{
    Assignment, DataTransfer, DistributionPlan, Phase, PlanPhase, WorkerInfo, schedule,
};
