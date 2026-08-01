//! Runner module — trait-based execution contracts.
//!
//! A [`Runner`] defines the contract for executing plans (fit + forward).
//! [`LocalRunner`] executes locally using the Executor.
//! The worker's `RemoteRunner` prepares the environment and delegates to `LocalRunner`.

pub mod local;
pub mod remote;

use somatize_compiler::ExecutionPlan;
use somatize_core::cache::CacheStore;
use somatize_core::error::Result;
use somatize_core::value::Value;
use std::collections::HashMap;

use crate::EventBus;
use crate::executor::GraphInfo;
use crate::node_catalog::NodeCatalog;
use std::sync::Arc;

/// Everything a runner needs besides the plan and the data.
///
/// A struct rather than six more parameters, and one of them is the point:
/// `graph_info`. Both runner methods used to build
/// `GraphInfo::for_linear(plan.node_ids())` — chaining the plan's nodes in
/// flattened order as if every graph were a chain. On a diamond that is
/// simply wrong: `GraphSession::forward` on `a → {b, c} → d` answered with
/// `d(c(...))`, `d` never seeing `b` and `a` never seeing the input.
///
/// The caller supplies the real topology now. A caller that genuinely has
/// only a plan can still pass `GraphInfo::for_linear`, but it has to say so.
pub struct RunContext<'a> {
    pub catalog: &'a NodeCatalog,
    pub cache: &'a dyn CacheStore,
    pub events: &'a Arc<EventBus>,
    /// Tags every node event of this run — callers that emit a
    /// `RunStarted`/`RunCompleted` bracket pass the same id so readers can
    /// group a run's events.
    pub run_id: &'a str,
    pub graph_info: GraphInfo,
}

impl<'a> RunContext<'a> {
    pub fn new(
        catalog: &'a NodeCatalog,
        cache: &'a dyn CacheStore,
        events: &'a Arc<EventBus>,
        run_id: &'a str,
        graph_info: GraphInfo,
    ) -> Self {
        Self {
            catalog,
            cache,
            events,
            run_id,
            graph_info,
        }
    }

    /// For a caller that has only a plan: treat it as a chain.
    ///
    /// Correct for a linear pipeline and a fabrication for anything else,
    /// which is why it is spelled out at the call site rather than being
    /// what you get by default.
    pub fn linear(
        catalog: &'a NodeCatalog,
        cache: &'a dyn CacheStore,
        events: &'a Arc<EventBus>,
        run_id: &'a str,
        plan: &ExecutionPlan,
    ) -> Self {
        let ids = plan.node_ids();
        Self::new(catalog, cache, events, run_id, GraphInfo::for_linear(&ids))
    }
}

/// Contract for executing plans. Every execution mode (local, remote, stream)
/// implements this trait. One interface, polymorphic dispatch.
pub trait Runner: Send + Sync {
    /// Train: fit each filter, forward to propagate outputs.
    /// Returns (last output, all node outputs).
    fn fit(
        &self,
        plan: &ExecutionPlan,
        ctx: &RunContext<'_>,
        input: &Value,
        y: Option<&Value>,
    ) -> Result<(Value, HashMap<String, Value>)>;

    /// Inference: forward data through the compiled plan.
    fn forward(&self, plan: &ExecutionPlan, ctx: &RunContext<'_>, input: &Value) -> Result<Value>;
}

pub use local::LocalRunner;
pub use remote::{RemoteRunner, Transport};
