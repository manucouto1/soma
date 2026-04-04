//! Runner module — trait-based execution contracts.
//!
//! A [`Runner`] defines the contract for executing plans (fit + forward).
//! [`LocalRunner`] executes locally using the Executor.
//! The worker's `RemoteRunner` prepares the environment and delegates to `LocalRunner`.

pub mod local;

use somatize_compiler::ExecutionPlan;
use somatize_core::cache::CacheStore;
use somatize_core::error::Result;
use somatize_core::value::Value;
use std::collections::HashMap;

use crate::EventBus;
use crate::filter_library::FilterLibrary;
use std::sync::Arc;

/// Contract for executing plans. Every execution mode (local, remote, stream)
/// implements this trait. One interface, polymorphic dispatch.
pub trait Runner: Send + Sync {
    /// Train: fit each filter, forward to propagate outputs.
    /// Returns (last output, all node outputs).
    fn fit(
        &self,
        plan: &ExecutionPlan,
        filters: &FilterLibrary,
        cache: &dyn CacheStore,
        event_bus: &Arc<EventBus>,
        input: &Value,
        y: Option<&Value>,
    ) -> Result<(Value, HashMap<String, Value>)>;

    /// Inference: forward data through the compiled plan.
    fn forward(
        &self,
        plan: &ExecutionPlan,
        filters: &FilterLibrary,
        cache: &dyn CacheStore,
        event_bus: &Arc<EventBus>,
        input: &Value,
    ) -> Result<Value>;
}

pub use local::LocalRunner;
