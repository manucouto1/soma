//! RemoteRunner — executes plans on remote workers via a Transport abstraction.
//!
//! The Transport trait abstracts HOW to communicate with workers (WS, HTTP, gRPC, etc.).
//! RemoteRunner implements Runner by serializing fit/forward calls and sending them
//! through the transport layer.

use super::Runner;
use crate::EventBus;
use crate::node_catalog::NodeCatalog;

use somatize_compiler::ExecutionPlan;
use somatize_core::cache::CacheStore;
use somatize_core::error::Result;
use somatize_core::value::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// Abstraction for communicating with remote workers.
/// Implemented by WsTransport (WebSocket), but could be HTTP, gRPC, etc.
pub trait Transport: Send + Sync {
    /// Send a plan for execution and receive the output + trained states.
    fn execute(
        &self,
        plan: &ExecutionPlan,
        filters: &NodeCatalog,
        input: &Value,
        y: Option<&Value>,
        fit_mode: bool,
    ) -> Result<(Value, HashMap<String, Value>)>;

    /// Request trained states from the remote worker.
    fn get_state(&self, node_ids: &[String]) -> Result<HashMap<String, Value>>;

    /// Load states on the remote worker.
    fn set_state(&self, states: &HashMap<String, Value>) -> Result<()>;

    /// Request gradients from the remote worker.
    fn get_gradients(&self, node_ids: &[String]) -> Result<HashMap<String, Value>>;

    /// Apply aggregated gradients on the remote worker.
    fn apply_gradients(&self, gradients: &HashMap<String, Value>) -> Result<()>;

    /// Convenience: execute a single node remotely (used by the plan executor).
    fn execute_node(&self, node_id: &str, input: Option<&Value>) -> Result<Value> {
        let plan = ExecutionPlan::Execute {
            node_id: node_id.to_string(),
        };
        let input_val = input.cloned().unwrap_or(Value::Empty);
        let filters = crate::node_catalog::NodeCatalog::new();
        let (output, _) = self.execute(&plan, &filters, &input_val, None, false)?;
        Ok(output)
    }
}

/// A Runner that delegates execution to a remote worker via Transport.
pub struct RemoteRunner {
    transport: Box<dyn Transport>,
}

impl RemoteRunner {
    pub fn new(transport: impl Transport + 'static) -> Self {
        Self {
            transport: Box::new(transport),
        }
    }

    /// Access the underlying transport (for strategy methods).
    pub fn transport(&self) -> &dyn Transport {
        self.transport.as_ref()
    }
}

impl Runner for RemoteRunner {
    fn fit(
        &self,
        plan: &ExecutionPlan,
        filters: &NodeCatalog,
        _cache: &dyn CacheStore,
        _event_bus: &Arc<EventBus>,
        _run_id: &str,
        input: &Value,
        y: Option<&Value>,
    ) -> Result<(Value, HashMap<String, Value>)> {
        self.transport.execute(plan, filters, input, y, true)
    }

    fn forward(
        &self,
        plan: &ExecutionPlan,
        filters: &NodeCatalog,
        _cache: &dyn CacheStore,
        _event_bus: &Arc<EventBus>,
        input: &Value,
    ) -> Result<Value> {
        let (output, _states) = self.transport.execute(plan, filters, input, None, false)?;
        Ok(output)
    }
}
