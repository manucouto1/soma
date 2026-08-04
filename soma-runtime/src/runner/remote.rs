//! RemoteRunner — executes plans on remote workers via a Transport abstraction.
//!
//! The Transport trait abstracts HOW to communicate with workers (WS, HTTP, gRPC, etc.).
//! RemoteRunner implements Runner by serializing fit/forward calls and sending them
//! through the transport layer.

use super::{RunContext, Runner};
use crate::node_catalog::NodeCatalog;

use crate::executor::RunMode;
use somatize_compiler::ExecutionPlan;
use somatize_core::error::Result;
use somatize_core::value::Value;
use std::collections::HashMap;

/// Abstraction for communicating with remote workers.
/// Implemented by WsTransport (WebSocket), but could be HTTP, gRPC, etc.
pub trait Transport: Send + Sync {
    /// Send a plan for execution and receive the output + trained states.
    ///
    /// `mode` says what to do with the nodes, and carries the labels when
    /// there are any. It replaced a `fit_mode: bool` sitting beside an
    /// `y: Option<&Value>` — a flag selecting between two operations with
    /// differently shaped results, and a parameter that meant nothing
    /// unless the flag was set. It is the same [`RunMode`] the local
    /// executor reads, so the two paths cannot disagree about what a fit is.
    fn execute(
        &self,
        plan: &ExecutionPlan,
        filters: &NodeCatalog,
        input: &Value,
        mode: &RunMode,
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
        let (output, _) = self.execute(&plan, &filters, &input_val, &RunMode::Forward)?;
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
        ctx: &RunContext<'_>,
        input: &Value,
        y: Option<&Value>,
    ) -> Result<(Value, HashMap<String, Value>)> {
        self.transport
            .execute(plan, ctx.catalog, input, &RunMode::Fit { y: y.cloned() })
    }

    fn forward(&self, plan: &ExecutionPlan, ctx: &RunContext<'_>, input: &Value) -> Result<Value> {
        let (output, _states) =
            self.transport
                .execute(plan, ctx.catalog, input, &RunMode::Forward)?;
        Ok(output)
    }
}
