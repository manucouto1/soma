//! `Transport` — how a plan reaches a remote worker.
//!
//! The trait abstracts the wire (WebSocket today; HTTP or gRPC would fit
//! the same shape) and nothing more. There is no `RemoteRunner`: the
//! second `Runner` implementation that once sat here was constructed by
//! nothing for its whole life, and the compiler's `ExecutionPlan::Remote`
//! arm is what actually sends work out.

use crate::executor::RunMode;
use crate::node_catalog::NodeCatalog;
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
    ///
    /// `seed` is the run's experiment seed, and it is a parameter rather
    /// than something the transport digs out because the transport has no
    /// [`RunContext`](super::RunContext) to dig in. Without it the worker salts nothing, and a
    /// five-seed sweep run remotely shares one cache line across all five —
    /// the worker protocol's `SerializedPlan::seed` documents that as the
    /// bug it exists to close, and this path was still passing `None`.
    fn execute(
        &self,
        plan: &ExecutionPlan,
        filters: &NodeCatalog,
        input: &Value,
        mode: &RunMode,
        seed: Option<i64>,
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
    ///
    /// Unseeded, and it has to be: this takes a node id and nothing else,
    /// so there is no run to take a seed from. Callers that have a
    /// [`RunContext`](super::RunContext) should go through [`Transport::execute`] with
    /// `ctx.seed` instead of reaching for this.
    fn execute_node(&self, node_id: &str, input: Option<&Value>) -> Result<Value> {
        let plan = ExecutionPlan::Execute {
            node_id: node_id.to_string(),
        };
        let input_val = input.cloned().unwrap_or(Value::Empty);
        let filters = crate::node_catalog::NodeCatalog::new();
        let (output, _) = self.execute(&plan, &filters, &input_val, &RunMode::Forward, None)?;
        Ok(output)
    }
}
