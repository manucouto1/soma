//! WebSocket-based RemoteExecutor — routes nodes to workers and delegates via WsTransport.

use somatize_core::error::{Result, SomaError};
use somatize_core::filter::RemoteTarget;
use somatize_core::value::Value;
use somatize_runtime::executor::RemoteExecutor;
use somatize_runtime::runner::Transport;

use crate::ws_transport::WsTransport;
use std::sync::RwLock;

/// Routes remote execution requests to the appropriate worker via WebSocket.
///
/// Workers are registered by address + optional token + tags.
/// When `execute_remote` is called, it finds a matching worker by target,
/// creates a WsTransport, and delegates the execution.
pub struct WsRemoteExecutor {
    workers: RwLock<Vec<WorkerEntry>>,
}

#[derive(Clone)]
struct WorkerEntry {
    address: String,
    token: Option<String>,
    tags: Vec<String>,
}

impl WsRemoteExecutor {
    pub fn new() -> Self {
        Self {
            workers: RwLock::new(Vec::new()),
        }
    }

    pub fn add_worker(&self, address: impl Into<String>, token: Option<String>, tags: Vec<String>) {
        self.workers.write().unwrap().push(WorkerEntry {
            address: address.into(),
            token,
            tags,
        });
    }

    fn find_worker(&self, target: &RemoteTarget) -> Option<WorkerEntry> {
        let workers = self.workers.read().unwrap();
        match target {
            RemoteTarget::WorkerId(id) => workers.iter().find(|w| w.address.contains(id)).cloned(),
            RemoteTarget::Tag(tag) => workers.iter().find(|w| w.tags.contains(tag)).cloned(),
        }
    }

    pub fn has_workers(&self) -> bool {
        !self.workers.read().unwrap().is_empty()
    }
}

impl Default for WsRemoteExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl RemoteExecutor for WsRemoteExecutor {
    fn execute_remote(
        &self,
        node_id: &str,
        target: &RemoteTarget,
        input: Option<&Value>,
    ) -> Result<Value> {
        let worker = self
            .find_worker(target)
            .ok_or_else(|| SomaError::Other(format!("no worker found for target {target:?}")))?;

        tracing::info!(
            "Dispatching node '{node_id}' to worker at {}",
            worker.address
        );

        // Create a WsTransport for this worker and delegate
        let transport = WsTransport::new(&worker.address, worker.token.clone());
        let plan = somatize_compiler::ExecutionPlan::Execute {
            node_id: node_id.to_string(),
        };
        let empty_filters = somatize_runtime::FilterLibrary::new();
        let input_val = input.cloned().unwrap_or(Value::Empty);

        let (output, _states) =
            transport.execute(&plan, &empty_filters, &input_val, None, false)?;
        Ok(output)
    }
}
