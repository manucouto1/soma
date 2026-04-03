//! WebSocket-based RemoteExecutor — sends plans to workers and collects results.

use somatize_core::error::{Result, SomaError};
use somatize_core::filter::RemoteTarget;
use somatize_core::value::Value;
use somatize_runtime::executor::RemoteExecutor;

use crate::protocol::*;
use std::sync::RwLock;

/// A remote executor that dispatches work to workers via WebSocket.
///
/// Workers are registered by address + optional token.
/// When `execute_remote` is called, it finds a matching worker,
/// connects via WS, sends the plan, and waits for the result.
pub struct WsRemoteExecutor {
    /// Registered workers: (address, token, tags)
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

    /// Register a worker endpoint.
    pub fn add_worker(&self, address: impl Into<String>, token: Option<String>, tags: Vec<String>) {
        let mut workers = self.workers.write().unwrap();
        workers.push(WorkerEntry {
            address: address.into(),
            token,
            tags,
        });
    }

    /// Find a worker matching the given target.
    fn find_worker(&self, target: &RemoteTarget) -> Option<WorkerEntry> {
        let workers = self.workers.read().unwrap();
        match target {
            RemoteTarget::WorkerId(id) => workers.iter().find(|w| w.address.contains(id)).cloned(),
            RemoteTarget::Tag(tag) => workers.iter().find(|w| w.tags.contains(tag)).cloned(),
        }
    }

    /// Send a plan to a worker via WebSocket and wait for the result.
    fn execute_on_worker(
        &self,
        worker: &WorkerEntry,
        node_id: &str,
        input: Option<&Value>,
    ) -> Result<Value> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| SomaError::Other(format!("tokio runtime: {e}")))?;

        rt.block_on(async {
            let url = if let Some(token) = &worker.token {
                format!("{}/ws?token={}", worker.address, token)
            } else {
                format!("{}/ws", worker.address)
            };

            let (mut ws, _) = tokio_tungstenite::connect_async(&url)
                .await
                .map_err(|e| SomaError::Other(format!("WS connect to {}: {e}", worker.address)))?;

            use futures_util::{SinkExt, StreamExt};
            use tokio_tungstenite::tungstenite::Message;

            // Build a simple plan: execute this one node
            let plan = SerializedPlan {
                plan_id: format!("remote_{node_id}"),
                plan: somatize_compiler::ExecutionPlan::Execute {
                    node_id: node_id.to_string(),
                },
                input: input.map(|v| InputSource::Inline { value: v.clone() }),
                metadata: serde_json::json!({}),
            };

            let msg = CoordinatorToWorker::AssignPlan { plan };
            let json = serde_json::to_string(&msg)
                .map_err(|e| SomaError::Other(format!("serialize: {e}")))?;

            ws.send(Message::Text(json.into()))
                .await
                .map_err(|e| SomaError::Other(format!("WS send: {e}")))?;

            // Wait for result
            while let Some(Ok(Message::Text(response))) = ws.next().await {
                if let Ok(result) = serde_json::from_str::<WorkerToCoordinator>(&response) {
                    match result {
                        WorkerToCoordinator::PlanResult { result, .. } => match result {
                            PlanResult::Success { output, .. } => {
                                let _ = ws.close(None).await;
                                return Ok(output);
                            }
                            PlanResult::Failed { error, .. } => {
                                let _ = ws.close(None).await;
                                return Err(SomaError::Execution {
                                    node_id: node_id.to_string(),
                                    message: error,
                                });
                            }
                        },
                        // Skip progress/event messages
                        _ => continue,
                    }
                }
            }

            let _ = ws.close(None).await;
            Err(SomaError::Other(format!(
                "worker {} closed without result",
                worker.address
            )))
        })
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

        self.execute_on_worker(&worker, node_id, input)
    }
}
