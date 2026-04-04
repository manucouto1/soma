//! WebSocket-based StrategyContext implementation.
//!
//! Provides the concrete worker communication layer for strategy execution.
//! The strategy logic lives in `TrainingStrategy::fit()` (soma-core);
//! this module only implements HOW to talk to workers.

use somatize_core::error::{Result, SomaError};
use somatize_core::strategy::StrategyContext;
use somatize_core::value::Value;
use std::collections::HashMap;

use crate::protocol::*;

/// A pool of workers connected via WebSocket.
/// Implements `StrategyContext` so training strategies can coordinate workers.
pub struct WsWorkerPool {
    workers: Vec<WorkerConnection>,
    /// Serialized filters to send with each plan.
    filters: Vec<SerializedFilter>,
    /// The compiled execution plan.
    plan: somatize_compiler::ExecutionPlan,
}

/// A single WebSocket connection to a worker.
pub struct WorkerConnection {
    pub address: String,
    pub token: Option<String>,
    pub tags: Vec<String>,
}

impl WsWorkerPool {
    pub fn new(
        workers: Vec<WorkerConnection>,
        plan: somatize_compiler::ExecutionPlan,
        filters: Vec<SerializedFilter>,
    ) -> Self {
        Self {
            workers,
            filters,
            plan,
        }
    }
}

impl WorkerConnection {
    /// Send a CoordinatorToWorker message and wait for a response.
    fn send(&self, msg: &CoordinatorToWorker) -> Result<WorkerToCoordinator> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| SomaError::Other(format!("tokio: {e}")))?;

        rt.block_on(async {
            let url = if let Some(t) = &self.token {
                format!("{}/ws?token={t}", self.address)
            } else {
                format!("{}/ws", self.address)
            };

            let ws_config = {
                let mut c = tokio_tungstenite::tungstenite::protocol::WebSocketConfig::default();
                c.max_message_size = None;
                c.max_frame_size = None;
                c
            };

            let (mut ws, _) =
                tokio_tungstenite::connect_async_with_config(&url, Some(ws_config), false)
                    .await
                    .map_err(|e| SomaError::Other(format!("WS connect: {e}")))?;

            use futures_util::{SinkExt, StreamExt};
            use tokio_tungstenite::tungstenite::Message;

            let json = serde_json::to_string(msg)
                .map_err(|e| SomaError::Other(format!("serialize: {e}")))?;

            ws.send(Message::Text(json.into()))
                .await
                .map_err(|e| SomaError::Other(format!("WS send: {e}")))?;

            while let Some(Ok(Message::Text(response))) = ws.next().await {
                if let Ok(result) = serde_json::from_str::<WorkerToCoordinator>(&response) {
                    let _ = ws.close(None).await;
                    return Ok(result);
                }
            }

            Err(SomaError::Other("worker closed without response".into()))
        })
    }
}

impl StrategyContext for WsWorkerPool {
    fn num_workers(&self) -> usize {
        self.workers.len()
    }

    fn execute_on_worker(
        &self,
        worker_idx: usize,
        _plan_meta: &serde_json::Value,
        input: &Value,
        y: Option<&Value>,
    ) -> Result<HashMap<String, Value>> {
        let worker = self
            .workers
            .get(worker_idx)
            .ok_or_else(|| SomaError::Other(format!("worker {worker_idx} not found")))?;

        let plan_id = somatize_core::util::timestamp_id("strategy");
        let serialized = SerializedPlan {
            plan_id,
            plan: self.plan.clone(),
            input: Some(InputSource::Inline {
                value: input.clone(),
            }),
            filters: self.filters.clone(),
            mode: ExecutionMode::Fit { y: y.cloned() },
            metadata: serde_json::json!({}),
        };

        let msg = CoordinatorToWorker::AssignPlan { plan: serialized };
        match worker.send(&msg)? {
            WorkerToCoordinator::PlanResult { result, .. } => match result {
                PlanResult::Success { states, .. } => Ok(states),
                PlanResult::Failed { error, .. } => {
                    Err(SomaError::Other(format!("worker {worker_idx}: {error}")))
                }
            },
            other => Err(SomaError::Other(format!(
                "expected PlanResult, got: {other:?}"
            ))),
        }
    }

    fn get_state(&self, worker_idx: usize, node_ids: &[String]) -> Result<HashMap<String, Value>> {
        let worker = self
            .workers
            .get(worker_idx)
            .ok_or_else(|| SomaError::Other(format!("worker {worker_idx} not found")))?;

        let msg = CoordinatorToWorker::GetState {
            plan_id: String::new(),
            node_ids: node_ids.to_vec(),
        };
        match worker.send(&msg)? {
            WorkerToCoordinator::StateResult { states, .. } => Ok(states),
            other => Err(SomaError::Other(format!(
                "expected StateResult, got: {other:?}"
            ))),
        }
    }

    fn set_state(&self, worker_idx: usize, states: &HashMap<String, Value>) -> Result<()> {
        let worker = self
            .workers
            .get(worker_idx)
            .ok_or_else(|| SomaError::Other(format!("worker {worker_idx} not found")))?;

        let msg = CoordinatorToWorker::SetState {
            plan_id: String::new(),
            states: states.clone(),
        };
        worker.send(&msg)?;
        Ok(())
    }

    fn get_gradients(
        &self,
        worker_idx: usize,
        node_ids: &[String],
    ) -> Result<HashMap<String, Value>> {
        let worker = self
            .workers
            .get(worker_idx)
            .ok_or_else(|| SomaError::Other(format!("worker {worker_idx} not found")))?;

        let msg = CoordinatorToWorker::GetGradients {
            plan_id: String::new(),
            node_ids: node_ids.to_vec(),
        };
        match worker.send(&msg)? {
            WorkerToCoordinator::GradientsResult { gradients, .. } => Ok(gradients),
            other => Err(SomaError::Other(format!(
                "expected GradientsResult, got: {other:?}"
            ))),
        }
    }

    fn apply_gradients(&self, worker_idx: usize, gradients: &HashMap<String, Value>) -> Result<()> {
        let worker = self
            .workers
            .get(worker_idx)
            .ok_or_else(|| SomaError::Other(format!("worker {worker_idx} not found")))?;

        let msg = CoordinatorToWorker::ApplyGradients {
            plan_id: String::new(),
            gradients: gradients.clone(),
        };
        worker.send(&msg)?;
        Ok(())
    }
}
