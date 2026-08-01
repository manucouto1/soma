//! WebSocket-based Transport implementation.
//!
//! Implements the `Transport` trait from soma-runtime, sending plans to
//! remote workers via WebSocket and receiving results.

use somatize_compiler::ExecutionPlan;
use somatize_core::error::{Result, SomaError};
use somatize_core::value::Value;
use somatize_runtime::node_catalog::NodeCatalog;
use somatize_runtime::runner::Transport;
use std::collections::HashMap;

use crate::protocol::*;

/// Transport implementation using WebSocket.
pub struct WsTransport {
    pub address: String,
    pub token: Option<String>,
}

impl WsTransport {
    pub fn new(address: impl Into<String>, token: Option<String>) -> Self {
        Self {
            address: address.into(),
            token,
        }
    }

    /// Send a CoordinatorToWorker message and wait for response.
    fn send_msg(&self, msg: &CoordinatorToWorker) -> Result<WorkerToCoordinator> {
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

    /// Resolve OutputDelivery — inline or download via HTTP.
    fn resolve_output(&self, delivery: &OutputDelivery) -> Result<Value> {
        match delivery {
            OutputDelivery::Inline { value } => Ok(value.clone()),
            OutputDelivery::Reference { data_ref } => {
                let http_addr = self
                    .address
                    .replace("ws://", "http://")
                    .replace("wss://", "https://");
                let url = format!("{http_addr}/download");
                let ref_json = serde_json::to_string(data_ref)
                    .map_err(|e| SomaError::Other(format!("serialize ref: {e}")))?;
                let token = self.token.clone();

                std::thread::spawn(move || {
                    let client = reqwest::blocking::Client::new();
                    let mut req = client.get(&url).query(&[("ref", &ref_json)]);
                    if let Some(t) = &token {
                        req = req.query(&[("token", t.as_str())]);
                    }
                    let resp = req
                        .send()
                        .map_err(|e| SomaError::Other(format!("HTTP download: {e}")))?;
                    let bytes = resp
                        .bytes()
                        .map_err(|e| SomaError::Other(format!("read response: {e}")))?;
                    serde_json::from_slice(&bytes)
                        .map_err(|e| SomaError::Other(format!("deserialize: {e}")))
                })
                .join()
                .map_err(|_| SomaError::Other("download thread panicked".into()))?
            }
        }
    }
}

impl Transport for WsTransport {
    fn execute(
        &self,
        plan: &ExecutionPlan,
        _filters: &NodeCatalog,
        input: &Value,
        y: Option<&Value>,
        fit_mode: bool,
    ) -> Result<(Value, HashMap<String, Value>)> {
        let serialized = SerializedPlan {
            plan_id: somatize_core::util::timestamp_id("remote"),
            plan: plan.clone(),
            input: Some(InputSource::Inline {
                value: input.clone(),
            }),
            filters: vec![], // TODO: serialize from NodeCatalog if needed
            mode: if fit_mode {
                ExecutionMode::Fit {
                    y: y.cloned(),
                    batch_size: None,
                }
            } else {
                ExecutionMode::Forward
            },
            metadata: serde_json::json!({}),
        };

        let msg = CoordinatorToWorker::AssignPlan { plan: serialized };
        match self.send_msg(&msg)? {
            WorkerToCoordinator::PlanResult { result, .. } => match result {
                PlanResult::Success { output, states, .. } => {
                    let value = self.resolve_output(&output)?;
                    Ok((value, states))
                }
                PlanResult::Failed { error, .. } => {
                    Err(SomaError::Other(format!("remote: {error}")))
                }
            },
            other => Err(SomaError::Other(format!(
                "expected PlanResult, got: {other:?}"
            ))),
        }
    }

    fn get_state(&self, node_ids: &[String]) -> Result<HashMap<String, Value>> {
        let msg = CoordinatorToWorker::GetState {
            plan_id: String::new(),
            node_ids: node_ids.to_vec(),
        };
        match self.send_msg(&msg)? {
            WorkerToCoordinator::StateResult { states, .. } => Ok(states),
            other => Err(SomaError::Other(format!(
                "expected StateResult, got: {other:?}"
            ))),
        }
    }

    fn set_state(&self, states: &HashMap<String, Value>) -> Result<()> {
        let msg = CoordinatorToWorker::SetState {
            plan_id: String::new(),
            states: states.clone(),
        };
        self.send_msg(&msg)?;
        Ok(())
    }

    fn get_gradients(&self, node_ids: &[String]) -> Result<HashMap<String, Value>> {
        let msg = CoordinatorToWorker::GetGradients {
            plan_id: String::new(),
            node_ids: node_ids.to_vec(),
        };
        match self.send_msg(&msg)? {
            WorkerToCoordinator::GradientsResult { gradients, .. } => Ok(gradients),
            other => Err(SomaError::Other(format!(
                "expected GradientsResult, got: {other:?}"
            ))),
        }
    }

    fn apply_gradients(&self, gradients: &HashMap<String, Value>) -> Result<()> {
        let msg = CoordinatorToWorker::ApplyGradients {
            plan_id: String::new(),
            gradients: gradients.clone(),
        };
        self.send_msg(&msg)?;
        Ok(())
    }
}
