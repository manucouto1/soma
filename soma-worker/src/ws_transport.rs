//! WebSocket-based Transport implementation.
//!
//! Implements the `Transport` trait from soma-runtime, sending plans to
//! remote workers via WebSocket and receiving results.

use crate::error::{Result, WorkerError};
use somatize_compiler::ExecutionPlan;
use somatize_core::value::Value;
use somatize_runtime::executor::RunMode;
use somatize_runtime::node_catalog::NodeCatalog;
use somatize_runtime::runner::Transport;
use std::collections::HashMap;

use crate::protocol::*;

/// Transport implementation using WebSocket.
pub struct WsTransport {
    pub address: String,
    pub token: Option<String>,
}

/// Drive `fut` to completion from a synchronous caller.
///
/// On a thread of our own, with a runtime of its own, always. [`Transport`]
/// is a synchronous trait — the effect driver calls it from
/// `std::thread::scope` — but a caller may equally already be inside a
/// tokio runtime: `soma.Worker.serve()` runs an axum server on a thread of
/// the user's Python process, and the Python bindings dispatch plans from
/// there. `block_on` inside a runtime is a *panic*, not an error.
///
/// This file used to hold two contradictory answers to that. `upload` and
/// `resolve_output` paid for a thread and said why in a comment;
/// `send_msg`, `notify` and `stream_plan` built a bare current-thread
/// runtime and blocked on it, so they worked until they were called from
/// the wrong place. One rule now: never assume you are outside a runtime.
///
/// Scoped rather than detached, so the future may borrow `self`.
fn on_own_runtime<F, T>(fut: F) -> Result<T>
where
    F: std::future::Future<Output = Result<T>> + Send,
    T: Send,
{
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|e| WorkerError::Concurrency(format!("tokio: {e}")))?
                    .block_on(fut)
            })
            .join()
            .map_err(|_| WorkerError::Concurrency("transport thread panicked".into()))?
    })
}

impl WsTransport {
    pub fn new(address: impl Into<String>, token: Option<String>) -> Self {
        Self {
            address: address.into(),
            token,
        }
    }

    /// The worker's HTTP address, for the bulk endpoints.
    fn http_addr(&self) -> String {
        self.address
            .replace("ws://", "http://")
            .replace("wss://", "https://")
    }

    /// Send a `CoordinatorToWorker` message and wait for the response.
    ///
    /// Public because a caller that builds its own plan — the Python
    /// bindings decide which worker gets which filters, which is policy,
    /// not transport — should not have to open its own socket to ship it.
    /// A second `connect_async` elsewhere is a second place to get the
    /// frame-size configuration wrong.
    pub fn send_msg(&self, msg: &CoordinatorToWorker) -> Result<WorkerToCoordinator> {
        on_own_runtime(async {
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
                    .map_err(|e| WorkerError::Transport(format!("WS connect: {e}")))?;

            use futures_util::{SinkExt, StreamExt};
            use tokio_tungstenite::tungstenite::Message;

            let json = serde_json::to_string(msg)
                .map_err(|e| WorkerError::Encoding(format!("serialize: {e}")))?;

            ws.send(Message::Text(json.into()))
                .await
                .map_err(|e| WorkerError::Transport(format!("WS send: {e}")))?;

            while let Some(Ok(Message::Text(response))) = ws.next().await {
                if let Ok(result) = serde_json::from_str::<WorkerToCoordinator>(&response) {
                    let _ = ws.close(None).await;
                    return Ok(result);
                }
            }

            Err(WorkerError::Transport(
                "worker closed without response".into(),
            ))
        })
    }

    /// Send a message without waiting for an answer.
    ///
    /// `Shutdown` is the one that needs this: the worker is not going to
    /// reply, so [`WsTransport::send_msg`] would block until the socket
    /// closed.
    pub fn notify(&self, msg: &CoordinatorToWorker) -> Result<()> {
        let url = match &self.token {
            Some(t) => format!("{}/ws?token={t}", self.address),
            None => format!("{}/ws", self.address),
        };
        let json = serde_json::to_string(msg)
            .map_err(|e| WorkerError::Encoding(format!("serialize: {e}")))?;

        on_own_runtime(async move {
            use futures_util::SinkExt;
            use tokio_tungstenite::tungstenite::Message;

            let (mut ws, _) = tokio_tungstenite::connect_async(&url)
                .await
                .map_err(|e| WorkerError::Transport(format!("WS connect: {e}")))?;
            ws.send(Message::Text(json.into()))
                .await
                .map_err(|e| WorkerError::Transport(format!("WS send: {e}")))
        })
    }

    /// Upload a value to the worker's `/upload` endpoint, for payloads too
    /// large to travel inline in a WebSocket message.
    pub fn upload(&self, value: &Value) -> Result<somatize_core::store::DataRef> {
        let url = format!("{}/upload", self.http_addr());
        let body = serde_json::to_vec(value)
            .map_err(|e| WorkerError::Encoding(format!("serialize upload: {e}")))?;
        let token = self.token.clone();

        // Blocking HTTP on its own thread: this may be called from inside
        // a tokio runtime, and nesting one is a panic.
        std::thread::spawn(move || {
            let client = reqwest::blocking::Client::new();
            let mut req = client
                .post(&url)
                .header("Content-Type", "application/json")
                .body(body);
            if let Some(t) = &token {
                req = req.query(&[("token", t.as_str())]);
            }
            let resp = req
                .send()
                .map_err(|e| WorkerError::Transport(format!("HTTP upload: {e}")))?;
            if !resp.status().is_success() {
                return Err(WorkerError::Transport(format!(
                    "HTTP upload failed: {}",
                    resp.status()
                )));
            }
            resp.json::<somatize_core::store::DataRef>()
                .map_err(|e| WorkerError::Encoding(format!("parse upload response: {e}")))
        })
        .join()
        .map_err(|_| WorkerError::Concurrency("upload thread panicked".into()))?
    }

    /// Ship a plan and a stream of chunks over one WebSocket, collecting
    /// results as they come back.
    ///
    /// The binary side of the protocol. It lived in the Python bindings,
    /// which meant a second hand-rolled `connect_async` and a second copy
    /// of the msgpack `StreamMessage` framing, a crate away from the enum
    /// that defines it.
    pub fn stream_plan(&self, plan: SerializedPlan, chunks: Vec<Value>) -> Result<Value> {
        let stream_id = plan.plan_id.clone();
        let total_chunks = chunks.len();
        let url = match &self.token {
            Some(t) => format!("{}/ws?token={t}", self.address),
            None => format!("{}/ws", self.address),
        };

        on_own_runtime(async move {
            use futures_util::StreamExt;
            use tokio_tungstenite::tungstenite::Message;

            let (mut ws, _) = tokio_tungstenite::connect_async(&url)
                .await
                .map_err(|e| WorkerError::Transport(format!("WS connect: {e}")))?;

            send_frame(
                &mut ws,
                StreamMessage::StreamBegin {
                    stream_id: stream_id.clone(),
                    plan_id: stream_id.clone(),
                    total_chunks: Some(total_chunks),
                    plan: Box::new(plan),
                },
            )
            .await?;

            let mut results: Vec<Value> = Vec::new();

            for (i, chunk) in chunks.into_iter().enumerate() {
                send_frame(
                    &mut ws,
                    StreamMessage::ChunkData {
                        stream_id: stream_id.clone(),
                        chunk_index: i,
                        value: chunk,
                    },
                )
                .await?;

                // Drain whatever has come back so far, so a long stream
                // does not queue every result until the end.
                while let Ok(Some(Ok(Message::Binary(resp)))) =
                    tokio::time::timeout(std::time::Duration::from_millis(1), ws.next()).await
                {
                    if let Ok(StreamMessage::ChunkResult { value, .. }) =
                        crate::protocol::decode_frame(&resp)
                    {
                        results.push(value);
                    }
                }
            }

            send_frame(
                &mut ws,
                StreamMessage::StreamEnd {
                    stream_id: stream_id.clone(),
                },
            )
            .await?;

            // Whatever is left, then the barrier filters' flush.
            let mut flushed: Option<Value> = None;
            while let Some(Ok(Message::Binary(resp))) = ws.next().await {
                match crate::protocol::decode_frame(&resp) {
                    Ok(StreamMessage::ChunkResult { value, .. }) => results.push(value),
                    Ok(StreamMessage::StreamComplete { result, .. }) => match result {
                        PlanResult::Success { output, .. } => {
                            let v = self.resolve_output(&output)?;
                            if !v.is_empty() {
                                flushed = Some(v);
                            }
                            break;
                        }
                        PlanResult::Failed { error, .. } => {
                            return Err(WorkerError::Remote(format!("stream error: {error}")));
                        }
                    },
                    _ => {}
                }
            }

            if let Some(v) = flushed {
                results.push(v);
            }
            match results.len() {
                0 => Ok(Value::Empty),
                1 => Ok(results.into_iter().next().unwrap()),
                _ => Ok(somatize_runtime::executors::materialize_buffer(&results)?),
            }
        })
    }

    /// Resolve OutputDelivery — inline or download via HTTP.
    pub fn resolve_output(&self, delivery: &OutputDelivery) -> Result<Value> {
        match delivery {
            OutputDelivery::Inline { value } => Ok(value.clone()),
            OutputDelivery::Reference { data_ref } => {
                let url = format!("{}/download", self.http_addr());
                let ref_json = serde_json::to_string(data_ref)
                    .map_err(|e| WorkerError::Encoding(format!("serialize ref: {e}")))?;
                let token = self.token.clone();

                std::thread::spawn(move || {
                    let client = reqwest::blocking::Client::new();
                    let mut req = client.get(&url).query(&[("ref", &ref_json)]);
                    if let Some(t) = &token {
                        req = req.query(&[("token", t.as_str())]);
                    }
                    let resp = req
                        .send()
                        .map_err(|e| WorkerError::Transport(format!("HTTP download: {e}")))?;
                    let bytes = resp
                        .bytes()
                        .map_err(|e| WorkerError::Transport(format!("read response: {e}")))?;
                    serde_json::from_slice(&bytes)
                        .map_err(|e| WorkerError::Encoding(format!("deserialize: {e}")))
                })
                .join()
                .map_err(|_| WorkerError::Concurrency("download thread panicked".into()))?
            }
        }
    }
}

/// One msgpack frame down a socket.
async fn send_frame<S>(ws: &mut S, msg: StreamMessage) -> Result<()>
where
    S: futures_util::Sink<tokio_tungstenite::tungstenite::Message> + Unpin,
    S::Error: std::fmt::Display,
{
    use futures_util::SinkExt;
    let bytes = crate::protocol::encode_frame(&msg)?;
    ws.send(tokio_tungstenite::tungstenite::Message::Binary(
        bytes.into(),
    ))
    .await
    .map_err(|e| WorkerError::Transport(format!("WS send: {e}")))
}

/// The seam.
///
/// `Transport` is a `soma-runtime` trait, so these return `SomaError`
/// while everything behind them is a typed [`WorkerError`]. A refused
/// socket and a reply that would not decode stay distinguishable inside
/// this crate, which is where the retry decision is made.
impl Transport for WsTransport {
    fn execute(
        &self,
        plan: &ExecutionPlan,
        _filters: &NodeCatalog,
        input: &Value,
        mode: &RunMode,
    ) -> somatize_core::error::Result<(Value, HashMap<String, Value>)> {
        let serialized = SerializedPlan {
            protocol_version: PROTOCOL_VERSION,
            plan_id: somatize_core::util::timestamp_id("remote"),
            plan: plan.clone(),
            input: Some(InputSource::Inline {
                value: input.clone(),
            }),
            filters: vec![], // TODO: serialize from NodeCatalog if needed
            mode: match mode {
                RunMode::Fit { y } => ExecutionMode::Fit {
                    y: y.clone(),
                    batch_size: None,
                },
                RunMode::Forward => ExecutionMode::Forward,
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
                    Err(WorkerError::Remote(format!("remote: {error}")).into())
                }
            },
            other => {
                Err(WorkerError::Transport(format!("expected PlanResult, got: {other:?}")).into())
            }
        }
    }

    fn get_state(
        &self,
        node_ids: &[String],
    ) -> somatize_core::error::Result<HashMap<String, Value>> {
        let msg = CoordinatorToWorker::GetState {
            plan_id: String::new(),
            node_ids: node_ids.to_vec(),
        };
        match self.send_msg(&msg)? {
            WorkerToCoordinator::StateResult { states, .. } => Ok(states),
            other => {
                Err(WorkerError::Transport(format!("expected StateResult, got: {other:?}")).into())
            }
        }
    }

    fn set_state(&self, states: &HashMap<String, Value>) -> somatize_core::error::Result<()> {
        let msg = CoordinatorToWorker::SetState {
            plan_id: String::new(),
            states: states.clone(),
        };
        self.send_msg(&msg)?;
        Ok(())
    }

    fn get_gradients(
        &self,
        node_ids: &[String],
    ) -> somatize_core::error::Result<HashMap<String, Value>> {
        let msg = CoordinatorToWorker::GetGradients {
            plan_id: String::new(),
            node_ids: node_ids.to_vec(),
        };
        match self.send_msg(&msg)? {
            WorkerToCoordinator::GradientsResult { gradients, .. } => Ok(gradients),
            other => Err(WorkerError::Transport(format!(
                "expected GradientsResult, got: {other:?}"
            ))
            .into()),
        }
    }

    fn apply_gradients(
        &self,
        gradients: &HashMap<String, Value>,
    ) -> somatize_core::error::Result<()> {
        let msg = CoordinatorToWorker::ApplyGradients {
            plan_id: String::new(),
            gradients: gradients.clone(),
        };
        self.send_msg(&msg)?;
        Ok(())
    }
}
