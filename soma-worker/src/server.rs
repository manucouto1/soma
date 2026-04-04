//! Axum HTTP/WebSocket server for the worker process.
//!
//! Supports optional bearer token authentication on WebSocket connections.
//! Set a token via [`worker_router_authenticated`] or the `--token` CLI flag.

use crate::env_manager::{EnvManager, EnvType};
use crate::protocol::*;
use crate::worker::Worker;
use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::extract::ws::{Message, WebSocket};
use axum::extract::{Query, State, WebSocketUpgrade};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use somatize_core::cache::CacheKey;
use somatize_core::store::{DataStore, LocalDataStore};
use somatize_core::value::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Shared state for the worker HTTP/WebSocket server.
struct ServerState {
    worker: Mutex<Worker>,
    env_manager: EnvManager,
    work_dir: PathBuf,
    /// Optional bearer token for authentication.
    token: Option<String>,
    /// Temporary local store for HTTP bulk uploads.
    temp_store: Arc<LocalDataStore>,
    /// Track upload times for automatic cleanup.
    temp_uploads: Mutex<HashMap<CacheKey, Instant>>,
    /// Active streaming sessions: stream_id → StreamExecutor.
    active_streams: Mutex<HashMap<String, somatize_runtime::executors::stream::StreamExecutor>>,
}

/// Build a worker server router (no authentication).
pub fn worker_router(worker: Worker) -> Router {
    worker_router_full(worker, "/tmp/soma-envs", "/tmp/soma-work", None)
}

/// Build a worker server router with custom directories.
pub fn worker_router_with_dirs(
    worker: Worker,
    env_dir: impl Into<PathBuf>,
    work_dir: impl Into<PathBuf>,
) -> Router {
    worker_router_full(worker, env_dir, work_dir, None)
}

/// Build a worker server router with authentication.
pub fn worker_router_authenticated(
    worker: Worker,
    env_dir: impl Into<PathBuf>,
    work_dir: impl Into<PathBuf>,
    token: impl Into<String>,
) -> Router {
    worker_router_full(worker, env_dir, work_dir, Some(token.into()))
}

fn worker_router_full(
    worker: Worker,
    env_dir: impl Into<PathBuf>,
    work_dir: impl Into<PathBuf>,
    token: Option<String>,
) -> Router {
    let work = work_dir.into();
    std::fs::create_dir_all(&work).ok();
    let temp_store = worker.temp_store().clone();
    let state = Arc::new(ServerState {
        worker: Mutex::new(worker),
        env_manager: EnvManager::new(env_dir, EnvType::Venv),
        work_dir: work,
        token,
        temp_store,
        temp_uploads: Mutex::new(HashMap::new()),
        active_streams: Mutex::new(HashMap::new()),
    });
    // Background cleanup: remove temp uploads older than 1 hour
    let cleanup_state = state.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(300));
        loop {
            interval.tick().await;
            let cutoff = Instant::now() - std::time::Duration::from_secs(3600);
            let expired: Vec<CacheKey> = {
                let uploads = cleanup_state
                    .temp_uploads
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                uploads
                    .iter()
                    .filter(|(_, created)| **created < cutoff)
                    .map(|(k, _)| k.clone())
                    .collect()
            };
            if !expired.is_empty() {
                let mut uploads = cleanup_state
                    .temp_uploads
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                for key in &expired {
                    let data_ref = somatize_core::store::DataRef::Cached {
                        cache_key: key.clone(),
                    };
                    let _ = cleanup_state.temp_store.remove(&data_ref);
                    uploads.remove(key);
                }
                tracing::info!("Cleaned up {} expired temp uploads", expired.len());
            }
        }
    });

    Router::new()
        .route("/health", get(health))
        .route("/info", get(info))
        .route("/upload", post(upload_data))
        .route("/download", get(download_data))
        .route("/ws", get(ws_handler))
        .layer(DefaultBodyLimit::disable()) // No limit — workers handle arbitrary data sizes
        .with_state(state)
}

/// Start a worker server on the given address.
pub async fn serve_worker(worker: Worker, addr: &str) -> Result<(), Box<dyn std::error::Error>> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("Worker server listening on {addr}");
    axum::serve(listener, worker_router(worker)).await?;
    Ok(())
}

/// Start a worker server with authentication.
pub async fn serve_worker_authenticated(
    worker: Worker,
    addr: &str,
    token: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("Worker server listening on {addr} (authenticated)");
    let router = worker_router_authenticated(worker, "/tmp/soma-envs", "/tmp/soma-work", token);
    axum::serve(listener, router).await?;
    Ok(())
}

async fn health() -> &'static str {
    "ok"
}

async fn info(State(state): State<Arc<ServerState>>) -> impl IntoResponse {
    let worker = state.worker.lock().unwrap_or_else(|e| e.into_inner());
    let msg = worker.registration_message();
    axum::Json(serde_json::to_value(msg).unwrap_or_default())
}

/// Upload data via HTTP for large payloads that exceed WebSocket limits.
///
/// Accepts msgpack or JSON body, stores in temp_store, returns DataRef as JSON.
/// Token auth via `?token=` query param (same as WebSocket).
async fn upload_data(
    Query(params): Query<WsParams>,
    State(state): State<Arc<ServerState>>,
    body: axum::body::Bytes,
) -> Result<impl IntoResponse, StatusCode> {
    // Validate token
    if let Some(expected) = &state.token {
        match &params.token {
            Some(provided) if provided == expected => {}
            _ => return Err(StatusCode::UNAUTHORIZED),
        }
    }

    // Deserialize: try msgpack first, then JSON
    let value: Value = rmp_serde::from_slice(&body)
        .or_else(|_| serde_json::from_slice(&body))
        .map_err(|_| StatusCode::BAD_REQUEST)?;

    let key = CacheKey::hash_data(&body);
    let data_ref = state
        .temp_store
        .put(&key, &value)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    // Track for cleanup
    state
        .temp_uploads
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(key, Instant::now());

    tracing::info!("Uploaded {} bytes → {data_ref:?}", body.len());

    Ok(axum::Json(
        serde_json::to_value(&data_ref).unwrap_or_default(),
    ))
}

/// Query params for data download.
#[derive(serde::Deserialize)]
struct DownloadParams {
    /// JSON-serialized DataRef (same format returned by /upload).
    #[serde(rename = "ref")]
    data_ref: String,
    token: Option<String>,
}

/// Download data from the worker's temp store by DataRef.
///
/// Returns msgpack-encoded Value. Used by clients to resolve
/// `OutputDelivery::Reference` results after plan execution.
async fn download_data(
    Query(params): Query<DownloadParams>,
    State(state): State<Arc<ServerState>>,
) -> Result<impl IntoResponse, StatusCode> {
    // Validate token
    if let Some(expected) = &state.token {
        match &params.token {
            Some(provided) if provided == expected => {}
            _ => return Err(StatusCode::UNAUTHORIZED),
        }
    }

    let data_ref: somatize_core::store::DataRef =
        serde_json::from_str(&params.data_ref).map_err(|_| StatusCode::BAD_REQUEST)?;

    let value = state
        .temp_store
        .get(&data_ref)
        .map_err(|_| StatusCode::NOT_FOUND)?;

    let bytes = serde_json::to_vec(&value).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok((
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        bytes,
    ))
}

/// Query params for WebSocket authentication.
#[derive(serde::Deserialize, Default)]
struct WsParams {
    token: Option<String>,
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    Query(params): Query<WsParams>,
    State(state): State<Arc<ServerState>>,
) -> Result<impl IntoResponse, StatusCode> {
    // Validate token if server requires one
    if let Some(expected) = &state.token {
        match &params.token {
            Some(provided) if provided == expected => {}
            _ => return Err(StatusCode::UNAUTHORIZED),
        }
    }
    Ok(ws
        .max_message_size(usize::MAX) // No limit on incoming WS messages
        .max_frame_size(usize::MAX)
        .on_upgrade(move |socket| handle_ws(socket, state)))
}

async fn handle_ws(mut socket: WebSocket, state: Arc<ServerState>) {
    loop {
        match socket.recv().await {
            Some(Ok(Message::Text(text))) => {
                let response = match serde_json::from_str::<CoordinatorToWorker>(&text) {
                    Ok(CoordinatorToWorker::AssignPlan { plan }) => {
                        let mut worker = state.worker.lock().unwrap_or_else(|e| e.into_inner());
                        let plan_id = plan.plan_id.clone();
                        let worker_id = worker.id.clone();
                        let result = worker.execute_plan(&plan);
                        let msg = WorkerToCoordinator::PlanResult {
                            worker_id,
                            plan_id,
                            result,
                        };
                        serde_json::to_string(&msg).unwrap_or_default()
                    }
                    Ok(CoordinatorToWorker::StatusRequest) => {
                        let worker = state.worker.lock().unwrap_or_else(|e| e.into_inner());
                        serde_json::to_string(&worker.registration_message()).unwrap_or_default()
                    }
                    Ok(CoordinatorToWorker::CancelPlan { .. }) => {
                        r#"{"status": "cancel_not_implemented"}"#.to_string()
                    }
                    Ok(CoordinatorToWorker::AssignPythonJob { job }) => {
                        // Send progress messages during execution
                        let messages = execute_python_job_with_progress(&state, &job);
                        // Send all but the last as intermediate messages
                        for msg in &messages[..messages.len().saturating_sub(1)] {
                            if socket
                                .send(Message::Text(msg.clone().into()))
                                .await
                                .is_err()
                            {
                                break;
                            }
                        }
                        // Return the last message (result) through the normal path
                        messages.into_iter().last().unwrap_or_default()
                    }
                    Ok(CoordinatorToWorker::Ping) => r#"{"type":"Pong"}"#.to_string(),
                    Ok(CoordinatorToWorker::Registered { .. }) => continue,
                    Ok(CoordinatorToWorker::Shutdown { reason }) => {
                        tracing::info!("Shutdown requested: {reason}");
                        let _ = socket
                            .send(Message::Text(r#"{"type":"ShutdownAck"}"#.into()))
                            .await;
                        std::process::exit(0);
                    }
                    Ok(
                        CoordinatorToWorker::GetState { .. }
                        | CoordinatorToWorker::SetState { .. }
                        | CoordinatorToWorker::GetGradients { .. }
                        | CoordinatorToWorker::ApplyGradients { .. },
                    ) => {
                        r#"{"error":"get_state/set_state/get_gradients/apply_gradients not yet implemented for SubprocessFilter"}"#.to_string()
                    }
                    Err(e) => {
                        format!(r#"{{"error": "invalid message: {e}"}}"#)
                    }
                };

                if socket.send(Message::Text(response.into())).await.is_err() {
                    break;
                }
            }
            Some(Ok(Message::Binary(bytes))) => {
                if let Ok(stream_msg) = rmp_serde::from_slice::<StreamMessage>(&bytes) {
                    let reply = handle_stream_message(stream_msg, &state);
                    if let Some(reply_msg) = reply {
                        let reply_bytes = rmp_serde::to_vec(&reply_msg).unwrap_or_default();
                        if socket
                            .send(Message::Binary(reply_bytes.into()))
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                }
            }
            Some(Ok(Message::Close(_))) | None => break,
            _ => {}
        }
    }
}

/// Handle a streaming protocol message. Returns an optional reply.
fn handle_stream_message(msg: StreamMessage, state: &Arc<ServerState>) -> Option<StreamMessage> {
    use somatize_runtime::executors::stream::{FittedFilter, StreamExecutor};

    match msg {
        StreamMessage::StreamBegin {
            stream_id, plan, ..
        } => {
            // Build StreamExecutor from the plan's filters
            let mut worker = state.worker.lock().unwrap_or_else(|e| e.into_inner());

            // Register filters via SubprocessFilter backed by a shared PythonProcess
            let filter_specs: Vec<(String, Vec<u8>, bool)> = plan
                .filters
                .iter()
                .map(|sf| (sf.node_id.clone(), sf.pickled_filter.clone(), sf.trainable))
                .collect();

            if !filter_specs.is_empty() {
                let process = Arc::new(std::sync::Mutex::new(
                    crate::python_process::PythonProcess::spawn("python3", &filter_specs)
                        .expect("PythonProcess spawn failed"),
                ));

                for sf in &plan.filters {
                    let filter: Box<dyn somatize_core::filter::Filter> =
                        Box::new(crate::python_process::SubprocessFilter::new(
                            process.clone(),
                            sf.node_id.clone(),
                            sf.trainable,
                        ));
                    worker.register_filter(&sf.node_id, filter);
                    if let Some(s) = &sf.state {
                        worker.set_filter_state(&sf.node_id, s.clone());
                    }
                }
            }

            // Build FittedFilter list from plan node order
            let node_ids = plan.plan.node_ids();
            let fitted: Vec<FittedFilter> = node_ids
                .iter()
                .filter_map(|id| {
                    let filter = worker.get_filter(id)?;
                    let filter_state = worker.get_filter_state(id);
                    Some(FittedFilter {
                        name: id.to_string(),
                        filter,
                        state: filter_state,
                    })
                })
                .collect();

            let executor = StreamExecutor::new(fitted);
            state
                .active_streams
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(stream_id, executor);

            None // No reply for StreamBegin
        }
        StreamMessage::ChunkData {
            stream_id,
            chunk_index,
            value,
        } => {
            let mut streams = state
                .active_streams
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if let Some(executor) = streams.get_mut(&stream_id) {
                match executor.process_chunk(value) {
                    Ok(Some(result)) => Some(StreamMessage::ChunkResult {
                        stream_id,
                        chunk_index,
                        value: result,
                    }),
                    Ok(None) => None, // Barrier mode — no result yet
                    Err(e) => Some(StreamMessage::StreamComplete {
                        stream_id,
                        result: PlanResult::Failed {
                            error: e.to_string(),
                            duration_ms: 0,
                        },
                    }),
                }
            } else {
                Some(StreamMessage::StreamComplete {
                    stream_id,
                    result: PlanResult::Failed {
                        error: "unknown stream_id".to_string(),
                        duration_ms: 0,
                    },
                })
            }
        }
        StreamMessage::StreamEnd { stream_id } => {
            let mut streams = state
                .active_streams
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if let Some(mut executor) = streams.remove(&stream_id) {
                // Flush barrier filters
                let output = executor
                    .flush()
                    .unwrap_or(None)
                    .unwrap_or(somatize_core::value::Value::Empty);
                Some(StreamMessage::StreamComplete {
                    stream_id,
                    result: PlanResult::Success {
                        output: OutputDelivery::Inline { value: output },
                        duration_ms: 0,
                        states: std::collections::HashMap::new(),
                    },
                })
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Execute a Python pipeline job with progress reporting.
fn execute_python_job_with_progress(state: &ServerState, job: &PythonPipelineJob) -> Vec<String> {
    let start = Instant::now();
    let mut messages = Vec::new();
    let worker_id = {
        let w = state.worker.lock().unwrap_or_else(|e| e.into_inner());
        w.id.clone()
    };

    let progress = |wid: &str, jid: &str, phase: &str, step: u32, total: u32| -> String {
        serde_json::to_string(&WorkerToCoordinator::JobProgress {
            worker_id: wid.into(),
            job_id: jid.into(),
            phase: phase.into(),
            step,
            total,
            metrics: serde_json::json!({}),
        })
        .unwrap_or_default()
    };

    // Phase 1/4: Environment setup
    messages.push(progress(&worker_id, &job.job_id, "environment", 1, 4));

    let python = match state
        .env_manager
        .ensure_env(&job.pipeline_id, &job.requirements)
    {
        Ok(p) => p,
        Err(e) => {
            tracing::error!("Failed to create env for pipeline {}: {e}", job.pipeline_id);
            let msg = WorkerToCoordinator::JobResult {
                worker_id,
                job_id: job.job_id.clone(),
                success: false,
                metrics: serde_json::json!({}),
                output: format!("Environment setup failed: {e}"),
                duration_ms: start.elapsed().as_millis() as u64,
            };
            messages.push(serde_json::to_string(&msg).unwrap_or_default());
            return messages;
        }
    };

    // Phase 2/4: Write files
    messages.push(progress(&worker_id, &job.job_id, "write_files", 2, 4));

    let job_dir = state.work_dir.join(format!("job-{}", job.job_id));
    if let Err(e) = std::fs::create_dir_all(&job_dir) {
        let msg = WorkerToCoordinator::JobResult {
            worker_id,
            job_id: job.job_id.clone(),
            success: false,
            metrics: serde_json::json!({}),
            output: format!("Failed to create work dir: {e}"),
            duration_ms: start.elapsed().as_millis() as u64,
        };
        messages.push(serde_json::to_string(&msg).unwrap_or_default());
        return messages;
    }

    for file in &job.files {
        let file_path = job_dir.join(&file.path);
        if let Some(parent) = file_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        if let Err(e) = std::fs::write(&file_path, &file.content) {
            tracing::error!("Failed to write {}: {e}", file.path);
        }
    }

    // Phase 3/4: Execute
    messages.push(progress(&worker_id, &job.job_id, "execute", 3, 4));

    tracing::info!(
        "Executing job {} with python: {}",
        job.job_id,
        python.display()
    );

    let output = std::process::Command::new(&python)
        .arg(&job.entry_point)
        .current_dir(&job_dir)
        .env("PYTHONPATH", &job_dir)
        .output();

    let duration_ms = start.elapsed().as_millis() as u64;

    // Phase 4/4: Collect results
    let _ = std::fs::remove_dir_all(&job_dir);
    messages.push(progress(&worker_id, &job.job_id, "collect_results", 4, 4));

    let result_msg = match output {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout).to_string();
            let stderr = String::from_utf8_lossy(&out.stderr).to_string();
            let success = out.status.success();

            let metrics = stdout
                .lines()
                .rev()
                .find_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
                .unwrap_or(serde_json::json!({}));

            if !success {
                tracing::warn!(
                    "Job {} failed: {}",
                    job.job_id,
                    stderr.chars().take(200).collect::<String>()
                );
            }

            WorkerToCoordinator::JobResult {
                worker_id,
                job_id: job.job_id.clone(),
                success,
                metrics,
                output: if success {
                    stdout
                } else {
                    format!("STDERR:\n{stderr}\nSTDOUT:\n{stdout}")
                },
                duration_ms,
            }
        }
        Err(e) => WorkerToCoordinator::JobResult {
            worker_id,
            job_id: job.job_id.clone(),
            success: false,
            metrics: serde_json::json!({}),
            output: format!("Failed to execute: {e}"),
            duration_ms,
        },
    };
    messages.push(serde_json::to_string(&result_msg).unwrap_or_default());
    messages
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::Capabilities;
    fn make_worker() -> Worker {
        Worker::new(
            "test_worker",
            Capabilities {
                cpu_cores: 4,
                ram_bytes: 8_000_000_000,
                gpus: vec![],
                python_envs: vec![],
                tags: vec!["test".into()],
            },
        )
    }

    #[tokio::test]
    async fn router_builds() {
        let _router = worker_router(make_worker());
    }

    #[tokio::test]
    async fn health_returns_ok() {
        let resp = health().await;
        assert_eq!(resp, "ok");
    }

    #[tokio::test]
    async fn full_server_starts_and_stops() {
        let worker = make_worker();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server = tokio::spawn(async move {
            axum::serve(listener, worker_router(worker)).await.unwrap();
        });

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let client = reqwest::Client::new();
        let resp = client
            .get(format!("http://{addr}/health"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.text().await.unwrap(), "ok");

        let resp = client
            .get(format!("http://{addr}/info"))
            .send()
            .await
            .unwrap();
        let json: serde_json::Value = resp.json().await.unwrap();
        assert!(json.get("type").is_some() || json.get("worker_id").is_some());

        server.abort();
    }
}
