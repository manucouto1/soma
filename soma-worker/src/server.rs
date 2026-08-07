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
use somatize_core::data::store::{DataStore, LocalDataStore};
use somatize_core::data::value::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Resolves when the worker has been asked to stop.
///
/// A `Shutdown` message used to call `std::process::exit(0)` from inside the
/// WebSocket handler. That is wrong twice over: it skips the graceful
/// shutdown the binary wires up, and this crate is *embedded* — `Worker.serve()`
/// runs this server on a thread of the user's Python process, so exiting
/// killed their interpreter. Asking the serve loop to stop is the only
/// thing a library may do.
#[derive(Clone)]
pub struct ShutdownSignal(Arc<tokio::sync::Notify>);

impl ShutdownSignal {
    /// Wait until shutdown is requested.
    pub async fn wait(&self) {
        self.0.notified().await;
    }

    /// Ask the worker to stop.
    pub fn trigger(&self) {
        self.0.notify_waiters();
    }
}

/// Shared state for the worker HTTP/WebSocket server.
struct ServerState {
    worker: Mutex<Worker>,
    /// Notified when a `Shutdown` message arrives.
    shutdown: ShutdownSignal,
    env_manager: EnvManager,
    work_dir: PathBuf,
    /// Optional bearer token for authentication.
    token: Option<String>,
    /// Temporary local store for HTTP bulk uploads.
    temp_store: Arc<LocalDataStore>,
    /// Track upload times for automatic cleanup.
    temp_uploads: Mutex<HashMap<CacheKey, Instant>>,
    /// Active streaming sessions, one driver + context alive between
    /// WS messages — the state a chunked run must carry across RPCs.
    active_streams: Mutex<HashMap<String, StreamSession>>,
}

/// One in-flight streaming run: the driver, its execution context, and
/// the cache it reads/writes — held between `StreamBegin`, N ×
/// `ChunkData` and `StreamEnd`.
struct StreamSession {
    run: somatize_runtime::StreamRun,
    ctx: somatize_runtime::Context,
    cache: std::sync::Arc<dyn somatize_core::cache::CacheStore>,
    started: Instant,
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
    worker_router_with_shutdown(worker, env_dir, work_dir, token).0
}

/// Build a worker router together with the handle that stops it.
///
/// Callers that own the serve loop (the binary, [`serve_worker`]) pass the
/// signal to `with_graceful_shutdown` so a `Shutdown` message ends the
/// server the same way Ctrl+C does.
pub fn worker_router_with_shutdown(
    worker: Worker,
    env_dir: impl Into<PathBuf>,
    work_dir: impl Into<PathBuf>,
    token: Option<String>,
) -> (Router, ShutdownSignal) {
    let work = work_dir.into();
    std::fs::create_dir_all(&work).ok();
    let temp_store = worker.temp_store().clone();
    let shutdown = ShutdownSignal(Arc::new(tokio::sync::Notify::new()));
    let state = Arc::new(ServerState {
        worker: Mutex::new(worker),
        shutdown: shutdown.clone(),
        env_manager: EnvManager::new(env_dir, EnvType::Venv),
        work_dir: work,
        token,
        temp_store,
        temp_uploads: Mutex::new(HashMap::new()),
        active_streams: Mutex::new(HashMap::new()),
    });
    // Background cleanup: remove temp uploads older than 1 hour. It stops
    // with the server rather than outliving it — this task used to run
    // forever, so a test that built a router leaked one per router.
    let cleanup_state = state.clone();
    let cleanup_shutdown = shutdown.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(300));
        loop {
            tokio::select! {
                _ = interval.tick() => {}
                _ = cleanup_shutdown.wait() => break,
            }
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
                    let data_ref = somatize_core::data::store::DataRef::Cached {
                        cache_key: key.clone(),
                    };
                    let _ = cleanup_state.temp_store.remove(&data_ref);
                    uploads.remove(key);
                }
                tracing::info!("Cleaned up {} expired temp uploads", expired.len());
            }
        }
    });

    let router = Router::new()
        .route("/health", get(health))
        .route("/info", get(info))
        .route("/upload", post(upload_data))
        .route("/download", get(download_data))
        .route("/ws", get(ws_handler))
        .layer(DefaultBodyLimit::disable()) // No limit — workers handle arbitrary data sizes
        .with_state(state);
    (router, shutdown)
}

/// Start a worker server on the given address.
pub async fn serve_worker(worker: Worker, addr: &str) -> Result<(), Box<dyn std::error::Error>> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("Worker server listening on {addr}");
    let (router, shutdown) =
        worker_router_with_shutdown(worker, "/tmp/soma-envs", "/tmp/soma-work", None);
    axum::serve(listener, router)
        .with_graceful_shutdown(async move { shutdown.wait().await })
        .await?;
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
    let (router, shutdown) = worker_router_with_shutdown(
        worker,
        "/tmp/soma-envs",
        "/tmp/soma-work",
        Some(token.to_string()),
    );
    axum::serve(listener, router)
        .with_graceful_shutdown(async move { shutdown.wait().await })
        .await?;
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

    let data_ref: somatize_core::data::store::DataRef =
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

/// A well-formed error reply.
///
/// These used to be built by interpolating the error straight into a JSON
/// string literal, so a message containing a quote or a backslash — which
/// serde's own parse errors do contain — produced invalid JSON. The client
/// then reported a parse failure instead of the failure that happened.
///
/// Still not a `WorkerToCoordinator` variant, so a client cannot match on
/// it; that is a wire-protocol change and belongs with versioning it.
fn error_reply(message: &str) -> String {
    // A real protocol variant, not a bare `{"error": …}`. The client
    // silently skips what it cannot parse, so an unparseable failure left
    // it waiting for a reply that had already gone.
    serde_json::to_string(&WorkerToCoordinator::Error {
        message: message.to_string(),
    })
    .unwrap_or_else(|_| r#"{"type":"Error","message":"unserializable error"}"#.to_string())
}

async fn handle_ws(mut socket: WebSocket, state: Arc<ServerState>) {
    loop {
        match socket.recv().await {
            Some(Ok(Message::Text(text))) => {
                let response = match serde_json::from_str::<CoordinatorToWorker>(&text) {
                    Ok(CoordinatorToWorker::AssignPlan { plan }) => {
                        // Off the reactor: executing a plan creates a venv,
                        // pip-installs into it and drives a Python
                        // subprocess to completion. Running that inline in
                        // an async handler parked a tokio worker thread for
                        // the whole plan, so `/health` and every other
                        // connection stalled behind it.
                        let st = state.clone();
                        let joined = tokio::task::spawn_blocking(move || {
                            let mut worker = st.worker.lock().unwrap_or_else(|e| e.into_inner());
                            let plan_id = plan.plan_id.clone();
                            let worker_id = worker.id.clone();
                            let result = worker.execute_plan(&plan);
                            WorkerToCoordinator::PlanResult {
                                worker_id,
                                plan_id,
                                result,
                            }
                        })
                        .await;
                        match joined {
                            Ok(msg) => serde_json::to_string(&msg).unwrap_or_default(),
                            // A panic no longer reaches the reactor; report
                            // it as a failure of this plan and keep serving.
                            Err(e) => error_reply(&format!("plan execution panicked: {e}")),
                        }
                    }
                    Ok(CoordinatorToWorker::StatusRequest) => {
                        let worker = state.worker.lock().unwrap_or_else(|e| e.into_inner());
                        serde_json::to_string(&worker.registration_message()).unwrap_or_default()
                    }
                    Ok(CoordinatorToWorker::CancelPlan { .. }) => {
                        error_reply("cancelling a running plan is not implemented")
                    }
                    Ok(CoordinatorToWorker::AssignPythonJob { job }) => {
                        // Same reasoning as AssignPlan: this creates an env
                        // and runs a subprocess.
                        let st = state.clone();
                        let messages = match tokio::task::spawn_blocking(move || {
                            execute_python_job_with_progress(&st, &job)
                        })
                        .await
                        {
                            Ok(messages) => messages,
                            Err(e) => vec![error_reply(&format!("python job panicked: {e}"))],
                        };
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
                        // Ask the serve loop to wind down; do not exit the
                        // process. This crate also runs inside the user's
                        // Python interpreter.
                        state.shutdown.trigger();
                        break;
                    }
                    // These four used to be refused together with "not
                    // implemented for SubprocessFilter". Every piece they
                    // need was already written — the daemon script has
                    // GET_STATE/SET_STATE/GET_GRADIENTS/APPLY_GRADIENTS and
                    // `PythonProcess` has the methods — and nothing called
                    // them, which is what kept DataParallel from running.
                    //
                    // Off the reactor, for the same reason `AssignPlan` is:
                    // each one takes the worker's mutex and then talks to a
                    // Python subprocess over a pipe. Doing that inline in an
                    // async handler parks a tokio worker thread, and with
                    // two of these in flight the runtime deadlocks — which
                    // is exactly what it did.
                    Ok(
                        msg @ (CoordinatorToWorker::GetState { .. }
                        | CoordinatorToWorker::SetState { .. }
                        | CoordinatorToWorker::GetGradients { .. }
                        | CoordinatorToWorker::ApplyGradients { .. }),
                    ) => {
                        let st = state.clone();
                        let joined = tokio::task::spawn_blocking(move || {
                            let mut worker = st.worker.lock().unwrap_or_else(|e| e.into_inner());
                            let id = worker.id.clone();
                            match msg {
                                CoordinatorToWorker::GetState { plan_id, node_ids } => {
                                    match worker.read_states(&node_ids) {
                                        Ok(states) => serde_json::to_string(
                                            &WorkerToCoordinator::StateResult {
                                                worker_id: id,
                                                plan_id,
                                                states,
                                            },
                                        )
                                        .unwrap_or_default(),
                                        Err(e) => error_reply(&e.to_string()),
                                    }
                                }
                                CoordinatorToWorker::SetState { states, .. } => {
                                    match worker.write_states(&states) {
                                        Ok(()) => {
                                            serde_json::to_string(&WorkerToCoordinator::Ack {
                                                worker_id: id,
                                            })
                                            .unwrap_or_default()
                                        }
                                        Err(e) => error_reply(&e.to_string()),
                                    }
                                }
                                CoordinatorToWorker::GetGradients { plan_id, node_ids } => {
                                    match worker.read_gradients(&node_ids) {
                                        Ok(gradients) => serde_json::to_string(
                                            &WorkerToCoordinator::GradientsResult {
                                                worker_id: id,
                                                plan_id,
                                                gradients,
                                            },
                                        )
                                        .unwrap_or_default(),
                                        Err(e) => error_reply(&e.to_string()),
                                    }
                                }
                                CoordinatorToWorker::ApplyGradients { gradients, .. } => {
                                    match worker.write_gradients(&gradients) {
                                        Ok(()) => {
                                            serde_json::to_string(&WorkerToCoordinator::Ack {
                                                worker_id: id,
                                            })
                                            .unwrap_or_default()
                                        }
                                        Err(e) => error_reply(&e.to_string()),
                                    }
                                }
                                _ => unreachable!("guarded by the match arm above"),
                            }
                        })
                        .await;
                        joined.unwrap_or_else(|e| error_reply(&format!("worker task: {e}")))
                    }
                    Err(e) => error_reply(&format!("invalid message: {e}")),
                };

                if socket.send(Message::Text(response.into())).await.is_err() {
                    break;
                }
            }
            Some(Ok(Message::Binary(bytes))) => {
                // A frame that will not decode is reported, not dropped.
                // Swallowing it behind `if let Ok(..)` is what let a
                // months-old encoding bug look like a chunk that simply
                // never arrived.
                let stream_msg = match crate::protocol::decode_frame(&bytes) {
                    Ok(msg) => msg,
                    Err(e) => {
                        tracing::error!("{e}");
                        let _ = socket
                            .send(Message::Text(error_reply(&e.to_string()).into()))
                            .await;
                        continue;
                    }
                };

                // Chunk processing drives the same Python subprocess as
                // a plan does, so it belongs off the reactor too.
                let st = state.clone();
                let reply =
                    tokio::task::spawn_blocking(move || handle_stream_message(stream_msg, &st))
                        .await
                        .unwrap_or_else(|e| {
                            tracing::error!("stream message handler panicked: {e}");
                            None
                        });
                if let Some(reply_msg) = reply {
                    match crate::protocol::encode_frame(&reply_msg) {
                        Ok(reply_bytes) => {
                            if socket
                                .send(Message::Binary(reply_bytes.into()))
                                .await
                                .is_err()
                            {
                                break;
                            }
                        }
                        Err(e) => {
                            tracing::error!("{e}");
                            let _ = socket
                                .send(Message::Text(error_reply(&e.to_string()).into()))
                                .await;
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
    use somatize_runtime::{Context, StreamRun};

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
                    let config_hash = sf.config_hash.clone().unwrap_or_else(|| {
                        crate::python_process::SubprocessFilter::fallback_config_hash(
                            &sf.node_id,
                            &sf.pickled_filter,
                        )
                    });
                    let filter: Box<dyn somatize_core::graph::filter::Filter> =
                        Box::new(crate::python_process::SubprocessFilter::new(
                            process.clone(),
                            sf.node_id.clone(),
                            sf.trainable,
                            config_hash,
                        ));
                    worker.register_filter(&sf.node_id, filter);
                    // A stream that cannot restore the state it was sent is
                    // the same failure as a plan that cannot (D-25): it would
                    // run from random weights and report chunks that look
                    // exactly like a correct run's.
                    if let Some(s) = &sf.state
                        && let Err(e) = worker.set_filter_state(&sf.node_id, s.clone())
                    {
                        return Some(StreamMessage::StreamComplete {
                            stream_id,
                            result: PlanResult::Failed {
                                error: e.to_string(),
                                duration_ms: 0,
                            },
                        });
                    }
                }
            }

            // Build the stream driver over the registered filters. A node
            // the worker cannot resolve fails the stream up front, rather
            // than silently streaming a shorter chain.
            let node_ids: Vec<String> =
                plan.plan.node_ids().into_iter().map(String::from).collect();
            let run = match StreamRun::new(&node_ids, worker.catalog()) {
                Ok(run) => run,
                Err(e) => {
                    return Some(StreamMessage::StreamComplete {
                        stream_id,
                        result: PlanResult::Failed {
                            error: e.to_string(),
                            duration_ms: 0,
                        },
                    });
                }
            };

            let run_id = format!("worker_stream_{stream_id}");
            let ctx = Context::new(worker.event_bus().clone(), run_id).with_seed(plan.seed);
            let session = StreamSession {
                run,
                ctx,
                cache: worker.cache().clone(),
                started: Instant::now(),
            };
            state
                .active_streams
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(stream_id, session);

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
            if let Some(session) = streams.get_mut(&stream_id) {
                let outcome =
                    session
                        .run
                        .process_chunk(value, &mut session.ctx, session.cache.as_ref());
                match outcome {
                    Ok(Some(result)) => Some(StreamMessage::ChunkResult {
                        stream_id,
                        chunk_index,
                        value: result,
                    }),
                    Ok(None) => None, // Barrier mode — no result yet
                    Err(e) => {
                        // The run is dead; drop the session so a retry
                        // does not resume a half-failed one.
                        let duration_ms = session.started.elapsed().as_millis() as u64;
                        streams.remove(&stream_id);
                        Some(StreamMessage::StreamComplete {
                            stream_id,
                            result: PlanResult::Failed {
                                error: e.to_string(),
                                duration_ms,
                            },
                        })
                    }
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
            if let Some(mut session) = streams.remove(&stream_id) {
                let duration_ms = session.started.elapsed().as_millis() as u64;
                // Flush barrier filters, then close each node's event
                // bracket with its chunk/hit/miss aggregate.
                let flushed = session.run.flush(&mut session.ctx, session.cache.as_ref());
                let output = match flushed {
                    Ok(v) => v.unwrap_or(somatize_core::data::value::Value::Empty),
                    Err(e) => {
                        return Some(StreamMessage::StreamComplete {
                            stream_id,
                            result: PlanResult::Failed {
                                error: format!("stream flush: {e}"),
                                duration_ms,
                            },
                        });
                    }
                };
                session.run.finish(&session.ctx);
                Some(StreamMessage::StreamComplete {
                    stream_id,
                    result: PlanResult::Success {
                        output: OutputDelivery::Inline { value: output },
                        duration_ms,
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
