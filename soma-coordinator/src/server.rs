//! HTTP/WebSocket server for the Coordinator.
//!
//! Endpoints:
//! - `GET  /health` — liveness check
//! - `GET  /workers` — list active workers with capabilities
//! - `GET  /summary` — cluster summary (total CPUs, GPUs, RAM)
//! - `POST /register` — worker self-registration (JSON body)
//! - `POST /submit` — client submits a SerializedPlan for execution
//! - `POST /heartbeat` — worker heartbeat with load metrics
//!
//! All mutating endpoints require `?token=sk-xxx` when a token is configured.

use crate::registry::WorkerRegistry;
use axum::Router;
use axum::extract::{Json, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use serde::{Deserialize, Serialize};
use somatize_worker::protocol::{Capabilities, LoadMetrics};
use std::sync::Arc;

/// Shared coordinator server state.
struct CoordinatorState {
    registry: WorkerRegistry,
    token: Option<String>,
}

/// Query params for token authentication.
#[derive(Deserialize, Default)]
struct AuthParams {
    token: Option<String>,
}

/// Build the coordinator router.
///
/// Also starts the reaper: without it a worker that dies leaves its entry
/// in the registry forever, because nothing ever called `prune_stale`.
pub fn coordinator_router(registry: WorkerRegistry, token: Option<String>) -> Router {
    let state = Arc::new(CoordinatorState { registry, token });

    let reaping = state.registry.clone();
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(std::time::Duration::from_secs(10));
        loop {
            tick.tick().await;
            for id in reaping.prune_stale() {
                tracing::warn!("worker {id} stopped sending heartbeats; dropped");
            }
        }
    });

    Router::new()
        .route("/health", get(health))
        .route("/workers", get(list_workers))
        .route("/summary", get(summary))
        .route("/register", post(register_worker))
        .route("/heartbeat", post(heartbeat))
        .route("/submit", post(submit_plan))
        .route("/complete", post(complete_plan))
        .with_state(state)
}

/// Start the coordinator server.
pub async fn serve_coordinator(
    registry: WorkerRegistry,
    addr: &str,
    token: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("Coordinator listening on {addr}");
    if token.is_some() {
        tracing::info!("Authentication enabled");
    }
    axum::serve(listener, coordinator_router(registry, token)).await?;
    Ok(())
}

/// Validate the token if one is configured.
///
/// `Authorization: Bearer <token>` is the supported form. `?token=` is
/// still accepted so existing workers keep working, but it is deprecated:
/// a query string ends up in access logs, proxy logs and browser history,
/// which is not where a credential belongs.
///
/// The comparison is constant-time. A byte-by-byte `==` leaks the length
/// of the matching prefix through timing, which is enough to recover a
/// token one character at a time.
fn check_auth(
    state: &CoordinatorState,
    headers: &HeaderMap,
    params: &AuthParams,
) -> Result<(), StatusCode> {
    let Some(expected) = &state.token else {
        return Ok(());
    };

    let from_header = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(str::to_string);

    if from_header.is_none() && params.token.is_some() {
        tracing::warn!("a client authenticated with ?token=; use `Authorization: Bearer` instead");
    }

    match from_header.or_else(|| params.token.clone()) {
        Some(provided) if constant_time_eq(provided.as_bytes(), expected.as_bytes()) => Ok(()),
        _ => Err(StatusCode::UNAUTHORIZED),
    }
}

/// Compare without leaking where two secrets first differ.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

// ── Handlers ──

async fn health() -> &'static str {
    "ok"
}

async fn list_workers(State(state): State<Arc<CoordinatorState>>) -> impl IntoResponse {
    let workers = state.registry.active_workers();
    axum::Json(workers)
}

async fn summary(State(state): State<Arc<CoordinatorState>>) -> impl IntoResponse {
    state.registry.summary()
}

/// Worker registration request body.
#[derive(Deserialize)]
struct RegisterRequest {
    worker_id: String,
    address: String,
    capabilities: Capabilities,
}

/// Worker registration response.
#[derive(Serialize)]
struct RegisterResponse {
    status: String,
    worker_id: String,
}

async fn register_worker(
    Query(params): Query<AuthParams>,
    State(state): State<Arc<CoordinatorState>>,
    headers: HeaderMap,
    Json(req): Json<RegisterRequest>,
) -> Result<impl IntoResponse, StatusCode> {
    check_auth(&state, &headers, &params)?;

    tracing::info!(
        "Worker registered: {} at {} ({})",
        req.worker_id,
        req.address,
        req.capabilities.summary()
    );

    state
        .registry
        .register(&req.worker_id, &req.address, req.capabilities);

    Ok(axum::Json(RegisterResponse {
        status: "registered".into(),
        worker_id: req.worker_id,
    }))
}

/// Heartbeat request body.
#[derive(Deserialize)]
struct HeartbeatRequest {
    worker_id: String,
    load: LoadMetrics,
}

async fn heartbeat(
    Query(params): Query<AuthParams>,
    State(state): State<Arc<CoordinatorState>>,
    headers: HeaderMap,
    Json(req): Json<HeartbeatRequest>,
) -> Result<impl IntoResponse, StatusCode> {
    check_auth(&state, &headers, &params)?;
    // An unknown worker is told to register rather than silently ignored:
    // it may have been reaped while it was busy, and it has no other way
    // to find out.
    if state.registry.get(&req.worker_id).is_none() {
        return Err(StatusCode::NOT_FOUND);
    }
    state.registry.heartbeat(&req.worker_id, req.load);
    Ok(StatusCode::OK)
}

/// A request to place a plan on a worker.
///
/// It used to carry the whole `SerializedPlan` — cloudpickled filters and
/// inline input included — and then throw it away, returning only an
/// address. That is a large upload parsed for nothing, and it made the
/// endpoint look like it executed the plan when it does not.
///
/// The coordinator places work; the client then talks to the worker
/// directly, which is what keeps tensor-sized payloads off this hop. All
/// it needs to do that is the plan's id, to hold the lease under.
#[derive(Deserialize)]
struct SubmitRequest {
    plan_id: String,
    /// Required tags for worker selection.
    #[serde(default)]
    required_tags: Vec<String>,
    /// Max concurrent plans per worker (for capacity check).
    #[serde(default = "default_max_concurrent")]
    max_concurrent: usize,
}

fn default_max_concurrent() -> usize {
    4
}

/// Plan submission response.
#[derive(Serialize)]
struct SubmitResponse {
    status: String,
    worker_id: Option<String>,
    worker_address: Option<String>,
    error: Option<String>,
}

async fn submit_plan(
    Query(params): Query<AuthParams>,
    State(state): State<Arc<CoordinatorState>>,
    headers: HeaderMap,
    Json(req): Json<SubmitRequest>,
) -> Result<impl IntoResponse, StatusCode> {
    check_auth(&state, &headers, &params)?;

    // Find a suitable worker
    let candidates = state
        .registry
        .find_workers(&req.required_tags, req.max_concurrent);

    if candidates.is_empty() {
        return Ok(axum::Json(SubmitResponse {
            status: "no_workers".into(),
            worker_id: None,
            worker_address: None,
            error: Some("No workers available matching requirements".into()),
        }));
    }

    // Pick the least loaded worker
    let best = candidates
        .iter()
        .min_by_key(|w| w.active_plans.len())
        .unwrap();

    tracing::info!(
        "Placing plan {} on worker {} ({})",
        req.plan_id,
        best.id,
        best.address
    );

    // Hold the lease before answering. Without it the next caller sees the
    // same worker as equally idle and every plan lands on one machine.
    state.registry.claim(&best.id, &req.plan_id);

    Ok(axum::Json(SubmitResponse {
        status: "routed".into(),
        worker_id: Some(best.id.clone()),
        worker_address: Some(best.address.clone()),
        error: None,
    }))
}

/// Release a placement, whether the plan finished or failed.
#[derive(Deserialize)]
struct CompleteRequest {
    worker_id: String,
    plan_id: String,
}

async fn complete_plan(
    Query(params): Query<AuthParams>,
    State(state): State<Arc<CoordinatorState>>,
    headers: HeaderMap,
    Json(req): Json<CompleteRequest>,
) -> Result<impl IntoResponse, StatusCode> {
    check_auth(&state, &headers, &params)?;
    if !state.registry.release(&req.worker_id, &req.plan_id) {
        return Err(StatusCode::NOT_FOUND);
    }
    Ok(StatusCode::OK)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::WorkerStatus;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    fn test_caps() -> Capabilities {
        Capabilities {
            cpu_cores: 4,
            ram_bytes: 8_000_000_000,
            gpus: vec![],
            python_envs: vec![],
            tags: vec!["cpu".into()],
        }
    }

    #[tokio::test]
    async fn health_endpoint() {
        let registry = WorkerRegistry::new();
        let app = coordinator_router(registry, None);

        let resp = app
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn register_and_list() {
        let registry = WorkerRegistry::new();
        let app = coordinator_router(registry.clone(), None);

        // Register a worker
        let body = serde_json::json!({
            "worker_id": "w1",
            "address": "ws://host1:8080",
            "capabilities": test_caps()
        });

        let resp = app
            .clone()
            .oneshot(
                Request::post("/register")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // List workers
        let resp = app
            .oneshot(Request::get("/workers").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 10_000)
            .await
            .unwrap();
        let workers: Vec<WorkerStatus> = serde_json::from_slice(&body).unwrap();
        assert_eq!(workers.len(), 1);
        assert_eq!(workers[0].id, "w1");
    }

    #[tokio::test]
    async fn auth_rejects_without_token() {
        let registry = WorkerRegistry::new();
        let app = coordinator_router(registry, Some("sk-secret".into()));

        let body = serde_json::json!({
            "worker_id": "w1",
            "address": "ws://host:8080",
            "capabilities": test_caps()
        });

        // Without token → 401
        let resp = app
            .clone()
            .oneshot(
                Request::post("/register")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

        // With token → 200
        let resp = app
            .oneshot(
                Request::post("/register?token=sk-secret")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_string(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn summary_endpoint() {
        let registry = WorkerRegistry::new();
        registry.register("w1", "ws://h1:8080", test_caps());

        let app = coordinator_router(registry, None);
        let resp = app
            .oneshot(Request::get("/summary").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 10_000)
            .await
            .unwrap();
        let text = String::from_utf8_lossy(&body);
        assert!(text.contains("1 workers"));
    }
}
