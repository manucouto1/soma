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

use crate::coordinator::WorkerRegistry;
use crate::protocol::{Capabilities, LoadMetrics, SerializedPlan};
use axum::Router;
use axum::extract::{Json, Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use serde::{Deserialize, Serialize};
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
pub fn coordinator_router(registry: WorkerRegistry, token: Option<String>) -> Router {
    let state = Arc::new(CoordinatorState { registry, token });
    Router::new()
        .route("/health", get(health))
        .route("/workers", get(list_workers))
        .route("/summary", get(summary))
        .route("/register", post(register_worker))
        .route("/heartbeat", post(heartbeat))
        .route("/submit", post(submit_plan))
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

/// Validate token if required. Returns Err(401) on mismatch.
fn check_auth(state: &CoordinatorState, params: &AuthParams) -> Result<(), StatusCode> {
    if let Some(expected) = &state.token {
        match &params.token {
            Some(provided) if provided == expected => Ok(()),
            _ => Err(StatusCode::UNAUTHORIZED),
        }
    } else {
        Ok(())
    }
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
    Json(req): Json<RegisterRequest>,
) -> Result<impl IntoResponse, StatusCode> {
    check_auth(&state, &params)?;

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
    Json(req): Json<HeartbeatRequest>,
) -> Result<impl IntoResponse, StatusCode> {
    check_auth(&state, &params)?;
    state.registry.heartbeat(&req.worker_id, req.load);
    Ok(StatusCode::OK)
}

/// Plan submission request body.
#[derive(Deserialize)]
struct SubmitRequest {
    plan: SerializedPlan,
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
    Json(req): Json<SubmitRequest>,
) -> Result<impl IntoResponse, StatusCode> {
    check_auth(&state, &params)?;

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
        "Routing plan {} to worker {} ({})",
        req.plan.plan_id,
        best.id,
        best.address
    );

    // Return the worker address — client connects directly
    // (In a full implementation, coordinator would forward via WS)
    Ok(axum::Json(SubmitResponse {
        status: "routed".into(),
        worker_id: Some(best.id.clone()),
        worker_address: Some(best.address.clone()),
        error: None,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coordinator::WorkerStatus;
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
