use crate::protocol::*;
use crate::worker::Worker;
use axum::Router;
use axum::extract::ws::{Message, WebSocket};
use axum::extract::{State, WebSocketUpgrade};
use axum::response::IntoResponse;
use axum::routing::get;
use std::sync::{Arc, Mutex};

/// Shared state for the worker HTTP/WebSocket server.
struct ServerState {
    worker: Mutex<Worker>,
}

/// Build a worker server router.
pub fn worker_router(worker: Worker) -> Router {
    let state = Arc::new(ServerState {
        worker: Mutex::new(worker),
    });
    Router::new()
        .route("/health", get(health))
        .route("/info", get(info))
        .route("/ws", get(ws_handler))
        .with_state(state)
}

/// Start a worker server on the given address.
pub async fn serve_worker(worker: Worker, addr: &str) -> Result<(), Box<dyn std::error::Error>> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("Worker server listening on {addr}");
    axum::serve(listener, worker_router(worker)).await?;
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

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<ServerState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_ws(socket, state))
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
                        // Python pipeline execution — TODO: wire EnvManager
                        let worker = state.worker.lock().unwrap_or_else(|e| e.into_inner());
                        let msg = WorkerToCoordinator::JobResult {
                            worker_id: worker.id.clone(),
                            job_id: job.job_id.clone(),
                            success: false,
                            metrics: serde_json::json!({}),
                            output: "Python job execution not yet wired in server".into(),
                            duration_ms: 0,
                        };
                        serde_json::to_string(&msg).unwrap_or_default()
                    }
                    Ok(CoordinatorToWorker::Ping) => {
                        r#"{"type":"Pong"}"#.to_string()
                    }
                    Ok(CoordinatorToWorker::Registered { .. }) => continue,
                    Err(e) => {
                        format!(r#"{{"error": "invalid message: {e}"}}"#)
                    }
                };

                if socket.send(Message::Text(response.into())).await.is_err() {
                    break;
                }
            }
            Some(Ok(Message::Close(_))) | None => break,
            _ => {}
        }
    }
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

    #[test]
    fn router_builds() {
        let _router = worker_router(make_worker());
    }

    #[tokio::test]
    async fn health_returns_ok() {
        let resp = health().await;
        assert_eq!(resp, "ok");
    }

    // info endpoint tested via full_server_starts_and_stops

    #[tokio::test]
    async fn full_server_starts_and_stops() {
        let worker = make_worker();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server = tokio::spawn(async move {
            axum::serve(listener, worker_router(worker)).await.unwrap();
        });

        // Give server a moment to start
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Make a health check request
        let client = reqwest::Client::new();
        let resp = client
            .get(format!("http://{addr}/health"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.text().await.unwrap(), "ok");

        // Make an info request
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
