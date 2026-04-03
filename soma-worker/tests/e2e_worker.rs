//! End-to-end worker tests: start server, connect WebSocket, execute plans.

use somatize_compiler::ExecutionPlan;
use somatize_core::cache::CacheKey;
use somatize_core::error::Result as SomaResult;
use somatize_core::filter::{Filter, FilterKind, FilterMeta, StreamMode};
use somatize_core::value::Value;
use somatize_worker::protocol::*;
use somatize_worker::worker::Worker;
use somatize_worker::{worker_router, worker_router_authenticated};
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;

struct TestDoubler;

impl Filter for TestDoubler {
    fn config_hash(&self) -> CacheKey {
        CacheKey::from_parts(&[b"TestDoubler"])
    }
    fn fit(&self, _x: &Value, _y: Option<&Value>) -> SomaResult<Value> {
        Ok(Value::Empty)
    }
    fn forward(&self, x: &Value, _state: &Value) -> SomaResult<Value> {
        match x {
            Value::Tensor { values, shape } => {
                let doubled: Vec<f64> = values.iter().map(|v| v * 2.0).collect();
                Ok(Value::tensor(doubled, shape.clone()))
            }
            _ => Ok(x.clone()),
        }
    }
    fn meta(&self) -> FilterMeta {
        FilterMeta {
            name: "TestDoubler".into(),
            kind: FilterKind::Stateless,
            cacheable: true,
            differentiable: true,
            stream_mode: StreamMode::FixedState,
            distribution: somatize_core::filter::Distribution::Local,
            input_schema: None,
            output_schema: None,
        }
    }
}

fn make_worker() -> Worker {
    let mut w = Worker::new(
        "e2e_worker",
        Capabilities {
            cpu_cores: 2,
            ram_bytes: 4_000_000_000,
            gpus: vec![],
            python_envs: vec![],
            tags: vec!["test".into()],
        },
    );
    w.register_filter("doubler", Box::new(TestDoubler));
    w
}

#[tokio::test]
async fn worker_ws_execute_plan() {
    let worker = make_worker();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let server = tokio::spawn(async move {
        axum::serve(listener, worker_router(worker)).await.unwrap();
    });

    // Give server time to start
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // Connect WebSocket
    let url = format!("ws://{addr}/ws");
    let (mut ws, _) = connect_async(&url).await.expect("WS connect failed");

    // Send a plan
    use futures_util::{SinkExt, StreamExt};
    let plan = SerializedPlan {
        plan_id: "test_001".into(),
        plan: ExecutionPlan::Execute {
            node_id: "doubler".into(),
        },
        input: Some(InputSource::Inline {
            value: Value::tensor(vec![1.0, 2.0, 3.0], vec![3]),
        }),
        filters: vec![],
        mode: somatize_worker::protocol::ExecutionMode::default(),
        metadata: serde_json::json!({}),
    };

    let msg = CoordinatorToWorker::AssignPlan { plan };
    let json = serde_json::to_string(&msg).unwrap();
    ws.send(Message::Text(json.into())).await.unwrap();

    // Receive result
    if let Some(Ok(Message::Text(response))) = ws.next().await {
        let result: WorkerToCoordinator = serde_json::from_str(&response).unwrap();
        if let WorkerToCoordinator::PlanResult {
            worker_id,
            plan_id,
            result,
        } = result
        {
            assert_eq!(worker_id, "e2e_worker");
            assert_eq!(plan_id, "test_001");
            if let PlanResult::Success {
                output,
                duration_ms,
                ..
            } = result
            {
                let value = match output {
                    somatize_worker::protocol::OutputDelivery::Inline { value } => value,
                    _ => panic!("expected inline output"),
                };
                let (data, _) = value.as_tensor().unwrap();
                assert_eq!(data, &[2.0, 4.0, 6.0]);
                assert!(duration_ms < 5000);
            } else {
                panic!("expected success, got {result:?}");
            }
        } else {
            panic!("expected PlanResult, got {result:?}");
        }
    } else {
        panic!("no response received");
    }

    ws.close(None).await.ok();
    server.abort();
}

#[tokio::test]
async fn worker_ws_sequence_plan() {
    let mut worker = make_worker();
    worker.register_filter("d2", Box::new(TestDoubler));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let server = tokio::spawn(async move {
        axum::serve(listener, worker_router(worker)).await.unwrap();
    });

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let url = format!("ws://{addr}/ws");
    let (mut ws, _) = connect_async(&url).await.unwrap();

    use futures_util::{SinkExt, StreamExt};
    let plan = SerializedPlan {
        plan_id: "seq_001".into(),
        plan: ExecutionPlan::Sequence(vec![
            ExecutionPlan::Execute {
                node_id: "doubler".into(),
            },
            ExecutionPlan::Execute {
                node_id: "d2".into(),
            },
        ]),
        input: Some(InputSource::Inline {
            value: Value::tensor(vec![5.0], vec![1]),
        }),
        filters: vec![],
        mode: somatize_worker::protocol::ExecutionMode::default(),
        metadata: serde_json::json!({}),
    };

    let msg = CoordinatorToWorker::AssignPlan { plan };
    ws.send(Message::Text(serde_json::to_string(&msg).unwrap().into()))
        .await
        .unwrap();

    if let Some(Ok(Message::Text(response))) = ws.next().await {
        let result: WorkerToCoordinator = serde_json::from_str(&response).unwrap();
        if let WorkerToCoordinator::PlanResult { result, .. } = result {
            if let PlanResult::Success { output, .. } = result {
                let value = match output {
                    somatize_worker::protocol::OutputDelivery::Inline { value } => value,
                    _ => panic!("expected inline output"),
                };
                let (data, _) = value.as_tensor().unwrap();
                assert_eq!(data, &[20.0]); // 5 * 2 * 2
            } else {
                panic!("expected success");
            }
        }
    }

    ws.close(None).await.ok();
    server.abort();
}

#[tokio::test]
async fn worker_ws_auth_rejects_no_token() {
    let worker = make_worker();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let server = tokio::spawn(async move {
        let router = worker_router_authenticated(
            worker,
            "/tmp/soma-test-envs",
            "/tmp/soma-test-work",
            "sk-test-secret",
        );
        axum::serve(listener, router).await.unwrap();
    });

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // Without token → should fail
    let url = format!("ws://{addr}/ws");
    let result = connect_async(&url).await;
    assert!(result.is_err(), "should reject unauthenticated connection");

    // With token → should succeed
    let url_auth = format!("ws://{addr}/ws?token=sk-test-secret");
    let result = connect_async(&url_auth).await;
    assert!(result.is_ok(), "should accept authenticated connection");

    server.abort();
}

#[tokio::test]
async fn worker_health_and_info() {
    let worker = make_worker();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let server = tokio::spawn(async move {
        axum::serve(listener, worker_router(worker)).await.unwrap();
    });

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let client = reqwest::Client::new();

    // Health
    let resp = client
        .get(format!("http://{addr}/health"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.text().await.unwrap(), "ok");

    // Info
    let resp = client
        .get(format!("http://{addr}/info"))
        .send()
        .await
        .unwrap();
    let json: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(json["worker_id"], "e2e_worker");

    server.abort();
}
