//! End-to-end worker tests: start server, connect WebSocket, execute plans.

use somatize_compiler::ExecutionPlan;
use somatize_core::cache::CacheKey;
use somatize_core::data::value::Value;
use somatize_core::error::Result as SomaResult;
use somatize_core::graph::filter::{Filter, FilterKind, FilterMeta, StreamMode};
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
            deterministic: true,
            stream_mode: StreamMode::FixedState,
            distribution: somatize_core::graph::filter::Distribution::Local,
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
        protocol_version: PROTOCOL_VERSION,
        plan_id: "test_001".into(),
        plan: ExecutionPlan::Execute {
            node_id: "doubler".into(),
        },
        input: Some(InputSource::Inline {
            value: Value::tensor(vec![1.0, 2.0, 3.0], vec![3]),
        }),
        filters: vec![],
        mode: somatize_worker::protocol::ExecutionMode::default(),
        seed: None,
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
        protocol_version: PROTOCOL_VERSION,
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
        seed: None,
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

/// A `Shutdown` message must stop the *server*, not the process.
///
/// This test could not exist before: the handler called
/// `std::process::exit(0)`, which would have taken the test binary down
/// mid-run. The same call also killed the user's interpreter, because
/// `soma.Worker.serve()` runs this server on a thread inside their Python
/// process.
#[tokio::test]
async fn a_shutdown_message_stops_the_server_and_not_the_process() {
    use futures_util::{SinkExt, StreamExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (router, shutdown) = somatize_worker::worker_router_with_shutdown(
        make_worker(),
        "/tmp/soma-envs-test",
        "/tmp/soma-work-test",
        None,
    );

    let server = tokio::spawn(async move {
        axum::serve(listener, router)
            .with_graceful_shutdown(async move { shutdown.wait().await })
            .await
            .unwrap();
    });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let url = format!("ws://{addr}/ws");
    let (mut ws, _) = connect_async(&url).await.expect("WS connect failed");
    let msg = CoordinatorToWorker::Shutdown {
        reason: "test".into(),
    };
    ws.send(Message::Text(serde_json::to_string(&msg).unwrap().into()))
        .await
        .unwrap();

    let ack = ws.next().await.expect("no ack").expect("ws error");
    assert!(ack.to_text().unwrap().contains("ShutdownAck"));

    // The server task finishes on its own; reaching this line at all means
    // the process survived.
    tokio::time::timeout(std::time::Duration::from_secs(5), server)
        .await
        .expect("server did not shut down within 5s")
        .expect("server task panicked");
}

/// The transport works when the caller is already inside a runtime.
///
/// `send_msg`, `notify` and `stream_plan` used to build a current-thread
/// runtime and `block_on` it. Inside an existing runtime that is a panic,
/// not an error — and this is not a hypothetical caller:
/// `soma.Worker.serve()` runs an axum server on a thread of the user's
/// Python process, and the bindings dispatch plans from there.
///
/// Multi-threaded on purpose, and called directly rather than through
/// `spawn_blocking`: `spawn_blocking` moves the work off the runtime's
/// worker threads, which is exactly what hides the bug.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_transport_does_not_panic_when_called_from_inside_a_runtime() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (router, shutdown) = somatize_worker::worker_router_with_shutdown(
        make_worker(),
        "/tmp/soma-envs-rt-test",
        "/tmp/soma-work-rt-test",
        None,
    );
    let server = tokio::spawn(async move {
        axum::serve(listener, router)
            .with_graceful_shutdown(async move { shutdown.wait().await })
            .await
            .unwrap();
    });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    let transport = somatize_worker::WsTransport::new(format!("ws://{addr}"), None);
    let status = transport.send_msg(&CoordinatorToWorker::StatusRequest);
    assert!(status.is_ok(), "send_msg from inside a runtime: {status:?}");

    transport
        .notify(&CoordinatorToWorker::Shutdown {
            reason: "done".into(),
        })
        .expect("notify from inside a runtime");

    let _ = tokio::time::timeout(std::time::Duration::from_secs(5), server).await;
}

/// A worker-side failure must reach the caller, not hang it.
///
/// `error_reply` used to send a bare `{"error": …}`, which is not a
/// `WorkerToCoordinator` at all — and `WsTransport::send_msg` skipped what
/// it could not parse and kept waiting. So every error the worker reported
/// over WebSocket blocked its caller until the socket closed, saying
/// nothing about why. It cost an afternoon to find, because the symptom
/// was a hang in a completely different feature.
#[tokio::test]
async fn a_worker_error_reaches_the_caller_instead_of_hanging_it() {
    let worker = make_worker();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, worker_router(worker)).await.unwrap();
    });
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // `CancelPlan` is answered with an error, and any error will do.
    let transport = somatize_worker::WsTransport::new(format!("ws://{addr}"), None);
    let outcome = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        tokio::task::spawn_blocking(move || {
            transport.send_msg(&CoordinatorToWorker::CancelPlan {
                plan_id: "nothing-is-running".into(),
            })
        }),
    )
    .await
    .expect("the call hung: an unparseable reply is being skipped again")
    .expect("task panicked");

    let err = outcome.expect_err("cancelling a plan that is not running is an error");
    assert!(
        err.to_string().contains("not implemented"),
        "the worker's own words should come through: {err}"
    );
}

/// D-25: a resume that cannot resume fails, instead of retraining from
/// random weights and saying so only in a log line.
///
/// The state sent is a `Value::Tensor`, which `set_state` refuses to encode
/// before it sends anything — so the refusal under test is the worker's, not
/// something the Python side decided, and the test cannot go green because a
/// filter happened to accept a bad state.
///
/// What made this worth a High: the old path logged `warn!` and carried on,
/// so the epoch restarted from random initialization and the returned
/// metrics were indistinguishable from a genuinely bad run.
#[test]
fn a_state_that_cannot_be_restored_fails_the_plan() {
    let python = std::env::var("SOMA_PYTHON").unwrap_or_else(|_| "python3".into());

    // Both are needed to get as far as the state load: the daemon imports
    // cloudpickle at startup, and LOAD unpickles the filter.
    let usable = std::process::Command::new(&python)
        .args(["-c", "import cloudpickle"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !usable {
        eprintln!("Skipping: {python} has no cloudpickle");
        return;
    }

    let dir = tempfile::tempdir().unwrap();
    let pickle_path = dir.path().join("filter.pkl");
    let script = format!(
        "import cloudpickle\n\
         class Trivial:\n\
         \x20   def fit(self, x, y=None): return {{}}\n\
         \x20   def forward(self, x, state): return x\n\
         open({:?}, 'wb').write(cloudpickle.dumps(Trivial()))\n",
        pickle_path.to_string_lossy()
    );
    let dumped = std::process::Command::new(&python)
        .args(["-c", &script])
        .output()
        .expect("failed to run python");
    assert!(
        dumped.status.success(),
        "could not pickle the test filter: {}",
        String::from_utf8_lossy(&dumped.stderr)
    );
    let pickled = std::fs::read(&pickle_path).unwrap();

    let mut worker = make_worker().with_python(&python);
    let plan = SerializedPlan {
        protocol_version: PROTOCOL_VERSION,
        plan_id: "resume_001".into(),
        plan: ExecutionPlan::Execute {
            node_id: "trained".into(),
        },
        input: Some(InputSource::Inline {
            value: Value::tensor(vec![1.0, 2.0, 3.0], vec![3]),
        }),
        filters: vec![SerializedFilter {
            node_id: "trained".into(),
            pickled_filter: pickled,
            // A tensor is a state the wire format has no encoding for.
            state: Some(Value::tensor(vec![0.5], vec![1])),
            requirements: vec![],
            trainable: true,
            config_hash: None,
        }],
        mode: somatize_worker::protocol::ExecutionMode::default(),
        seed: None,
        metadata: serde_json::json!({}),
    };

    match worker.execute_plan(&plan) {
        PlanResult::Failed { error, .. } => {
            assert!(
                error.contains("trained"),
                "the failure must name the node whose state was lost: {error}"
            );
            assert!(
                error.contains("random weights"),
                "and say what continuing would have cost: {error}"
            );
        }
        PlanResult::Success { .. } => {
            panic!("a plan that could not restore its trained state reported success")
        }
    }
}
