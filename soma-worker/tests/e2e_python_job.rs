//! End-to-end test: execute a Python job via WebSocket.
//!
//! Sends an AssignPythonJob message, verifies progress events and final result.
//! Requires python3 on the system.

use soma_worker::protocol::*;
use soma_worker::worker::Worker;
use soma_worker::worker_router;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;

fn has_python3() -> bool {
    std::process::Command::new("python3")
        .args(["--version"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn make_worker() -> Worker {
    Worker::new(
        "python_test_worker",
        Capabilities {
            cpu_cores: 2,
            ram_bytes: 4_000_000_000,
            gpus: vec![],
            python_envs: vec![],
            tags: vec!["test".into()],
        },
    )
}

#[tokio::test]
async fn python_job_executes_and_returns_result() {
    if !has_python3() {
        eprintln!("Skipping: python3 not found");
        return;
    }

    let worker = make_worker();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let server = tokio::spawn(async move {
        axum::serve(listener, worker_router(worker)).await.unwrap();
    });

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    let url = format!("ws://{addr}/ws");
    let (mut ws, _) = connect_async(&url).await.expect("WS connect failed");

    use futures_util::{SinkExt, StreamExt};

    // Create a simple Python job
    let job = PythonPipelineJob {
        job_id: "py_test_001".into(),
        pipeline_id: "test_pipeline".into(),
        investigation_id: "test_inv".into(),
        files: vec![PipelineFile {
            path: "main.py".into(),
            content: r#"
import json
result = {"accuracy": 0.95, "loss": 0.05}
print(json.dumps(result))
"#
            .into(),
        }],
        requirements: "".into(), // no extra packages needed
        entry_point: "main.py".into(),
        input_data: None,
        params: serde_json::json!({}),
    };

    let msg = CoordinatorToWorker::AssignPythonJob { job };
    ws.send(Message::Text(serde_json::to_string(&msg).unwrap().into()))
        .await
        .unwrap();

    // Collect all messages (progress + result)
    let mut messages = Vec::new();
    let timeout = tokio::time::timeout(std::time::Duration::from_secs(60), async {
        while let Some(Ok(Message::Text(text))) = ws.next().await {
            let parsed: serde_json::Value = serde_json::from_str(&text).unwrap_or_default();
            let msg_type = parsed.get("type").and_then(|t| t.as_str()).unwrap_or("");
            messages.push(parsed.clone());

            // JobResult is the last message
            if msg_type == "JobResult" {
                break;
            }
        }
    });

    timeout.await.expect("timed out waiting for job result");

    // Should have progress messages + final result
    assert!(
        messages.len() >= 2,
        "expected progress + result, got {} messages",
        messages.len()
    );

    // Last message should be the result
    let result = messages.last().unwrap();
    assert_eq!(result["type"], "JobResult");
    assert_eq!(result["job_id"], "py_test_001");
    assert_eq!(result["success"], true);

    // Metrics should contain our JSON output
    let metrics = &result["metrics"];
    assert_eq!(metrics["accuracy"], 0.95);
    assert_eq!(metrics["loss"], 0.05);

    ws.close(None).await.ok();
    server.abort();
}

#[tokio::test]
async fn python_job_reports_failure() {
    if !has_python3() {
        eprintln!("Skipping: python3 not found");
        return;
    }

    let worker = make_worker();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let server = tokio::spawn(async move {
        axum::serve(listener, worker_router(worker)).await.unwrap();
    });

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    let url = format!("ws://{addr}/ws");
    let (mut ws, _) = connect_async(&url).await.unwrap();

    use futures_util::{SinkExt, StreamExt};

    // Job that will fail (syntax error)
    let job = PythonPipelineJob {
        job_id: "py_fail_001".into(),
        pipeline_id: "fail_pipeline".into(),
        investigation_id: "test_inv".into(),
        files: vec![PipelineFile {
            path: "bad.py".into(),
            content: "this is not valid python!!!".into(),
        }],
        requirements: "".into(),
        entry_point: "bad.py".into(),
        input_data: None,
        params: serde_json::json!({}),
    };

    let msg = CoordinatorToWorker::AssignPythonJob { job };
    ws.send(Message::Text(serde_json::to_string(&msg).unwrap().into()))
        .await
        .unwrap();

    let mut messages = Vec::new();
    let timeout = tokio::time::timeout(std::time::Duration::from_secs(60), async {
        while let Some(Ok(Message::Text(text))) = ws.next().await {
            let parsed: serde_json::Value = serde_json::from_str(&text).unwrap_or_default();
            let msg_type = parsed.get("type").and_then(|t| t.as_str()).unwrap_or("");
            messages.push(parsed.clone());
            if msg_type == "JobResult" {
                break;
            }
        }
    });

    timeout.await.expect("timed out");

    let result = messages.last().unwrap();
    assert_eq!(result["type"], "JobResult");
    assert_eq!(
        result["success"], false,
        "job with syntax error should fail"
    );

    ws.close(None).await.ok();
    server.abort();
}
