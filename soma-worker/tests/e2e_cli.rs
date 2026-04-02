//! End-to-end test for the soma-worker CLI binary.
//!
//! Spawns the binary, waits for the health endpoint, verifies it responds,
//! then kills the process.

use std::process::{Command, Stdio};
use std::time::Duration;

/// Find the built binary path.
fn binary_path() -> std::path::PathBuf {
    let mut path = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    path.push("soma-worker");
    path
}

#[test]
fn cli_help_flag() {
    let output = Command::new(binary_path())
        .args(["--help"])
        .output()
        .expect("failed to run soma-worker --help");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Soma distributed execution worker"));
    assert!(stdout.contains("--port"));
    assert!(stdout.contains("--cpus"));
    assert!(stdout.contains("--token"));
    assert!(stdout.contains("--coordinator"));
}

#[tokio::test]
async fn cli_starts_and_responds_to_health() {
    let binary = binary_path();
    if !binary.exists() {
        eprintln!("Skipping: binary not found at {}", binary.display());
        return;
    }

    // Find a free port
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener); // free the port

    // Spawn the worker
    let mut child = Command::new(&binary)
        .args(["--port", &port.to_string(), "--id", "cli_test_worker"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn soma-worker");

    // Wait for it to start
    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{port}/health");

    let mut started = false;
    for _ in 0..20 {
        tokio::time::sleep(Duration::from_millis(200)).await;
        if let Ok(resp) = client.get(&url).send().await {
            if resp.status().is_success() {
                let text = resp.text().await.unwrap_or_default();
                assert_eq!(text, "ok");
                started = true;
                break;
            }
        }
    }

    // Verify /info endpoint
    if started {
        let info_url = format!("http://127.0.0.1:{port}/info");
        let resp = client.get(&info_url).send().await.unwrap();
        let json: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(json["worker_id"], "cli_test_worker");
    }

    // Kill the process
    child.kill().ok();
    child.wait().ok();

    assert!(
        started,
        "worker should have started and responded to /health within 4 seconds"
    );
}
