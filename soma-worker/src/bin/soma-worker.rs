//! soma-worker — distributed execution daemon.
//!
//! Auto-detects hardware capabilities and exposes them via HTTP/WebSocket.
//! Optionally registers with a coordinator for auto-discovery.
//!
//! ```bash
//! # Basic: auto-detect everything, no auth
//! soma-worker
//!
//! # With resource limits (Slurm-style)
//! soma-worker --cpus 4 --memory 8G --gpus 1 --max-concurrent 2
//!
//! # With authentication
//! soma-worker --token sk-my-secret-token
//!
//! # With coordinator auto-registration
//! soma-worker --coordinator http://coord:9090 --token sk-xxx --tags gpu,training
//!
//! # Full example
//! soma-worker --port 8080 --cpus 4 --memory 8G --gpus 1 \
//!   --tags gpu,training --token sk-xxx \
//!   --coordinator http://coord:9090
//! ```

use clap::Parser;
use somatize_worker::detect::ResourceLimits;
use somatize_worker::protocol::Capabilities;
use somatize_worker::worker::Worker;

#[derive(Parser, Debug)]
#[command(name = "soma-worker", about = "Soma distributed execution worker")]
struct Args {
    /// Port to listen on.
    #[arg(short, long, default_value = "8080")]
    port: u16,

    /// Max CPU cores to expose (default: all detected).
    #[arg(long)]
    cpus: Option<usize>,

    /// Max memory to expose (e.g. "8G", "512M"). Default: all detected.
    #[arg(long)]
    memory: Option<String>,

    /// Max GPUs to expose (default: all detected).
    #[arg(long)]
    gpus: Option<usize>,

    /// Max concurrent plans to accept.
    #[arg(long, default_value = "4")]
    max_concurrent: usize,

    /// Comma-separated tags for routing (e.g. "gpu,training").
    #[arg(long, value_delimiter = ',')]
    tags: Vec<String>,

    /// Bearer token for authentication. Connections without this token are rejected.
    #[arg(long, env = "SOMA_TOKEN")]
    token: Option<String>,

    /// Coordinator URL for auto-registration (e.g. "http://coord:9090").
    #[arg(long, env = "SOMA_COORDINATOR")]
    coordinator: Option<String>,

    /// Worker ID (default: hostname or random).
    #[arg(long)]
    id: Option<String>,

    /// Custom environment directory for Python envs.
    #[arg(long, default_value = "/tmp/soma-envs")]
    env_dir: String,

    /// Custom working directory for job execution.
    #[arg(long, default_value = "/tmp/soma-work")]
    work_dir: String,

    /// Directory for temporary HTTP bulk uploads (auto-cleaned after 1h).
    #[arg(long)]
    temp_dir: Option<String>,
}

fn parse_memory(s: &str) -> u64 {
    let s = s.trim().to_uppercase();
    if let Some(n) = s.strip_suffix('G') {
        n.parse::<u64>().unwrap_or(0) * 1024 * 1024 * 1024
    } else if let Some(n) = s.strip_suffix('M') {
        n.parse::<u64>().unwrap_or(0) * 1024 * 1024
    } else if let Some(n) = s.strip_suffix('T') {
        n.parse::<u64>().unwrap_or(0) * 1024 * 1024 * 1024 * 1024
    } else {
        s.parse::<u64>().unwrap_or(0) // assume bytes
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    // Auto-detect capabilities
    let mut caps = Capabilities::detect();

    // Apply resource limits
    let limits = ResourceLimits {
        max_cpus: args.cpus,
        max_memory_bytes: args.memory.as_deref().map(parse_memory),
        max_gpus: args.gpus,
        max_concurrent: args.max_concurrent,
    };
    caps = caps.with_limits(&limits);

    // Add user tags
    for tag in &args.tags {
        if !caps.tags.contains(tag) {
            caps.tags.push(tag.clone());
        }
    }

    // Generate worker ID
    let worker_id = args.id.unwrap_or_else(|| {
        hostname::get()
            .map(|h| h.to_string_lossy().to_string())
            .unwrap_or_else(|_| format!("worker_{}", std::process::id()))
    });

    tracing::info!("Starting worker: {worker_id}");
    tracing::info!("Capabilities: {}", caps.summary());

    let mut worker = Worker::new(&worker_id, caps.clone());
    if let Some(temp_dir) = args.temp_dir {
        worker = worker.with_temp_dir(temp_dir.into());
    }
    let addr = format!("0.0.0.0:{}", args.port);

    // Register with coordinator if configured
    if let Some(coordinator_url) = &args.coordinator {
        let url = format!("{coordinator_url}/register");
        let body = serde_json::json!({
            "worker_id": worker_id,
            "address": format!("ws://{}:{}", local_ip(), args.port),
            "capabilities": caps,
        });

        let mut request = reqwest::Client::new().post(&url).json(&body);
        if let Some(token) = &args.token {
            request = request.query(&[("token", token.as_str())]);
        }

        match request.send().await {
            Ok(resp) if resp.status().is_success() => {
                tracing::info!("Registered with coordinator at {coordinator_url}");
            }
            Ok(resp) => {
                tracing::warn!(
                    "Coordinator registration failed: {} {}",
                    resp.status(),
                    resp.text().await.unwrap_or_default()
                );
            }
            Err(e) => {
                tracing::warn!("Could not reach coordinator at {coordinator_url}: {e}");
            }
        }
    }

    // Start server
    if let Some(token) = args.token {
        tracing::info!("Authentication enabled");
        somatize_worker::serve_worker_authenticated(worker, &addr, &token)
            .await
            .unwrap();
    } else {
        somatize_worker::serve_worker(worker, &addr).await.unwrap();
    }
}

fn local_ip() -> String {
    // Best-effort local IP detection
    std::net::UdpSocket::bind("0.0.0.0:0")
        .and_then(|s| {
            s.connect("8.8.8.8:80")?;
            s.local_addr()
        })
        .map(|addr| addr.ip().to_string())
        .unwrap_or_else(|_| "127.0.0.1".to_string())
}
