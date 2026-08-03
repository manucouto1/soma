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

    /// Persistent DataStore path for shared data (e.g. "/data/soma").
    /// When set, workers can resolve DataRef::Local from this store.
    #[arg(long, env = "SOMA_DATA_STORE")]
    data_store: Option<String>,
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
    if let Some(store_path) = &args.data_store {
        let store = somatize_core::store::LocalDataStore::new(store_path);
        worker = worker.with_data_store(std::sync::Arc::new(store));
        tracing::info!("DataStore configured: {store_path}");
    }
    let addr = format!("0.0.0.0:{}", args.port);

    // Register with the coordinator, then keep saying so.
    //
    // Registration used to happen once and that was the end of it. The
    // coordinator drops a worker it has not heard from in 30 seconds, so
    // every worker vanished from `/workers`, `/summary` and routing half a
    // minute after start-up while still running perfectly well.
    if let Some(coordinator_url) = &args.coordinator {
        let address = format!("ws://{}:{}", local_ip(), args.port);
        register_with(coordinator_url, &worker_id, &address, &caps, &args.token).await;

        let url = coordinator_url.clone();
        let id = worker_id.clone();
        let token = args.token.clone();
        let caps = caps.clone();
        tokio::spawn(async move {
            // Comfortably inside the coordinator's 30s window, so a single
            // dropped request is not enough to be declared dead.
            let mut tick = tokio::time::interval(std::time::Duration::from_secs(10));
            loop {
                tick.tick().await;
                match send_heartbeat(&url, &id, &token).await {
                    HeartbeatOutcome::Ok => {}
                    // The coordinator does not know us: it restarted, or
                    // reaped us during a long plan. Registering again is
                    // the only way back, and it is idempotent.
                    HeartbeatOutcome::Unknown => {
                        tracing::warn!("coordinator no longer knows this worker; re-registering");
                        register_with(&url, &id, &address, &caps, &token).await;
                    }
                    HeartbeatOutcome::Unreachable(e) => {
                        tracing::warn!("heartbeat to {url} failed: {e}");
                    }
                }
            }
        });
    }

    if args.token.is_some() {
        tracing::info!("Authentication enabled");
    }
    let (router, requested) = somatize_worker::worker_router_with_shutdown(
        worker,
        &args.env_dir,
        &args.work_dir,
        args.token.clone(),
    );

    // Two ways to stop, one path out: Ctrl+C, or a `Shutdown` message over
    // the WebSocket. The latter used to call `std::process::exit(0)` from
    // inside the handler, which skipped this graceful shutdown entirely.
    let shutdown = async move {
        tokio::select! {
            result = tokio::signal::ctrl_c() => {
                match result {
                    Ok(()) => tracing::info!("Ctrl+C received, shutting down..."),
                    Err(e) => tracing::warn!("could not listen for Ctrl+C: {e}"),
                }
            }
            _ = requested.wait() => tracing::info!("shutdown requested by coordinator"),
        }
    };

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    tracing::info!("Worker listening on {addr}");
    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown)
        .await
        .unwrap();

    tracing::info!("Worker stopped.");
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

/// What a heartbeat attempt told us.
enum HeartbeatOutcome {
    Ok,
    /// The coordinator has no record of this worker.
    Unknown,
    Unreachable(String),
}

/// Announce this worker to a coordinator. Idempotent.
async fn register_with(
    coordinator_url: &str,
    worker_id: &str,
    address: &str,
    caps: &somatize_worker::protocol::Capabilities,
    token: &Option<String>,
) {
    let url = format!("{coordinator_url}/register");
    let body = serde_json::json!({
        "worker_id": worker_id,
        "address": address,
        "capabilities": caps,
    });

    let mut request = reqwest::Client::new().post(&url).json(&body);
    if let Some(token) = token {
        request = request.bearer_auth(token);
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

/// Tell the coordinator this worker is still here, and how loaded it is.
async fn send_heartbeat(
    coordinator_url: &str,
    worker_id: &str,
    token: &Option<String>,
) -> HeartbeatOutcome {
    let load = somatize_worker::protocol::LoadMetrics {
        cpu_usage: 0.0,
        memory_usage: 0.0,
        gpu_usage: vec![],
        active_plans: 0,
        queue_depth: 0,
        timestamp: chrono::Utc::now(),
    };
    let body = serde_json::json!({ "worker_id": worker_id, "load": load });

    let mut request = reqwest::Client::new()
        .post(format!("{coordinator_url}/heartbeat"))
        .json(&body);
    if let Some(token) = token {
        request = request.bearer_auth(token);
    }

    match request.send().await {
        Ok(resp) if resp.status().is_success() => HeartbeatOutcome::Ok,
        Ok(resp) if resp.status() == reqwest::StatusCode::NOT_FOUND => HeartbeatOutcome::Unknown,
        Ok(resp) => HeartbeatOutcome::Unreachable(resp.status().to_string()),
        Err(e) => HeartbeatOutcome::Unreachable(e.to_string()),
    }
}
