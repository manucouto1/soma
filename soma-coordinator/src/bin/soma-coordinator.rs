//! soma-coordinator — worker registry and placement service.
//!
//! The crate's own documentation has advertised this command since it was
//! written; there was no binary behind it. Workers point at it with
//! `--coordinator`, and it answers three questions: who is alive, what can
//! they do, and which of them should take the next plan.
//!
//! It does not execute plans and does not carry their data. A client asks
//! for a placement, then talks to the worker directly — which is what keeps
//! tensor-sized payloads off this hop.
//!
//! ```bash
//! soma-coordinator --port 9090 --token sk-xxx
//! ```

use clap::Parser;
use somatize_coordinator::{WorkerRegistry, coordinator_router};

#[derive(Parser, Debug)]
#[command(
    name = "soma-coordinator",
    about = "Soma worker registry and placement"
)]
struct Args {
    /// Port to listen on.
    #[arg(short, long, default_value = "9090")]
    port: u16,

    /// Bearer token. Requests without it are rejected when set.
    #[arg(long, env = "SOMA_TOKEN")]
    token: Option<String>,

    /// Seconds without a heartbeat before a worker is considered gone.
    ///
    /// Workers beat every 10s, so the default tolerates two dropped
    /// requests before declaring anyone dead.
    #[arg(long, default_value = "30")]
    heartbeat_timeout: i64,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let registry = WorkerRegistry::new().with_heartbeat_timeout(args.heartbeat_timeout);
    let addr = format!("0.0.0.0:{}", args.port);

    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            tracing::error!("could not bind {addr}: {e}");
            std::process::exit(1);
        }
    };

    tracing::info!("Coordinator listening on {addr}");
    if args.token.is_some() {
        tracing::info!("Authentication enabled");
    }

    let shutdown = async {
        match tokio::signal::ctrl_c().await {
            Ok(()) => tracing::info!("Ctrl+C received, shutting down..."),
            Err(e) => tracing::warn!("could not listen for Ctrl+C: {e}"),
        }
    };

    if let Err(e) = axum::serve(listener, coordinator_router(registry, args.token))
        .with_graceful_shutdown(shutdown)
        .await
    {
        tracing::error!("coordinator stopped: {e}");
        std::process::exit(1);
    }
    tracing::info!("Coordinator stopped.");
}
