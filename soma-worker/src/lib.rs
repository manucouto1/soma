// The crate is fully documented and clippy runs with -D warnings in CI,
// so this makes "public API without docs" a build error from here on.
#![warn(missing_docs)]

//! Remote worker for distributed pipeline execution.
//!
//! Receives execution plans from a coordinator, runs them locally,
//! and reports results. Each worker manages isolated Python environments
//! ([`EnvManager`]) and communicates via WebSocket ([`protocol`]).
//!
//! Python filters execute in a **child subprocess** — the GIL is completely
//! isolated from Rust/Tokio threads. No PyO3, no segfaults.

pub mod detect;
pub mod env_manager;
pub mod error;
pub mod protocol;
pub mod python_process;
pub mod server;
pub mod worker;
pub mod ws_transport;

pub use detect::ResourceLimits;
pub use env_manager::EnvManager;
pub use error::WorkerError;
pub use protocol::*;
pub use python_process::{PythonProcess, SubprocessFilter};
pub use server::{
    ShutdownSignal, serve_worker, serve_worker_authenticated, worker_router,
    worker_router_authenticated, worker_router_with_shutdown,
};
pub use worker::Worker;
pub use ws_transport::WsTransport;
