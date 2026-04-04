//! Remote worker for distributed pipeline execution.
//!
//! Receives execution plans from a coordinator, runs them locally,
//! and reports results. Each worker manages isolated Python environments
//! ([`EnvManager`]) and communicates via WebSocket ([`protocol`]).

pub mod coordinator;
pub mod coordinator_server;
pub mod detect;
pub mod env_manager;
pub mod protocol;
#[cfg(feature = "pyo3")]
pub mod py_filter;
pub mod remote_executor;
pub mod server;
pub mod strategy_executor;
pub mod worker;

pub use coordinator::{WorkerRegistry, WorkerStatus};
pub use coordinator_server::{coordinator_router, serve_coordinator};
pub use detect::ResourceLimits;
pub use env_manager::EnvManager;
pub use protocol::*;
#[cfg(feature = "pyo3")]
pub use py_filter::EmbeddedPyFilter;
pub use remote_executor::WsRemoteExecutor;
pub use server::{
    serve_worker, serve_worker_authenticated, worker_router, worker_router_authenticated,
};
pub use worker::Worker;
