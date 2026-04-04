//! Remote worker for distributed pipeline execution.
//!
//! Receives execution plans from a coordinator, runs them locally,
//! and reports results. Each worker manages isolated Python environments
//! ([`EnvManager`]) and communicates via WebSocket ([`protocol`]).

pub mod detect;
pub mod env_manager;
pub mod protocol;
#[cfg(feature = "pyo3")]
pub mod py_filter;
pub mod server;
pub mod worker;
pub mod ws_transport;

pub use detect::ResourceLimits;
pub use env_manager::EnvManager;
pub use protocol::*;
#[cfg(feature = "pyo3")]
pub use py_filter::EmbeddedPyFilter;
pub use server::{
    serve_worker, serve_worker_authenticated, worker_router, worker_router_authenticated,
};
pub use worker::Worker;
pub use ws_transport::WsTransport;
