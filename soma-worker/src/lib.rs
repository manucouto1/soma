//! Remote worker for distributed pipeline execution.
//!
//! Receives execution plans from a coordinator, runs them locally,
//! and reports results. Each worker manages isolated Python environments
//! ([`EnvManager`]) and communicates via WebSocket ([`protocol`]).

pub mod env_manager;
pub mod protocol;
pub mod server;
pub mod worker;

pub use env_manager::EnvManager;
pub use protocol::*;
pub use server::{serve_worker, worker_router};
pub use worker::Worker;
