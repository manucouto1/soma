//! Coordinator service for Soma distributed workers.
//!
//! Manages worker registration, health monitoring, load balancing,
//! and plan routing. Separate from the worker process itself.

pub mod registry;
pub mod server;

pub use registry::{WorkerRegistry, WorkerStatus};
pub use server::{coordinator_router, serve_coordinator};
