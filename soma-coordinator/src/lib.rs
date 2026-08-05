// The crate is fully documented and clippy runs with -D warnings in CI,
// so this makes "public API without docs" a build error from here on.
#![warn(missing_docs)]

//! Coordinator service for Soma distributed workers.
//!
//! Manages worker registration, health monitoring, load balancing,
//! and plan routing. Separate from the worker process itself.

pub mod registry;
pub mod server;

pub use registry::{WorkerRegistry, WorkerStatus};
pub use server::{coordinator_router, serve_coordinator};
