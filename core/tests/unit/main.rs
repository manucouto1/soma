//! The unit test binary. One `mod` per module of `src/`.
//!
//! The tests live outside `src/` on purpose: they are another crate, so they
//! only see the public API and cannot lean on anything private to pass.

mod build;
mod device;
mod doubles;
mod execution;
mod graph;
mod placement;
mod plan;
mod value;
