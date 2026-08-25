//! The unit test binary. One `mod` per module of `src/`, mirroring its shape.
//!
//! The tests live outside `src/` on purpose: they are another crate, so they
//! only see the public API and cannot lean on anything private to pass.

mod goal;
mod invariants;
mod partition;
mod point;
mod pruner;
mod sampler;
mod samples;
mod space;
