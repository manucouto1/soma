//! The unit test binary, one `mod` per module of `src/`. Outside `src/` on
//! purpose: another crate, so it can only lean on the public API.

mod build;
mod device;
mod doubles;
mod execution;
mod fact;
mod graph;
mod host;
mod key;
mod memory;
mod placement;
mod plan;
mod value;
mod watcher;
