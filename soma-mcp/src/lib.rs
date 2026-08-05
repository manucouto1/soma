// The crate is fully documented and clippy runs with -D warnings in CI,
// so this makes "public API without docs" a build error from here on.
#![warn(missing_docs)]

//! MCP (Model Context Protocol) server for Soma.
//!
//! Exposes 20 tools for code agents: filter CRUD, knowledge base
//! queries, report generation, and the seven experiment-pool tools
//! (`kb_*`) that let a model find what has already been tried, follow a
//! lineage, compare two runs and retain what it concluded.
//!
//! `run_pipeline` and `run_study` execute. They used to echo their
//! arguments back, on the grounds that this server cannot load user
//! code — true of the server, which is Rust, and beside the point: it
//! runs the graph in a Python subprocess rooted at the project
//! directory, which is what `soma-worker` has always done. See
//! [`exec`].

pub mod context;
pub mod exec;
pub mod protocol;
pub mod render;
pub mod server;
pub mod tools;

pub use context::SomaContext;
pub use server::run_stdio;
