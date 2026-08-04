//! MCP (Model Context Protocol) server for Soma.
//!
//! Exposes 20 tools for code agents: filter CRUD, knowledge base
//! queries, report generation, and the seven experiment-pool tools
//! (`kb_*`) that let a model find what has already been tried, follow a
//! lineage, compare two runs and retain what it concluded.
//!
//! `run_pipeline` and `run_study` are declared but not implemented —
//! executing user code is out of reach for this server, and their
//! descriptions say so rather than returning a stub that reads like
//! success.

pub mod context;
pub mod protocol;
pub mod render;
pub mod server;
pub mod tools;

pub use context::SomaContext;
pub use server::run_stdio;
