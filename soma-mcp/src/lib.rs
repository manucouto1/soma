//! MCP (Model Context Protocol) server for Soma.
//!
//! Exposes 13 tools for code agents: filter CRUD, pipeline execution,
//! study management, knowledge base queries, and report generation.

pub mod context;
pub mod protocol;
pub mod server;
pub mod tools;

pub use context::SomaContext;
pub use server::run_stdio;
