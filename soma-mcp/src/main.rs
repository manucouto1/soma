//! The `soma-mcp` binary: an MCP server over stdio.
//!
//! Takes the project directory as the first argument, falling back to
//! `SOMA_PROJECT_DIR` and then the current directory, and serves until
//! stdin closes. Diagnostics go to stderr — stdout belongs to the
//! protocol.

use somatize_mcp::{SomaContext, run_stdio};
use std::env;

fn main() {
    let project_dir = env::args()
        .nth(1)
        .or_else(|| env::var("SOMA_PROJECT_DIR").ok())
        .unwrap_or_else(|| ".".into());

    eprintln!("soma-mcp: starting with project_dir={project_dir}");
    let ctx = SomaContext::new(project_dir);
    run_stdio(ctx);
}
