use soma_mcp::{SomaContext, run_stdio};
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
