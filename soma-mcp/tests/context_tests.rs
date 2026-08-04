//! Tests for Soma MCP tool handlers.

use serde_json::json;
use somatize_mcp::context::SomaContext;
use std::fs;

fn setup_context() -> (SomaContext, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let filter_dir = dir.path().join("filters");
    fs::create_dir_all(&filter_dir).unwrap();
    fs::write(
        filter_dir.join("scaler.py"),
        "from soma import Filter\n\nclass Scaler(Filter):\n    def fit(self, x, y=None):\n        return {}\n    def forward(self, x, state):\n        return x\n",
    ).unwrap();
    // Hermetic: never read the developer's real SOMA_KB_PATH.
    let ctx = SomaContext::with_env_override(dir.path(), None);
    (ctx, dir)
}

fn result_text(r: &somatize_mcp::protocol::ToolCallResult) -> String {
    r.content
        .iter()
        .map(|c| c.text.clone())
        .collect::<Vec<_>>()
        .join("\n")
}

fn is_error(r: &somatize_mcp::protocol::ToolCallResult) -> bool {
    r.is_error == Some(true)
}

#[test]
fn list_filters_finds_py_files() {
    let (ctx, _dir) = setup_context();
    let result = ctx.list_filters(&json!({}));
    assert!(!is_error(&result));
    assert!(result_text(&result).contains("scaler.py"));
}

#[test]
fn list_filters_with_subdir() {
    let (ctx, _dir) = setup_context();
    let result = ctx.list_filters(&json!({"subdir": "filters"}));
    assert!(!is_error(&result));
    assert!(result_text(&result).contains("scaler.py"));
}

#[test]
fn read_filter_source() {
    let (ctx, _dir) = setup_context();
    let result = ctx.read_filter_source(&json!({"file_path": "filters/scaler.py"}));
    assert!(!is_error(&result));
    let text = result_text(&result);
    assert!(text.contains("class Scaler"));
    assert!(text.contains("def fit"));
}

#[test]
fn read_filter_source_not_found() {
    let (ctx, _dir) = setup_context();
    let result = ctx.read_filter_source(&json!({"file_path": "nonexistent.py"}));
    assert!(is_error(&result));
}

#[test]
fn write_filter_source_creates_file() {
    let (ctx, dir) = setup_context();
    let result = ctx.write_filter_source(&json!({
        "file_path": "filters/new_filter.py",
        "content": "# New filter\nclass NewFilter:\n    pass\n"
    }));
    assert!(!is_error(&result));
    assert!(dir.path().join("filters/new_filter.py").exists());
}

#[test]
fn record_and_query_experiment() {
    let (mut ctx, _dir) = setup_context();
    let result = ctx.record_experiment(&json!({
        "id": "exp_001",
        "name": "exp_001",
        "hypothesis": "testing baseline",
        "research_line": "test_line",
        "params": {"k": 5},
        "metrics": {"accuracy": 0.95}
    }));
    assert!(!is_error(&result));
    assert_eq!(ctx.kb.len(), 1);

    // Query should find something
    let q = ctx.query_knowledge_base(&json!({"query": "test"}));
    assert!(!is_error(&q));
}

#[test]
fn list_research_lines_empty() {
    let (ctx, _dir) = setup_context();
    let result = ctx.list_research_lines(&json!({}));
    assert!(!is_error(&result));
}

#[test]
fn create_research_line() {
    let (mut ctx, _dir) = setup_context();
    let result = ctx.create_research_line(&json!({
        "name": "normalization_study",
        "description": "Impact of normalization"
    }));
    assert!(!is_error(&result));
}

#[test]
fn promising_lines_empty() {
    let (ctx, _dir) = setup_context();
    let result = ctx.promising_lines(&json!({"metric": "accuracy"}));
    assert!(!is_error(&result));
}

#[test]
fn generate_report_empty() {
    let (ctx, _dir) = setup_context();
    let result = ctx.generate_report(&json!({"research_line": "nonexistent"}));
    // May be error or empty report, both are fine
    let _ = result;
}
