//! Knowledge-base backend selection and persistence across contexts.
//!
//! All tests use `SomaContext::with_env_override` so they neither read
//! nor leak through the real `SOMA_KB_PATH` environment variable.

use serde_json::json;
use somatize_mcp::context::SomaContext;
use std::fs;

fn record(ctx: &mut SomaContext, id: &str) {
    let result = ctx.record_experiment(&json!({
        "id": id,
        "name": format!("experiment {id}"),
        "research_line": "mos",
        "metrics": {"f1": 0.8},
    }));
    assert!(
        !result.is_error.unwrap_or(false),
        "record failed: {result:?}"
    );
}

fn found(ctx: &SomaContext, query: &str) -> bool {
    let result = ctx.query_knowledge_base(&json!({"query": query, "max_results": 10}));
    let text = result
        .content
        .iter()
        .map(|c| c.text.clone())
        .collect::<String>();
    text.contains(query)
}

#[test]
fn env_override_wins_over_project_soma_dir() {
    let project = tempfile::tempdir().unwrap();
    fs::create_dir_all(project.path().join(".soma")).unwrap();
    let elsewhere = tempfile::tempdir().unwrap();
    let override_path = elsewhere.path().join("custom-kb.jsonl");

    let mut ctx =
        SomaContext::with_env_override(project.path(), Some(override_path.display().to_string()));
    record(&mut ctx, "exp-env");

    assert!(override_path.exists(), "record went to the override path");
    assert!(
        !project.path().join(".soma/experiments.jsonl").exists(),
        "project .soma untouched when the override is set"
    );
    let line = fs::read_to_string(&override_path).unwrap();
    assert!(line.contains("exp-env"));
}

#[test]
fn project_soma_dir_enables_persistence() {
    let project = tempfile::tempdir().unwrap();
    fs::create_dir_all(project.path().join(".soma")).unwrap();

    let mut ctx = SomaContext::with_env_override(project.path(), None);
    record(&mut ctx, "exp-dir");
    assert!(project.path().join(".soma/experiments.jsonl").exists());
}

#[test]
fn soma_as_a_file_falls_back_to_memory() {
    let project = tempfile::tempdir().unwrap();
    fs::write(project.path().join(".soma"), "not a dir").unwrap();

    let mut ctx = SomaContext::with_env_override(project.path(), None);
    record(&mut ctx, "exp-file"); // must not panic, lands in memory
    assert!(found(&ctx, "exp-file"));
    drop(ctx);
    let rebuilt = SomaContext::with_env_override(project.path(), None);
    assert!(!found(&rebuilt, "exp-file"), "in-memory records are lost");
}

#[test]
fn no_soma_dir_means_in_memory_only() {
    let project = tempfile::tempdir().unwrap();
    let mut ctx = SomaContext::with_env_override(project.path(), None);
    record(&mut ctx, "exp-mem");
    assert!(found(&ctx, "exp-mem"));

    let rebuilt = SomaContext::with_env_override(project.path(), None);
    assert!(!found(&rebuilt, "exp-mem"));
}

#[test]
fn corrupt_kb_file_falls_back_to_memory_and_server_still_works() {
    let project = tempfile::tempdir().unwrap();
    let soma = project.path().join(".soma");
    fs::create_dir_all(&soma).unwrap();
    // Corruption in the MIDDLE of the file is a hard open error for
    // FileKnowledgeBase — the context must fall back, not die.
    fs::write(
        soma.join("experiments.jsonl"),
        "garbage line\n{\"id\":\"x\",\"name\":\"n\",\"pipeline_summary\":\"\",\"params\":{},\"metrics\":{},\"timestamp\":\"2026-07-26T10:00:00Z\",\"duration\":{\"secs\":0,\"nanos\":0},\"tags\":[]}\n",
    )
    .unwrap();

    let mut ctx = SomaContext::with_env_override(project.path(), None);
    record(&mut ctx, "exp-after-corruption");
    assert!(found(&ctx, "exp-after-corruption"));
}

#[test]
fn records_persist_across_context_rebuilds() {
    // The user-visible feature: an MCP server restart keeps the KB.
    let project = tempfile::tempdir().unwrap();
    fs::create_dir_all(project.path().join(".soma")).unwrap();

    {
        let mut ctx = SomaContext::with_env_override(project.path(), None);
        record(&mut ctx, "exp-persistent-1");
        record(&mut ctx, "exp-persistent-2");
    } // server "shuts down"

    let ctx = SomaContext::with_env_override(project.path(), None);
    assert!(found(&ctx, "exp-persistent-1"));
    assert!(found(&ctx, "exp-persistent-2"));

    // And the research-line tools see rehydrated records too.
    let lines = ctx.list_research_lines(&json!({}));
    let text = lines
        .content
        .iter()
        .map(|c| c.text.clone())
        .collect::<String>();
    assert!(text.contains("mos"), "research line from disk: {text}");
}
