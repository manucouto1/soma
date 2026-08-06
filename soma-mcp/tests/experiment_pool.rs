//! The seven experiment-pool tools, over a real `.soma` fixture.
//!
//! These go through `dispatch` rather than calling handlers directly,
//! because dispatch is what a client reaches: the refresh-before-read,
//! the argument names in the schema and the rendered text are all part
//! of the contract being tested here.

use serde_json::json;
use somatize_core::tracking::summary::{RunConclusion, RunOutcome};
use somatize_mcp::SomaContext;
use somatize_mcp::tools::dispatch;
use somatize_memory::{DerivationMove, ExperimentRecord, MetricDelta};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use tempfile::TempDir;

/// A project with `.soma/`, so the context uses a file-backed journal.
fn project() -> TempDir {
    let dir = TempDir::new().unwrap();
    fs::create_dir_all(dir.path().join(".soma")).unwrap();
    dir
}

fn ctx(dir: &TempDir) -> SomaContext {
    SomaContext::with_env_override(dir.path(), None)
}

fn call(ctx: &mut SomaContext, tool: &str, params: serde_json::Value) -> String {
    dispatch(ctx, tool, &params).content_text()
}

fn call_expecting_error(ctx: &mut SomaContext, tool: &str, params: serde_json::Value) -> String {
    let result = dispatch(ctx, tool, &params);
    assert!(result.is_error(), "expected an error from {tool}");
    result.content_text()
}

/// Append a journal line directly — standing in for a training run
/// that finished in another process.
fn append(root: &Path, record: &ExperimentRecord) {
    use std::io::Write;
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(root.join(".soma").join("experiments.jsonl"))
        .unwrap();
    writeln!(file, "{}", serde_json::to_string(record).unwrap()).unwrap();
}

fn experiment(id: &str, name: &str, f1: f64) -> ExperimentRecord {
    let mut r = ExperimentRecord::new(id, name);
    r.research_line = Some("mos".into());
    r.pipeline_summary = "scaler(Scaler) → model(SVM)".into();
    r.metrics.insert("val_f1".into(), f1);
    r.run_dir = Some(format!("/proj/.soma/runs/{id}"));
    r.conclusion = Some(RunConclusion {
        headline: format!("completed in 2m 00s · val_f1={f1}"),
        outcome: Some(RunOutcome::Completed),
        cache_hit_ratio: Some(0.5),
        ..RunConclusion::default()
    });
    r
}

fn descended(mut child: ExperimentRecord, parent: &str, summary: &str) -> ExperimentRecord {
    child.parent = Some(parent.to_string());
    child.derivation = Some(DerivationMove {
        from: parent.to_string(),
        to: child.id.clone(),
        changes: vec![somatize_memory::Change::ParamChanged {
            key: "lr".into(),
            from: json!(0.01),
            to: json!(0.05),
        }],
        metric_delta: BTreeMap::from([(
            "val_f1".to_string(),
            MetricDelta {
                before: 0.81,
                after: 0.87,
                delta: 0.06,
            },
        )]),
        summary: summary.to_string(),
    });
    child
}

/// A minimal run directory: a manifest is all `summarize` requires.
fn run_dir(root: &Path, run_id: &str) -> std::path::PathBuf {
    let dir = root.join(".soma").join("runs").join(run_id);
    fs::create_dir_all(&dir).unwrap();
    fs::write(
        dir.join("manifest.json"),
        json!({
            "schema_version": 1,
            "run_id": run_id,
            "kind": "train",
            "name": "from-disk",
            "created_at": "2026-07-20T10:00:00Z",
        })
        .to_string(),
    )
    .unwrap();
    dir
}

fn seeded() -> TempDir {
    let dir = project();
    append(dir.path(), &experiment("run_a", "mos-baseline", 0.81));
    append(
        dir.path(),
        &descended(
            experiment("run_b", "mos-wider", 0.87),
            "run_a",
            "lr: 0.01 → 0.05 ⇒ val_f1 +0.06",
        ),
    );
    dir
}

// ── kb_find_similar ─────────────────────────────────────────────────

#[test]
fn find_similar_ranks_and_points_at_the_next_call() {
    let dir = seeded();
    let text = call(&mut ctx(&dir), "kb_find_similar", json!({"query": "wider"}));

    assert!(text.contains("mos-wider"), "{text}");
    assert!(
        text.contains("move: lr: 0.01 → 0.05 ⇒ val_f1 +0.06"),
        "{text}"
    );
    assert!(text.contains("run_dir: /proj/.soma/runs/run_b"), "{text}");
    assert!(text.contains("next: kb_lineage(id=\"run_b\")"), "{text}");
}

#[test]
fn find_similar_honors_limit_and_line_filters() {
    let dir = seeded();
    let mut ctx = ctx(&dir);
    let text = call(
        &mut ctx,
        "kb_find_similar",
        json!({"query": "mos", "limit": 1}),
    );
    assert!(text.starts_with("# 1 experiment"), "{text}");

    let text = call(
        &mut ctx,
        "kb_find_similar",
        json!({"query": "mos", "research_line": "does-not-exist"}),
    );
    assert!(text.contains("No experiments match"), "{text}");
}

#[test]
fn find_similar_needs_something_to_go_on() {
    let dir = seeded();
    let text = call_expecting_error(&mut ctx(&dir), "kb_find_similar", json!({}));
    assert!(text.contains("needs a `query`"), "{text}");
}

#[test]
fn find_similar_refuses_to_match_an_architecture_that_was_never_recorded() {
    let dir = seeded();
    let text = call_expecting_error(
        &mut ctx(&dir),
        "kb_find_similar",
        json!({"like_run": "run_a"}),
    );
    assert!(text.contains("no architecture recorded"), "{text}");
}

// ── kb_lineage ──────────────────────────────────────────────────────

#[test]
fn lineage_renders_the_tree_with_labeled_edges() {
    let dir = seeded();
    let text = call(&mut ctx(&dir), "kb_lineage", json!({"id": "run_b"}));

    assert!(text.contains("# Lineage of run_b"), "{text}");
    assert!(text.contains("· run_a — mos-baseline"), "{text}");
    assert!(
        text.contains("▶ run_b — mos-wider  ← lr: 0.01 → 0.05"),
        "{text}"
    );
    assert!(text.contains("1 ancestor, 0 descendants."), "{text}");
    assert!(text.contains("kb_diff(a=\"run_a\", b=\"run_b\")"), "{text}");
}

#[test]
fn lineage_of_an_unknown_id_says_where_to_look_instead() {
    let dir = seeded();
    let text = call_expecting_error(&mut ctx(&dir), "kb_lineage", json!({"id": "nope"}));
    assert!(text.contains("no experiment 'nope'"), "{text}");
    assert!(text.contains("kb_stats"), "{text}");
}

// ── kb_diff ─────────────────────────────────────────────────────────

#[test]
fn diff_reports_metrics_and_cost() {
    let dir = seeded();
    let text = call(
        &mut ctx(&dir),
        "kb_diff",
        json!({"a": "run_a", "b": "run_b"}),
    );

    assert!(text.contains("# run_a → run_b"), "{text}");
    assert!(text.contains("## Metrics"), "{text}");
    assert!(text.contains("- val_f1: 0.81 → 0.87 (+0.06)"), "{text}");
    assert!(text.contains("## Cost"), "{text}");
    assert!(text.contains("- cache hits: 50% → 50%"), "{text}");
}

#[test]
fn diff_needs_both_ids_to_exist() {
    let dir = seeded();
    let mut ctx = ctx(&dir);
    assert!(
        call_expecting_error(&mut ctx, "kb_diff", json!({"a": "run_a"}))
            .contains("needs two experiment ids")
    );
    assert!(
        call_expecting_error(&mut ctx, "kb_diff", json!({"a": "run_a", "b": "ghost"}))
            .contains("no experiment 'ghost'")
    );
}

// ── kb_record_conclusion ────────────────────────────────────────────

#[test]
fn recording_a_conclusion_appends_without_rewriting() {
    let dir = seeded();
    let journal = dir.path().join(".soma").join("experiments.jsonl");
    let before = fs::read_to_string(&journal).unwrap();

    let mut ctx = ctx(&dir);
    let text = call(
        &mut ctx,
        "kb_record_conclusion",
        json!({"run_id": "run_b", "notes": "the gain came from the schedule, not the width"}),
    );
    assert!(text.contains("Recorded an amendment to run_b"), "{text}");
    assert!(text.contains("append-only"), "{text}");

    let after = fs::read_to_string(&journal).unwrap();
    assert!(
        after.starts_with(&before),
        "existing lines must be untouched"
    );
    assert_eq!(after.lines().count(), before.lines().count() + 1);

    let amendment: serde_json::Value = serde_json::from_str(after.lines().last().unwrap()).unwrap();
    assert_eq!(amendment["kind"], "amendment");
    assert_eq!(amendment["amends"], "run_b");
    assert_eq!(amendment["research_line"], "mos", "inherits the line");

    // And it is retrievable, which is the whole point of retaining it.
    let text = call(&mut ctx, "kb_find_similar", json!({"query": "schedule"}));
    assert!(text.contains("the gain came from the schedule"), "{text}");
}

#[test]
fn recording_a_conclusion_needs_a_target_and_something_to_say() {
    let dir = seeded();
    let mut ctx = ctx(&dir);
    assert!(
        call_expecting_error(&mut ctx, "kb_record_conclusion", json!({"run_id": "run_b"}))
            .contains("needs `notes`")
    );
    assert!(
        call_expecting_error(
            &mut ctx,
            "kb_record_conclusion",
            json!({"run_id": "ghost", "notes": "x"})
        )
        .contains("no experiment 'ghost'")
    );
}

// ── kb_branch_from ──────────────────────────────────────────────────

#[test]
fn branching_writes_head_only_for_a_run_that_exists() {
    let dir = seeded();
    run_dir(dir.path(), "run_a");
    let head = dir.path().join(".soma").join("HEAD");
    let mut ctx = ctx(&dir);

    let text = call(&mut ctx, "kb_branch_from", json!({"run_id": "run_a"}));
    assert!(text.starts_with("HEAD → run_a"), "{text}");
    assert!(text.contains("mos-baseline"), "names what it branched from");
    assert_eq!(fs::read_to_string(&head).unwrap().trim(), "run_a");

    // A run with a journal line but no run directory is still a typo
    // risk: HEAD must not move to something checkout cannot verify.
    let text = call_expecting_error(&mut ctx, "kb_branch_from", json!({"run_id": "run_b"}));
    assert!(text.contains("no run 'run_b'"), "{text}");
    assert!(text.contains("HEAD was not moved"), "{text}");
    assert_eq!(fs::read_to_string(&head).unwrap().trim(), "run_a");
}

// ── kb_summarize_run ────────────────────────────────────────────────

#[test]
fn summarize_run_reads_a_directory_the_journal_never_saw() {
    let dir = project();
    run_dir(dir.path(), "run_orphan");
    let mut ctx = ctx(&dir);

    let text = call(
        &mut ctx,
        "kb_summarize_run",
        json!({"run_id": "run_orphan"}),
    );
    assert!(text.contains("# run_orphan — from-disk"), "{text}");
    assert!(text.contains("outcome: "), "{text}");
    assert!(text.contains("run_dir: "), "{text}");
    assert!(text.contains("next: kb_lineage"), "{text}");
}

#[test]
fn summarize_run_accepts_a_path_and_reports_a_missing_run() {
    let dir = project();
    let path = run_dir(dir.path(), "run_x");
    let mut ctx = ctx(&dir);

    let text = call(
        &mut ctx,
        "kb_summarize_run",
        json!({"run_id": path.to_string_lossy()}),
    );
    assert!(text.contains("# run_x"), "{text}");

    let text = call_expecting_error(&mut ctx, "kb_summarize_run", json!({"run_id": "ghost"}));
    assert!(text.contains("no run directory for 'ghost'"), "{text}");
}

// ── kb_stats ────────────────────────────────────────────────────────

#[test]
fn stats_report_coverage_and_orient_an_empty_pool() {
    let empty = project();
    let text = call(&mut ctx(&empty), "kb_stats", json!({}));
    assert!(text.contains("empty"), "{text}");
    assert!(text.contains("soma kb reindex"), "{text}");

    let dir = seeded();
    let text = call(&mut ctx(&dir), "kb_stats", json!({}));
    assert!(text.contains("experiments: 2"), "{text}");
    assert!(text.contains("- with a conclusion: 2/2 (100%)"), "{text}");
    assert!(
        text.contains("- with a parent (in a lineage): 1/2 (50%)"),
        "{text}"
    );
    assert!(text.contains("mos — 2 experiments"), "{text}");
    assert!(text.contains("experiments.jsonl"), "names the journal");
}

// ── refresh ─────────────────────────────────────────────────────────

#[test]
fn a_run_finishing_in_another_process_is_visible_on_the_next_call() {
    // The failure this prevents: an MCP server that has been up all
    // day answering "no such experiment" for a run that finished five
    // minutes ago in the user's terminal.
    let dir = seeded();
    let mut ctx = ctx(&dir);

    let text = call(&mut ctx, "kb_stats", json!({}));
    assert!(text.contains("experiments: 2"), "{text}");
    assert!(
        dispatch(&mut ctx, "kb_lineage", &json!({"id": "run_c"})).is_error(),
        "run_c does not exist yet"
    );

    // Another process finishes a run mid-session.
    append(
        dir.path(),
        &descended(
            experiment("run_c", "mos-deeper", 0.91),
            "run_b",
            "+depth=4 ⇒ val_f1 +0.04",
        ),
    );

    let text = call(&mut ctx, "kb_stats", json!({}));
    assert!(text.contains("experiments: 3"), "{text}");
    let text = call(&mut ctx, "kb_lineage", json!({"id": "run_c"}));
    assert!(text.contains("▶ run_c — mos-deeper"), "{text}");
    assert!(text.contains("2 ancestors, 0 descendants."), "{text}");
}
