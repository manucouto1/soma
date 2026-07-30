//! The experiment-pool tools — Case-Based Reasoning over past runs.
//!
//! The four CBR steps map onto the tools directly: **Retrieve** is
//! [`find_similar`], **Reuse** is reading the `run_dir` a hit points at,
//! **Revise** is [`branch_from`] followed by an actual run, and
//! **Retain** is [`record_conclusion`].
//!
//! Handlers stay thin on purpose: they parse arguments, ask the
//! knowledge base, and hand the answer to [`crate::render`]. All the
//! text a model sees is produced by pure functions over there, where it
//! is snapshot-tested.

use crate::context::SomaContext;
use crate::protocol::ToolCallResult;
use crate::render;
use chrono::Utc;
use somatize_core::fingerprint::ArchitectureFingerprint;
use somatize_memory::{ExperimentRecord, RetrievalQuery, derive};
use somatize_runtime::tracking::{RunReader, summarize};
use std::path::PathBuf;

/// Retrieve: rank past experiments against a description of the
/// problem at hand.
pub fn find_similar(ctx: &SomaContext, params: &serde_json::Value) -> ToolCallResult {
    let query_text = params
        .get("query")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let like_run = params.get("like_run").and_then(|v| v.as_str());
    if query_text.is_empty() && like_run.is_none() {
        return ToolCallResult::error(
            "kb_find_similar needs a `query` (free text) or a `like_run` (an experiment id \
             whose architecture to match), or both.",
        );
    }

    let mut query = RetrievalQuery::new(&query_text, Utc::now());
    query.limit = params
        .get("limit")
        .and_then(|v| v.as_u64())
        .unwrap_or(5)
        .clamp(1, 50) as usize;
    if let Some(line) = params.get("research_line").and_then(|v| v.as_str()) {
        query.research_line = Some(line.to_string());
    }
    if let Some(tags) = params.get("tags").and_then(|v| v.as_array()) {
        query.tags = tags
            .iter()
            .filter_map(|t| t.as_str().map(String::from))
            .collect();
    }
    if let Some(days) = params.get("half_life_days").and_then(|v| v.as_f64())
        && days > 0.0
    {
        query.half_life_days = days;
    }

    if let Some(run_id) = like_run {
        match architecture_of(ctx, run_id) {
            Some(architecture) => query.architecture = Some(architecture),
            None => {
                return ToolCallResult::error(format!(
                    "no architecture recorded for '{run_id}' — it may predate fingerprinting, \
                     or its run directory may be gone. Retry with `query` alone."
                ));
            }
        }
    }

    match ctx.kb.retrieve(&query) {
        Ok(hits) => {
            let label = if query_text.is_empty() {
                format!("like {}", like_run.unwrap_or_default())
            } else {
                query_text
            };
            ToolCallResult::text(render::find_similar(&hits, &label))
        }
        Err(e) => ToolCallResult::error(format!("retrieval failed: {e}")),
    }
}

/// The experiment tree around one run, with the move on every edge.
pub fn lineage(ctx: &SomaContext, params: &serde_json::Value) -> ToolCallResult {
    let Some(id) = params.get("id").and_then(|v| v.as_str()) else {
        return ToolCallResult::error("kb_lineage needs an `id`.");
    };
    match ctx.kb.lineage(id) {
        Ok(Some(lineage)) => ToolCallResult::text(render::lineage(&lineage)),
        Ok(None) => ToolCallResult::error(unknown_id(id)),
        Err(e) => ToolCallResult::error(format!("lineage failed: {e}")),
    }
}

/// Compare any two experiments — related or not.
pub fn diff(ctx: &SomaContext, params: &serde_json::Value) -> ToolCallResult {
    let (Some(a_id), Some(b_id)) = (
        params.get("a").and_then(|v| v.as_str()),
        params.get("b").and_then(|v| v.as_str()),
    ) else {
        return ToolCallResult::error("kb_diff needs two experiment ids, `a` and `b`.");
    };
    let (Some(a), Some(b)) = (fetch(ctx, a_id), fetch(ctx, b_id)) else {
        let missing = if fetch(ctx, a_id).is_none() {
            a_id
        } else {
            b_id
        };
        return ToolCallResult::error(unknown_id(missing));
    };
    // The same pure diff the capture path uses, so an on-demand
    // comparison and a recorded derivation can never disagree.
    let move_ = derive(&a, &b);
    ToolCallResult::text(render::diff(&a, &b, &move_))
}

/// Retain: append a conclusion to an existing experiment.
///
/// Written as a separate `Amendment` line. The journal is strictly
/// append-only: an earlier record is never rewritten, so a note added
/// today cannot corrupt what was recorded when the run happened.
pub fn record_conclusion(ctx: &mut SomaContext, params: &serde_json::Value) -> ToolCallResult {
    let Some(run_id) = params.get("run_id").and_then(|v| v.as_str()) else {
        return ToolCallResult::error("kb_record_conclusion needs a `run_id`.");
    };
    let Some(notes) = params.get("notes").and_then(|v| v.as_str()) else {
        return ToolCallResult::error(
            "kb_record_conclusion needs `notes` — what you concluded, in your own words.",
        );
    };
    let Some(target) = fetch(ctx, run_id) else {
        return ToolCallResult::error(unknown_id(run_id));
    };

    let id = somatize_core::util::timestamp_id("amend");
    let mut amendment = ExperimentRecord::amendment(&id, run_id, notes);
    amendment.research_line = target.research_line.clone();
    if let Some(hypothesis) = params.get("hypothesis").and_then(|v| v.as_str()) {
        amendment = amendment.with_hypothesis(hypothesis);
    }
    if let Some(tags) = params.get("tags").and_then(|v| v.as_array()) {
        amendment = amendment.with_tags(
            tags.iter()
                .filter_map(|t| t.as_str().map(String::from))
                .collect(),
        );
    }

    match ctx.kb.record(amendment) {
        Ok(()) => ToolCallResult::text(render::conclusion_recorded(&id, &target)),
        Err(e) => ToolCallResult::error(format!("failed to record the conclusion: {e}")),
    }
}

/// Revise: point `.soma/HEAD` at a run so the next one branches from it.
pub fn branch_from(ctx: &SomaContext, params: &serde_json::Value) -> ToolCallResult {
    let Some(run_id) = params.get("run_id").and_then(|v| v.as_str()) else {
        return ToolCallResult::error("kb_branch_from needs a `run_id`.");
    };
    let root = ctx.tracking_root();
    match somatize_runtime::tracking::checkout(&root, run_id) {
        Ok(()) => ToolCallResult::text(render::branched(run_id, fetch(ctx, run_id).as_ref())),
        Err(e) => ToolCallResult::error(format!(
            "{e}\n\nHEAD was not moved. `kb_stats` lists what this project has recorded."
        )),
    }
}

/// Summarize a run directory on demand.
///
/// Reads the directory rather than the journal, so it works on runs
/// recorded long before the experiment pool existed — and on runs that
/// never got a journal line at all, because they crashed.
pub fn summarize_run(ctx: &SomaContext, params: &serde_json::Value) -> ToolCallResult {
    let Some(run_id) = params.get("run_id").and_then(|v| v.as_str()) else {
        return ToolCallResult::error("kb_summarize_run needs a `run_id` or a run directory path.");
    };
    let Some(dir) = resolve_run_dir(ctx, run_id) else {
        return ToolCallResult::error(format!(
            "no run directory for '{run_id}' under {}. Pass an absolute path if the run \
             lives elsewhere.",
            ctx.tracking_root().join("runs").display()
        ));
    };
    let summary = RunReader::open(&dir).and_then(|reader| summarize(&reader));
    match summary {
        Ok(summary) => ToolCallResult::text(render::summarize_run(&summary)),
        Err(e) => ToolCallResult::error(format!("cannot read {}: {e}", dir.display())),
    }
}

/// Orientation: how big the pool is and how much of it is usable.
pub fn stats(ctx: &SomaContext, _params: &serde_json::Value) -> ToolCallResult {
    let records = match ctx.kb.all() {
        Ok(records) => records,
        Err(e) => return ToolCallResult::error(format!("cannot read the pool: {e}")),
    };
    let lines = ctx.kb.research_lines().unwrap_or_default();
    ToolCallResult::text(render::stats(
        &records,
        &lines,
        ctx.kb_location().as_deref(),
    ))
}

// ── helpers ─────────────────────────────────────────────────────────

fn fetch(ctx: &SomaContext, id: &str) -> Option<ExperimentRecord> {
    ctx.kb.get(id).ok().flatten()
}

fn unknown_id(id: &str) -> String {
    format!(
        "no experiment '{id}' in this pool. `kb_find_similar` searches by text; `kb_stats` \
         says how much has been recorded at all."
    )
}

/// The architecture to match against: from the record if it has one,
/// else from the run directory's `fingerprint.json`.
fn architecture_of(ctx: &SomaContext, run_id: &str) -> Option<ArchitectureFingerprint> {
    if let Some(architecture) = fetch(ctx, run_id).and_then(|r| r.architecture) {
        return Some(architecture);
    }
    let dir = resolve_run_dir(ctx, run_id)?;
    let reader = RunReader::open(dir).ok()?;
    summarize(&reader).ok()?.architecture
}

/// Accept a run id, a path relative to the project, or an absolute path.
fn resolve_run_dir(ctx: &SomaContext, run_id: &str) -> Option<PathBuf> {
    let as_path = PathBuf::from(run_id);
    if as_path.is_dir() {
        return Some(as_path);
    }
    if let Some(dir) = fetch(ctx, run_id).and_then(|r| r.run_dir) {
        let dir = PathBuf::from(dir);
        if dir.is_dir() {
            return Some(dir);
        }
    }
    let under_root = ctx.tracking_root().join("runs").join(run_id);
    under_root.is_dir().then_some(under_root)
}
