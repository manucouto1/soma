//! Rendering the experiment pool for a model to read.
//!
//! MCP carries text. There is no structured result a client will render
//! for us, so **the text is the API**: what these functions emit is
//! what the model sees, and its shape decides whether the model can
//! follow a lineage, compare two runs, or notice that it is looking at
//! a dead end.
//!
//! Three rules hold everywhere below.
//!
//! - **Every result ends with a `next:` line.** A model that has just
//!   read a hit should not have to guess what the follow-up call is
//!   named or which argument it takes.
//! - **Every experiment shows its `run_dir`.** The pool summarizes; the
//!   run directory has the events, the diagnostics and the figures. A
//!   model with file tools can go and read them.
//! - **Absence is stated, never faked.** "no conclusion recorded" is a
//!   useful sentence; a blank line is not.
//!
//! Every function here is pure — a value in, a `String` out — so the
//! output is snapshot-tested rather than merely eyeballed.

use somatize_core::summary::{RunSummary, human_duration, round4};
use somatize_memory::knowledge_base::Lineage;
use somatize_memory::{
    Change, DerivationMove, ExperimentRecord, ResearchLine, ScoredRecord, is_dead_end,
};
use std::fmt::Write as _;

/// Metrics listed inline before the list is cut short.
const MAX_METRICS: usize = 6;

/// Ranked hits for `kb_find_similar`.
pub fn find_similar(hits: &[ScoredRecord], query: &str) -> String {
    if hits.is_empty() {
        return format!(
            "No experiments match \"{query}\".\n\n\
             The pool may simply be empty — `kb_stats` says how much is in it, and \
             `soma kb reindex` rebuilds it from the run directories if the journal was lost.\n\n\
             next: kb_stats()"
        );
    }
    let mut out = format!(
        "# {} experiment{} matching \"{query}\"\n",
        hits.len(),
        plural(hits.len())
    );
    for (i, hit) in hits.iter().enumerate() {
        let _ = write!(
            out,
            "\n## {}. {} — {:.2}\n{}",
            i + 1,
            hit.record.name,
            hit.score,
            record_body(&hit.record)
        );
        let _ = writeln!(out, "why: {}", hit.why());
    }
    out.push_str(&next(&[
        &format!("kb_lineage(id=\"{}\")", hits[0].record.id),
        &format!("kb_summarize_run(run_id=\"{}\")", hits[0].record.id),
        &format!("kb_branch_from(run_id=\"{}\")", hits[0].record.id),
    ]));
    out
}

/// One experiment, as the body of a list entry.
fn record_body(record: &ExperimentRecord) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "id: {}", record.id);

    if let Some(derivation) = &record.derivation
        && !derivation.summary.is_empty()
    {
        let _ = writeln!(out, "move: {}", derivation.summary);
    }
    match &record.conclusion {
        Some(c) if !c.headline.is_empty() => {
            let _ = writeln!(out, "outcome: {}", c.headline);
        }
        _ => out.push_str("outcome: no conclusion recorded\n"),
    }
    if is_dead_end(record) {
        out.push_str("⚠ dead end — worth reading before trying this again\n");
    }
    if !record.pipeline_summary.is_empty() {
        let _ = writeln!(out, "pipeline: {}", record.pipeline_summary);
    }
    if let Some(hypothesis) = &record.hypothesis {
        let _ = writeln!(out, "hypothesis: {hypothesis}");
    }
    if let Some(notes) = &record.notes {
        let _ = writeln!(out, "notes: {notes}");
    }
    let mut context = Vec::new();
    if let Some(line) = &record.research_line {
        context.push(format!("line {line}"));
    }
    if let Some(parent) = &record.parent {
        context.push(format!("parent {parent}"));
    }
    if !record.tags.is_empty() {
        context.push(format!("tags {}", record.tags.join(", ")));
    }
    if !context.is_empty() {
        let _ = writeln!(out, "context: {}", context.join(" · "));
    }
    if !record.metrics.is_empty() {
        let _ = writeln!(out, "metrics: {}", metrics_line(record));
    }
    if !record.params.is_empty() {
        let _ = writeln!(out, "params: {}", params_line(record));
    }
    match &record.run_dir {
        Some(dir) => {
            let _ = writeln!(out, "run_dir: {dir}");
        }
        None => out.push_str("run_dir: none (recorded without a run directory)\n"),
    }
    out
}

/// A lineage tree with the move on every edge — the whole point of
/// recording derivations rather than only parents.
pub fn lineage(lineage: &Lineage) -> String {
    let mut out = format!(
        "# Lineage of {} — {}\n\n",
        lineage.focus.id, lineage.focus.name
    );

    for (depth, ancestor) in lineage.ancestors.iter().enumerate() {
        out.push_str(&tree_line(depth, ancestor, false));
    }
    let focus_depth = lineage.ancestors.len();
    out.push_str(&tree_line(focus_depth, &lineage.focus, true));
    for node in &lineage.descendants {
        out.push_str(&tree_line(focus_depth + node.depth, &node.record, false));
    }

    let _ = write!(
        out,
        "\n{} ancestor{}, {} descendant{}.\n",
        lineage.ancestors.len(),
        plural(lineage.ancestors.len()),
        lineage.descendants.len(),
        plural(lineage.descendants.len())
    );
    if lineage.ancestors.is_empty() && lineage.descendants.is_empty() {
        out.push_str(
            "This experiment stands alone. Runs get a parent from `.soma/HEAD`; \
             `kb_branch_from` points HEAD at a run so the next one descends from it.\n",
        );
    }

    let mut follow_ups = vec![format!("kb_summarize_run(run_id=\"{}\")", lineage.focus.id)];
    if let Some(parent) = lineage.ancestors.last() {
        follow_ups.push(format!(
            "kb_diff(a=\"{}\", b=\"{}\")",
            parent.id, lineage.focus.id
        ));
    }
    follow_ups.push(format!("kb_branch_from(run_id=\"{}\")", lineage.focus.id));
    out.push_str(&next(
        &follow_ups.iter().map(String::as_str).collect::<Vec<_>>(),
    ));
    out
}

/// One indented node, with the move that produced it.
fn tree_line(depth: usize, record: &ExperimentRecord, is_focus: bool) -> String {
    let indent = "  ".repeat(depth);
    let marker = if is_focus { "▶" } else { "·" };
    let mut line = format!("{indent}{marker} {} — {}", record.id, record.name);
    if let Some(derivation) = &record.derivation
        && !derivation.summary.is_empty()
    {
        let _ = write!(line, "  ← {}", derivation.summary);
    }
    line.push('\n');
    if let Some(conclusion) = &record.conclusion
        && !conclusion.headline.is_empty()
    {
        let _ = writeln!(line, "{indent}    {}", conclusion.headline);
    }
    line
}

/// Two experiments side by side: what changed, what it did to the
/// metrics, and what it cost.
pub fn diff(a: &ExperimentRecord, b: &ExperimentRecord, move_: &DerivationMove) -> String {
    let mut out = format!(
        "# {} → {}\n\n{} — {}\n{} — {}\n\n",
        a.id, b.id, a.id, a.name, b.id, b.name
    );

    out.push_str("## Changes\n\n");
    if move_.changes.is_empty() {
        out.push_str("- (none detected)\n");
    }
    for change in &move_.changes {
        let _ = writeln!(out, "- {}", change.describe());
        if let Change::Unspecified { reason } = change {
            let _ = writeln!(
                out,
                "  (soma cannot describe this move: {reason}. The run directories hold the \
                 raw graph.json if they still exist.)"
            );
        }
    }

    out.push_str("\n## Metrics\n\n");
    if move_.metric_delta.is_empty() {
        out.push_str("- no metric appears in both runs\n");
    }
    for (name, delta) in &move_.metric_delta {
        let _ = writeln!(
            out,
            "- {name}: {} → {} ({}{})",
            round4(delta.before),
            round4(delta.after),
            if delta.delta >= 0.0 { "+" } else { "−" },
            round4(delta.delta.abs())
        );
    }
    out.push_str(
        "\nSigns are raw differences. Whether up is good depends on the objective — \
         soma does not guess.\n",
    );

    out.push_str("\n## Cost\n\n");
    out.push_str(&cost_rows(a, b));

    for record in [a, b] {
        if let Some(dir) = &record.run_dir {
            let _ = writeln!(out, "\nrun_dir ({}): {dir}", record.id);
        }
    }
    out.push_str(&next(&[
        &format!("kb_lineage(id=\"{}\")", b.id),
        &format!("kb_record_conclusion(run_id=\"{}\", notes=\"...\")", b.id),
    ]));
    out
}

/// Wall time and cache effectiveness, which are as much a result as the
/// metrics are — a variant that matches the baseline in half the time
/// is a win the metric table cannot show.
fn cost_rows(a: &ExperimentRecord, b: &ExperimentRecord) -> String {
    let mut out = String::new();
    let (ms_a, ms_b) = (a.duration.as_millis() as u64, b.duration.as_millis() as u64);
    let _ = write!(
        out,
        "- duration: {} → {}",
        human_duration(ms_a),
        human_duration(ms_b)
    );
    if ms_a > 0 {
        let ratio = ms_b as f64 / ms_a as f64;
        let _ = write!(out, " ({ratio:.2}×)");
    }
    out.push('\n');

    let hit_ratio = |r: &ExperimentRecord| r.conclusion.as_ref().and_then(|c| c.cache_hit_ratio);
    match (hit_ratio(a), hit_ratio(b)) {
        (Some(x), Some(y)) => {
            let _ = writeln!(
                out,
                "- cache hits: {}% → {}%",
                (x * 100.0).round() as i64,
                (y * 100.0).round() as i64
            );
        }
        _ => out.push_str("- cache hits: not recorded for both runs\n"),
    }
    out
}

/// A run directory summarized on demand — works on runs recorded long
/// before the pool existed.
pub fn summarize_run(summary: &RunSummary) -> String {
    let mut out = format!("# {} — {}\n\n", summary.run_id, summary.name);
    let _ = writeln!(out, "kind: {}", summary.kind);
    let _ = writeln!(out, "started: {}", summary.created_at.to_rfc3339());
    if let Some(ms) = summary.duration_ms {
        let _ = writeln!(out, "duration: {}", human_duration(ms));
    }
    let _ = writeln!(out, "outcome: {}", summary.conclusion.headline);
    if !summary.pipeline_summary.is_empty() {
        let _ = writeln!(out, "pipeline: {}", summary.pipeline_summary);
    }
    if let Some(architecture) = &summary.architecture {
        let _ = writeln!(
            out,
            "architecture: {} ({} nodes, {} edges)",
            architecture.short(),
            architecture.n_nodes,
            architecture.n_edges
        );
    }
    if let Some(parent) = &summary.parent_run_id {
        let _ = writeln!(out, "parent: {parent}");
    }
    if let Some(hypothesis) = &summary.hypothesis {
        let _ = writeln!(out, "hypothesis: {hypothesis}");
    }

    if !summary.metrics.is_empty() {
        out.push_str("\n## Metrics\n\n");
        for (name, value) in &summary.metrics {
            let _ = writeln!(out, "- {name}: {}", round4(*value));
        }
    }
    if let Some(trials) = &summary.conclusion.trials {
        let _ = write!(
            out,
            "\n## Trials\n\n- {} total, {} completed, {} pruned, {} failed\n",
            trials.total, trials.completed, trials.pruned, trials.failed
        );
        if let (Some(objective), Some(best)) = (&trials.objective, trials.best_value) {
            let _ = writeln!(out, "- best {objective} = {}", round4(best));
        }
    }
    let flags = somatize_core::summary::FlagCount::merge_all(
        &summary.conclusion.health_flags,
        &summary.conclusion.audit_flags,
    );
    if !flags.is_empty() {
        out.push_str("\n## Health flags\n\n");
        for flag in &flags {
            let _ = writeln!(
                out,
                "- {} ×{} at {}",
                flag.flag,
                flag.count,
                flag.nodes.join(", ")
            );
        }
    }
    if !summary.conclusion.warnings.is_empty() {
        out.push_str("\n## What could not be read\n\n");
        for warning in &summary.conclusion.warnings {
            let _ = writeln!(out, "- {warning}");
        }
    }

    let _ = write!(out, "\nrun_dir: {}\n", summary.run_dir);
    out.push_str(&next(&[
        &format!("kb_lineage(id=\"{}\")", summary.run_id),
        &format!(
            "kb_record_conclusion(run_id=\"{}\", notes=\"...\")",
            summary.run_id
        ),
    ]));
    out
}

/// Orientation, with honest coverage: how much of the pool actually
/// carries the things the other tools depend on.
pub fn stats(
    records: &[ExperimentRecord],
    lines: &[ResearchLine],
    kb_path: Option<&str>,
) -> String {
    let total = records.len();
    if total == 0 {
        return format!(
            "The experiment pool is empty{}.\n\n\
             Runs are recorded automatically when `graph.track_run(...)` or `study.run(...)` \
             finishes successfully. If runs exist under `.soma/runs/` but the journal does not, \
             `soma kb reindex` rebuilds it.\n\n\
             next: kb_stats()",
            kb_path.map_or(String::new(), |p| format!(" ({p})"))
        );
    }
    let count = |f: fn(&ExperimentRecord) -> bool| records.iter().filter(|r| f(r)).count();
    let pct = |n: usize| (n as f64 * 100.0 / total as f64).round() as i64;

    let with_conclusion = count(|r| r.conclusion.as_ref().is_some_and(|c| !c.is_empty()));
    let with_lineage = count(|r| r.parent.is_some());
    let with_architecture = count(|r| r.architecture.is_some());
    let with_run_dir = count(|r| r.run_dir.is_some());
    let dead_ends = count(is_dead_end);
    let human_notes = count(|r| r.hypothesis.is_some() || r.notes.is_some());

    let mut out = String::from("# Experiment pool\n\n");
    if let Some(path) = kb_path {
        let _ = writeln!(out, "journal: {path}");
    }
    let _ = writeln!(out, "experiments: {total}");
    if let (Some(first), Some(last)) = (
        records.iter().map(|r| r.timestamp).min(),
        records.iter().map(|r| r.timestamp).max(),
    ) {
        let _ = writeln!(
            out,
            "span: {} → {}",
            first.format("%Y-%m-%d"),
            last.format("%Y-%m-%d")
        );
    }

    out.push_str("\n## Coverage\n\n");
    for (label, n) in [
        ("with a conclusion", with_conclusion),
        ("with a parent (in a lineage)", with_lineage),
        ("with an architecture fingerprint", with_architecture),
        ("with a run directory to read", with_run_dir),
        ("with a human hypothesis or note", human_notes),
    ] {
        let _ = writeln!(out, "- {label}: {n}/{total} ({}%)", pct(n));
    }
    let _ = writeln!(out, "- dead ends recorded: {dead_ends}");
    if with_lineage == 0 && total > 1 {
        out.push_str(
            "\nNothing in this pool has a parent, so `kb_lineage` and `kb_diff` have \
             nothing to work with. Runs inherit a parent from `.soma/HEAD`, which advances \
             after every successful run; `kb_branch_from` rewinds it.\n",
        );
    }

    if !lines.is_empty() {
        out.push_str("\n## Research lines\n\n");
        for line in lines {
            let _ = write!(
                out,
                "- {} — {} experiment{}, {}",
                line.name,
                line.experiments.len(),
                plural(line.experiments.len()),
                line.trend
            );
            if let (Some(name), Some(value)) = (&line.best_metric_name, line.best_metric_value) {
                let _ = write!(out, ", best {name}={}", round4(value));
            }
            out.push('\n');
        }
    }

    out.push_str(&next(&[
        "kb_find_similar(query=\"...\")",
        "list_research_lines()",
    ]));
    out
}

/// Confirmation for `kb_record_conclusion`.
pub fn conclusion_recorded(amendment_id: &str, target: &ExperimentRecord) -> String {
    format!(
        "Recorded an amendment to {} — {}.\n\n\
         amendment id: {amendment_id}\n\n\
         The journal is append-only: the original record is untouched, and this note is \
         layered on top of it. It is indexed for retrieval like any other text, so the next \
         `kb_find_similar` can surface it.\n\n{}",
        target.id,
        target.name,
        next(&[
            &format!("kb_lineage(id=\"{}\")", target.id),
            &format!("kb_branch_from(run_id=\"{}\")", target.id),
        ])
    )
}

/// Confirmation for `kb_branch_from`.
pub fn branched(run_id: &str, record: Option<&ExperimentRecord>) -> String {
    let mut out = format!("HEAD → {run_id}\n\n");
    if let Some(record) = record {
        let _ = writeln!(out, "{} — {}", record.id, record.name);
        if let Some(conclusion) = &record.conclusion
            && !conclusion.headline.is_empty()
        {
            let _ = writeln!(out, "{}", conclusion.headline);
        }
        out.push('\n');
    }
    out.push_str(
        "The next run in this project will record itself as a child of that run, with the \
         difference between them as the edge. Anything already descended from it stays where \
         it is — this creates a sibling branch, it does not move history.\n\n",
    );
    out.push_str(&next(&[&format!("kb_lineage(id=\"{run_id}\")")]));
    out
}

/// The follow-up line every result ends with.
fn next(calls: &[&str]) -> String {
    format!("\nnext: {}\n", calls.join(" · "))
}

fn metrics_line(record: &ExperimentRecord) -> String {
    let mut names: Vec<&String> = record.metrics.keys().collect();
    names.sort();
    let mut rendered: Vec<String> = names
        .iter()
        .take(MAX_METRICS)
        .map(|name| format!("{name}={}", round4(record.metrics[*name])))
        .collect();
    if names.len() > MAX_METRICS {
        rendered.push(format!("+{} more", names.len() - MAX_METRICS));
    }
    rendered.join(", ")
}

fn params_line(record: &ExperimentRecord) -> String {
    let mut names: Vec<&String> = record.params.keys().collect();
    names.sort();
    let mut rendered: Vec<String> = names
        .iter()
        .take(MAX_METRICS)
        .map(|name| {
            let value = &record.params[*name];
            let text = match value {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            format!("{name}={}", somatize_core::summary::one_line(&text, 40))
        })
        .collect();
    if names.len() > MAX_METRICS {
        rendered.push(format!("+{} more", names.len() - MAX_METRICS));
    }
    rendered.join(", ")
}

fn plural(n: usize) -> &'static str {
    if n == 1 { "" } else { "s" }
}

// ── Execution ──

/// A truncated preview of an output, for a model that needs the shape.
fn preview(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Object(map) if map.contains_key("truncated") => format!(
            "{} values (showing the first {})",
            map.get("length").and_then(|v| v.as_u64()).unwrap_or(0),
            map.get("head")
                .and_then(|h| h.as_array())
                .map(|a| a.len())
                .unwrap_or(0),
        ),
        serde_json::Value::Array(items) => {
            let shown: Vec<String> = items.iter().take(8).map(|v| v.to_string()).collect();
            if items.len() > shown.len() {
                format!("[{}, … {} total]", shown.join(", "), items.len())
            } else {
                format!("[{}]", shown.join(", "))
            }
        }
        other => {
            let text = other.to_string();
            if text.chars().count() > 200 {
                format!("{}…", text.chars().take(200).collect::<String>())
            } else {
                text
            }
        }
    }
}

/// What the driver reported for a failed run: the error, then whatever
/// the traceback and the filter's own prints said. A model debugging its
/// own graph needs the traceback more than it needs a tidy sentence.
fn render_failure(payload: &serde_json::Value, what: &str) -> String {
    let mut out = format!(
        "{what} failed\n\n{}\n",
        payload
            .get("error")
            .and_then(|v| v.as_str())
            .unwrap_or("the driver reported no reason")
    );
    for (label, key) in [
        ("traceback", "detail"),
        ("stdout", "stdout"),
        ("stderr", "stderr"),
    ] {
        if let Some(text) = payload.get(key).and_then(|v| v.as_str())
            && !text.trim().is_empty()
        {
            let _ = writeln!(out, "\n{label}:\n{text}");
        }
    }
    let _ = write!(
        out,
        "\nnext: read_filter_source(file_path=…) to see the code that failed"
    );
    out
}

/// One pipeline run, as the model reads it.
pub fn render_pipeline_run(payload: &serde_json::Value) -> String {
    if payload.get("ok").and_then(|v| v.as_bool()) != Some(true) {
        return render_failure(payload, "run_pipeline");
    }
    let mut out = String::from("run_pipeline: ok\n");
    if let Some(plan) = payload.get("plan").and_then(|v| v.as_str()) {
        let _ = writeln!(out, "\nplan:\n{}", plan.trim_end());
    }
    if let Some(output) = payload.get("output") {
        let _ = writeln!(out, "\noutput: {}", preview(output));
    }
    if let Some(state) = payload.get("state").and_then(|v| v.as_object())
        && !state.is_empty()
    {
        let names: Vec<&str> = state.keys().take(8).map(|s| s.as_str()).collect();
        let _ = writeln!(out, "state learned by: {}", names.join(", "));
    }
    for (label, key) in [("stdout", "stdout"), ("stderr", "stderr")] {
        if let Some(text) = payload.get(key).and_then(|v| v.as_str())
            && !text.trim().is_empty()
        {
            let _ = writeln!(out, "\n{label}:\n{text}");
        }
    }
    match payload.get("run_dir").and_then(|v| v.as_str()) {
        Some(dir) => {
            let _ = write!(
                out,
                "\nrun_dir: {dir}\nnext: kb_summarize_run(run_id=…) for what the \
                 pool recorded, or kb_find_similar(query=…) for what it resembles"
            );
        }
        None => {
            let _ = write!(
                out,
                "\nrun_dir: none — this run was not tracked, so the pool did not \
                 record it\nnext: run_pipeline(track=true) to keep the next one"
            );
        }
    }
    out
}

/// One study, as the model reads it.
pub fn render_study_run(payload: &serde_json::Value) -> String {
    if payload.get("ok").and_then(|v| v.as_bool()) != Some(true) {
        return render_failure(payload, "run_study");
    }
    let mut out = format!(
        "run_study: {} trials\n",
        payload
            .get("n_trials")
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
    );
    if let Some(objectives) = payload.get("objectives").and_then(|v| v.as_array()) {
        let pairs: Vec<String> = objectives
            .iter()
            .filter_map(|o| o.as_array())
            .map(|o| {
                format!(
                    "{} ({})",
                    o.first().and_then(|v| v.as_str()).unwrap_or("?"),
                    o.get(1).and_then(|v| v.as_str()).unwrap_or("?")
                )
            })
            .collect();
        let _ = writeln!(out, "optimizing: {}", pairs.join(", "));
    }
    match payload.get("best_trial") {
        Some(serde_json::Value::Object(best)) => {
            out.push_str("\nbest trial:\n");
            if let Some(params) = best.get("params").and_then(|v| v.as_object()) {
                for (key, value) in params.iter().take(MAX_METRICS) {
                    let _ = writeln!(out, "  {key} = {value}");
                }
            }
            if let Some(metrics) = best.get("metrics").and_then(|v| v.as_object()) {
                for (key, value) in metrics.iter().take(MAX_METRICS) {
                    let _ = writeln!(out, "  → {key} = {value}");
                }
            }
        }
        // A study that ran and chose nothing is worth saying out loud:
        // every trial pruned, or every one failed.
        _ => out.push_str("\nno best trial: every trial was pruned or errored\n"),
    }
    if let Some(text) = payload.get("stderr").and_then(|v| v.as_str())
        && !text.trim().is_empty()
    {
        let _ = writeln!(out, "\nstderr:\n{text}");
    }
    let _ = write!(
        out,
        "\nrun_dir: {}\nnext: kb_find_similar(query=…) to place this against \
         earlier work, or run_pipeline(...) with the best params to keep one run",
        payload
            .get("run_dir")
            .and_then(|v| v.as_str())
            .unwrap_or("none — the study was not tracked")
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};
    use somatize_core::summary::{RunConclusion, RunOutcome, TrialSummary};
    use somatize_memory::knowledge_base::LineageNode;
    use somatize_memory::{MetricDelta, RetrievalQuery, Trend, rank};
    use std::collections::BTreeMap;
    use std::time::Duration;

    fn at(day: u32) -> chrono::DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 7, day, 12, 0, 0).unwrap()
    }

    fn record(id: &str, name: &str) -> ExperimentRecord {
        let mut r = ExperimentRecord::new(id, name);
        r.timestamp = at(20);
        r.run_dir = Some(format!("/proj/.soma/runs/{id}"));
        r.pipeline_summary = "scaler(Scaler) → model(SVM)".into();
        r.research_line = Some("mos".into());
        r.duration = Duration::from_secs(120);
        r.metrics.insert("val_f1".into(), 0.87);
        r.conclusion = Some(RunConclusion {
            headline: "completed in 2m 00s · val_f1=0.87".into(),
            outcome: Some(RunOutcome::Completed),
            cache_hit_ratio: Some(0.5),
            ..RunConclusion::default()
        });
        r
    }

    fn move_from(from: &str, to: &str) -> DerivationMove {
        DerivationMove {
            from: from.into(),
            to: to.into(),
            changes: vec![Change::ParamChanged {
                key: "lr".into(),
                from: serde_json::json!(0.01),
                to: serde_json::json!(0.05),
            }],
            metric_delta: BTreeMap::from([(
                "val_f1".to_string(),
                MetricDelta {
                    before: 0.81,
                    after: 0.87,
                    delta: 0.06,
                },
            )]),
            summary: "lr: 0.01 → 0.05 ⇒ val_f1 +0.06".into(),
        }
    }

    /// Assertions shared by every rendering: a model must always be
    /// able to find the next call, and never see an empty result.
    fn assert_navigable(text: &str) {
        assert!(!text.trim().is_empty());
        let last = text.trim_end().lines().last().unwrap();
        assert!(last.starts_with("next: "), "no follow-up line: {last:?}");
        assert!(last.contains("("), "follow-ups must be callable: {last:?}");
    }

    #[test]
    fn a_hit_shows_its_move_outcome_and_run_dir() {
        let mut hit = record("run_b", "mos-wider");
        hit.derivation = Some(move_from("run_a", "run_b"));
        hit.parent = Some("run_a".into());
        hit.tags = vec!["mos".into()];
        hit.params.insert("lr".into(), serde_json::json!(0.05));

        let hits = rank(&[hit], &RetrievalQuery::new("wider", at(21)));
        let text = find_similar(&hits, "wider");

        assert!(
            text.starts_with("# 1 experiment matching \"wider\""),
            "{text}"
        );
        assert!(
            text.contains("move: lr: 0.01 → 0.05 ⇒ val_f1 +0.06"),
            "{text}"
        );
        assert!(text.contains("outcome: completed in 2m 00s · val_f1=0.87"));
        assert!(text.contains("pipeline: scaler(Scaler) → model(SVM)"));
        assert!(text.contains("context: line mos · parent run_a · tags mos"));
        assert!(text.contains("metrics: val_f1=0.87"));
        assert!(text.contains("params: lr=0.05"));
        assert!(text.contains("run_dir: /proj/.soma/runs/run_b"));
        assert!(text.contains("why: score "));
        assert_navigable(&text);
    }

    #[test]
    fn a_missing_conclusion_is_stated_not_hidden() {
        let mut bare = record("run_x", "unexplained");
        bare.conclusion = None;
        bare.run_dir = None;
        let hits = rank(&[bare], &RetrievalQuery::new("unexplained", at(21)));
        let text = find_similar(&hits, "unexplained");
        assert!(text.contains("outcome: no conclusion recorded"), "{text}");
        assert!(
            text.contains("run_dir: none (recorded without a run directory)"),
            "{text}"
        );
    }

    #[test]
    fn a_dead_end_is_flagged_for_the_model() {
        let mut failed = record("run_f", "collapsed");
        failed.conclusion = Some(RunConclusion {
            headline: "failed after 12.0s · error: loss became NaN".into(),
            outcome: Some(RunOutcome::Failed),
            ..RunConclusion::default()
        });
        let hits = rank(&[failed], &RetrievalQuery::new("collapsed", at(21)));
        let text = find_similar(&hits, "collapsed");
        assert!(text.contains("⚠ dead end"), "{text}");
    }

    #[test]
    fn no_hits_explains_itself_instead_of_returning_nothing() {
        let text = find_similar(&[], "nothing like this");
        assert!(text.contains("No experiments match"));
        assert!(text.contains("soma kb reindex"), "tells the model the fix");
        assert_navigable(&text);
    }

    #[test]
    fn a_lineage_puts_the_move_on_every_edge() {
        let root = record("run_a", "baseline");
        let mut focus = record("run_b", "wider");
        focus.derivation = Some(move_from("run_a", "run_b"));
        let mut child = record("run_c", "wider+deeper");
        child.derivation = Some(DerivationMove {
            summary: "+depth=4 ⇒ val_f1 −0.02".into(),
            ..move_from("run_b", "run_c")
        });

        let text = lineage(&Lineage {
            focus: focus.clone(),
            ancestors: vec![root],
            descendants: vec![LineageNode {
                record: child,
                depth: 1,
            }],
        });

        assert!(text.contains("· run_a — baseline"), "{text}");
        assert!(
            text.contains("▶ run_b — wider  ← lr: 0.01 → 0.05"),
            "{text}"
        );
        assert!(
            text.contains("· run_c — wider+deeper  ← +depth=4"),
            "{text}"
        );
        assert!(text.contains("1 ancestor, 1 descendant."));
        // Indentation grows with depth, so the tree reads as a tree.
        let focus_line = text.lines().find(|l| l.contains("▶")).unwrap();
        let child_line = text.lines().find(|l| l.contains("run_c")).unwrap();
        assert!(
            child_line.len() - child_line.trim_start().len()
                > focus_line.len() - focus_line.trim_start().len()
        );
        assert!(text.contains("kb_diff(a=\"run_a\", b=\"run_b\")"));
        assert_navigable(&text);
    }

    #[test]
    fn a_lone_experiment_is_told_how_to_get_a_lineage() {
        let text = lineage(&Lineage {
            focus: record("run_solo", "alone"),
            ancestors: Vec::new(),
            descendants: Vec::new(),
        });
        assert!(text.contains("0 ancestors, 0 descendants."));
        assert!(text.contains("stands alone"));
        assert!(text.contains("kb_branch_from"));
        assert_navigable(&text);
    }

    #[test]
    fn a_diff_reports_cost_as_well_as_metrics() {
        let mut a = record("run_a", "baseline");
        a.duration = Duration::from_secs(240);
        let mut b = record("run_b", "wider");
        b.duration = Duration::from_secs(120);
        b.conclusion = Some(RunConclusion {
            cache_hit_ratio: Some(0.75),
            ..a.conclusion.clone().unwrap()
        });

        let text = diff(&a, &b, &move_from("run_a", "run_b"));
        assert!(text.contains("- lr: 0.01 → 0.05"), "{text}");
        assert!(text.contains("- val_f1: 0.81 → 0.87 (+0.06)"), "{text}");
        assert!(
            text.contains("- duration: 4m 00s → 2m 00s (0.50×)"),
            "{text}"
        );
        assert!(text.contains("- cache hits: 50% → 75%"), "{text}");
        assert!(text.contains("run_dir (run_a):"));
        assert!(text.contains("soma does not guess"));
        assert_navigable(&text);
    }

    #[test]
    fn a_diff_says_so_when_it_cannot_describe_the_move() {
        let a = record("run_a", "baseline");
        let b = record("run_b", "variant");
        let unspecified = DerivationMove {
            from: "run_a".into(),
            to: "run_b".into(),
            changes: vec![Change::Unspecified {
                reason: "no architecture recorded for parent run_a".into(),
            }],
            metric_delta: BTreeMap::new(),
            summary: String::new(),
        };
        let text = diff(&a, &b, &unspecified);
        assert!(text.contains("soma cannot describe this move"), "{text}");
        assert!(text.contains("no metric appears in both runs"), "{text}");
    }

    #[test]
    fn a_run_summary_reports_what_it_could_not_read() {
        let summary = RunSummary {
            run_id: "run_old".into(),
            run_dir: "/proj/.soma/runs/run_old".into(),
            name: "ancient".into(),
            kind: "train".into(),
            created_at: at(1),
            finished_at: None,
            duration_ms: Some(65_000),
            tags: Vec::new(),
            git: Default::default(),
            seeds: BTreeMap::new(),
            params: BTreeMap::new(),
            hypothesis: None,
            parent_run_id: None,
            architecture: None,
            pipeline_summary: String::new(),
            metrics: BTreeMap::from([("f1".to_string(), 0.5)]),
            conclusion: RunConclusion {
                headline: "completed in 1m 05s · f1=0.5".into(),
                outcome: Some(RunOutcome::Completed),
                trials: Some(TrialSummary {
                    total: 4,
                    completed: 3,
                    pruned: 1,
                    objective: Some("f1".into()),
                    best_value: Some(0.5),
                    ..TrialSummary::default()
                }),
                warnings: vec!["graph.json is unreadable: unexpected EOF".into()],
                ..RunConclusion::default()
            },
        };
        let text = summarize_run(&summary);
        assert!(text.contains("# run_old — ancient"), "{text}");
        assert!(text.contains("duration: 1m 05s"));
        assert!(text.contains("- f1: 0.5"));
        assert!(text.contains("4 total, 3 completed, 1 pruned, 0 failed"));
        assert!(text.contains("- best f1 = 0.5"));
        assert!(text.contains("## What could not be read"));
        assert!(text.contains("graph.json is unreadable"));
        assert!(text.contains("run_dir: /proj/.soma/runs/run_old"));
        assert_navigable(&text);
    }

    #[test]
    fn stats_report_coverage_honestly() {
        let mut with_everything = record("run_a", "complete");
        with_everything.parent = Some("run_0".into());
        with_everything.hypothesis = Some("wider helps".into());
        with_everything.architecture = Some(Default::default());
        let mut bare = ExperimentRecord::new("run_b", "bare");
        bare.timestamp = at(25);

        let lines = vec![ResearchLine {
            name: "mos".into(),
            experiments: vec!["run_a".into()],
            trend: Trend::Improving,
            best_metric_value: Some(0.87),
            best_metric_name: Some("val_f1".into()),
        }];
        let text = stats(
            &[with_everything, bare],
            &lines,
            Some("/proj/.soma/experiments.jsonl"),
        );

        assert!(text.contains("experiments: 2"));
        assert!(text.contains("span: 2026-07-20 → 2026-07-25"));
        assert!(text.contains("- with a conclusion: 1/2 (50%)"), "{text}");
        assert!(text.contains("- with a parent (in a lineage): 1/2 (50%)"));
        assert!(text.contains("- with an architecture fingerprint: 1/2 (50%)"));
        assert!(text.contains("mos — 1 experiment, improving, best val_f1=0.87"));
        assert_navigable(&text);
    }

    #[test]
    fn an_empty_pool_says_how_to_fill_it() {
        let text = stats(&[], &[], Some("/proj/.soma/experiments.jsonl"));
        assert!(text.contains("empty"));
        assert!(text.contains("track_run"));
        assert!(text.contains("soma kb reindex"));
        assert_navigable(&text);
    }

    #[test]
    fn a_pool_with_no_lineage_at_all_is_told_why() {
        let text = stats(&[record("a", "one"), record("b", "two")], &[], None);
        assert!(text.contains("Nothing in this pool has a parent"), "{text}");
        assert!(text.contains(".soma/HEAD"));
    }

    #[test]
    fn confirmations_point_at_the_next_move() {
        let target = record("run_a", "baseline");
        let text = conclusion_recorded("amend_1", &target);
        assert!(text.contains("append-only"));
        assert_navigable(&text);

        let text = branched("run_a", Some(&target));
        assert!(text.starts_with("HEAD → run_a"));
        assert!(text.contains("sibling branch, it does not move history"));
        assert_navigable(&text);

        // Branching to a run the journal has never seen still explains
        // itself rather than rendering a blank.
        let text = branched("run_unknown", None);
        assert!(text.starts_with("HEAD → run_unknown"));
        assert_navigable(&text);
    }

    #[test]
    fn long_metric_and_param_lists_are_capped() {
        let mut wide = record("run_w", "many");
        for i in 0..12 {
            wide.metrics.insert(format!("m{i:02}"), i as f64);
            wide.params.insert(format!("p{i:02}"), serde_json::json!(i));
        }
        let hits = rank(&[wide], &RetrievalQuery::new("many", at(21)));
        let text = find_similar(&hits, "many");
        assert!(text.contains("+7 more"), "metrics capped: {text}");
        assert!(text.contains("+6 more"), "params capped: {text}");
    }
}
