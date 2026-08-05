//! Soma MCP server context: holds state and implements tool handlers.

use crate::protocol::ToolCallResult;
use serde_json::json;
use somatize_memory::{ExperimentRecord, FileKnowledgeBase, KnowledgeBase, MemoryKnowledgeBase};
use std::path::{Path, PathBuf};

/// Server-side state for the Soma MCP server.
pub struct SomaContext {
    /// Project directory (where user's filter code lives).
    pub project_dir: PathBuf,
    /// Knowledge base for experiment tracking.
    pub kb: Box<dyn KnowledgeBase>,
    /// Where the knowledge base is persisted, when it is.
    kb_path: Option<PathBuf>,
}

impl SomaContext {
    /// Create a context with a persistent knowledge base when one is
    /// available: `SOMA_KB_PATH` if set, else the project's
    /// `.soma/experiments.jsonl` when a `.soma/` directory exists.
    /// Falls back to the in-memory KB otherwise (records are lost on
    /// server exit).
    pub fn new(project_dir: impl Into<PathBuf>) -> Self {
        Self::with_env_override(project_dir, std::env::var("SOMA_KB_PATH").ok())
    }

    /// Deterministic constructor: `env_override` plays the role of the
    /// `SOMA_KB_PATH` environment variable. Used by `new()` and by
    /// tests, which must never depend on (or leak through) the real
    /// process environment.
    pub fn with_env_override(
        project_dir: impl Into<PathBuf>,
        env_override: Option<String>,
    ) -> Self {
        let project_dir = project_dir.into();
        let resolved = Self::kb_path(&project_dir, env_override);
        let mut kb_path = None;
        let kb: Box<dyn KnowledgeBase> = match &resolved {
            Some(path) => match FileKnowledgeBase::open(path) {
                Ok(kb) => {
                    kb_path = resolved.clone();
                    Box::new(kb)
                }
                Err(e) => {
                    eprintln!(
                        "soma-mcp: failed to open knowledge base at {} ({e}); using in-memory",
                        path.display()
                    );
                    Box::new(MemoryKnowledgeBase::new())
                }
            },
            None => Box::new(MemoryKnowledgeBase::new()),
        };
        Self {
            project_dir,
            kb,
            kb_path,
        }
    }

    /// Pick up experiments another process appended since the last
    /// call. An MCP server outlives many training runs; without this it
    /// answers every question from the snapshot it loaded at startup.
    pub fn refresh_kb(&mut self) {
        if let Err(e) = self.kb.refresh() {
            eprintln!("soma-mcp: could not refresh the knowledge base: {e}");
        }
    }

    /// Human-readable location of the journal, when it has one.
    pub fn kb_location(&self) -> Option<String> {
        self.kb_path.as_ref().map(|p| p.display().to_string())
    }

    /// The tracking root this project's runs live under (`.soma`).
    pub fn tracking_root(&self) -> PathBuf {
        self.project_dir.join(".soma")
    }

    /// Where the persistent KB lives, if anywhere: the explicit
    /// override wins; otherwise a `.soma/` DIRECTORY in the project
    /// enables `.soma/experiments.jsonl`.
    fn kb_path(project_dir: &Path, env_override: Option<String>) -> Option<PathBuf> {
        if let Some(path) = env_override.filter(|p| !p.is_empty()) {
            return Some(PathBuf::from(path));
        }
        let soma_dir = project_dir.join(".soma");
        soma_dir
            .is_dir()
            .then(|| soma_dir.join("experiments.jsonl"))
    }

    // ═══════════════════════════════════════
    // Code tools
    // ═══════════════════════════════════════

    /// `list_filters`: every `.py`/`.rs` file under `path` (default:
    /// the project directory), recursing but skipping hidden
    /// directories. Returns a JSON array of `{"path": ...}` objects.
    ///
    /// Like every handler below, it takes the raw `arguments` object
    /// from `tools/call` and reports a missing or bad argument as a
    /// [`ToolCallResult::error`] — the model reads the message and can
    /// retry, which a protocol-level error would not allow.
    pub fn list_filters(&self, params: &serde_json::Value) -> ToolCallResult {
        let dir = params
            .get("path")
            .and_then(|v| v.as_str())
            .map(PathBuf::from)
            .unwrap_or_else(|| self.project_dir.clone());

        match find_filter_files(&dir) {
            Ok(files) => {
                let result: Vec<serde_json::Value> = files
                    .iter()
                    .map(|f| json!({ "path": f.to_string_lossy() }))
                    .collect();
                ToolCallResult::text(serde_json::to_string_pretty(&result).unwrap_or_default())
            }
            Err(e) => ToolCallResult::error(format!("Failed to list filters: {e}")),
        }
    }

    /// `read_filter_source`: the contents of `file_path`, resolved
    /// against the project directory when relative.
    pub fn read_filter_source(&self, params: &serde_json::Value) -> ToolCallResult {
        let Some(path) = params.get("file_path").and_then(|v| v.as_str()) else {
            return ToolCallResult::error("Missing required parameter: file_path");
        };

        let full_path = self.resolve_path(path);
        match std::fs::read_to_string(&full_path) {
            Ok(content) => ToolCallResult::text(content),
            Err(e) => ToolCallResult::error(format!("Failed to read {}: {e}", full_path.display())),
        }
    }

    /// `write_filter_source`: write `content` to `file_path`, creating
    /// parent directories as needed. An existing file is copied to a
    /// `.bak` sibling first — a model editing code deserves one level
    /// of undo.
    pub fn write_filter_source(&self, params: &serde_json::Value) -> ToolCallResult {
        let Some(path) = params.get("file_path").and_then(|v| v.as_str()) else {
            return ToolCallResult::error("Missing required parameter: file_path");
        };
        let Some(content) = params.get("content").and_then(|v| v.as_str()) else {
            return ToolCallResult::error("Missing required parameter: content");
        };

        let full_path = self.resolve_path(path);

        // Create backup if file exists
        if full_path.exists() {
            let backup = full_path.with_extension("bak");
            if let Err(e) = std::fs::copy(&full_path, &backup) {
                return ToolCallResult::error(format!("Failed to create backup: {e}"));
            }
        }

        // Ensure parent directory exists
        if let Some(parent) = full_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        match std::fs::write(&full_path, content) {
            Ok(()) => ToolCallResult::text(format!(
                "Written {} bytes to {}",
                content.len(),
                full_path.display()
            )),
            Err(e) => ToolCallResult::error(format!("Failed to write: {e}")),
        }
    }

    // ═══════════════════════════════════════
    // Execution tools
    // ═══════════════════════════════════════

    /// `run_pipeline`: declared but NOT implemented. Executing a
    /// pipeline means loading user filter code (Python via PyO3, or a
    /// Rust dylib), which this server cannot do; it answers with a
    /// `"status": "stub"` payload that says so, and the tool's
    /// description says so too — better than a stub that reads like
    /// success.
    pub fn run_pipeline(&self, params: &serde_json::Value) -> ToolCallResult {
        // For now, return a stub. Full execution requires loading user filters
        // dynamically (Python via PyO3 or Rust dylib).
        let config = params.get("pipeline_config").cloned().unwrap_or(json!({}));
        let input = params.get("input_data").cloned().unwrap_or(json!([]));

        ToolCallResult::text(json!({
            "status": "stub",
            "message": "Pipeline execution via MCP requires dynamic filter loading. Use soma-python for now.",
            "config": config,
            "input_shape": input.as_array().map(|a| a.len()).unwrap_or(0)
        }).to_string())
    }

    /// `run_study`: declared but NOT implemented, for the same reason
    /// as [`run_pipeline`](Self::run_pipeline) — a study runs user
    /// code. The stub echoes the study name and trial count so the
    /// model sees its arguments were understood, not executed.
    pub fn run_study(&self, params: &serde_json::Value) -> ToolCallResult {
        let name = params
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("unnamed");
        let n_trials = params
            .get("n_trials")
            .and_then(|v| v.as_u64())
            .unwrap_or(10);

        ToolCallResult::text(json!({
            "status": "stub",
            "message": "Study execution via MCP requires dynamic filter loading. Use soma-python for now.",
            "study_name": name,
            "n_trials": n_trials
        }).to_string())
    }

    // ═══════════════════════════════════════
    // Knowledge tools
    // ═══════════════════════════════════════

    /// `record_experiment`: append an experiment to the knowledge base.
    /// `id` and `name` are required; hypothesis, research line,
    /// pipeline summary, parent, notes, tags, metrics and params are
    /// taken when present and silently skipped when absent or of the
    /// wrong JSON type — a partial record beats no record.
    pub fn record_experiment(&mut self, params: &serde_json::Value) -> ToolCallResult {
        let Some(id) = params.get("id").and_then(|v| v.as_str()) else {
            return ToolCallResult::error("Missing required parameter: id");
        };
        let Some(name) = params.get("name").and_then(|v| v.as_str()) else {
            return ToolCallResult::error("Missing required parameter: name");
        };

        let mut record = ExperimentRecord::new(id, name);

        if let Some(h) = params.get("hypothesis").and_then(|v| v.as_str()) {
            record = record.with_hypothesis(h);
        }
        if let Some(l) = params.get("research_line").and_then(|v| v.as_str()) {
            record = record.with_research_line(l);
        }
        if let Some(p) = params.get("pipeline_summary").and_then(|v| v.as_str()) {
            record = record.with_pipeline(p);
        }
        if let Some(parent) = params.get("parent").and_then(|v| v.as_str()) {
            record = record.with_parent(parent);
        }
        if let Some(notes) = params.get("notes").and_then(|v| v.as_str()) {
            record = record.with_notes(notes);
        }
        if let Some(tags) = params.get("tags").and_then(|v| v.as_array()) {
            let tags: Vec<String> = tags
                .iter()
                .filter_map(|t| t.as_str().map(String::from))
                .collect();
            record = record.with_tags(tags);
        }
        if let Some(metrics) = params.get("metrics").and_then(|v| v.as_object()) {
            let m: std::collections::BTreeMap<String, f64> = metrics
                .iter()
                .filter_map(|(k, v)| v.as_f64().map(|val| (k.clone(), val)))
                .collect();
            record = record.with_metrics(m);
        }
        if let Some(p) = params.get("params").and_then(|v| v.as_object()) {
            let params_map: std::collections::BTreeMap<String, serde_json::Value> =
                p.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
            record = record.with_params(params_map);
        }

        match self.kb.record(record) {
            Ok(()) => ToolCallResult::text(format!("Experiment '{id}' recorded successfully")),
            Err(e) => ToolCallResult::error(format!("Failed to record experiment: {e}")),
        }
    }

    /// `query_knowledge_base`: free-text search over recorded
    /// experiments (`query`, plus `max_results`, default 10). Returns a
    /// JSON array of experiment summaries — id, name, hypothesis, line,
    /// metrics, tags — enough to decide which one to ask more about.
    pub fn query_knowledge_base(&self, params: &serde_json::Value) -> ToolCallResult {
        let Some(query) = params.get("query").and_then(|v| v.as_str()) else {
            return ToolCallResult::error("Missing required parameter: query");
        };
        let max = params
            .get("max_results")
            .and_then(|v| v.as_u64())
            .unwrap_or(10) as usize;

        match self.kb.search(query, max) {
            Ok(results) => {
                let items: Vec<serde_json::Value> = results
                    .iter()
                    .map(|e| {
                        json!({
                            "id": e.id,
                            "name": e.name,
                            "hypothesis": e.hypothesis,
                            "research_line": e.research_line,
                            "metrics": e.metrics,
                            "tags": e.tags,
                        })
                    })
                    .collect();
                ToolCallResult::text(serde_json::to_string_pretty(&items).unwrap_or_default())
            }
            Err(e) => ToolCallResult::error(format!("Query failed: {e}")),
        }
    }

    /// `get_trajectory`: the values of `metric` across the experiments
    /// of `research_line`, in recording order — how the line has been
    /// moving, as `{experiment_id, value}` pairs.
    pub fn get_trajectory(&self, params: &serde_json::Value) -> ToolCallResult {
        let Some(line) = params.get("research_line").and_then(|v| v.as_str()) else {
            return ToolCallResult::error("Missing: research_line");
        };
        let Some(metric) = params.get("metric").and_then(|v| v.as_str()) else {
            return ToolCallResult::error("Missing: metric");
        };

        match self.kb.trajectory(line, metric) {
            Ok(traj) => {
                let items: Vec<serde_json::Value> = traj
                    .iter()
                    .map(|(id, val)| json!({"experiment_id": id, "value": val}))
                    .collect();
                ToolCallResult::text(serde_json::to_string_pretty(&items).unwrap_or_default())
            }
            Err(e) => ToolCallResult::error(format!("Failed: {e}")),
        }
    }

    /// `get_change_points`: the experiments where `metric` jumped by
    /// more than `threshold` (default 0.05) within `research_line` —
    /// the moments worth reading closely, with before/after values and
    /// a description of each.
    pub fn get_change_points(&self, params: &serde_json::Value) -> ToolCallResult {
        let Some(line) = params.get("research_line").and_then(|v| v.as_str()) else {
            return ToolCallResult::error("Missing: research_line");
        };
        let Some(metric) = params.get("metric").and_then(|v| v.as_str()) else {
            return ToolCallResult::error("Missing: metric");
        };
        let threshold = params
            .get("threshold")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.05);

        match self.kb.change_points(line, metric, threshold) {
            Ok(points) => {
                let items: Vec<serde_json::Value> = points
                    .iter()
                    .map(|cp| {
                        json!({
                            "experiment_id": cp.experiment_id,
                            "metric": cp.metric_name,
                            "before": cp.value_before,
                            "after": cp.value_after,
                            "description": cp.description,
                        })
                    })
                    .collect();
                ToolCallResult::text(serde_json::to_string_pretty(&items).unwrap_or_default())
            }
            Err(e) => ToolCallResult::error(format!("Failed: {e}")),
        }
    }

    /// `list_research_lines`: every research line in the pool, with its
    /// trend, experiment count and best metric so far. Takes no
    /// arguments.
    pub fn list_research_lines(&self, _params: &serde_json::Value) -> ToolCallResult {
        match self.kb.research_lines() {
            Ok(lines) => {
                let items: Vec<serde_json::Value> = lines
                    .iter()
                    .map(|l| {
                        json!({
                            "name": l.name,
                            "trend": l.trend.to_string(),
                            "experiments": l.experiments.len(),
                            "best_metric": l.best_metric_value,
                            "best_metric_name": l.best_metric_name,
                        })
                    })
                    .collect();
                ToolCallResult::text(serde_json::to_string_pretty(&items).unwrap_or_default())
            }
            Err(e) => ToolCallResult::error(format!("Failed: {e}")),
        }
    }

    /// `promising_lines`: the research lines with an improving trend,
    /// or whose best result is on `metric` — where the next experiment
    /// is most likely to pay off.
    pub fn promising_lines(&self, params: &serde_json::Value) -> ToolCallResult {
        let Some(metric) = params.get("metric").and_then(|v| v.as_str()) else {
            return ToolCallResult::error("Missing: metric");
        };

        match self.kb.promising_lines(metric) {
            Ok(lines) => {
                let items: Vec<serde_json::Value> = lines
                    .iter()
                    .map(|l| json!({"name": l.name, "trend": l.trend.to_string(), "best": l.best_metric_value}))
                    .collect();
                ToolCallResult::text(serde_json::to_string_pretty(&items).unwrap_or_default())
            }
            Err(e) => ToolCallResult::error(format!("Failed: {e}")),
        }
    }

    // ═══════════════════════════════════════
    // Project tools
    // ═══════════════════════════════════════

    /// `create_research_line`: start a named line by recording a
    /// `<name>_init` marker experiment carrying the description. Lines
    /// have no existence of their own in the pool — they are the set of
    /// experiments tagged with them, so creating one means recording
    /// one.
    pub fn create_research_line(&mut self, params: &serde_json::Value) -> ToolCallResult {
        let Some(name) = params.get("name").and_then(|v| v.as_str()) else {
            return ToolCallResult::error("Missing: name");
        };
        let description = params
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        // Record a marker experiment for the line
        let record =
            ExperimentRecord::new(format!("{name}_init"), format!("Research line: {name}"))
                .with_research_line(name)
                .with_notes(format!("Research line created. {description}"));

        match self.kb.record(record) {
            Ok(()) => ToolCallResult::text(format!("Research line '{name}' created")),
            Err(e) => ToolCallResult::error(format!("Failed: {e}")),
        }
    }

    /// `generate_report`: a Markdown report for `research_line` — the
    /// experiment table, the trajectory of its first metric, the
    /// significant changes, and the line's trend. Markdown because the
    /// consumer is a model (or a human pasted the output): structure
    /// survives, no renderer needed.
    pub fn generate_report(&self, params: &serde_json::Value) -> ToolCallResult {
        let Some(line) = params.get("research_line").and_then(|v| v.as_str()) else {
            return ToolCallResult::error("Missing: research_line");
        };

        let experiments = self.kb.experiments_in_line(line).unwrap_or_default();
        if experiments.is_empty() {
            return ToolCallResult::text(format!("# {line}\n\nNo experiments found."));
        }

        let mut report = format!("# Research Report: {line}\n\n");
        report.push_str(&format!("**Experiments**: {}\n\n", experiments.len()));

        // Summary table
        report.push_str("## Experiments\n\n");
        report.push_str("| ID | Hypothesis | Metrics | Notes |\n");
        report.push_str("|---|---|---|---|\n");
        for exp in &experiments {
            let metrics_str: String = exp
                .metrics
                .iter()
                .map(|(k, v)| format!("{k}={v:.4}"))
                .collect::<Vec<_>>()
                .join(", ");
            report.push_str(&format!(
                "| {} | {} | {} | {} |\n",
                exp.id,
                exp.hypothesis.as_deref().unwrap_or("-"),
                metrics_str,
                exp.notes.as_deref().unwrap_or("-"),
            ));
        }

        // Trajectory
        if let Some(first_metric) = experiments[0].metrics.keys().next() {
            let traj = self.kb.trajectory(line, first_metric).unwrap_or_default();
            if !traj.is_empty() {
                report.push_str(&format!("\n## Trajectory ({first_metric})\n\n"));
                for (id, val) in &traj {
                    report.push_str(&format!("- {id}: {val:.4}\n"));
                }
            }

            // Change points
            let cps = self
                .kb
                .change_points(line, first_metric, 0.05)
                .unwrap_or_default();
            if !cps.is_empty() {
                report.push_str("\n## Significant Changes\n\n");
                for cp in &cps {
                    report.push_str(&format!("- **{}**: {}\n", cp.experiment_id, cp.description));
                }
            }
        }

        // Research lines status
        if let Ok(lines) = self.kb.research_lines()
            && let Some(this_line) = lines.iter().find(|l| l.name == line)
        {
            report.push_str(&format!(
                "\n## Status\n\n- **Trend**: {}\n- **Best metric**: {:?}\n",
                this_line.trend, this_line.best_metric_value
            ));
        }

        ToolCallResult::text(report)
    }

    // ═══════════════════════════════════════
    // Helpers
    // ═══════════════════════════════════════

    fn resolve_path(&self, path: &str) -> PathBuf {
        let p = PathBuf::from(path);
        if p.is_absolute() {
            p
        } else {
            self.project_dir.join(p)
        }
    }
}

/// Find Python and Rust filter files in a directory.
fn find_filter_files(dir: &Path) -> Result<Vec<PathBuf>, std::io::Error> {
    let mut files = Vec::new();
    if !dir.exists() {
        return Ok(files);
    }
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file()
            && let Some(ext) = path.extension()
            && (ext == "py" || ext == "rs")
        {
            files.push(path);
        } else if path.is_dir()
            && !path
                .file_name()
                .is_some_and(|n| n.to_string_lossy().starts_with('.'))
        {
            files.extend(find_filter_files(&path)?);
        }
    }
    files.sort();
    Ok(files)
}
