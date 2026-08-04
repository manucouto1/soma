//! Tool definitions and dispatch for the Soma MCP server.
//!
//! Descriptions here are the only documentation a model gets, so they
//! say what a tool actually does — including when that is "not much".

pub mod knowledge;

use crate::context::SomaContext;
use crate::protocol::{ToolCallResult, ToolDefinition};
use serde_json::json;

/// Register all available tools.
pub fn all_tools() -> Vec<ToolDefinition> {
    vec![
        // ── Code tools ──
        ToolDefinition {
            name: "list_filters".into(),
            description: "List available filter source files in the project directory.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Project directory path (optional, uses configured default)" }
                }
            }),
        },
        ToolDefinition {
            name: "read_filter_source".into(),
            description: "Read the source code of a filter file.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "file_path": { "type": "string", "description": "Path to the filter source file" }
                },
                "required": ["file_path"]
            }),
        },
        ToolDefinition {
            name: "write_filter_source".into(),
            description: "Write or update filter source code. Creates a backup before writing."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "file_path": { "type": "string", "description": "Path to the filter source file" },
                    "content": { "type": "string", "description": "New file content" }
                },
                "required": ["file_path", "content"]
            }),
        },
        // ── Execution tools ──
        ToolDefinition {
            name: "run_pipeline".into(),
            description: "NOT IMPLEMENTED — echoes its arguments back and executes nothing. \
                 Running a pipeline needs the user's filter code loaded (Python via PyO3 or a \
                 Rust dylib), which this server cannot do. To actually run something, ask the \
                 user to run it from Python; then read the result with kb_summarize_run."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "pipeline_config": { "type": "object", "description": "Pipeline configuration (filter names and params)" },
                    "input_data": { "type": "array", "items": { "type": "number" }, "description": "Input data as array of numbers" }
                },
                "required": ["pipeline_config", "input_data"]
            }),
        },
        ToolDefinition {
            name: "run_study".into(),
            description: "NOT IMPLEMENTED — echoes its arguments back and runs no trials. \
                 Same reason as run_pipeline: the objective function is user code this server \
                 cannot load. Ask the user to run the study from Python; its results land in \
                 the experiment pool automatically and kb_find_similar will find them."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "Study name" },
                    "search_space": { "type": "array", "description": "Search space dimensions" },
                    "strategy": { "type": "string", "enum": ["grid", "random", "bayesian"] },
                    "n_trials": { "type": "integer", "description": "Number of trials" },
                    "objectives": { "type": "array", "description": "Optimization objectives [(metric, direction)]" }
                },
                "required": ["name", "search_space", "strategy", "n_trials", "objectives"]
            }),
        },
        // ── Knowledge tools ──
        ToolDefinition {
            name: "record_experiment".into(),
            description: "Record an experiment by hand. Runs started with graph.track_run() or \
                 study.run() record themselves — with a conclusion, an architecture fingerprint \
                 and a lineage — so use this only for work soma did not execute. To add a \
                 finding to an existing experiment, use kb_record_conclusion instead."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "id": { "type": "string" },
                    "name": { "type": "string" },
                    "hypothesis": { "type": "string", "description": "What you expected, and why" },
                    "research_line": { "type": "string", "description": "Groups related experiments; inherited from the parent when one is given" },
                    "pipeline_summary": { "type": "string", "description": "One line describing the topology, e.g. 'scaler → encoder → head'" },
                    "params": { "type": "object", "description": "Hyperparameters; a later variant diffs against these" },
                    "metrics": { "type": "object", "description": "Final numeric results" },
                    "parent": { "type": "string", "description": "Experiment this one was derived from. Setting it computes the move between them." },
                    "objective": { "type": "string", "description": "Metric being optimized, so improvement can be judged" },
                    "run_dir": { "type": "string", "description": "Directory holding raw artifacts, if any" },
                    "tags": { "type": "array", "items": { "type": "string" } },
                    "notes": { "type": "string", "description": "What you concluded" }
                },
                "required": ["id", "name"]
            }),
        },
        // ── Experiment pool ──
        ToolDefinition {
            name: "kb_find_similar".into(),
            description: "Find past experiments bearing on the problem at hand — the first \
                 thing to call before designing anything. Ranks by text relevance, \
                 architectural resemblance, recency and importance. Dead ends rank too: not \
                 repeating a failure saves as much time as repeating a success. Each hit \
                 carries its conclusion, the move that produced it, and a run_dir you can read \
                 directly."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "What you are trying to do, in words. Names, filters, metrics and symptoms all match." },
                    "like_run": { "type": "string", "description": "Experiment id whose architecture to match. Combine with query, or use alone to find structurally similar work." },
                    "limit": { "type": "integer", "default": 5, "description": "1-50" },
                    "research_line": { "type": "string", "description": "Restrict to one line" },
                    "tags": { "type": "array", "items": { "type": "string" }, "description": "Every listed tag must be present" },
                    "half_life_days": { "type": "number", "default": 30, "description": "Raise it to weight old work more heavily" }
                }
            }),
        },
        ToolDefinition {
            name: "kb_lineage".into(),
            description: "The experiment tree around one run: ancestors above, descendants \
                 below, and on every edge the change that produced the child from its parent \
                 plus what it did to the metrics. This is how you see what has already been \
                 tried from a given starting point."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "id": { "type": "string", "description": "Experiment or run id" }
                },
                "required": ["id"]
            }),
        },
        ToolDefinition {
            name: "kb_diff".into(),
            description: "Compare two experiments: what differs in architecture, parameters \
                 and code, what each metric did, and what it cost (wall time, cache hits). \
                 They need not be related — this works on any two ids."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "a": { "type": "string", "description": "Baseline experiment id" },
                    "b": { "type": "string", "description": "Variant experiment id" }
                },
                "required": ["a", "b"]
            }),
        },
        ToolDefinition {
            name: "kb_record_conclusion".into(),
            description: "Retain what you learned about a run: why it worked, why it did not, \
                 what to try next. Appended as a separate amendment — the original record is \
                 never rewritten — and indexed so a later kb_find_similar surfaces it. Worth \
                 doing for failures especially."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "run_id": { "type": "string", "description": "Experiment being annotated" },
                    "notes": { "type": "string", "description": "What you concluded" },
                    "hypothesis": { "type": "string", "description": "The hypothesis this run turned out to be testing" },
                    "tags": { "type": "array", "items": { "type": "string" } }
                },
                "required": ["run_id", "notes"]
            }),
        },
        ToolDefinition {
            name: "kb_branch_from".into(),
            description: "Point .soma/HEAD at an existing run so the NEXT run records itself \
                 as its child. Use it to go back and try a different variation instead of \
                 continuing from the last thing that happened to run. Creates a sibling \
                 branch; it never moves or rewrites existing history."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "run_id": { "type": "string", "description": "Run to branch from. Must exist under .soma/runs/." }
                },
                "required": ["run_id"]
            }),
        },
        ToolDefinition {
            name: "kb_summarize_run".into(),
            description: "Read a run directory and summarize it: outcome, metrics, slowest \
                 node, cache effectiveness, health flags, trials. Works on runs recorded \
                 before the experiment pool existed and on runs that crashed before writing a \
                 journal line, and reports what it could not read rather than staying silent."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "run_id": { "type": "string", "description": "Run id under .soma/runs/, or a path to a run directory" }
                },
                "required": ["run_id"]
            }),
        },
        ToolDefinition {
            name: "kb_stats".into(),
            description: "How much this project has recorded and how usable it is: totals, \
                 date span, research lines, and honest coverage — what fraction of records \
                 carry a conclusion, a lineage, an architecture. Call it first when you do not \
                 know whether the pool is worth querying."
                .into(),
            input_schema: json!({ "type": "object", "properties": {} }),
        },
        ToolDefinition {
            name: "query_knowledge_base".into(),
            description: "Search experiments in the knowledge base by text query.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Search query" },
                    "max_results": { "type": "integer", "default": 10 }
                },
                "required": ["query"]
            }),
        },
        ToolDefinition {
            name: "get_trajectory".into(),
            description: "Get the metric trajectory for a research line.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "research_line": { "type": "string" },
                    "metric": { "type": "string" }
                },
                "required": ["research_line", "metric"]
            }),
        },
        ToolDefinition {
            name: "get_change_points".into(),
            description: "Detect significant changes in experiment metrics.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "research_line": { "type": "string" },
                    "metric": { "type": "string" },
                    "threshold": { "type": "number", "default": 0.05 }
                },
                "required": ["research_line", "metric"]
            }),
        },
        ToolDefinition {
            name: "list_research_lines".into(),
            description: "List all research lines with trend analysis.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {}
            }),
        },
        ToolDefinition {
            name: "promising_lines".into(),
            description: "Get research lines that are improving.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "metric": { "type": "string", "description": "Metric to evaluate" }
                },
                "required": ["metric"]
            }),
        },
        // ── Project tools ──
        ToolDefinition {
            name: "create_research_line".into(),
            description: "Create a new research line for tracking experiments.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string" },
                    "description": { "type": "string" }
                },
                "required": ["name"]
            }),
        },
        ToolDefinition {
            name: "generate_report".into(),
            description: "Generate a markdown report for a research line.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "research_line": { "type": "string" }
                },
                "required": ["research_line"]
            }),
        },
    ]
}

/// Dispatch a tool call to the appropriate handler.
///
/// Every knowledge read is preceded by a refresh: this server outlives
/// the training runs it is asked about, and a stale snapshot silently
/// answers "no such experiment" for a run that finished five minutes
/// ago in another terminal.
pub fn dispatch(
    ctx: &mut SomaContext,
    tool_name: &str,
    params: &serde_json::Value,
) -> ToolCallResult {
    if reads_knowledge(tool_name) {
        ctx.refresh_kb();
    }
    match tool_name {
        // Experiment pool
        "kb_find_similar" => knowledge::find_similar(ctx, params),
        "kb_lineage" => knowledge::lineage(ctx, params),
        "kb_diff" => knowledge::diff(ctx, params),
        "kb_record_conclusion" => knowledge::record_conclusion(ctx, params),
        "kb_branch_from" => knowledge::branch_from(ctx, params),
        "kb_summarize_run" => knowledge::summarize_run(ctx, params),
        "kb_stats" => knowledge::stats(ctx, params),
        // Code tools
        "list_filters" => ctx.list_filters(params),
        "read_filter_source" => ctx.read_filter_source(params),
        "write_filter_source" => ctx.write_filter_source(params),
        // Execution tools
        "run_pipeline" => ctx.run_pipeline(params),
        "run_study" => ctx.run_study(params),
        // Knowledge tools
        "record_experiment" => ctx.record_experiment(params),
        "query_knowledge_base" => ctx.query_knowledge_base(params),
        "get_trajectory" => ctx.get_trajectory(params),
        "get_change_points" => ctx.get_change_points(params),
        "list_research_lines" => ctx.list_research_lines(params),
        "promising_lines" => ctx.promising_lines(params),
        // Project tools
        "create_research_line" => ctx.create_research_line(params),
        "generate_report" => ctx.generate_report(params),
        _ => ToolCallResult::error(format!("Unknown tool: {tool_name}")),
    }
}

/// Whether a tool reads the experiment journal, and therefore needs to
/// see what other processes have appended.
fn reads_knowledge(tool_name: &str) -> bool {
    tool_name.starts_with("kb_")
        || matches!(
            tool_name,
            "query_knowledge_base"
                | "get_trajectory"
                | "get_change_points"
                | "list_research_lines"
                | "promising_lines"
                | "generate_report"
                | "record_experiment"
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_tool_is_defined_exactly_once_and_dispatches() {
        let tools = all_tools();
        let mut names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        names.sort();
        let unique = {
            let mut u = names.clone();
            u.dedup();
            u
        };
        assert_eq!(names, unique, "duplicate tool name");

        let dir = tempfile::tempdir().unwrap();
        for tool in &tools {
            let mut ctx = SomaContext::with_env_override(dir.path(), None);
            let result = dispatch(&mut ctx, &tool.name, &serde_json::json!({}));
            // No arguments: a tool may refuse, but never as "unknown".
            let text = result.content_text();
            assert!(
                !text.contains("Unknown tool"),
                "{} is defined but not dispatched",
                tool.name
            );
        }
        assert!(
            dispatch(
                &mut SomaContext::with_env_override(dir.path(), None),
                "kb_nope",
                &serde_json::json!({})
            )
            .content_text()
            .contains("Unknown tool")
        );
    }

    #[test]
    fn the_pool_tools_are_all_registered() {
        let names: Vec<String> = all_tools().into_iter().map(|t| t.name).collect();
        for expected in [
            "kb_find_similar",
            "kb_lineage",
            "kb_diff",
            "kb_record_conclusion",
            "kb_branch_from",
            "kb_summarize_run",
            "kb_stats",
        ] {
            assert!(names.contains(&expected.to_string()), "missing {expected}");
        }
        assert_eq!(names.len(), 20, "13 original tools + 7 pool tools");
    }

    #[test]
    fn the_unimplemented_tools_say_so_up_front() {
        // These used to promise execution and return a stub, which
        // actively misleads a model into thinking it ran something.
        for name in ["run_pipeline", "run_study"] {
            let tool = all_tools().into_iter().find(|t| t.name == name).unwrap();
            assert!(
                tool.description.starts_with("NOT IMPLEMENTED"),
                "{name}: {}",
                tool.description
            );
        }
    }

    #[test]
    fn knowledge_reads_are_refreshed_and_code_tools_are_not() {
        for name in ["kb_find_similar", "query_knowledge_base", "generate_report"] {
            assert!(reads_knowledge(name), "{name} should refresh");
        }
        for name in ["list_filters", "read_filter_source", "run_pipeline"] {
            assert!(!reads_knowledge(name), "{name} should not refresh");
        }
    }
}
