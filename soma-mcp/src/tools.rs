//! Tool definitions and dispatch for the Soma MCP server.

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
            description: "Run a pipeline with given data. Returns output values.".into(),
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
            description: "Run a hyperparameter optimization study.".into(),
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
            description: "Record an experiment result in the knowledge base.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "id": { "type": "string" },
                    "name": { "type": "string" },
                    "hypothesis": { "type": "string" },
                    "research_line": { "type": "string" },
                    "pipeline_summary": { "type": "string" },
                    "params": { "type": "object" },
                    "metrics": { "type": "object" },
                    "parent": { "type": "string" },
                    "tags": { "type": "array", "items": { "type": "string" } },
                    "notes": { "type": "string" }
                },
                "required": ["id", "name"]
            }),
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
pub fn dispatch(
    ctx: &mut SomaContext,
    tool_name: &str,
    params: &serde_json::Value,
) -> ToolCallResult {
    match tool_name {
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
