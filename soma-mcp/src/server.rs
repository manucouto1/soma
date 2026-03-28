//! MCP stdio server: reads JSON-RPC from stdin, writes to stdout.

use crate::context::SomaContext;
use crate::protocol::*;
use crate::tools;
use serde_json::json;
use std::io::{self, BufRead, Write};

/// Run the MCP server on stdin/stdout.
pub fn run_stdio(mut ctx: SomaContext) {
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };

        if line.trim().is_empty() {
            continue;
        }

        let request: JsonRpcRequest = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                let resp = JsonRpcResponse::error(
                    serde_json::Value::Null,
                    -32700,
                    format!("Parse error: {e}"),
                );
                write_response(&mut stdout, &resp);
                continue;
            }
        };

        let response = handle_request(&mut ctx, &request);
        write_response(&mut stdout, &response);
    }
}

fn handle_request(ctx: &mut SomaContext, req: &JsonRpcRequest) -> JsonRpcResponse {
    match req.method.as_str() {
        "initialize" => {
            let result = InitializeResult {
                protocol_version: "2024-11-05".to_string(),
                capabilities: ServerCapabilities {
                    tools: ToolsCapability {
                        list_changed: false,
                    },
                },
                server_info: ServerInfo {
                    name: "soma-mcp".to_string(),
                    version: env!("CARGO_PKG_VERSION").to_string(),
                },
            };
            JsonRpcResponse::success(req.id.clone(), serde_json::to_value(result).unwrap())
        }

        "notifications/initialized" => {
            // Client acknowledgment, no response needed for notifications
            // but we return success anyway for non-notification calls
            JsonRpcResponse::success(req.id.clone(), json!({}))
        }

        "tools/list" => {
            let tool_list = tools::all_tools();
            JsonRpcResponse::success(req.id.clone(), json!({ "tools": tool_list }))
        }

        "tools/call" => {
            let tool_name = req
                .params
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let arguments = req.params.get("arguments").cloned().unwrap_or(json!({}));

            let result = tools::dispatch(ctx, tool_name, &arguments);
            JsonRpcResponse::success(req.id.clone(), serde_json::to_value(result).unwrap())
        }

        _ => JsonRpcResponse::error(
            req.id.clone(),
            METHOD_NOT_FOUND,
            format!("Method not found: {}", req.method),
        ),
    }
}

fn write_response(out: &mut impl Write, resp: &JsonRpcResponse) {
    if let Ok(json) = serde_json::to_string(resp) {
        let _ = writeln!(out, "{json}");
        let _ = out.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn test_ctx() -> SomaContext {
        SomaContext::new(env::temp_dir())
    }

    fn call(ctx: &mut SomaContext, method: &str, params: serde_json::Value) -> JsonRpcResponse {
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: json!(1),
            method: method.into(),
            params,
        };
        handle_request(ctx, &req)
    }

    #[test]
    fn initialize() {
        let mut ctx = test_ctx();
        let resp = call(&mut ctx, "initialize", json!({}));
        assert!(resp.result.is_some());
        let result = resp.result.unwrap();
        assert_eq!(result["serverInfo"]["name"], "soma-mcp");
    }

    #[test]
    fn tools_list() {
        let mut ctx = test_ctx();
        let resp = call(&mut ctx, "tools/list", json!({}));
        let result = resp.result.unwrap();
        let tools = result["tools"].as_array().unwrap();
        assert!(tools.len() >= 10);

        let names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
        assert!(names.contains(&"record_experiment"));
        assert!(names.contains(&"query_knowledge_base"));
        assert!(names.contains(&"generate_report"));
        assert!(names.contains(&"list_filters"));
    }

    #[test]
    fn record_and_query_experiment() {
        let mut ctx = test_ctx();

        // Record
        let resp = call(
            &mut ctx,
            "tools/call",
            json!({
                "name": "record_experiment",
                "arguments": {
                    "id": "exp_001",
                    "name": "SVM test",
                    "hypothesis": "SVM works well on iris",
                    "research_line": "svm_exploration",
                    "metrics": { "f1": 0.85 },
                    "tags": ["svm", "classification"]
                }
            }),
        );
        assert!(resp.error.is_none());

        // Query
        let resp = call(
            &mut ctx,
            "tools/call",
            json!({
                "name": "query_knowledge_base",
                "arguments": { "query": "SVM", "max_results": 5 }
            }),
        );
        let result = resp.result.unwrap();
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("exp_001"));
    }

    #[test]
    fn research_line_workflow() {
        let mut ctx = test_ctx();

        // Record experiments (line is created implicitly)
        for (id, f1) in [("e1", 0.7), ("e2", 0.8), ("e3", 0.9)] {
            call(
                &mut ctx,
                "tools/call",
                json!({
                    "name": "record_experiment",
                    "arguments": {
                        "id": id,
                        "name": format!("Experiment {id}"),
                        "research_line": "norm_study",
                        "metrics": { "f1": f1 }
                    }
                }),
            );
        }

        // Get trajectory
        let resp = call(
            &mut ctx,
            "tools/call",
            json!({
                "name": "get_trajectory",
                "arguments": { "research_line": "norm_study", "metric": "f1" }
            }),
        );
        let text = resp.result.unwrap()["content"][0]["text"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(text.contains("0.7"));
        assert!(text.contains("0.9"));

        // List lines
        let resp = call(
            &mut ctx,
            "tools/call",
            json!({"name": "list_research_lines", "arguments": {}}),
        );
        let text = resp.result.unwrap()["content"][0]["text"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(text.contains("norm_study"));

        // Generate report
        let resp = call(
            &mut ctx,
            "tools/call",
            json!({"name": "generate_report", "arguments": {"research_line": "norm_study"}}),
        );
        let text = resp.result.unwrap()["content"][0]["text"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(text.contains("# Research Report: norm_study"));
        assert!(text.contains("Trajectory"));
    }

    #[test]
    fn unknown_method() {
        let mut ctx = test_ctx();
        let resp = call(&mut ctx, "unknown/method", json!({}));
        assert!(resp.error.is_some());
    }

    #[test]
    fn unknown_tool() {
        let mut ctx = test_ctx();
        let resp = call(
            &mut ctx,
            "tools/call",
            json!({"name": "nonexistent_tool", "arguments": {}}),
        );
        let result = resp.result.unwrap();
        assert_eq!(result["isError"], true);
    }
}
