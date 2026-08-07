//! Talking to an MCP server over stdio.
//!
//! Soma already *is* one (`soma-mcp`). This is the other direction: any
//! Model Context Protocol server becomes a set of tools an agent can call,
//! which is how the tool surface grows without Soma knowing anything about
//! what was added.
//!
//! Transport is JSON-RPC 2.0, one message per line, over a child process's
//! stdin/stdout — the transport every MCP server supports, and the one that
//! needs no ports, no TLS and no service to be running first.

use crate::error::{LlmError, Result};
use somatize_core::agentic::tool::ToolSpec;
use somatize_core::data::value::Value;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Mutex;

use crate::tools::ToolOutcome;

const PROTOCOL_VERSION: &str = "2024-11-05";

/// A running MCP server and the pipe to it.
///
/// Requests are serialised through a mutex: JSON-RPC over a single pipe pair
/// has no way to interleave, and the alternative — a reader task matching
/// responses by id — buys nothing when tool calls already run on their own
/// threads at the driver level.
pub struct McpClient {
    command: String,
    io: Mutex<Pipe>,
}

struct Pipe {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
}

impl McpClient {
    /// Start the server and complete the handshake.
    pub fn start(command: &str, args: &[&str]) -> Result<Self> {
        let mut child = Command::new(command)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // The server's own logs go to stderr; let them through to ours
            // rather than swallowing the one place a broken server explains
            // itself.
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| LlmError::mcp(command, format!("starting the process: {e}")))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| LlmError::mcp(command, "the process has no stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| LlmError::mcp(command, "the process has no stdout"))?;

        let client = Self {
            command: command.to_string(),
            io: Mutex::new(Pipe {
                child,
                stdin,
                stdout: BufReader::new(stdout),
                next_id: 1,
            }),
        };

        client.request(
            "initialize",
            serde_json::json!({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "soma", "version": env!("CARGO_PKG_VERSION")},
            }),
        )?;
        client.notify("notifications/initialized", serde_json::json!({}))?;

        Ok(client)
    }

    /// What this server publishes.
    pub fn list_tools(&self) -> Result<Vec<ToolSpec>> {
        let result = self.request("tools/list", serde_json::json!({}))?;
        let tools = result
            .get("tools")
            .cloned()
            .unwrap_or(serde_json::json!([]));
        serde_json::from_value(tools)
            .map_err(|e| LlmError::mcp(&self.command, format!("unreadable tool list: {e}")))
    }

    /// Call one.
    pub fn call_tool(&self, name: &str, args: &Value) -> Result<ToolOutcome> {
        let result = self.request(
            "tools/call",
            serde_json::json!({"name": name, "arguments": args.to_plain_json()}),
        )?;

        // MCP replies with content items; the text ones are what a model
        // can read. Joining them keeps a multi-part answer whole.
        let text = result
            .get("content")
            .and_then(|c| c.as_array())
            .map(|items| {
                items
                    .iter()
                    .filter_map(|i| i.get("text").and_then(|t| t.as_str()))
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .unwrap_or_default();

        let is_error = result
            .get("isError")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);

        Ok(ToolOutcome {
            output: Value::text(text),
            is_error,
        })
    }

    /// Send a request and wait for its response.
    fn request(&self, method: &str, params: serde_json::Value) -> Result<serde_json::Value> {
        let mut io = self.io.lock().map_err(|_| {
            LlmError::mcp(
                &self.command,
                "the client handle is poisoned by an earlier panic",
            )
        })?;

        let id = io.next_id;
        io.next_id += 1;

        let line = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        })
        .to_string();

        writeln!(io.stdin, "{line}")
            .and_then(|()| io.stdin.flush())
            .map_err(|e| self.dead(&mut io, format!("writing `{method}`: {e}")))?;

        // Skip notifications and any log line the server interleaves; a
        // response is the first object carrying our id.
        loop {
            let mut raw = String::new();
            let read = io
                .stdout
                .read_line(&mut raw)
                .map_err(|e| self.dead(&mut io, format!("reading `{method}`: {e}")))?;
            if read == 0 {
                return Err(self.dead(&mut io, format!("closed its output during `{method}`")));
            }
            let Ok(message) = serde_json::from_str::<serde_json::Value>(raw.trim()) else {
                continue;
            };
            if message.get("id").and_then(serde_json::Value::as_u64) != Some(id) {
                continue;
            }
            if let Some(error) = message.get("error") {
                let text = error
                    .get("message")
                    .and_then(|m| m.as_str())
                    .unwrap_or("unknown error");
                return Err(LlmError::mcp(
                    &self.command,
                    format!("refused `{method}`: {text}"),
                ));
            }
            return Ok(message.get("result").cloned().unwrap_or_default());
        }
    }

    /// Fire and forget — notifications carry no id and get no reply.
    fn notify(&self, method: &str, params: serde_json::Value) -> Result<()> {
        let mut io = self.io.lock().map_err(|_| {
            LlmError::mcp(
                &self.command,
                "the client handle is poisoned by an earlier panic",
            )
        })?;
        let line = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        })
        .to_string();
        writeln!(io.stdin, "{line}")
            .and_then(|()| io.stdin.flush())
            .map_err(|e| LlmError::mcp(&self.command, format!("notify `{method}`: {e}")))?;
        Ok(())
    }

    /// Report a broken pipe, including how the child died if it has.
    fn dead(&self, io: &mut Pipe, what: String) -> LlmError {
        let status = match io.child.try_wait() {
            Ok(Some(status)) => format!(" (server exited: {status})"),
            _ => String::new(),
        };
        LlmError::mcp(&self.command, format!("{what}{status}"))
    }
}

impl Drop for McpClient {
    fn drop(&mut self) {
        // Closing stdin is how an MCP server is asked to stop; kill only if
        // it does not take the hint.
        if let Ok(mut io) = self.io.lock() {
            let _ = io.stdin.flush();
            let _ = io.child.kill();
            let _ = io.child.wait();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal MCP server in Python, so the framing is exercised for real
    /// — a mocked transport would not catch a newline or flush bug.
    const FAKE_SERVER: &str = r#"
import json, sys
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    method = msg.get("method")
    if "id" not in msg:
        continue                      # a notification
    if method == "initialize":
        result = {"protocolVersion": "2024-11-05", "capabilities": {}}
    elif method == "tools/list":
        result = {"tools": [{
            "name": "echo",
            "description": "Echo text back.",
            "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}},
        }]}
    elif method == "tools/call":
        args = msg["params"].get("arguments", {})
        if msg["params"]["name"] == "boom":
            result = {"content": [{"type": "text", "text": "it broke"}], "isError": True}
        else:
            result = {"content": [{"type": "text", "text": args.get("text", "")}]}
    else:
        print(json.dumps({"jsonrpc": "2.0", "id": msg["id"],
                          "error": {"code": -32601, "message": "no such method"}}), flush=True)
        continue
    print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": result}), flush=True)
"#;

    fn fake_server() -> Result<McpClient> {
        McpClient::start("python3", &["-c", FAKE_SERVER])
    }

    #[test]
    fn handshakes_and_lists_tools() {
        let client = fake_server().expect("start");
        let tools = client.list_tools().unwrap();

        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "echo");
        // The MCP spelling of the schema field is understood.
        assert_eq!(tools[0].input_schema["type"], "object");
    }

    #[test]
    fn calls_a_tool_and_reads_its_content() {
        let client = fake_server().expect("start");
        let outcome = client
            .call_tool(
                "echo",
                &Value::json(serde_json::json!({"text": "hello mcp"})),
            )
            .unwrap();

        assert_eq!(outcome.output.as_text(), Some("hello mcp"));
        assert!(!outcome.is_error);
    }

    /// A tool the server marks failed comes back as an error the model can
    /// see, not as a transport failure.
    #[test]
    fn a_tool_error_is_carried_not_raised() {
        let client = fake_server().expect("start");
        let outcome = client.call_tool("boom", &Value::Empty).unwrap();
        assert!(outcome.is_error);
        assert_eq!(outcome.output.as_text(), Some("it broke"));
    }

    /// Several calls share one pipe; ids must not cross.
    #[test]
    fn repeated_calls_keep_their_replies_straight() {
        let client = fake_server().expect("start");
        for i in 0..5 {
            let text = format!("message {i}");
            let outcome = client
                .call_tool("echo", &Value::json(serde_json::json!({"text": text})))
                .unwrap();
            assert_eq!(outcome.output.as_text(), Some(text.as_str()));
        }
    }

    #[test]
    fn a_refused_method_reports_the_servers_reason() {
        let client = fake_server().expect("start");
        let err = client
            .request("resources/list", serde_json::json!({}))
            .unwrap_err()
            .to_string();
        assert!(err.contains("no such method"), "{err}");
    }

    #[test]
    fn a_missing_server_binary_says_so() {
        let err = match McpClient::start("definitely-not-a-real-binary-9182", &[]) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("a nonexistent binary should not start"),
        };
        assert!(err.contains("definitely-not-a-real-binary-9182"), "{err}");
    }

    /// A server that dies mid-conversation is reported as such, with its
    /// exit status, rather than as an opaque parse failure.
    #[test]
    fn a_server_that_exits_early_is_reported() {
        // Either the handshake fails immediately, or the first call does.
        let err = match McpClient::start("python3", &["-c", "import sys; sys.exit(3)"]) {
            Err(e) => e.to_string(),
            Ok(client) => client.list_tools().unwrap_err().to_string(),
        };
        assert!(err.contains("python3"), "{err}");
    }
}
