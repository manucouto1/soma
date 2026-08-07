//! The whole stack, exercised as a user would.
//!
//! A `ReactStep` asks a model, the model asks for a tool, the tool runs, the
//! model answers — over real HTTP, with a real MCP subprocess supplying the
//! tool, journaled so the second run performs nothing. If this passes, the
//! agentic layer is usable rather than merely present.

use somatize_core::data::value::Value;
use somatize_llm::{Catalog, LlmHandler, ProviderConfig, ReactStep, Router, ToolOutcome, Toolbox};
use somatize_runtime::agentic::{EffectDriver, EffectJournal, NodeOutcome};
use somatize_runtime::cache::fs_store::FsActionStore;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Serves a scripted sequence of responses, one per request.
struct ScriptedServer {
    port: u16,
    calls: Arc<AtomicUsize>,
    bodies: Arc<std::sync::Mutex<Vec<serde_json::Value>>>,
}

impl ScriptedServer {
    fn start(script: Vec<String>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().unwrap().port();
        let calls = Arc::new(AtomicUsize::new(0));
        let bodies = Arc::new(std::sync::Mutex::new(Vec::new()));

        let counter = calls.clone();
        let seen = bodies.clone();
        std::thread::spawn(move || {
            for body in script {
                let Ok((mut stream, _)) = listener.accept() else {
                    return;
                };
                counter.fetch_add(1, Ordering::SeqCst);

                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut line = String::new();
                let _ = reader.read_line(&mut line);
                let mut length = 0usize;
                loop {
                    let mut header = String::new();
                    if reader.read_line(&mut header).unwrap_or(0) == 0 || header == "\r\n" {
                        break;
                    }
                    if let Some((name, value)) = header.trim_end().split_once(": ")
                        && name.eq_ignore_ascii_case("content-length")
                    {
                        length = value.parse().unwrap_or(0);
                    }
                }
                let mut raw = vec![0u8; length];
                let _ = reader.read_exact(&mut raw);
                if let Ok(json) = serde_json::from_slice(&raw) {
                    seen.lock().unwrap().push(json);
                }

                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                     Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes());
                let _ = stream.flush();
            }
        });

        Self {
            port,
            calls,
            bodies,
        }
    }

    fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}/v1", self.port)
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    fn requests(&self) -> Vec<serde_json::Value> {
        self.bodies.lock().unwrap().clone()
    }
}

/// "Call the `weather` tool for the city in the prompt."
fn wants_weather() -> String {
    serde_json::json!({
        "choices": [{
            "message": {
                "content": "let me check",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "weather", "arguments": "{\"city\":\"Vigo\"}"}
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 8}
    })
    .to_string()
}

fn final_answer() -> String {
    serde_json::json!({
        "choices": [{
            "message": {"content": "It is sunny in Vigo."},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 30, "completion_tokens": 6}
    })
    .to_string()
}

fn weather_toolbox() -> Toolbox {
    let mut toolbox = Toolbox::new();
    toolbox.add_fn(
        somatize_core::agentic::tool::ToolSpec::new(
            "weather",
            "Current weather for a city. Call this when asked about conditions somewhere.",
            serde_json::json!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }),
        ),
        |args| {
            let city = args.to_plain_json()["city"]
                .as_str()
                .unwrap_or("nowhere")
                .to_string();
            Ok(ToolOutcome::ok(Value::text(format!("sunny in {city}"))))
        },
    );
    toolbox
}

fn driver_for(server: &ScriptedServer, toolbox: Toolbox, dir: &std::path::Path) -> EffectDriver {
    let mut catalog = Catalog::builtin();
    catalog.insert("mock", ProviderConfig::local(server.base_url()));
    let router = Router::from_catalog(catalog).unwrap().with_default("mock");

    let store = Arc::new(FsActionStore::new(dir).unwrap());
    EffectDriver::new(EffectJournal::new(store.clone(), store))
        .with_handler(Arc::new(LlmHandler::new(router)))
        .with_handler(Arc::new(toolbox))
}

// ── Tests ──

/// Ask → tool call → tool result → answer. The loop, end to end.
#[test]
fn a_react_step_calls_a_tool_and_answers() {
    let server = ScriptedServer::start(vec![wants_weather(), final_answer()]);
    let dir = tempfile::tempdir().unwrap();
    let driver = driver_for(&server, weather_toolbox(), dir.path());

    let step = ReactStep::new("some-model")
        .with_system("You are a weather assistant.")
        .with_tools(weather_toolbox().specs())
        .text_only();

    let outcome = driver
        .run(
            &step,
            "run-react",
            "agent",
            &Value::text("weather in Vigo?"),
        )
        .unwrap();

    match outcome {
        NodeOutcome::Produced(v) => assert_eq!(v.as_text(), Some("It is sunny in Vigo.")),
        other => panic!("{other:?}"),
    }
    assert_eq!(server.calls(), 2, "one call to ask, one to answer");

    let sent = server.requests();

    // The tool was advertised, with its description — that text is what the
    // model reads to decide whether to reach for it.
    assert_eq!(sent[0]["tools"][0]["function"]["name"], "weather");
    assert!(
        sent[0]["tools"][0]["function"]["description"]
            .as_str()
            .unwrap()
            .contains("Call this when"),
    );

    // The second call carries the conversation so far: the assistant turn
    // that asked, then the tool result paired to it by id.
    let second = sent[1]["messages"].as_array().unwrap();
    let roles: Vec<&str> = second.iter().map(|m| m["role"].as_str().unwrap()).collect();
    assert_eq!(roles, ["system", "user", "assistant", "tool"]);
    assert_eq!(second[2]["tool_calls"][0]["id"], "call_1");
    assert_eq!(second[3]["tool_call_id"], "call_1");
    assert_eq!(second[3]["content"], "sunny in Vigo");
}

/// Replaying the run performs nothing — no model call, no tool call — and
/// lands on the same answer. A crashed agent resumes rather than restarts.
#[test]
fn replaying_a_react_run_performs_nothing() {
    let server = ScriptedServer::start(vec![wants_weather(), final_answer()]);
    let dir = tempfile::tempdir().unwrap();
    let driver = driver_for(&server, weather_toolbox(), dir.path());

    let step = ReactStep::new("some-model")
        .with_tools(weather_toolbox().specs())
        .text_only();

    let first = driver
        .run(&step, "run-same", "agent", &Value::text("weather in Vigo?"))
        .unwrap();
    assert_eq!(server.calls(), 2);

    let replayed = driver
        .run(&step, "run-same", "agent", &Value::text("weather in Vigo?"))
        .unwrap();

    assert_eq!(server.calls(), 2, "the replay went back to the model");
    match (first, replayed) {
        (NodeOutcome::Produced(a), NodeOutcome::Produced(b)) => assert_eq!(a, b),
        other => panic!("{other:?}"),
    }
}

/// A model asking for a tool that does not exist gets told so and can
/// continue, rather than the run failing.
#[test]
fn an_unknown_tool_is_reported_to_the_model() {
    let server = ScriptedServer::start(vec![wants_weather(), final_answer()]);
    let dir = tempfile::tempdir().unwrap();
    // A toolbox with no `weather` in it.
    let driver = driver_for(&server, Toolbox::new(), dir.path());

    let step = ReactStep::new("some-model").text_only();
    let outcome = driver
        .run(&step, "run-missing", "agent", &Value::text("weather?"))
        .unwrap();

    // The run completed: the model was told, and answered anyway.
    match outcome {
        NodeOutcome::Produced(v) => assert_eq!(v.as_text(), Some("It is sunny in Vigo.")),
        other => panic!("{other:?}"),
    }

    let sent = server.requests();
    let second = sent[1]["messages"].as_array().unwrap();
    let tool_turn = second.last().unwrap();
    assert_eq!(tool_turn["role"], "tool");
    assert!(
        tool_turn["content"]
            .as_str()
            .unwrap()
            .contains("no tool named"),
        "{tool_turn}"
    );
}

/// Tools published by an MCP server are indistinguishable from native ones
/// once registered — which is how the tool surface grows without Soma
/// knowing what was added.
#[test]
fn an_mcp_server_supplies_tools_to_the_loop() {
    const SERVER: &str = r#"
import json, sys
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    if "id" not in msg:
        continue
    method = msg.get("method")
    if method == "initialize":
        result = {"protocolVersion": "2024-11-05", "capabilities": {}}
    elif method == "tools/list":
        result = {"tools": [{
            "name": "weather",
            "description": "Current weather for a city. Call this when asked about conditions.",
            "inputSchema": {"type": "object", "properties": {"city": {"type": "string"}}},
        }]}
    elif method == "tools/call":
        city = msg["params"].get("arguments", {}).get("city", "nowhere")
        result = {"content": [{"type": "text", "text": "sunny in " + city}]}
    else:
        continue
    print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": result}), flush=True)
"#;

    let mut toolbox = Toolbox::new();
    let count = toolbox
        .add_mcp_server("python3", &["-c", SERVER])
        .expect("the MCP server should start and publish its tools");
    assert_eq!(count, 1);
    assert_eq!(toolbox.names(), vec!["weather"]);

    let server = ScriptedServer::start(vec![wants_weather(), final_answer()]);
    let dir = tempfile::tempdir().unwrap();
    let specs = toolbox.specs();
    let driver = driver_for(&server, toolbox, dir.path());

    let step = ReactStep::new("some-model").with_tools(specs).text_only();
    let outcome = driver
        .run(&step, "run-mcp", "agent", &Value::text("weather in Vigo?"))
        .unwrap();

    match outcome {
        NodeOutcome::Produced(v) => assert_eq!(v.as_text(), Some("It is sunny in Vigo.")),
        other => panic!("{other:?}"),
    }

    // The MCP tool's result reached the model, paired to its call.
    let sent = server.requests();
    let second = sent[1]["messages"].as_array().unwrap();
    let tool_turn = second.last().unwrap();
    assert_eq!(tool_turn["tool_call_id"], "call_1");
    assert_eq!(tool_turn["content"], "sunny in Vigo");
}
