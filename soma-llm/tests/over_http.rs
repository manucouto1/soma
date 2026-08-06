//! End-to-end over real HTTP, against a mock server on localhost.
//!
//! Unit tests cover the translation; these cover everything the translation
//! sits inside — headers, auth, status handling, and the whole path from an
//! agentic step through the effect driver to a socket and back. No network,
//! so they run anywhere CI runs.

use somatize_core::agentic::effect::{Effect, EffectResult, LlmRequest, StopReason};
use somatize_core::agentic::message::Message;
use somatize_core::cache::CacheKey;
use somatize_core::data::value::Value;
use somatize_core::error::Result;
use somatize_core::graph::step::{Step, StepCtx, StepMeta, Transition};
use somatize_llm::{
    Auth, Catalog, LlmHandler, LlmProvider, OpenAiCompatible, ProviderConfig, Router,
};
use somatize_runtime::agentic::{EffectDriver, EffectHandler, EffectJournal, NodeOutcome};
use somatize_runtime::cache::fs_store::FsActionStore;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

// ── A minimal HTTP server ──

/// Serves a canned body, recording what it was sent.
///
/// Hand-rolled rather than pulling in a mocking framework: the surface under
/// test is "did we send the right bytes to the right URL with the right
/// headers", and forty lines of `TcpListener` answers that without a
/// dependency or an async runtime.
struct MockServer {
    port: u16,
    requests: Arc<std::sync::Mutex<Vec<Recorded>>>,
    calls: Arc<AtomicUsize>,
}

#[derive(Debug, Clone)]
struct Recorded {
    path: String,
    authorization: Option<String>,
    headers: Vec<(String, String)>,
    body: serde_json::Value,
}

impl MockServer {
    /// Answer `status` with `body` for the next `serves` requests.
    fn start(status: u16, body: String, serves: usize) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().unwrap().port();
        let requests = Arc::new(std::sync::Mutex::new(Vec::new()));
        let calls = Arc::new(AtomicUsize::new(0));

        let recorder = requests.clone();
        let counter = calls.clone();
        std::thread::spawn(move || {
            for _ in 0..serves {
                let Ok((mut stream, _)) = listener.accept() else {
                    return;
                };
                counter.fetch_add(1, Ordering::SeqCst);

                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut request_line = String::new();
                let _ = reader.read_line(&mut request_line);
                let path = request_line
                    .split_whitespace()
                    .nth(1)
                    .unwrap_or("/")
                    .to_string();

                let mut headers = Vec::new();
                let mut content_length = 0usize;
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                        break;
                    }
                    if let Some((name, value)) = line.trim_end().split_once(": ") {
                        if name.eq_ignore_ascii_case("content-length") {
                            content_length = value.parse().unwrap_or(0);
                        }
                        headers.push((name.to_lowercase(), value.to_string()));
                    }
                }

                let mut raw = vec![0u8; content_length];
                if content_length > 0 {
                    let _ = reader.read_exact(&mut raw);
                }

                recorder.lock().unwrap().push(Recorded {
                    path,
                    authorization: headers
                        .iter()
                        .find(|(n, _)| n == "authorization")
                        .map(|(_, v)| v.clone()),
                    headers,
                    body: serde_json::from_slice(&raw).unwrap_or(serde_json::Value::Null),
                });

                let response = format!(
                    "HTTP/1.1 {status} OK\r\nContent-Type: application/json\r\n\
                     Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes());
                let _ = stream.flush();
            }
        });

        Self {
            port,
            requests,
            calls,
        }
    }

    fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}/v1", self.port)
    }

    fn recorded(&self) -> Vec<Recorded> {
        self.requests.lock().unwrap().clone()
    }

    fn call_count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

fn reply(text: &str) -> String {
    serde_json::json!({
        "model": "mock-model",
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4}
    })
    .to_string()
}

fn ask(model: &str) -> LlmRequest {
    LlmRequest::new(model, vec![Message::user("what is soma?")].into())
}

// ── Tests ──

#[test]
fn a_completion_goes_out_and_comes_back() {
    let server = MockServer::start(200, reply("a graph runtime"), 1);
    let provider = OpenAiCompatible::new("mock", ProviderConfig::local(server.base_url())).unwrap();

    let response = provider.complete(&ask("some-model")).unwrap();

    assert_eq!(response.message.text(), "a graph runtime");
    assert_eq!(response.stop_reason, StopReason::EndTurn);
    assert_eq!(response.usage.output_tokens, 4);

    let sent = server.recorded();
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].path, "/v1/chat/completions");
    assert_eq!(sent[0].body["model"], "some-model");
    assert_eq!(sent[0].body["messages"][0]["content"], "what is soma?");
    assert!(
        sent[0].authorization.is_none(),
        "a local provider should send no credentials"
    );
}

/// The key reaches the wire as a bearer token, and extra headers ride along —
/// OpenRouter wants a referer, some tenants want their own.
#[test]
fn credentials_and_extra_headers_reach_the_wire() {
    // SAFETY: single-threaded test; nothing else reads this variable.
    unsafe { std::env::set_var("SOMA_TEST_HTTP_KEY", "sk-test-123") };

    let server = MockServer::start(200, reply("ok"), 1);
    let config = ProviderConfig {
        auth: Auth::bearer("SOMA_TEST_HTTP_KEY"),
        ..ProviderConfig::local(server.base_url())
    }
    .with_header("X-Title", "soma");

    OpenAiCompatible::new("mock", config)
        .unwrap()
        .complete(&ask("m"))
        .unwrap();

    let sent = server.recorded();
    assert_eq!(sent[0].authorization.as_deref(), Some("Bearer sk-test-123"));
    assert!(
        sent[0]
            .headers
            .iter()
            .any(|(n, v)| n == "x-title" && v == "soma"),
        "extra header missing: {:?}",
        sent[0].headers
    );

    unsafe { std::env::remove_var("SOMA_TEST_HTTP_KEY") };
}

/// An error body is surfaced verbatim. Endpoints bury the useful part in
/// differently-shaped envelopes, so showing it beats guessing at a schema.
#[test]
fn an_error_status_surfaces_the_body() {
    let server = MockServer::start(
        400,
        serde_json::json!({"error": {"message": "model `nope` not found"}}).to_string(),
        1,
    );
    let provider = OpenAiCompatible::new("mock", ProviderConfig::local(server.base_url())).unwrap();

    let err = provider.complete(&ask("nope")).unwrap_err().to_string();
    assert!(err.contains("mock"), "should name the provider: {err}");
    assert!(err.contains("400"), "should give the status: {err}");
    assert!(err.contains("not found"), "should show the body: {err}");
}

/// A body that is not a chat completion — an HTML error page from a proxy,
/// say — says so and shows what arrived.
#[test]
fn an_unparseable_body_is_reported_with_a_sample() {
    let server = MockServer::start(200, "<html>502 Bad Gateway</html>".into(), 1);
    let provider = OpenAiCompatible::new("mock", ProviderConfig::local(server.base_url())).unwrap();

    let err = provider.complete(&ask("m")).unwrap_err().to_string();
    assert!(err.contains("not a chat completion"), "{err}");
    assert!(err.contains("502"), "should include the body: {err}");
}

#[test]
fn model_discovery_lists_what_the_endpoint_offers() {
    let server = MockServer::start(
        200,
        serde_json::json!({"data": [{"id": "llama3.2"}, {"id": "qwen3"}]}).to_string(),
        1,
    );
    let provider =
        OpenAiCompatible::new("ollama", ProviderConfig::local(server.base_url())).unwrap();

    let models = provider.models().unwrap();
    assert_eq!(models.len(), 2);
    assert_eq!(models[0].qualified(), "ollama/llama3.2");
    assert_eq!(server.recorded()[0].path, "/v1/models");
}

/// A catalog entry added at runtime is indistinguishable from a built-in —
/// which is the whole claim of the data-driven design.
#[test]
fn a_provider_added_to_the_catalog_is_routable() {
    let server = MockServer::start(200, reply("from my box"), 1);

    let mut catalog = Catalog::builtin();
    catalog.insert("my-vllm", ProviderConfig::local(server.base_url()));
    let router = Router::from_catalog(catalog).unwrap();

    let response = router.complete(&ask("my-vllm/mixtral")).unwrap();
    assert_eq!(response.message.text(), "from my box");
    assert_eq!(server.recorded()[0].body["model"], "mixtral");
}

/// The whole path: an agentic step, through the driver, over HTTP — and then
/// replayed from the journal without touching the socket again.
#[test]
fn a_step_reaches_a_provider_and_then_replays() {
    struct AskOnce;
    impl Step for AskOnce {
        fn config_hash(&self) -> CacheKey {
            CacheKey::from_parts(&[b"AskOnce"])
        }
        fn meta(&self) -> StepMeta {
            StepMeta::new("AskOnce")
        }
        fn poll(&self, ctx: &StepCtx<'_>) -> Result<Transition> {
            match ctx.result() {
                None => Ok(Transition::Await(vec![Effect::Llm(LlmRequest::new(
                    "mock/some-model",
                    vec![Message::user(ctx.input.as_text().unwrap_or_default())].into(),
                ))])),
                Some(EffectResult::Llm(r)) => Ok(Transition::Done(Value::text(r.message.text()))),
                Some(other) => Ok(Transition::Done(Value::text(format!("{other:?}")))),
            }
        }
    }

    // Serve at most twice, so a replay that wrongly calls out would be
    // answered rather than hanging — and the count assertion below catches it.
    let server = MockServer::start(200, reply("the recorded answer"), 2);

    let mut catalog = Catalog::builtin();
    catalog.insert("mock", ProviderConfig::local(server.base_url()));
    let handler = Arc::new(LlmHandler::new(Router::from_catalog(catalog).unwrap()));

    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(FsActionStore::new(dir.path()).unwrap());
    let driver = EffectDriver::new(EffectJournal::new(store.clone(), store)).with_handler(handler);

    let first = driver
        .run(&AskOnce, "run-http", "agent", &Value::text("hello"))
        .unwrap();
    assert_eq!(server.call_count(), 1);

    let replayed = driver
        .run(&AskOnce, "run-http", "agent", &Value::text("hello"))
        .unwrap();

    assert_eq!(
        server.call_count(),
        1,
        "the replay went back out to the provider"
    );
    match (first, replayed) {
        (NodeOutcome::Produced(a), NodeOutcome::Produced(b)) => {
            assert_eq!(a.as_text(), Some("the recorded answer"));
            assert_eq!(a, b);
        }
        other => panic!("{other:?}"),
    }
}

/// A provider that is unreachable reaches the step as a result it can act
/// on, rather than blowing up the run.
#[test]
fn an_unreachable_provider_becomes_a_failed_result() {
    // Nothing is listening on this port.
    let config = ProviderConfig::local("http://127.0.0.1:1/v1");
    let mut router = Router::new();
    router.register(Arc::new(OpenAiCompatible::new("dead", config).unwrap()));
    let handler = LlmHandler::new(router.with_default("dead"));

    let result = handler.perform(&Effect::Llm(ask("m"))).unwrap();
    match result {
        EffectResult::Failed { message } => assert!(message.contains("dead"), "{message}"),
        other => panic!("expected Failed, got {other:?}"),
    }
}
