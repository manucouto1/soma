//! What the client does when an endpoint pushes back.
//!
//! Against a scripted server written on a raw `TcpListener` — forty lines of
//! HTTP by hand, and no new dependency for something this small. What is
//! being tested is *how many times* the client knocks, which means the test
//! has to count the knocks itself.

use somatize_core::agentic::effect::LlmRequest;
use somatize_core::agentic::message::{Message, Messages};
use somatize_llm::LlmProvider;
use somatize_llm::catalog::{ProviderConfig, RetryPolicy};
use somatize_llm::openai_compat::OpenAiCompatible;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

// ── A server that answers from a script ──

/// One scripted reply. `None` means hang up without answering, which is what
/// a proxy timing out or an endpoint being restarted looks like from here.
#[derive(Clone)]
enum Reply {
    Send {
        status: u16,
        headers: Vec<(String, String)>,
        body: String,
    },
    Hangup,
}

impl Reply {
    fn ok(body: &str) -> Self {
        Self::Send {
            status: 200,
            headers: Vec::new(),
            body: body.into(),
        }
    }

    fn status(status: u16) -> Self {
        Self::Send {
            status,
            headers: Vec::new(),
            body: r#"{"error": "scripted"}"#.into(),
        }
    }

    fn with_header(mut self, name: &str, value: &str) -> Self {
        if let Self::Send { headers, .. } = &mut self {
            headers.push((name.into(), value.into()));
        }
        self
    }
}

struct Server {
    port: u16,
    hits: Arc<AtomicUsize>,
}

impl Server {
    /// Serve `script` in order; once exhausted, keep repeating the last one.
    fn new(script: Vec<Reply>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().unwrap().port();
        let hits = Arc::new(AtomicUsize::new(0));
        let counter = hits.clone();

        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let i = counter.fetch_add(1, Ordering::SeqCst);
                let reply = script
                    .get(i)
                    .or_else(|| script.last())
                    .cloned()
                    .unwrap_or_else(|| Reply::ok("{}"));

                // Drain the request first; answering before reading gives
                // the client a broken pipe instead of the scripted status.
                let _ = drain(&mut stream);

                match reply {
                    Reply::Hangup => drop(stream),
                    Reply::Send {
                        status,
                        headers,
                        body,
                    } => {
                        let mut head = format!(
                            "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\n\
                             Content-Length: {}\r\n",
                            body.len()
                        );
                        for (name, value) in headers {
                            head.push_str(&format!("{name}: {value}\r\n"));
                        }
                        head.push_str("Connection: close\r\n\r\n");
                        let _ = stream.write_all(head.as_bytes());
                        let _ = stream.write_all(body.as_bytes());
                        let _ = stream.flush();
                    }
                }
            }
        });

        Self { port, hits }
    }

    fn provider(&self, retry: RetryPolicy) -> OpenAiCompatible {
        let config =
            ProviderConfig::local(format!("http://127.0.0.1:{}/v1", self.port)).with_retry(retry);
        OpenAiCompatible::new("mock", config).expect("client")
    }

    fn hits(&self) -> usize {
        self.hits.load(Ordering::SeqCst)
    }
}

fn drain(stream: &mut TcpStream) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut length = 0usize;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            return Ok(());
        }
        if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
            length = value.trim().parse().unwrap_or(0);
        }
        if line == "\r\n" || line == "\n" {
            break;
        }
    }
    let mut body = vec![0u8; length];
    reader.read_exact(&mut body)
}

/// A policy with the waits collapsed, so the tests measure behaviour rather
/// than the clock.
fn fast(max_attempts: u32) -> RetryPolicy {
    RetryPolicy {
        max_attempts,
        base_ms: 1,
        max_ms: 5,
        budget_secs: 30,
        jitter: false,
    }
}

fn ask(provider: &OpenAiCompatible) -> somatize_core::error::Result<()> {
    let req = LlmRequest::new("any", Messages::from(vec![Message::user("hello")]));
    provider.complete(&req).map(|_| ())
}

const GOOD: &str = r#"{"choices":[{"message":{"content":"hi"},"finish_reason":"stop"}]}"#;

// ── What is retryable ──

#[test]
fn a_rate_limit_is_retried_until_it_clears() {
    let server = Server::new(vec![
        Reply::status(429),
        Reply::status(429),
        Reply::ok(GOOD),
    ]);
    assert!(ask(&server.provider(fast(4))).is_ok());
    assert_eq!(server.hits(), 3);
}

#[test]
fn a_server_error_is_retried() {
    let server = Server::new(vec![Reply::status(503), Reply::ok(GOOD)]);
    assert!(ask(&server.provider(fast(4))).is_ok());
    assert_eq!(server.hits(), 2);
}

/// The whole point of classifying: a bad request is not a transient
/// condition, and four round trips to reach the same error is four wasted.
#[test]
fn a_bad_request_is_not_retried() {
    let server = Server::new(vec![Reply::status(400)]);
    let err = ask(&server.provider(fast(4))).unwrap_err();

    assert_eq!(server.hits(), 1, "a 400 must be asked exactly once");
    assert!(err.to_string().contains("400"), "{err}");
}

#[test]
fn an_unauthorized_request_is_not_retried() {
    let server = Server::new(vec![Reply::status(401)]);
    assert!(ask(&server.provider(fast(4))).is_err());
    assert_eq!(server.hits(), 1);
}

#[test]
fn a_dropped_connection_is_retried() {
    let server = Server::new(vec![Reply::Hangup, Reply::ok(GOOD)]);
    assert!(ask(&server.provider(fast(4))).is_ok());
    assert_eq!(server.hits(), 2);
}

/// A 200 carrying something that is not a completion is a proxy or a
/// captive portal answering for the endpoint — worth one more try, and not
/// something the caller can fix by changing the request.
#[test]
fn a_body_that_is_not_a_completion_is_retried() {
    let server = Server::new(vec![Reply::ok("<html>gateway</html>"), Reply::ok(GOOD)]);
    assert!(ask(&server.provider(fast(4))).is_ok());
    assert_eq!(server.hits(), 2);
}

// ── Giving up ──

#[test]
fn giving_up_says_how_many_times_it_tried() {
    let server = Server::new(vec![Reply::status(429)]);
    let err = ask(&server.provider(fast(3))).unwrap_err();

    assert_eq!(server.hits(), 3);
    let text = err.to_string();
    // A configuration problem reads as one only if the count is in the
    // message; "failed" alone is a mystery.
    assert!(text.contains("3 attempt"), "{text}");
    assert!(text.contains("mock"), "{text}");
    assert!(text.contains("429"), "{text}");
}

/// When the attempts failed differently, the first one is usually the
/// diagnosis and the rest are consequences. An endpoint that answers with a
/// gateway page and then stops answering at all has said two things.
#[test]
fn a_failure_that_changed_shape_reports_both_ends() {
    let server = Server::new(vec![Reply::ok("<html>gateway</html>"), Reply::status(503)]);
    let err = ask(&server.provider(fast(2))).unwrap_err().to_string();

    assert!(err.contains("503"), "the last failure: {err}");
    assert!(err.contains("first attempt"), "{err}");
    assert!(
        err.contains("not a chat completion"),
        "the diagnosis: {err}"
    );
}

#[test]
fn one_attempt_means_no_retries() {
    let server = Server::new(vec![Reply::status(503)]);
    assert!(ask(&server.provider(fast(1))).is_err());
    assert_eq!(server.hits(), 1);
}

/// The budget is a guard against a study trial spending an hour on an
/// endpoint that is down. It is checked before waiting, so it stops the loop
/// rather than interrupting a request in flight.
#[test]
fn the_wall_clock_budget_stops_the_loop() {
    let server = Server::new(vec![Reply::status(503)]);
    let policy = RetryPolicy {
        max_attempts: 50,
        base_ms: 400,
        max_ms: 400,
        budget_secs: 1,
        jitter: false,
    };

    let started = std::time::Instant::now();
    assert!(ask(&server.provider(policy)).is_err());
    let elapsed = started.elapsed();

    assert!(elapsed < Duration::from_secs(3), "took {elapsed:?}");
    assert!(
        server.hits() < 10,
        "budget should stop it early, got {} attempts",
        server.hits()
    );
}

// ── Retry-After ──

#[test]
fn retry_after_in_seconds_is_honoured() {
    let server = Server::new(vec![
        Reply::status(429).with_header("Retry-After", "1"),
        Reply::ok(GOOD),
    ]);
    // Ceiling above the second being asked for, so this measures the
    // instruction being obeyed rather than the cap (which has its own test).
    let policy = RetryPolicy {
        max_attempts: 4,
        base_ms: 1,
        max_ms: 5_000,
        budget_secs: 30,
        jitter: false,
    };

    let started = std::time::Instant::now();
    assert!(ask(&server.provider(policy)).is_ok());

    // The backoff alone would have waited a millisecond; the endpoint asked
    // for a second, and the endpoint knows when its window resets.
    assert!(
        started.elapsed() >= Duration::from_millis(900),
        "waited {:?}",
        started.elapsed()
    );
}

/// Endpoints send both forms the RFC allows. Reading only delta-seconds
/// would silently ignore half of them and hammer straight back.
#[test]
fn retry_after_as_a_date_is_honoured() {
    let server = Server::new(vec![
        Reply::status(429).with_header("Retry-After", "Sun, 06 Nov 1994 08:49:37 GMT"),
        Reply::ok(GOOD),
    ]);
    // A date in the past means "now"; what matters is that it parsed and
    // did not become a fatal error or a wild wait.
    assert!(ask(&server.provider(fast(4))).is_ok());
    assert_eq!(server.hits(), 2);
}

#[test]
fn a_retry_after_longer_than_the_ceiling_is_capped() {
    let policy = RetryPolicy {
        max_attempts: 2,
        base_ms: 1,
        max_ms: 50, // an endpoint asking for an hour is told to come back later
        budget_secs: 30,
        jitter: false,
    };
    let server = Server::new(vec![
        Reply::status(429).with_header("Retry-After", "3600"),
        Reply::ok(GOOD),
    ]);

    let started = std::time::Instant::now();
    assert!(ask(&server.provider(policy)).is_ok());
    assert!(started.elapsed() < Duration::from_secs(2));
}

// ── The policy itself ──

#[test]
fn backoff_doubles_and_stops_at_the_ceiling() {
    let policy = RetryPolicy {
        max_attempts: 10,
        base_ms: 100,
        max_ms: 400,
        budget_secs: 60,
        jitter: false,
    };
    let waits: Vec<u64> = (0..5)
        .map(|i| policy.backoff(i, None).as_millis() as u64)
        .collect();
    assert_eq!(waits, [100, 200, 400, 400, 400]);
}

#[test]
fn jitter_stays_under_the_exponential() {
    let policy = RetryPolicy {
        base_ms: 1000,
        max_ms: 1000,
        jitter: true,
        ..RetryPolicy::default()
    };
    for i in 0..8 {
        assert!(policy.backoff(i, None) <= Duration::from_millis(1000));
    }
}

/// A budget below one request's timeout promises retries that can never
/// happen. Say so rather than quietly rewriting somebody's TOML.
#[test]
fn a_budget_below_the_timeout_is_reported() {
    let mut config = ProviderConfig::local("http://localhost:1/v1");
    config.timeout_secs = 300;
    config.retry.budget_secs = 60;

    let warnings = config.warnings("slow-one");
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(warnings[0].contains("budget_secs"), "{}", warnings[0]);
}

#[test]
fn a_provider_that_never_retries_reports_nothing() {
    let mut config = ProviderConfig::local("http://localhost:1/v1");
    config.timeout_secs = 300;
    config.retry = RetryPolicy::none();
    assert!(config.warnings("careful").is_empty());
}
