//! One client for every endpoint that speaks `POST /chat/completions`.
//!
//! Which, as of 2026, is nearly all of them: Ollama, the Hugging Face
//! router, NVIDIA NIM, Moonshot, Z.ai, DeepSeek, Mistral, Groq, Together,
//! OpenRouter, vLLM, llama.cpp. They differ in URL, auth and a few quirks —
//! all of which live in [`ProviderConfig`], not here.
//!
//! The work this module actually does is translation: Soma's [`Message`] and
//! [`ContentBlock`] in, the wire's flat `role`/`content`/`tool_calls` out,
//! and back. That translation is the only place provider shape leaks, and
//! keeping it in one file is the point.

use crate::catalog::{ProviderConfig, RetryPolicy};
use crate::error::{LlmError, Result};
use crate::{LlmProvider, ModelInfo};
use serde::{Deserialize, Serialize};
use somatize_core::effect::{LlmRequest, LlmResponse, StopReason, ToolSpec, Usage};
use somatize_core::message::{ContentBlock, Message, Role};
use std::time::Duration;

// ── Pushback ──
//
// Retrying belongs here, in the transport, not in the step that asked for
// the completion. A 429 is not a decision about the work: the step has no
// information the client lacks, and by the time a `Failed` reaches it the
// only honest reading is "this will not get better by asking again" — which
// is exactly what `ReactStep` already assumes when it stops.

/// What an endpoint's refusal means for trying again.
enum Pushback {
    /// Worth another attempt, after the delay the endpoint asked for (if it
    /// bothered to say).
    Retryable(Option<Duration>),
    /// Asking again will produce the same answer. A 401 does not improve
    /// with patience.
    Fatal,
}

/// Read a refusal. Anything not listed is fatal — the safe direction, since
/// retrying a 400 wastes four round trips to reach the same error.
fn classify(status: reqwest::StatusCode, headers: &reqwest::header::HeaderMap) -> Pushback {
    let retryable = matches!(status.as_u16(), 408 | 425 | 429 | 500 | 502 | 503 | 504);
    if retryable {
        Pushback::Retryable(retry_after(headers))
    } else {
        Pushback::Fatal
    }
}

/// `Retry-After`, in either form the RFC allows: delta-seconds, or an HTTP
/// date. Endpoints use both, so reading only the first would silently ignore
/// half of them and hammer straight back.
fn retry_after(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    let raw = headers.get(reqwest::header::RETRY_AFTER)?.to_str().ok()?;

    if let Ok(seconds) = raw.trim().parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }

    let when = httpdate_to_unix(raw.trim())?;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs();
    Some(Duration::from_secs(when.saturating_sub(now)))
}

/// Seconds since the epoch for an IMF-fixdate (`Sun, 06 Nov 1994 08:49:37 GMT`).
///
/// The only format RFC 9110 requires a sender to produce, so parsing it by
/// hand is a dozen lines rather than a dependency.
fn httpdate_to_unix(text: &str) -> Option<u64> {
    let parts: Vec<&str> = text.split_whitespace().collect();
    if parts.len() != 6 {
        return None;
    }
    let day: u64 = parts[1].parse().ok()?;
    let month = match parts[2] {
        "Jan" => 1,
        "Feb" => 2,
        "Mar" => 3,
        "Apr" => 4,
        "May" => 5,
        "Jun" => 6,
        "Jul" => 7,
        "Aug" => 8,
        "Sep" => 9,
        "Oct" => 10,
        "Nov" => 11,
        "Dec" => 12,
        _ => return None,
    };
    let year: u64 = parts[3].parse().ok()?;
    let clock: Vec<&str> = parts[4].split(':').collect();
    if clock.len() != 3 {
        return None;
    }
    let (h, m, s): (u64, u64, u64) = (
        clock[0].parse().ok()?,
        clock[1].parse().ok()?,
        clock[2].parse().ok()?,
    );

    // Days from the epoch to the start of `year`, then to the start of
    // `month`. Civil-from-days, the usual closed form.
    let (y, mth) = if month <= 2 {
        (year - 1, month + 9)
    } else {
        (year, month - 3)
    };
    let era = y / 400;
    let yoe = y - era * 400;
    let doy = (153 * mth + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe;
    // 719_468 shifts the 0000-03-01 epoch this form uses to 1970-01-01.
    Some((days - 719_468) * 86_400 + h * 3600 + m * 60 + s)
}

/// Run `attempt` until it succeeds, gives up, or runs out of budget.
///
/// `attempt` returns `Ok(value)` on success, or `Err((pushback, error))`.
fn retrying<T>(
    policy: &RetryPolicy,
    provider: &str,
    what: &str,
    mut attempt: impl FnMut() -> std::result::Result<T, (Pushback, LlmError)>,
) -> Result<T> {
    let started = std::time::Instant::now();
    let mut first: Option<String> = None;
    let mut last: Option<LlmError> = None;

    for tries in 0..policy.max_attempts.max(1) {
        match attempt() {
            Ok(value) => return Ok(value),
            Err((Pushback::Fatal, e)) => return Err(e),
            Err((Pushback::Retryable(asked), e)) => {
                first.get_or_insert_with(|| e.to_string());
                last = Some(e);
                let is_last = tries + 1 >= policy.max_attempts.max(1);
                let wait = policy.backoff(tries, asked);
                let spent = started.elapsed();
                // Checked before sleeping, not after: a wait that would
                // overrun the budget is a wait nobody wants to sit through.
                let would_overrun = spent + wait >= Duration::from_secs(policy.budget_secs);

                if is_last || would_overrun {
                    break;
                }
                tracing::warn!(
                    provider,
                    what,
                    attempt = tries + 1,
                    wait_ms = wait.as_millis() as u64,
                    error = %last.as_ref().map(|e| e.to_string()).unwrap_or_default(),
                    "retrying after pushback"
                );
                std::thread::sleep(wait);
            }
        }
    }

    // The count matters in the message: "failed after 4 attempts" is a
    // configuration problem, "failed" is a mystery.
    let last = last
        .map(|e| e.to_string())
        .unwrap_or_else(|| "no error recorded".into());

    // When the attempts failed differently, the first one is usually the
    // diagnosis and the rest are consequences — an endpoint that answers
    // with an HTML error page and then stops answering at all has told you
    // two things, and only reporting the second loses the useful one.
    let context = match &first {
        Some(f) if *f != last => format!(" (first attempt: {f})"),
        _ => String::new(),
    };

    Err(LlmError::provider(
        provider,
        format!(
            "{what} failed after {} attempt(s): {last}{context}",
            policy.max_attempts.max(1),
        ),
    ))
}

/// A provider reached over the OpenAI chat-completions shape.
pub struct OpenAiCompatible {
    id: String,
    config: ProviderConfig,
    client: reqwest::blocking::Client,
}

impl OpenAiCompatible {
    /// A client for one endpoint. Builds the HTTP client with the config's
    /// timeout; nothing is contacted until a request is made.
    pub fn new(id: impl Into<String>, config: ProviderConfig) -> Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| LlmError::Config(format!("building the http client: {e}")))?;
        Ok(Self {
            id: id.into(),
            config,
            client,
        })
    }

    /// The configuration this client was built from.
    pub fn config(&self) -> &ProviderConfig {
        &self.config
    }

    fn request(
        &self,
        method: reqwest::Method,
        url: &str,
    ) -> Result<reqwest::blocking::RequestBuilder> {
        let mut builder = self.client.request(method, url);
        if let Some((header, value)) = self.config.auth.resolve(&self.id)? {
            builder = builder.header(header, value);
        }
        for (name, value) in &self.config.headers {
            builder = builder.header(name.as_str(), value.as_str());
        }
        Ok(builder)
    }

    /// Build the JSON body for a completion.
    fn body(&self, req: &LlmRequest) -> serde_json::Value {
        let quirks = &self.config.quirks;
        let mut messages: Vec<WireMessage> = Vec::new();

        if let Some(system) = &req.system
            && quirks.system_as_message
        {
            messages.push(WireMessage::system(system));
        }
        for message in req.messages.iter() {
            messages.extend(to_wire(message));
        }
        // An endpoint that will not take a system role still needs the
        // instruction: fold it into the first user turn rather than drop it.
        if let Some(system) = &req.system
            && !quirks.system_as_message
            && let Some(first) = messages.iter_mut().find(|m| m.role == "user")
        {
            first.content = Some(match first.content.take() {
                Some(existing) => format!("{system}\n\n{existing}"),
                None => system.clone(),
            });
        }

        // A schema an endpoint can enforce is worth far more than one it is
        // merely asked to respect: constrained decoding cannot produce
        // prose, while a prompt can. Where the endpoint cannot, say it in
        // words — worse, but better than dropping the requirement and
        // letting the caller wonder why nothing validates.
        let mut prompt_schema: Option<String> = None;
        if let Some(schema) = &req.schema
            && !quirks.supports_json_schema
        {
            prompt_schema = Some(format!(
                "Reply with JSON only — no prose, no code fences — matching this \
                 JSON Schema:\n{schema}"
            ));
        }
        if let Some(instruction) = &prompt_schema {
            match messages.iter_mut().find(|m| m.role == "system") {
                Some(system) => {
                    system.content = Some(match system.content.take() {
                        Some(existing) => format!("{existing}\n\n{instruction}"),
                        None => instruction.clone(),
                    });
                }
                None => messages.insert(0, WireMessage::system(instruction)),
            }
        }

        let mut body = serde_json::json!({
            "model": self.config.wire_model(&req.model),
            "messages": messages,
        });

        if let Some(schema) = &req.schema
            && quirks.supports_json_schema
        {
            body["response_format"] = serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                    "strict": true,
                },
            });
        }

        if let Some(max) = req.max_tokens {
            body[quirks.max_tokens_field.as_str()] = serde_json::json!(max);
        }
        if quirks.supports_tools && !(req.tools.is_empty() && quirks.omit_empty_tools) {
            body["tools"] =
                serde_json::json!(req.tools.iter().map(to_wire_tool).collect::<Vec<_>>());
        }
        // `effort` is Anthropic's word for it; OpenAI-shaped endpoints that
        // have the concept call it `reasoning_effort`. Ones that don't
        // ignore an unknown field, which is the correct behaviour here.
        if let Some(effort) = &req.effort {
            body["reasoning_effort"] = serde_json::json!(effort);
        }
        body
    }

    fn parse(&self, raw: WireResponse) -> Result<LlmResponse> {
        let choice = raw
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| LlmError::provider(&self.id, "the reply carried no choices"))?;

        let mut content: Vec<ContentBlock> = Vec::new();
        if let Some(text) = choice.message.content.filter(|t| !t.is_empty()) {
            content.push(ContentBlock::text(text));
        }
        for call in choice.message.tool_calls.unwrap_or_default() {
            // Arguments arrive as a JSON *string*. A model that emits
            // malformed JSON is common enough that it must not abort the
            // turn: pass the raw text through so the step can decide.
            let input = serde_json::from_str(&call.function.arguments)
                .unwrap_or_else(|_| serde_json::json!({ "_raw": call.function.arguments }));
            content.push(ContentBlock::tool_use(call.id, call.function.name, input));
        }

        let stop_reason = match choice.finish_reason.as_deref() {
            Some("tool_calls") | Some("function_call") => StopReason::ToolUse,
            Some("length") | Some("max_tokens") => StopReason::MaxTokens,
            Some("content_filter") => StopReason::Refusal {
                category: Some("content_filter".into()),
            },
            _ => StopReason::EndTurn,
        };

        Ok(LlmResponse {
            message: Message::new(Role::Assistant, content),
            stop_reason,
            usage: raw.usage.map(Into::into).unwrap_or_default(),
            model: raw.model,
        })
    }
}

impl LlmProvider for OpenAiCompatible {
    fn id(&self) -> &str {
        &self.id
    }

    /// The seam: [`LlmProvider`] is used through the effect driver, which
    /// speaks `SomaError`. Everything behind this call is an [`LlmError`].
    fn complete(&self, req: &LlmRequest) -> somatize_core::error::Result<LlmResponse> {
        let url = self.config.chat_url();
        let body = self.body(req);

        Ok(retrying(
            &self.config.retry,
            &self.id,
            "completion",
            || {
                let response = self
                    .request(reqwest::Method::POST, &url)
                    .map_err(|e| (Pushback::Fatal, e))?
                    .json(&body)
                    .send()
                    // A connection refused, reset or timed out says nothing
                    // about the request — only that this attempt did not land.
                    .map_err(|e| {
                        (
                            Pushback::Retryable(None),
                            LlmError::provider(&self.id, format!("{url}: {e}")),
                        )
                    })?;

                let status = response.status();
                let pushback = classify(status, response.headers());
                let text = response.text().map_err(|e| {
                    (
                        Pushback::Retryable(None),
                        LlmError::provider(&self.id, format!("reading response: {e}")),
                    )
                })?;

                if !status.is_success() {
                    // Endpoints bury the useful part in differently-shaped error
                    // envelopes; show the body rather than guess at its schema.
                    return Err((
                        pushback,
                        LlmError::provider(
                            &self.id,
                            format!("returned {status}: {}", truncate(text.trim(), 400)),
                        ),
                    ));
                }

                // A 200 carrying something that is not a chat completion is a
                // proxy or a captive portal answering for the endpoint. Worth
                // one more try; it is not a request the caller can fix.
                let raw: WireResponse = serde_json::from_str(&text).map_err(|e| {
                    (
                        Pushback::Retryable(None),
                        LlmError::provider(
                            &self.id,
                            format!(
                                "returned a body that is not a chat completion: {e}. Body: {}",
                                truncate(&text, 400)
                            ),
                        ),
                    )
                })?;

                self.parse(raw).map_err(|e| (Pushback::Fatal, e))
            },
        )?)
    }

    fn models(&self) -> somatize_core::error::Result<Vec<ModelInfo>> {
        let url = self.config.models_url();

        let listing: WireModels = retrying(&self.config.retry, &self.id, "model listing", || {
            let response = self
                .request(reqwest::Method::GET, &url)
                .map_err(|e| (Pushback::Fatal, e))?
                .send()
                .map_err(|e| {
                    (
                        Pushback::Retryable(None),
                        LlmError::provider(&self.id, format!("{url}: {e}")),
                    )
                })?;

            let status = response.status();
            let pushback = classify(status, response.headers());
            if !status.is_success() {
                return Err((
                    pushback,
                    LlmError::provider(&self.id, format!("model listing returned {status}")),
                ));
            }

            response.json().map_err(|e| {
                (
                    Pushback::Retryable(None),
                    LlmError::provider(&self.id, format!("parsing model list: {e}")),
                )
            })
        })?;

        Ok(listing
            .data
            .into_iter()
            .map(|m| ModelInfo {
                id: m.id,
                provider: self.id.clone(),
            })
            .collect())
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let cut = s.char_indices().nth(max).map_or(s.len(), |(i, _)| i);
    format!("{}…", &s[..cut])
}

// ── Soma → wire ──

/// One Soma message can become several on the wire.
///
/// Tool *results* are their own `role: "tool"` messages in this shape, while
/// Soma keeps them as blocks inside the user turn they belong to — closer to
/// how the model actually sees a conversation, and the reason a call and its
/// result cannot drift apart in Soma's representation.
fn to_wire(message: &Message) -> Vec<WireMessage> {
    let mut out = Vec::new();
    let mut text = String::new();
    let mut tool_calls: Vec<WireToolCall> = Vec::new();

    for block in &message.content {
        match block {
            ContentBlock::Text { text: t } => text.push_str(t),
            ContentBlock::ToolUse { id, name, input } => tool_calls.push(WireToolCall {
                id: id.clone(),
                kind: "function".into(),
                function: WireFunctionCall {
                    name: name.clone(),
                    arguments: serde_json::to_string(input).unwrap_or_else(|_| "{}".into()),
                },
            }),
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                ..
            } => out.push(WireMessage {
                role: "tool".into(),
                content: Some(content.clone()),
                tool_calls: None,
                tool_call_id: Some(tool_use_id.clone()),
            }),
            _ => {}
        }
    }

    if !text.is_empty() || !tool_calls.is_empty() {
        let wire = WireMessage {
            role: match message.role {
                Role::System => "system",
                Role::User => "user",
                Role::Assistant => "assistant",
                _ => "user",
            }
            .into(),
            content: (!text.is_empty()).then_some(text),
            tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
            tool_call_id: None,
        };
        // Tool results must follow the assistant turn that requested them.
        out.insert(0, wire);
    }
    out
}

fn to_wire_tool(spec: &ToolSpec) -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.input_schema,
        }
    })
}

// ── Wire types ──

#[derive(Debug, Serialize)]
struct WireMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<WireToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

impl WireMessage {
    fn system(text: &str) -> Self {
        Self {
            role: "system".into(),
            content: Some(text.to_string()),
            tool_calls: None,
            tool_call_id: None,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct WireToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: String,
    function: WireFunctionCall,
}

#[derive(Debug, Serialize, Deserialize)]
struct WireFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct WireResponse {
    #[serde(default)]
    model: Option<String>,
    choices: Vec<WireChoice>,
    #[serde(default)]
    usage: Option<WireUsage>,
}

#[derive(Debug, Deserialize)]
struct WireChoice {
    message: WireResponseMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct WireResponseMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<WireToolCall>>,
}

#[derive(Debug, Default, Deserialize)]
struct WireUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
}

impl From<WireUsage> for Usage {
    fn from(u: WireUsage) -> Self {
        Self {
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
            ..Default::default()
        }
    }
}

#[derive(Debug, Deserialize)]
struct WireModels {
    #[serde(default)]
    data: Vec<WireModel>,
}

#[derive(Debug, Deserialize)]
struct WireModel {
    id: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::Quirks;

    fn provider(config: ProviderConfig) -> OpenAiCompatible {
        OpenAiCompatible::new("test", config).unwrap()
    }

    fn request() -> LlmRequest {
        LlmRequest::new("some-model", vec![Message::user("hello")].into())
    }

    #[test]
    fn a_plain_request_has_the_expected_shape() {
        let body = provider(ProviderConfig::local("http://x/v1")).body(&request());
        assert_eq!(body["model"], "some-model");
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"], "hello");
        assert!(body.get("tools").is_none(), "empty tools should be omitted");
    }

    #[test]
    fn the_system_prompt_is_its_own_message() {
        let req = request().with_system("be terse");
        let body = provider(ProviderConfig::local("http://x/v1")).body(&req);
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][0]["content"], "be terse");
        assert_eq!(body["messages"][1]["role"], "user");
    }

    /// Where an endpoint refuses a system role, the instruction is folded
    /// into the first user turn rather than silently dropped.
    #[test]
    fn a_system_prompt_survives_an_endpoint_without_a_system_role() {
        let config = ProviderConfig::local("http://x/v1").with_quirks(Quirks {
            system_as_message: false,
            ..Default::default()
        });
        let req = request().with_system("be terse");
        let body = provider(config).body(&req);

        assert_eq!(body["messages"][0]["role"], "user");
        let content = body["messages"][0]["content"].as_str().unwrap();
        assert!(content.contains("be terse"), "{content}");
        assert!(content.contains("hello"), "{content}");
    }

    #[test]
    fn the_max_tokens_field_can_be_renamed() {
        let config = ProviderConfig::local("http://x/v1").with_quirks(Quirks {
            max_tokens_field: "max_completion_tokens".into(),
            ..Default::default()
        });
        let body = provider(config).body(&request().with_max_tokens(100));
        assert_eq!(body["max_completion_tokens"], 100);
        assert!(body.get("max_tokens").is_none());
    }

    #[test]
    fn tools_are_omitted_for_an_endpoint_that_rejects_them() {
        let spec = ToolSpec {
            name: "search".into(),
            description: "search the web".into(),
            input_schema: serde_json::json!({"type": "object"}),
        };
        let req = request().with_tools(vec![spec]);

        let with_tools = provider(ProviderConfig::local("http://x/v1")).body(&req);
        assert_eq!(with_tools["tools"][0]["function"]["name"], "search");

        let config = ProviderConfig::local("http://x/v1").with_quirks(Quirks {
            supports_tools: false,
            ..Default::default()
        });
        assert!(provider(config).body(&req).get("tools").is_none());
    }

    /// A tool call and its result must stay paired and correctly ordered:
    /// the assistant turn that asks, then the `tool` message that answers.
    #[test]
    fn tool_calls_and_results_round_trip_into_wire_order() {
        let assistant = Message::new(
            Role::Assistant,
            vec![
                ContentBlock::text("let me look"),
                ContentBlock::tool_use("call_1", "search", serde_json::json!({"q": "soma"})),
            ],
        );
        let user = Message::new(
            Role::User,
            vec![ContentBlock::tool_result("call_1", "found it")],
        );

        let req = LlmRequest::new("m", vec![assistant, user].into());
        let body = provider(ProviderConfig::local("http://x/v1")).body(&req);
        let messages = body["messages"].as_array().unwrap();

        assert_eq!(messages[0]["role"], "assistant");
        assert_eq!(messages[0]["content"], "let me look");
        assert_eq!(messages[0]["tool_calls"][0]["id"], "call_1");
        assert_eq!(messages[0]["tool_calls"][0]["function"]["name"], "search");
        // Arguments go on the wire as a JSON string, not an object.
        assert_eq!(
            messages[0]["tool_calls"][0]["function"]["arguments"],
            r#"{"q":"soma"}"#
        );

        assert_eq!(messages[1]["role"], "tool");
        assert_eq!(messages[1]["tool_call_id"], "call_1");
        assert_eq!(messages[1]["content"], "found it");
    }

    #[test]
    fn a_reply_becomes_an_assistant_message() {
        let raw: WireResponse = serde_json::from_value(serde_json::json!({
            "model": "kimi-k2",
            "choices": [{
                "message": {"content": "the answer"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 12, "completion_tokens": 5}
        }))
        .unwrap();

        let parsed = provider(ProviderConfig::local("http://x/v1"))
            .parse(raw)
            .unwrap();

        assert_eq!(parsed.message.text(), "the answer");
        assert_eq!(parsed.stop_reason, StopReason::EndTurn);
        assert_eq!(parsed.usage.input_tokens, 12);
        assert_eq!(parsed.usage.output_tokens, 5);
        assert_eq!(parsed.model.as_deref(), Some("kimi-k2"));
    }

    #[test]
    fn a_tool_call_reply_reports_tool_use() {
        let raw: WireResponse = serde_json::from_value(serde_json::json!({
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "id": "call_9",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{\"q\":\"x\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }))
        .unwrap();

        let parsed = provider(ProviderConfig::local("http://x/v1"))
            .parse(raw)
            .unwrap();

        assert_eq!(parsed.stop_reason, StopReason::ToolUse);
        let calls: Vec<_> = parsed.message.tool_uses().collect();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "call_9");
        assert_eq!(calls[0].1, "search");
        assert_eq!(calls[0].2["q"], "x");
    }

    /// Models emit malformed tool arguments often enough that it must not
    /// abort the turn — hand the raw text to the step instead.
    #[test]
    fn malformed_tool_arguments_are_passed_through() {
        let raw: WireResponse = serde_json::from_value(serde_json::json!({
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "c",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{not json"}
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }))
        .unwrap();

        let parsed = provider(ProviderConfig::local("http://x/v1"))
            .parse(raw)
            .unwrap();
        let calls: Vec<_> = parsed.message.tool_uses().collect();
        assert_eq!(calls[0].2["_raw"], "{not json");
    }

    #[test]
    fn finish_reasons_map_to_stop_reasons() {
        for (wire, expected) in [
            ("stop", StopReason::EndTurn),
            ("length", StopReason::MaxTokens),
            ("tool_calls", StopReason::ToolUse),
        ] {
            let raw: WireResponse = serde_json::from_value(serde_json::json!({
                "choices": [{"message": {"content": "x"}, "finish_reason": wire}]
            }))
            .unwrap();
            let parsed = provider(ProviderConfig::local("http://x/v1"))
                .parse(raw)
                .unwrap();
            assert_eq!(parsed.stop_reason, expected, "for finish_reason {wire}");
        }
    }

    #[test]
    fn a_reply_with_no_choices_is_an_error() {
        let raw: WireResponse = serde_json::from_value(serde_json::json!({"choices": []})).unwrap();
        assert!(
            provider(ProviderConfig::local("http://x/v1"))
                .parse(raw)
                .is_err()
        );
    }
}
