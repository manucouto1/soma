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

use crate::catalog::ProviderConfig;
use crate::{LlmProvider, ModelInfo};
use serde::{Deserialize, Serialize};
use somatize_core::effect::{LlmRequest, LlmResponse, StopReason, ToolSpec, Usage};
use somatize_core::error::{Result, SomaError};
use somatize_core::message::{ContentBlock, Message, Role};
use std::time::Duration;

/// A provider reached over the OpenAI chat-completions shape.
pub struct OpenAiCompatible {
    id: String,
    config: ProviderConfig,
    client: reqwest::blocking::Client,
}

impl OpenAiCompatible {
    pub fn new(id: impl Into<String>, config: ProviderConfig) -> Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| SomaError::Other(format!("building http client: {e}")))?;
        Ok(Self {
            id: id.into(),
            config,
            client,
        })
    }

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

        let mut body = serde_json::json!({
            "model": self.config.wire_model(&req.model),
            "messages": messages,
        });

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
        let choice = raw.choices.into_iter().next().ok_or_else(|| {
            SomaError::Other(format!("provider `{}` returned no choices", self.id))
        })?;

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

    fn complete(&self, req: &LlmRequest) -> Result<LlmResponse> {
        let url = self.config.chat_url();
        let response = self
            .request(reqwest::Method::POST, &url)?
            .json(&self.body(req))
            .send()
            .map_err(|e| SomaError::Other(format!("`{}` at {url}: {e}", self.id)))?;

        let status = response.status();
        let text = response
            .text()
            .map_err(|e| SomaError::Other(format!("`{}`: reading response: {e}", self.id)))?;

        if !status.is_success() {
            // Endpoints bury the useful part in differently-shaped error
            // envelopes; show the body rather than guess at its schema.
            return Err(SomaError::Other(format!(
                "`{}` returned {status}: {}",
                self.id,
                text.trim()
            )));
        }

        let raw: WireResponse = serde_json::from_str(&text).map_err(|e| {
            SomaError::Other(format!(
                "`{}` returned a body that is not a chat completion: {e}. Body: {}",
                self.id,
                truncate(&text, 400)
            ))
        })?;
        self.parse(raw)
    }

    fn models(&self) -> Result<Vec<ModelInfo>> {
        let url = self.config.models_url();
        let response = self
            .request(reqwest::Method::GET, &url)?
            .send()
            .map_err(|e| SomaError::Other(format!("`{}` at {url}: {e}", self.id)))?;

        if !response.status().is_success() {
            return Err(SomaError::Other(format!(
                "`{}` model listing returned {}",
                self.id,
                response.status()
            )));
        }
        let listing: WireModels = response
            .json()
            .map_err(|e| SomaError::Other(format!("`{}`: parsing model list: {e}", self.id)))?;

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
