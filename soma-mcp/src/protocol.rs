//! MCP JSON-RPC 2.0 protocol types.

use serde::{Deserialize, Serialize};

/// One JSON-RPC 2.0 request, as read line-by-line off stdin.
#[derive(Debug, Deserialize)]
pub struct JsonRpcRequest {
    /// Protocol version marker; `"2.0"` for every client we speak to.
    pub jsonrpc: String,
    /// Request id to echo back in the response. JSON-RPC allows a
    /// string, a number or null here, so it stays an opaque
    /// [`serde_json::Value`] rather than committing to one shape.
    pub id: serde_json::Value,
    /// The method being invoked — `initialize`, `tools/list`,
    /// `tools/call`, ...
    pub method: String,
    /// Method parameters. Defaults to `Value::Null` because clients may
    /// omit the field entirely for parameterless methods.
    #[serde(default)]
    pub params: serde_json::Value,
}

/// One JSON-RPC 2.0 response, written as a single line to stdout.
///
/// Exactly one of `result` and `error` is set; the [`success`] and
/// [`error`] constructors are the only ways this crate builds one, so
/// the invariant holds by construction.
///
/// [`success`]: JsonRpcResponse::success
/// [`error`]: JsonRpcResponse::error
#[derive(Debug, Serialize)]
pub struct JsonRpcResponse {
    /// Protocol version marker, always `"2.0"`.
    pub jsonrpc: String,
    /// The id of the request this answers, echoed verbatim.
    pub id: serde_json::Value,
    /// The successful payload; absent (not null) on failure.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    /// The failure; absent on success.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

/// The `error` member of a failed [`JsonRpcResponse`].
///
/// Protocol-level failure (unknown method, bad params, panic). A *tool*
/// that fails still returns a successful response carrying a
/// [`ToolCallResult`] with `is_error: true` — that distinction is MCP's,
/// not ours: the model sees tool errors, the client sees protocol ones.
#[derive(Debug, Serialize)]
pub struct JsonRpcError {
    /// Numeric error code; see [`METHOD_NOT_FOUND`], [`INVALID_PARAMS`],
    /// [`INTERNAL_ERROR`].
    pub code: i64,
    /// Human-readable description of what went wrong.
    pub message: String,
    /// Optional structured detail; this server never sets it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

impl JsonRpcResponse {
    /// A successful response carrying `result` for request `id`.
    pub fn success(id: serde_json::Value, result: serde_json::Value) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id,
            result: Some(result),
            error: None,
        }
    }

    /// A failed response for request `id` with the given code and
    /// message; `data` is left unset.
    pub fn error(id: serde_json::Value, code: i64, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.into(),
                data: None,
            }),
        }
    }
}

/// The server's half of the MCP `initialize` handshake.
#[derive(Debug, Serialize)]
pub struct InitializeResult {
    /// MCP protocol revision the server speaks (a date string,
    /// e.g. `"2024-11-05"`).
    #[serde(rename = "protocolVersion")]
    pub protocol_version: String,
    /// What the server can do — see [`ServerCapabilities`].
    pub capabilities: ServerCapabilities,
    /// Who is answering — see [`ServerInfo`].
    #[serde(rename = "serverInfo")]
    pub server_info: ServerInfo,
}

/// What this server offers a client.
///
/// Tools only: soma-mcp serves no resources and no prompts, so those
/// capability fields do not exist here — MCP treats an absent field as
/// "not supported", which is exactly the claim.
#[derive(Debug, Serialize)]
pub struct ServerCapabilities {
    /// The tools capability; the 20 tools are the whole API.
    pub tools: ToolsCapability,
}

/// Details of the tools capability advertised in the handshake.
#[derive(Debug, Serialize)]
pub struct ToolsCapability {
    /// Whether the server emits `tools/list_changed` notifications.
    /// This server's tool list is fixed at compile time, so it
    /// advertises `false` and a client need never re-fetch the list.
    #[serde(rename = "listChanged")]
    pub list_changed: bool,
}

/// Server identity reported in the `initialize` handshake.
#[derive(Debug, Serialize)]
pub struct ServerInfo {
    /// Server name (`"soma-mcp"`).
    pub name: String,
    /// Crate version, taken from `CARGO_PKG_VERSION` so it cannot drift
    /// from the release.
    pub version: String,
}

/// What this server publishes, and what an agent consumes.
///
/// The same type either way: [`somatize_core::agentic::tool::ToolSpec`]. Soma is both
/// a tool provider (here) and a tool caller (`soma-llm`), and describing a
/// tool twice is how the two descriptions drift.
pub use somatize_core::agentic::tool::ToolSpec as ToolDefinition;

/// What a `tools/call` returns: the text a model will read.
///
/// Every handler in [`crate::context`] produces one of these, and the
/// renderers in [`crate::render`] decide what goes in it — the text IS
/// the API (each experiment-pool result ends with a `next:` line and
/// carries its `run_dir:`), so this type stays a thin envelope.
#[derive(Debug, Serialize)]
pub struct ToolCallResult {
    /// The content items, concatenated by [`content_text`] when a
    /// caller wants the single string a model sees.
    ///
    /// [`content_text`]: ToolCallResult::content_text
    pub content: Vec<ContentItem>,
    /// `Some(true)` when the tool failed. MCP keeps tool failure inside
    /// a *successful* response — the model reads the error text and can
    /// react to it, unlike a protocol-level [`JsonRpcError`].
    #[serde(rename = "isError", skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
}

/// One block of tool-result content.
///
/// This server only ever emits text, so the type is not an enum: a
/// `type` tag plus the text is the whole story.
#[derive(Debug, Serialize)]
pub struct ContentItem {
    /// MCP content discriminator; always `"text"` here.
    #[serde(rename = "type")]
    pub content_type: String,
    /// The text itself.
    pub text: String,
}

impl ContentItem {
    /// A text content block.
    pub fn text(s: impl Into<String>) -> Self {
        Self {
            content_type: "text".into(),
            text: s.into(),
        }
    }
}

impl ToolCallResult {
    /// A successful result carrying one text block.
    pub fn text(s: impl Into<String>) -> Self {
        Self {
            content: vec![ContentItem::text(s)],
            is_error: None,
        }
    }

    /// A failed result: the same text block, flagged `isError` so the
    /// model knows it is reading a failure, not an answer.
    pub fn error(s: impl Into<String>) -> Self {
        Self {
            content: vec![ContentItem::text(s)],
            is_error: Some(true),
        }
    }

    /// The text a model would see — every content item concatenated.
    /// The protocol carries text, so this is the whole result.
    pub fn content_text(&self) -> String {
        self.content
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Whether this result is an error; an absent flag means success.
    pub fn is_error(&self) -> bool {
        self.is_error.unwrap_or(false)
    }
}

/// JSON-RPC 2.0 spec code: the requested method does not exist. What
/// the server answers for any method it does not recognise.
pub const METHOD_NOT_FOUND: i64 = -32601;
/// JSON-RPC 2.0 spec code: the method exists but the parameters are
/// invalid. Currently unsent — a missing tool argument is reported as a
/// [`ToolCallResult::error`] instead, so the *model* sees it and can
/// retry with the argument filled in.
pub const INVALID_PARAMS: i64 = -32602;
/// JSON-RPC 2.0 spec code: the server itself failed. Currently unsent;
/// kept beside its siblings so a future handler does not reinvent the
/// number.
pub const INTERNAL_ERROR: i64 = -32603;
