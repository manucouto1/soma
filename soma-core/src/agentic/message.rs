//! The canonical conversation shape.
//!
//! One definition, in core, so that every layer agrees: a Python step, the
//! Rust provider, the schema validator, the journal, and the report renderer.
//! Provider-specific wire formats are converted at the edge (in `soma-llm`),
//! never leaked inwards.
//!
//! This exists because of what the failure data says. Across 1600+ annotated
//! multi-agent traces (MAST, NeurIPS 2025), ~37% of failures are inter-agent
//! misalignment — context lost at a handoff, formats that don't line up.
//! A shared message type plus a schema that can say "this edge carries
//! messages" moves a chunk of that from runtime surprise to compile error.

use crate::data::value::Value;
use crate::error::{Result, SomaError};
use serde::{Deserialize, Serialize};

/// Who produced a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Role {
    /// Instructions that frame the conversation.
    System,
    /// The human — or the calling program — side of the exchange, including
    /// tool results, which return to the model as user-role turns.
    User,
    /// The model's own turns.
    Assistant,
}

impl Role {
    /// The wire spelling — the same lowercase token serde reads and writes.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One piece of a message's content.
///
/// A message is a *list* of blocks rather than a string because a single
/// assistant turn routinely mixes prose with tool calls, and a user turn
/// mixes prose with tool results. Flattening that to text loses the pairing
/// between a call and its result — which is one of the concrete ways a
/// handoff drops context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ContentBlock {
    /// Plain prose.
    Text {
        /// The prose itself.
        text: String,
    },

    /// The model asking for a tool to be run.
    ToolUse {
        /// Correlates with the matching [`ContentBlock::ToolResult`].
        id: String,
        /// Which tool — a [`crate::agentic::tool::ToolSpec::name`].
        name: String,
        /// Arguments, shaped by the tool's declared schema.
        input: serde_json::Value,
    },

    /// The answer to a [`ContentBlock::ToolUse`].
    ToolResult {
        /// The `id` of the [`ContentBlock::ToolUse`] this answers.
        tool_use_id: String,
        /// The tool's output — or its error text — as the model will read it.
        content: String,
        /// The tool failed and `content` is its error text.
        #[serde(default)]
        is_error: bool,
    },
}

impl ContentBlock {
    /// A prose block.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    /// A tool call: run `name` with `input`; `id` pairs it with its result.
    pub fn tool_use(
        id: impl Into<String>,
        name: impl Into<String>,
        input: serde_json::Value,
    ) -> Self {
        Self::ToolUse {
            id: id.into(),
            name: name.into(),
            input,
        }
    }

    /// A successful tool result, answering the call identified by `tool_use_id`.
    pub fn tool_result(tool_use_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self::ToolResult {
            tool_use_id: tool_use_id.into(),
            content: content.into(),
            is_error: false,
        }
    }

    /// A failed tool call. Reported to the model rather than raised, so it can
    /// adapt — a tool that errors is information, not the end of the turn.
    pub fn tool_error(tool_use_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self::ToolResult {
            tool_use_id: tool_use_id.into(),
            content: content.into(),
            is_error: true,
        }
    }

    /// The prose in this block, if it is prose.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text { text } => Some(text),
            _ => None,
        }
    }
}

/// One turn in a conversation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    /// Who produced this turn.
    pub role: Role,
    /// The turn's blocks, in order.
    pub content: Vec<ContentBlock>,
}

impl Message {
    /// A turn with an explicit role and block list.
    pub fn new(role: Role, content: Vec<ContentBlock>) -> Self {
        Self { role, content }
    }

    /// A system turn holding one prose block.
    pub fn system(text: impl Into<String>) -> Self {
        Self::new(Role::System, vec![ContentBlock::text(text)])
    }

    /// A user turn holding one prose block — the common case.
    pub fn user(text: impl Into<String>) -> Self {
        Self::new(Role::User, vec![ContentBlock::text(text)])
    }

    /// An assistant turn holding one prose block.
    pub fn assistant(text: impl Into<String>) -> Self {
        Self::new(Role::Assistant, vec![ContentBlock::text(text)])
    }

    /// Concatenate this turn's prose, dropping tool blocks.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(ContentBlock::as_text)
            .collect::<Vec<_>>()
            .join("")
    }

    /// The tool calls this turn is asking for.
    pub fn tool_uses(&self) -> impl Iterator<Item = (&str, &str, &serde_json::Value)> {
        self.content.iter().filter_map(|b| match b {
            ContentBlock::ToolUse { id, name, input } => Some((id.as_str(), name.as_str(), input)),
            _ => None,
        })
    }
}

/// A conversation: an ordered list of turns.
///
/// Carried between nodes as a [`Value::Json`] under this exact shape, so any
/// consumer — Rust, Python, the report renderer — reads the same structure.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Messages(pub Vec<Message>);

impl Messages {
    /// An empty conversation.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a turn.
    pub fn push(&mut self, message: Message) {
        self.0.push(message);
    }

    /// How many turns so far.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether no turn has been added yet.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Iterate over the turns, oldest first.
    pub fn iter(&self) -> std::slice::Iter<'_, Message> {
        self.0.iter()
    }

    /// The most recent turn, if any.
    pub fn last(&self) -> Option<&Message> {
        self.0.last()
    }

    /// Encode as the `Value` that travels along an edge.
    pub fn to_value(&self) -> Value {
        Value::json(serde_json::to_value(self).unwrap_or(serde_json::Value::Null))
    }

    /// Read a conversation off an edge.
    ///
    /// Accepts three shapes, in decreasing specificity: a full message list,
    /// a bare string (promoted to a single user turn), and a `Value::Text`
    /// (likewise). The promotions exist so a plain prompt can feed a node
    /// expecting a conversation without ceremony — the common first hop.
    pub fn from_value(value: &Value) -> Result<Self> {
        match value {
            Value::Text(s) => Ok(Self(vec![Message::user(s.as_ref())])),
            Value::Json(j) => {
                if let Some(s) = j.as_str() {
                    return Ok(Self(vec![Message::user(s)]));
                }
                serde_json::from_value((**j).clone()).map_err(|e| SomaError::SchemaMismatch {
                    expected: "messages".into(),
                    got: format!("json that is not a conversation: {e}"),
                })
            }
            other => Err(SomaError::SchemaMismatch {
                expected: "messages".into(),
                got: other.type_name().to_string(),
            }),
        }
    }
}

impl From<Vec<Message>> for Messages {
    fn from(v: Vec<Message>) -> Self {
        Self(v)
    }
}

impl IntoIterator for Messages {
    type Item = Message;
    type IntoIter = std::vec::IntoIter<Message>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips_through_a_value() {
        let mut msgs = Messages::new();
        msgs.push(Message::system("You are terse."));
        msgs.push(Message::user("What is 2+2?"));
        msgs.push(Message::new(
            Role::Assistant,
            vec![
                ContentBlock::text("Let me compute that."),
                ContentBlock::tool_use("t1", "calc", serde_json::json!({"expr": "2+2"})),
            ],
        ));
        msgs.push(Message::new(
            Role::User,
            vec![ContentBlock::tool_result("t1", "4")],
        ));

        let decoded = Messages::from_value(&msgs.to_value()).unwrap();
        assert_eq!(decoded, msgs);
    }

    /// A bare prompt should feed a conversation-shaped node without the
    /// caller having to build a message list first.
    #[test]
    fn promotes_a_bare_string_to_a_user_turn() {
        for v in [
            Value::text("Summarize this."),
            Value::json(serde_json::json!("Summarize this.")),
        ] {
            let msgs = Messages::from_value(&v).unwrap();
            assert_eq!(msgs.len(), 1);
            assert_eq!(msgs.0[0].role, Role::User);
            assert_eq!(msgs.0[0].text(), "Summarize this.");
        }
    }

    #[test]
    fn rejects_values_that_are_not_conversations() {
        let err = Messages::from_value(&Value::tensor(vec![1.0], vec![1])).unwrap_err();
        assert!(err.to_string().contains("messages"), "{err}");

        let err = Messages::from_value(&Value::json(serde_json::json!({"a": 1}))).unwrap_err();
        assert!(err.to_string().contains("messages"), "{err}");
    }

    #[test]
    fn text_concatenates_prose_and_skips_tool_blocks() {
        let m = Message::new(
            Role::Assistant,
            vec![
                ContentBlock::text("a"),
                ContentBlock::tool_use("t", "n", serde_json::json!({})),
                ContentBlock::text("b"),
            ],
        );
        assert_eq!(m.text(), "ab");
        assert_eq!(m.tool_uses().count(), 1);
    }

    #[test]
    fn tool_errors_are_marked() {
        let ok = ContentBlock::tool_result("t", "fine");
        let bad = ContentBlock::tool_error("t", "boom");
        assert!(matches!(
            ok,
            ContentBlock::ToolResult {
                is_error: false,
                ..
            }
        ));
        assert!(matches!(
            bad,
            ContentBlock::ToolResult { is_error: true, .. }
        ));
    }
}
