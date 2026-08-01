//! Effects: the things a step can ask the runtime to do for it.
//!
//! A [`crate::filter::Filter`] computes. A step *decides*, and the deciding
//! needs the world: a model call, a tool, another graph. Rather than let a
//! step reach out and do that itself, it **describes** what it wants and
//! hands the description back. The runtime performs it.
//!
//! That indirection buys three things that are otherwise each a project of
//! their own:
//!
//! - **Durability.** Every performed effect is journaled by
//!   `(node, turn, effect hash)`. Replaying a run re-polls the step and
//!   serves recorded results instead of re-calling the model. This is the
//!   record-once-replay discipline of durable-execution engines, on top of
//!   the content-addressed store Soma already has.
//! - **A sync trait over async work.** The step never awaits; the driver
//!   does. No coloured functions, and the Python bridge stays a plain call.
//! - **Testability.** A fake effect handler is a `match` — no network, no
//!   mocking framework.

use crate::cache::CacheKey;
use crate::graph::{Graph, NodeId};
use crate::message::Messages;
use crate::value::Value;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Something the runtime does on a step's behalf.
// Adjacent tagging, matching `Value`: an internal tag would collide both with
// the `Custom { kind }` field and with the tag `Value` carries in its own
// payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", content = "data", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Effect {
    /// Call a language model.
    Llm(LlmRequest),

    /// Invoke a registered tool.
    Tool { name: String, args: Value },

    /// Run a Soma graph and hand back its output.
    ///
    /// This is the bridge that makes a computational pipeline a first-class
    /// tool for an agent: it runs through the ordinary compiler and executor,
    /// with the ordinary cache, and the agent just sees a result.
    Graph {
        graph: Box<Graph>,
        input: Value,
        #[serde(default)]
        mode: GraphEffectMode,
    },

    /// Wait. Journaled like anything else, so a replay does not sleep again.
    Sleep(Duration),

    /// An effect this runtime doesn't know about, for host-supplied handlers.
    Custom { kind: String, payload: Value },
}

impl Effect {
    /// A short label for events and logs. Never includes the payload — a
    /// prompt is not something to leak into a log line.
    pub fn label(&self) -> String {
        match self {
            Self::Llm(req) => format!("llm:{}", req.model),
            Self::Tool { name, .. } => format!("tool:{name}"),
            Self::Graph { graph, .. } => format!("graph:{} nodes", graph.nodes.len()),
            Self::Sleep(d) => format!("sleep:{}ms", d.as_millis()),
            Self::Custom { kind, .. } => format!("custom:{kind}"),
        }
    }

    /// Is this effect safe to memoize by content?
    ///
    /// Pure effects are cached like any filter output: same input, same
    /// result, reused forever. Impure ones are still journaled — recorded
    /// once per `(node, turn)` so a replay is faithful — but never reused
    /// across runs, because "the same question asked twice" is genuinely a
    /// different event for a model call or a clock read.
    pub fn is_pure(&self) -> bool {
        match self {
            Self::Llm(_) | Self::Sleep(_) => false,
            // A graph effect inherits Soma's own determinism guarantees:
            // its nodes are cached by content already.
            Self::Graph { .. } => true,
            Self::Tool { .. } | Self::Custom { .. } => false,
        }
    }

    /// Content hash, used as the journal key for this effect.
    pub fn cache_key(&self) -> CacheKey {
        let encoded = serde_json::to_vec(self).unwrap_or_default();
        CacheKey::from_parts(&[b"soma-effect-v1", &encoded])
    }
}

/// What to do with a graph run as an effect.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum GraphEffectMode {
    /// Run `forward` with whatever states are already fitted.
    #[default]
    Forward,
    /// Fit the graph on the supplied input first.
    Fit,
}

/// A model call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmRequest {
    pub model: String,
    pub messages: Messages,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Tools the model may call, as JSON Schema definitions.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolSpec>,
    /// Reasoning depth, in the provider's terms (`low` … `max`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
    /// JSON Schema the reply must satisfy.
    ///
    /// A request, not a guarantee: endpoints that support constrained
    /// decoding enforce it, and the rest are asked in the prompt and may
    /// still answer with prose. Whoever consumes the reply validates it —
    /// see [`crate::schema::Schema`] for the graph-level contract, which is
    /// a different thing: this constrains one model call, that one
    /// constrains an edge.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schema: Option<serde_json::Value>,
}

impl LlmRequest {
    pub fn new(model: impl Into<String>, messages: Messages) -> Self {
        Self {
            model: model.into(),
            messages,
            system: None,
            max_tokens: None,
            tools: Vec::new(),
            effort: None,
            schema: None,
        }
    }

    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    pub fn with_max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = Some(n);
        self
    }

    pub fn with_tools(mut self, tools: Vec<ToolSpec>) -> Self {
        self.tools = tools;
        self
    }

    pub fn with_effort(mut self, effort: impl Into<String>) -> Self {
        self.effort = Some(effort.into());
        self
    }

    /// Ask for a reply shaped like `schema`.
    pub fn with_schema(mut self, schema: serde_json::Value) -> Self {
        self.schema = Some(schema);
        self
    }

    /// Wrap as the effect a step awaits.
    pub fn into_effect(self) -> Effect {
        Effect::Llm(self)
    }
}

pub use crate::tool::ToolSpec;

/// Performs one kind of effect.
///
/// Handlers are tried in order; the first that claims an effect wins.
///
/// Lives here rather than beside the driver that runs them, so a crate can
/// implement a handler without depending on the execution engine —
/// `soma-llm` is one, and what it needs is a provider client, not a
/// scheduler.
pub trait EffectHandler: Send + Sync {
    /// Will this handler take that effect?
    fn handles(&self, effect: &Effect) -> bool;

    /// Perform it. Blocking.
    ///
    /// A transport failure should come back as [`EffectResult::Failed`],
    /// not `Err` — the step decides whether to retry, fall back, or stop.
    /// Reserve `Err` for conditions the step cannot act on.
    fn perform(&self, effect: &Effect) -> crate::error::Result<EffectResult>;
}

/// What came back from performing an effect.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", content = "data", rename_all = "snake_case")]
#[non_exhaustive]
pub enum EffectResult {
    Llm(LlmResponse),
    /// A tool's output, or its error text.
    Tool {
        output: Value,
        #[serde(default)]
        is_error: bool,
    },
    Graph(Value),
    /// What a node spawned by [`crate::step::Transition::Spawn`] produced.
    Node(Value),
    Slept,
    Custom(Value),
    /// The effect could not be performed. Handed to the step rather than
    /// raised, so it can retry, fall back, or give up deliberately.
    Failed {
        message: String,
    },
}

impl EffectResult {
    pub fn is_error(&self) -> bool {
        matches!(
            self,
            Self::Failed { .. } | Self::Tool { is_error: true, .. }
        )
    }

    /// The value this result carries, if it carries one.
    pub fn value(&self) -> Option<&Value> {
        match self {
            Self::Tool { output, .. }
            | Self::Graph(output)
            | Self::Node(output)
            | Self::Custom(output) => Some(output),
            _ => None,
        }
    }
}

/// A model's reply.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResponse {
    /// The assistant turn, blocks intact — prose and tool calls together.
    pub message: crate::message::Message,
    pub stop_reason: StopReason,
    #[serde(default)]
    pub usage: Usage,
    /// Which model actually served this, which may differ from the one asked
    /// for when a provider-side fallback kicked in.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

/// Why the model stopped.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    /// The model wants tools run before it continues.
    ToolUse,
    /// The provider declined. Carried as a value, not an error: the step
    /// decides whether to rephrase, fall back, or surface it.
    Refusal {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        category: Option<String>,
    },
}

/// Token accounting for one call.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,
    #[serde(default)]
    pub cache_read_tokens: u64,
    #[serde(default)]
    pub cache_write_tokens: u64,
}

impl Usage {
    pub fn total(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }
}

impl std::ops::AddAssign for Usage {
    fn add_assign(&mut self, rhs: Self) {
        self.input_tokens += rhs.input_tokens;
        self.output_tokens += rhs.output_tokens;
        self.cache_read_tokens += rhs.cache_read_tokens;
        self.cache_write_tokens += rhs.cache_write_tokens;
    }
}

/// A node to create while the run is in flight.
///
/// The unit of dynamic fan-out: a step that discovers it has N things to do
/// emits N specs, and the runtime runs them. There is no way to pre-declare
/// this in a static plan, which is why an orchestrator-workers shape cannot
/// be expressed by topology alone.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeSpec {
    /// Which registered step or filter to run.
    pub runs: NodeId,
    /// The input it receives.
    pub input: Value,
    /// Suffix distinguishing this instance from its siblings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

impl NodeSpec {
    pub fn new(runs: impl Into<NodeId>, input: Value) -> Self {
        Self {
            runs: runs.into(),
            input,
            label: None,
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

/// How spawned work is recombined.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum JoinPolicy {
    /// Wait for all; a failure fails the join.
    #[default]
    All,
    /// Wait for all, keeping whatever succeeded.
    AllSettled,
    /// Take the first success and cancel the rest.
    First,
}

/// Why a run stopped and what would restart it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum SuspendReason {
    /// Waiting on a person.
    Human {
        prompt: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        schema: Option<serde_json::Value>,
    },
    /// Waiting on something outside the run entirely.
    External { token: String },
}

impl SuspendReason {
    /// A short description, for an error message or an event payload.
    pub fn label(&self) -> String {
        match self {
            Self::Human { prompt, .. } => format!("waiting on a person: {prompt}"),
            Self::External { token } => format!("waiting on `{token}`"),
        }
    }

    /// The kind, as a stable token events can be grouped by.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Human { .. } => "human",
            Self::External { .. } => "external",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Message;

    fn llm() -> Effect {
        Effect::Llm(LlmRequest::new(
            "claude-opus-5",
            vec![Message::user("hi")].into(),
        ))
    }

    /// The journal key must follow the request's content, or a replay would
    /// serve one call's answer to a different call.
    #[test]
    fn effect_keys_follow_content() {
        let a = llm();
        let b = Effect::Llm(LlmRequest::new(
            "claude-opus-5",
            vec![Message::user("something else")].into(),
        ));
        assert_eq!(a.cache_key(), llm().cache_key());
        assert_ne!(a.cache_key(), b.cache_key());
    }

    /// Changing any request knob must change the key — otherwise a replay at
    /// a different effort would reuse the cheaper answer.
    #[test]
    fn request_options_are_part_of_the_key() {
        let base = LlmRequest::new("claude-opus-5", vec![Message::user("hi")].into());
        let keys = [
            Effect::Llm(base.clone()).cache_key(),
            Effect::Llm(base.clone().with_system("be terse")).cache_key(),
            Effect::Llm(base.clone().with_effort("high")).cache_key(),
            Effect::Llm(base.with_max_tokens(10)).cache_key(),
        ];
        for (i, a) in keys.iter().enumerate() {
            for b in &keys[i + 1..] {
                assert_ne!(a, b, "two distinct requests share a journal key");
            }
        }
    }

    /// Model calls must never be reused by content: asking twice is two
    /// events, and freezing the first answer is the `_deterministic=False`
    /// foot-gun in a new costume.
    #[test]
    fn model_calls_are_impure() {
        assert!(!llm().is_pure());
        assert!(!Effect::Sleep(Duration::from_secs(1)).is_pure());
        assert!(
            !Effect::Tool {
                name: "search".into(),
                args: Value::Empty
            }
            .is_pure()
        );
    }

    #[test]
    fn labels_do_not_leak_payloads() {
        let label = llm().label();
        assert!(label.contains("claude-opus-5"));
        assert!(!label.contains("hi"));
    }

    #[test]
    fn usage_accumulates() {
        let mut total = Usage::default();
        total += Usage {
            input_tokens: 10,
            output_tokens: 5,
            ..Default::default()
        };
        total += Usage {
            input_tokens: 1,
            output_tokens: 2,
            ..Default::default()
        };
        assert_eq!(total.input_tokens, 11);
        assert_eq!(total.total(), 18);
    }

    #[test]
    fn failed_and_errored_tools_read_as_errors() {
        assert!(
            EffectResult::Failed {
                message: "boom".into()
            }
            .is_error()
        );
        assert!(
            EffectResult::Tool {
                output: Value::text("nope"),
                is_error: true
            }
            .is_error()
        );
        assert!(
            !EffectResult::Tool {
                output: Value::text("fine"),
                is_error: false
            }
            .is_error()
        );
    }
}
