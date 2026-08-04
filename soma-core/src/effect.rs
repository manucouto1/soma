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
    Tool {
        /// The name the tool was registered under — a [`ToolSpec::name`].
        name: String,
        /// Arguments, as JSON shaped by the tool's [`ToolSpec::input_schema`].
        args: Value,
    },

    /// Run a Soma graph and hand back its output.
    ///
    /// This is the bridge that makes a computational pipeline a first-class
    /// tool for an agent: it runs through the ordinary compiler and executor,
    /// with the ordinary cache, and the agent just sees a result.
    Graph {
        /// The graph to run. Structure travels in the effect itself, so the
        /// handler needs no registry lookup to know what it is performing —
        /// only the node *implementations* are resolved on the other side.
        graph: Box<Graph>,
        /// The input handed to the sub-graph's root nodes.
        input: Value,
        /// Forward with the states already fitted, or fit first.
        #[serde(default)]
        mode: GraphEffectMode,
    },

    /// Wait. Journaled like anything else, so a replay does not sleep again.
    Sleep(Duration),

    /// An effect this runtime doesn't know about, for host-supplied handlers.
    Custom {
        /// Which handler this is meant for — the string
        /// [`EffectHandler::handles`] implementations match on.
        kind: String,
        /// Whatever that handler expects.
        payload: Value,
    },
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
            // A filter-only forward inherits Soma's own determinism
            // guarantees: its nodes are cached by content already. A graph
            // that contains a step does not — the step calls a model, so the
            // same question asked twice is a different event, like `Llm`
            // itself. `Fit` is impure for a second reason: replaying it must
            // re-write the fitted states, and serving the recorded summary
            // alone would leave the graph unfitted for the effects after it.
            Self::Graph { graph, mode, .. } => {
                matches!(mode, GraphEffectMode::Forward) && !graph.contains_steps()
            }
            Self::Tool { .. } | Self::Custom { .. } => false,
        }
    }

    /// Content hash, used as the journal key for this effect.
    ///
    /// Canonical CBOR, not raw serializer output: an effect carries
    /// user-supplied JSON (`Tool { args }`, `Custom { payload }`), and two
    /// encodings of the same object must not produce two journal keys.
    ///
    /// Fallible on purpose. The previous encoding fell back to empty bytes,
    /// which gave *every* unserializable effect the same key — a replay
    /// would then serve one effect's recorded result to another. A key that
    /// cannot be computed has to mean "not journalable", never a guess.
    pub fn cache_key(&self) -> crate::error::Result<CacheKey> {
        let encoded = crate::canon::canonical_bytes(self)?;
        Ok(CacheKey::from_parts(&[b"soma-effect-v2", &encoded]))
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
    /// Which model to ask, in the provider's naming.
    pub model: String,
    /// The conversation so far, oldest turn first.
    pub messages: Messages,
    /// System prompt, kept apart from `messages` because providers disagree
    /// about where it goes — a `system` turn, a top-level field, or folded
    /// into the first user turn. The client places it at its edge.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// Cap on generated tokens. A reply that hits it stops with
    /// [`StopReason::MaxTokens`], which steps treat as an error, not an answer.
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
    /// A request with only the essentials: a model and a conversation.
    ///
    /// Everything else is opt-in through the `with_*` builders — and because
    /// unset options are skipped when the request is serialized, they are
    /// also absent from the journal key ([`Effect::cache_key`]). Growing this
    /// struct therefore never moves the keys of requests that predate the
    /// new knob.
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

    /// Set the system prompt.
    ///
    /// Stated here once, placed by the provider client wherever this
    /// endpoint wants it — as its own turn, or prepended to the first user
    /// turn when the endpoint refuses a system role.
    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Cap the reply at `n` generated tokens.
    ///
    /// A budget, not a target: a reply that runs into it is a cut-off
    /// thought, and [`LlmResponse::reject_non_answers`] turns it into an
    /// error rather than letting the fragment flow downstream as an answer.
    pub fn with_max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = Some(n);
        self
    }

    /// Offer the model these tools, replacing any previous set.
    ///
    /// Offering is all this does. When the model wants one, the reply stops
    /// with [`StopReason::ToolUse`] and it is the step's job to perform the
    /// calls — typically as [`Effect::Tool`] — and ask the model to continue
    /// with the results appended to the conversation.
    pub fn with_tools(mut self, tools: Vec<ToolSpec>) -> Self {
        self.tools = tools;
        self
    }

    /// Ask for this much reasoning, in the provider's vocabulary
    /// (`low` … `max`).
    ///
    /// Passed through verbatim (`reasoning_effort` on OpenAI-shaped
    /// endpoints); endpoints without the concept ignore the unknown field.
    /// Like every other knob it is part of the journal key — the same
    /// question at a different effort is a different request, so a replay
    /// never serves the cheaper answer.
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
    /// A model's reply to [`Effect::Llm`].
    Llm(LlmResponse),
    /// A tool's output, or its error text.
    Tool {
        /// What the tool produced — its error text when `is_error` is set.
        output: Value,
        /// The tool ran and reported failure. Reported, not raised, so the
        /// step (or the model, told via a tool-result block) can adapt.
        #[serde(default)]
        is_error: bool,
    },
    /// The output of the sub-graph run requested by [`Effect::Graph`].
    Graph(Value),
    /// What a node spawned by [`crate::step::Transition::Spawn`] produced.
    Node(Value),
    /// The sleep elapsed — or was replayed from the journal, and nobody slept.
    Slept,
    /// Whatever a host-supplied handler returned for [`Effect::Custom`].
    Custom(Value),
    /// The effect could not be performed. Handed to the step rather than
    /// raised, so it can retry, fall back, or give up deliberately.
    Failed {
        /// What went wrong, as text the step can quote or act on.
        message: String,
    },
}

impl EffectResult {
    /// Did this effect fail — either outright ([`Self::Failed`]) or as a
    /// tool that ran and reported an error?
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
    /// Why generation ended. Checked before the text is trusted —
    /// see [`Self::reject_non_answers`].
    pub stop_reason: StopReason,
    /// Token accounting for this one call.
    #[serde(default)]
    pub usage: Usage,
    /// Which model actually served this, which may differ from the one asked
    /// for when a provider-side fallback kicked in.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

impl LlmResponse {
    /// Fail unless this reply is actually an answer.
    ///
    /// Two ways a turn can end are not answers: the model was cut off
    /// mid-sentence, or it declined. Both leave `message` populated with
    /// something that reads like a reply, so a step that only parses the
    /// text turns them into a *confident wrong result* — a truncated JSON
    /// object that fails to parse and scores 0.0, a half-written plan
    /// treated as the whole plan. Neither is recoverable by reading harder,
    /// so every step calls this before it looks at the text.
    ///
    /// `node_id` names the failing node in the error.
    pub fn reject_non_answers(&self, node_id: &str) -> crate::error::Result<()> {
        match &self.stop_reason {
            StopReason::Refusal { category } => Err(crate::error::SomaError::Execution {
                node_id: node_id.to_string(),
                message: format!(
                    "the model declined to answer{}. Rephrasing the request or \
                     changing model is the fix; there is no partial answer to \
                     salvage",
                    category
                        .as_deref()
                        .map(|c| format!(" ({c})"))
                        .unwrap_or_default()
                ),
            }),
            // The text so far may well be useful, so it goes in the
            // message — but it is a cut-off thought, and passing it
            // downstream as complete is a wrong answer nobody can see is
            // wrong.
            StopReason::MaxTokens => Err(crate::error::SomaError::Execution {
                node_id: node_id.to_string(),
                message: format!(
                    "the model ran out of tokens mid-answer. Raise `max_tokens` \
                     (or shorten the task). Partial answer: {}",
                    crate::util::truncate(&self.message.text(), 300)
                ),
            }),
            _ => Ok(()),
        }
    }
}

/// Why the model stopped.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum StopReason {
    /// The model finished its turn on its own — the one reason that is an answer.
    EndTurn,
    /// Generation hit `max_tokens` mid-answer; the text is a fragment.
    /// [`LlmResponse::reject_non_answers`] turns this into an error.
    MaxTokens,
    /// The model wants tools run before it continues.
    ToolUse,
    /// The provider declined. Carried as a value, not an error: the step
    /// decides whether to rephrase, fall back, or surface it.
    Refusal {
        /// The provider's refusal category, when it gave one.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        category: Option<String>,
    },
}

/// Token accounting for one call.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Tokens the model read.
    #[serde(default)]
    pub input_tokens: u64,
    /// Tokens the model generated.
    #[serde(default)]
    pub output_tokens: u64,
    /// Input tokens served from the provider's prompt cache.
    #[serde(default)]
    pub cache_read_tokens: u64,
    /// Input tokens written to the provider's prompt cache.
    #[serde(default)]
    pub cache_write_tokens: u64,
}

impl Usage {
    /// Input plus output tokens — the headline number for cost and for
    /// watching a conversation approach its context limit.
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
    /// A spec for one spawned instance of `runs`, unlabelled.
    pub fn new(runs: impl Into<NodeId>, input: Value) -> Self {
        Self {
            runs: runs.into(),
            input,
            label: None,
        }
    }

    /// Name this instance, so siblings spawned from the same node stay
    /// tellable apart in events and the journal.
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

impl JoinPolicy {
    /// A label for telemetry, in the spirit of [`Effect::label`].
    pub fn label(&self) -> &'static str {
        match self {
            JoinPolicy::All => "all",
            JoinPolicy::AllSettled => "all-settled",
            JoinPolicy::First => "first",
        }
    }
}

/// Why a run stopped and what would restart it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum SuspendReason {
    /// Waiting on a person.
    Human {
        /// What to ask them.
        prompt: String,
        /// JSON Schema the answer should satisfy, when a shape is expected.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        schema: Option<serde_json::Value>,
    },
    /// Waiting on something outside the run entirely.
    External {
        /// Opaque correlation token; whoever resumes the run quotes it back.
        token: String,
    },
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
        assert_eq!(a.cache_key().unwrap(), llm().cache_key().unwrap());
        assert_ne!(a.cache_key().unwrap(), b.cache_key().unwrap());
    }

    /// A tool's arguments are user JSON, and a JSON object has no inherent
    /// key order. Two spellings of the same call are the same call, so they
    /// must journal under one key — this is why the encoding is canonical
    /// CBOR and not raw serializer output.
    #[test]
    fn tool_args_key_is_independent_of_json_key_order() {
        let one = Effect::Tool {
            name: "search".into(),
            args: Value::json(serde_json::json!({"q": "soma", "limit": 3})),
        };
        let other = Effect::Tool {
            name: "search".into(),
            args: Value::json(serde_json::json!({"limit": 3, "q": "soma"})),
        };
        assert_eq!(one.cache_key().unwrap(), other.cache_key().unwrap());
    }

    /// Changing any request knob must change the key — otherwise a replay at
    /// a different effort would reuse the cheaper answer.
    #[test]
    fn request_options_are_part_of_the_key() {
        let base = LlmRequest::new("claude-opus-5", vec![Message::user("hi")].into());
        let keys = [
            Effect::Llm(base.clone()).cache_key().unwrap(),
            Effect::Llm(base.clone().with_system("be terse"))
                .cache_key()
                .unwrap(),
            Effect::Llm(base.clone().with_effort("high"))
                .cache_key()
                .unwrap(),
            Effect::Llm(base.with_max_tokens(10)).cache_key().unwrap(),
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

    /// A filter-only forward is the one graph effect safe to memoize by
    /// content: its nodes are deterministic and content-cached already.
    #[test]
    fn a_filter_only_forward_graph_stays_pure() {
        let mut graph = crate::graph::Graph::new();
        graph.add_node(crate::graph::Node::filter("scale"));
        let effect = Effect::Graph {
            graph: Box::new(graph),
            input: Value::Empty,
            mode: GraphEffectMode::Forward,
        };
        assert!(effect.is_pure());
    }

    /// A sub-graph with a step calls a model; reusing its first answer
    /// forever by content key would be the `Llm` mistake with extra steps —
    /// even when the step hides one sub-graph down.
    #[test]
    fn a_step_containing_graph_effect_is_impure() {
        let mut inner = crate::graph::Graph::new();
        inner.add_node(crate::graph::Node::step("agent", "ReactStep"));
        let mut graph = crate::graph::Graph::new();
        graph.add_node(crate::graph::Node::subgraph("nested", inner));
        let effect = Effect::Graph {
            graph: Box::new(graph),
            input: Value::Empty,
            mode: GraphEffectMode::Forward,
        };
        assert!(!effect.is_pure());
    }

    /// Fit writes states as a side effect; replaying the recorded summary
    /// without re-fitting would leave the graph unfitted for what follows.
    #[test]
    fn a_fit_mode_graph_effect_is_impure() {
        let mut graph = crate::graph::Graph::new();
        graph.add_node(crate::graph::Node::filter("scale"));
        let effect = Effect::Graph {
            graph: Box::new(graph),
            input: Value::Empty,
            mode: GraphEffectMode::Fit,
        };
        assert!(!effect.is_pure());
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
