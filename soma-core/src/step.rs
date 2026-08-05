//! The effectful counterpart to [`crate::filter::Filter`].
//!
//! A `Filter` is a function: same config, same state, same input, same
//! output — which is what makes content-addressed caching sound. A `Step` is
//! not a function. It calls models, reads the world, decides what to run
//! next, and may pause for a person. Forcing that into `forward()` would
//! either make the trait async (colouring the whole runtime and complicating
//! the GIL story) or make caching lie.
//!
//! So `Step` gets its own shape: **advance one turn, describe what you need,
//! hand control back.**
//!
//! ```ignore
//! loop {
//!     match step.poll(&ctx, resume)? {
//!         Transition::Await(effects) => resume = Some(driver.perform(effects)),
//!         Transition::Done(value)    => break value,
//!         // Spawn / Goto / Suspend hand control to the runtime
//!     }
//! }
//! ```
//!
//! `poll` is synchronous and cheap. The driver owns the concurrency. The
//! five [`Transition`] variants are, deliberately, the union of the control
//! primitives every agent framework converges on: awaiting work, dynamic
//! fan-out, handoff, interrupt, and finishing.

use crate::cache::CacheKey;
use crate::effect::{Effect, EffectResult, JoinPolicy, NodeSpec, SuspendReason};
use crate::error::Result;
use crate::graph::NodeId;
use crate::schema::Schema;
use crate::value::Value;
use serde::{Deserialize, Serialize};

/// What a step wants to happen next.
///
/// Deliberately NOT `#[non_exhaustive]`, for the same reason as
/// [`crate::node::NodeOutcome`], its other half: the driver decides control
/// flow off this value, and a wildcard arm there is a silent wrong answer.
/// Adding a variant *should* break every consumer — each one has to decide
/// what the new transition means for it.
pub enum Transition {
    /// Perform these effects — concurrently — then poll again with the
    /// results in the same order.
    Await(Vec<Effect>),

    /// Create and run these nodes now, then poll again with their outputs.
    ///
    /// The map half of map-reduce, where the fan-out width is only known at
    /// runtime. LangGraph calls this `Send`.
    Spawn {
        /// The nodes to create — one per unit of discovered work.
        specs: Vec<NodeSpec>,
        /// How their outputs recombine, and what one failure does to the rest.
        join: JoinPolicy,
    },

    /// Hand control to another node, with a value. The current step is done.
    ///
    /// A handoff, in the sense the OpenAI Agents SDK uses: delegation that
    /// transfers control rather than nesting a call.
    Goto {
        /// Where control goes — a node reachable over a declared handoff edge.
        target: NodeId,
        /// The value it receives as its input.
        carry: Value,
    },

    /// Stop the run and persist it. Resuming replays the journal up to here
    /// and continues with whatever came back.
    Suspend {
        /// What the run is waiting for, and what would restart it.
        reason: SuspendReason,
    },

    /// Finished, with this output.
    Done(Value),
}

impl Transition {
    /// A short label for events. Payload-free by construction.
    pub fn label(&self) -> String {
        match self {
            Self::Await(effects) => {
                let names: Vec<String> = effects.iter().map(Effect::label).collect();
                format!("await[{}]", names.join(", "))
            }
            Self::Spawn { specs, join } => format!("spawn[{} x {join:?}]", specs.len()),
            Self::Goto { target, .. } => format!("goto:{target}"),
            Self::Suspend { .. } => "suspend".to_string(),
            Self::Done(_) => "done".to_string(),
        }
    }

    /// Is this the step's last word? `Done` and `Goto` are — nothing polls
    /// the step again this run. The other three expect another poll: after
    /// the effects, after the spawned nodes, or after a resume.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Done(_) | Self::Goto { .. })
    }
}

impl std::fmt::Debug for Transition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.label())
    }
}

/// What the runtime tells a step about where it is.
///
/// Deliberately small. A step's own reasoning state belongs in the value it
/// carries, not here — that is what makes replay work: re-polling with the
/// same journal reproduces the same decisions.
pub struct StepCtx<'a> {
    /// This node's id in the graph.
    pub node_id: &'a str,
    /// The run this belongs to.
    pub run_id: &'a str,
    /// Input resolved from predecessors.
    pub input: &'a Value,
    /// Which turn this is, counting from 0. Part of the journal key, so a
    /// step that asks the same question twice gets two recorded answers.
    pub turn: usize,
    /// Results of the effects requested last turn, in request order.
    /// Empty on turn 0.
    pub results: &'a [EffectResult],
    /// Every turn's results so far, oldest first.
    ///
    /// A step that accumulates — a conversation, a running total — rebuilds
    /// it from here rather than holding it in `self`. That is what keeps
    /// replay honest: the history is replayed identically, so the same
    /// decisions follow, whereas hidden state would not survive a restart.
    pub history: &'a [Vec<EffectResult>],
}

impl<'a> StepCtx<'a> {
    /// A context with no results yet — what a first poll sees. Later turns
    /// attach theirs with [`Self::with_history`] or [`Self::with_results`].
    pub fn new(node_id: &'a str, run_id: &'a str, input: &'a Value, turn: usize) -> Self {
        Self {
            node_id,
            run_id,
            input,
            turn,
            results: &[],
            history: &[],
        }
    }

    /// Set the full history; `results` becomes its last entry.
    pub fn with_history(mut self, history: &'a [Vec<EffectResult>]) -> Self {
        self.history = history;
        self.results = history.last().map(Vec::as_slice).unwrap_or(&[]);
        self
    }

    /// Set only the current turn's results — for tests and single-turn steps.
    pub fn with_results(mut self, results: &'a [EffectResult]) -> Self {
        self.results = results;
        self
    }

    /// The single result of a one-effect turn.
    pub fn result(&self) -> Option<&EffectResult> {
        self.results.first()
    }

    /// Every result from every turn, oldest first, flattened.
    pub fn all_results(&self) -> impl Iterator<Item = &EffectResult> {
        self.history.iter().flatten()
    }
}

/// What the compiler and runtime need to know about a step without running it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepMeta {
    /// The step type's name, for events and error messages.
    pub name: String,

    /// Cap on turns before the runtime gives up.
    ///
    /// Not a safety net so much as a budget: an agent that has not finished
    /// in this many turns is looping, and the honest response is to stop and
    /// say so rather than burn tokens.
    pub max_turns: usize,

    /// May this step's effects be written to the journal?
    ///
    /// Journaling is what makes a run replayable, but it also means prompts
    /// land on disk under `$SOMA_CACHE_DIR`. Turn it off for a step handling
    /// anything that must not be persisted.
    pub journal: bool,

    /// What it accepts (`None` = anything).
    pub input_schema: Option<Schema>,
    /// What it produces (`None` = unknown).
    pub output_schema: Option<Schema>,

    /// Where it may run.
    pub distribution: crate::filter::Distribution,
}

impl StepMeta {
    /// Defaults: a 24-turn budget, journaling on, schemas undeclared, runs
    /// locally.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            max_turns: 24,
            journal: true,
            input_schema: None,
            output_schema: None,
            distribution: crate::filter::Distribution::Local,
        }
    }

    /// Set the turn budget — see [`StepMeta::max_turns`] for what running
    /// out means.
    pub fn with_max_turns(mut self, n: usize) -> Self {
        self.max_turns = n;
        self
    }

    /// Keep this step's effects out of the journal — and give up replay for
    /// it in exchange.
    pub fn without_journal(mut self) -> Self {
        self.journal = false;
        self
    }

    /// Declare what this step accepts, so an impossible edge into it fails
    /// at compile rather than mid-run.
    pub fn with_input_schema(mut self, schema: Schema) -> Self {
        self.input_schema = Some(schema);
        self
    }

    /// Declare what this step produces, for its successors' input checks.
    pub fn with_output_schema(mut self, schema: Schema) -> Self {
        self.output_schema = Some(schema);
        self
    }
}

/// An effectful node.
///
/// Implementors are registered in the runtime's step library by node id, the
/// same way filters are.
pub trait Step: crate::any::AsAny + Send + Sync {
    /// Hash of this step's configuration. Same config, same hash — it is part
    /// of every journal key this step writes.
    fn config_hash(&self) -> CacheKey;

    /// This step's [`StepMeta`]: turn budget, journaling, schemas, placement.
    fn meta(&self) -> StepMeta;

    /// Advance one turn.
    ///
    /// Called with `ctx.turn == 0` and no results to start; thereafter with
    /// the results of whatever it last asked for. Must be **deterministic
    /// given the same context and results** — that is the whole basis of
    /// replay. Put the non-determinism in an [`Effect`], where it gets
    /// recorded.
    fn poll(&self, ctx: &StepCtx<'_>) -> Result<Transition>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effect::LlmRequest;
    use crate::message::Message;

    /// A step that asks a model once and returns its text.
    struct Once;

    impl Step for Once {
        fn config_hash(&self) -> CacheKey {
            CacheKey::from_parts(&[b"Once"])
        }
        fn meta(&self) -> StepMeta {
            StepMeta::new("Once")
        }
        fn poll(&self, ctx: &StepCtx<'_>) -> Result<Transition> {
            match ctx.result() {
                None => Ok(Transition::Await(vec![Effect::Llm(LlmRequest::new(
                    "claude-opus-5",
                    vec![Message::user(ctx.input.as_text().unwrap_or_default())].into(),
                ))])),
                Some(EffectResult::Llm(resp)) => {
                    Ok(Transition::Done(Value::text(resp.message.text())))
                }
                Some(other) => Ok(Transition::Done(Value::text(format!(
                    "unexpected {other:?}"
                )))),
            }
        }
    }

    #[test]
    fn first_poll_asks_for_the_model() {
        let input = Value::text("hello");
        let ctx = StepCtx::new("n", "r", &input, 0);
        let t = Once.poll(&ctx).unwrap();
        match t {
            Transition::Await(effects) => {
                assert_eq!(effects.len(), 1);
                assert!(matches!(effects[0], Effect::Llm(_)));
            }
            other => panic!("expected Await, got {other:?}"),
        }
    }

    #[test]
    fn second_poll_finishes_with_the_reply() {
        use crate::effect::{LlmResponse, StopReason, Usage};

        let input = Value::text("hello");
        let results = [EffectResult::Llm(LlmResponse {
            message: Message::assistant("hi there"),
            stop_reason: StopReason::EndTurn,
            usage: Usage::default(),
            model: None,
        })];
        let ctx = StepCtx::new("n", "r", &input, 1).with_results(&results);

        match Once.poll(&ctx).unwrap() {
            Transition::Done(v) => assert_eq!(v.as_text(), Some("hi there")),
            other => panic!("expected Done, got {other:?}"),
        }
    }

    /// Same context in, same decision out — the property replay rests on.
    #[test]
    fn polling_is_deterministic() {
        let input = Value::text("hello");
        let ctx = StepCtx::new("n", "r", &input, 0);
        let a = Once.poll(&ctx).unwrap();
        let b = Once.poll(&ctx).unwrap();
        assert_eq!(a.label(), b.label());
    }

    #[test]
    fn labels_describe_without_leaking() {
        let input = Value::text("a secret prompt");
        let ctx = StepCtx::new("n", "r", &input, 0);
        let label = Once.poll(&ctx).unwrap().label();
        assert!(label.starts_with("await["), "{label}");
        assert!(!label.contains("secret"), "{label}");
    }

    #[test]
    fn terminal_transitions() {
        assert!(Transition::Done(Value::Empty).is_terminal());
        assert!(
            Transition::Goto {
                target: "next".into(),
                carry: Value::Empty
            }
            .is_terminal()
        );
        assert!(!Transition::Await(vec![]).is_terminal());
    }

    #[test]
    fn journal_can_be_declined() {
        assert!(StepMeta::new("s").journal);
        assert!(!StepMeta::new("s").without_journal().journal);
    }
}
