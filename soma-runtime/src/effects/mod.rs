//! Running effectful steps.
//!
//! The loop is small on purpose:
//!
//! ```text
//! poll ──► Await(effects) ──► perform (concurrently, journaled) ──┐
//!   ▲                                                             │
//!   └─────────────────────────────────────────────────────────────┘
//!   └──► Done(value) ──► finished
//! ```
//!
//! **On concurrency.** Effects within a turn run on scoped OS threads, the
//! same mechanism [`crate::executor`] already uses for parallel branches —
//! not an async runtime. A model call is a blocking socket read; a handful
//! of them in flight is a handful of parked threads, which costs nothing and
//! keeps `Step::poll` synchronous, the Python bridge a plain call, and the
//! GIL-release discipline identical to the one the executor already proved.
//! An async runtime would earn its keep at thousands of concurrent calls; a
//! step awaiting three tools is not that, and paying for it up front would
//! colour every signature in the crate.

pub mod graph_handler;
pub mod journal;

pub use graph_handler::GraphHandler;
pub use journal::{EffectJournal, EffectSite};

use crate::event_bus::EventBus;
use somatize_core::effect::{Effect, EffectResult, Usage};
use somatize_core::error::{Result, SomaError};
use somatize_core::event::Event;
use somatize_core::step::{Step, StepCtx, Transition};
use somatize_core::value::Value;
use std::sync::Arc;
use std::time::Instant;

/// The journal entry a suspension reads and a resume writes.
///
/// Modelling the pause as an effect is what makes resuming free: it lands in
/// the same store, under the same site key, and replays by the same rule as
/// a model call.
fn suspension_effect(reason: &somatize_core::effect::SuspendReason) -> Effect {
    Effect::Custom {
        kind: "soma.suspend".into(),
        payload: Value::json(serde_json::to_value(reason).unwrap_or(serde_json::Value::Null)),
    }
}

pub use somatize_core::effect::EffectHandler;

pub use somatize_core::node::NodeOutcome;

/// Drives steps: performs their effects, journals them, emits their events.
#[derive(Clone)]
pub struct EffectDriver {
    handlers: Vec<Arc<dyn EffectHandler>>,
    journal: EffectJournal,
    event_bus: Option<Arc<EventBus>>,
    /// Needed only to satisfy [`Transition::Spawn`], which names nodes to
    /// create mid-run.
    catalog: Option<Arc<crate::node_catalog::NodeCatalog>>,
}

impl EffectDriver {
    pub fn new(journal: EffectJournal) -> Self {
        Self {
            handlers: Vec::new(),
            journal,
            event_bus: None,
            catalog: None,
        }
    }

    /// Provide the step library that dynamic fan-out draws from.
    pub fn with_catalog(mut self, catalog: Arc<crate::node_catalog::NodeCatalog>) -> Self {
        self.catalog = Some(catalog);
        self
    }

    pub fn with_handler(mut self, handler: Arc<dyn EffectHandler>) -> Self {
        self.handlers.push(handler);
        self
    }

    pub fn with_event_bus(mut self, bus: Arc<EventBus>) -> Self {
        self.event_bus = Some(bus);
        self
    }

    fn emit(&self, event: Event) {
        if let Some(bus) = &self.event_bus {
            bus.emit(event);
        }
    }

    /// Run a step to completion.
    ///
    /// Bounded by [`somatize_core::step::StepMeta::max_turns`]: a step that
    /// has not finished by then is looping, and stopping with a clear error
    /// beats burning tokens until something else gives out.
    pub fn run(
        &self,
        step: &dyn Step,
        run_id: &str,
        node_id: &str,
        input: &Value,
    ) -> Result<NodeOutcome> {
        let meta = step.meta();
        // A step may decline journaling; honour it for this step only.
        let journal = self
            .journal
            .clone()
            .with_enabled(self.journal.is_enabled() && meta.journal);

        let started = Instant::now();
        // Every turn's results, kept so a step can rebuild what it has
        // accumulated instead of holding it in itself — see `StepCtx::history`.
        let mut history: Vec<Vec<EffectResult>> = Vec::new();
        let mut usage = Usage::default();

        for turn in 0..meta.max_turns {
            self.emit(Event::AgentTurnStarted {
                run_id: run_id.to_string(),
                node_id: node_id.to_string(),
                turn,
            });

            let ctx = StepCtx::new(node_id, run_id, input, turn).with_history(&history);
            let transition = step.poll(&ctx)?;

            match transition {
                Transition::Await(effects) => {
                    if effects.is_empty() {
                        return Err(SomaError::Execution {
                            node_id: node_id.to_string(),
                            message: format!(
                                "step awaited nothing on turn {turn}; it would spin. \
                                 Return `Done` to finish, or ask for at least one effect"
                            ),
                        });
                    }
                    history.push(
                        self.perform_all(&journal, run_id, node_id, turn, &effects, &mut usage)?,
                    );
                }

                Transition::Done(value) => {
                    self.finish(run_id, node_id, turn + 1, started, usage);
                    return Ok(NodeOutcome::Produced(value));
                }

                Transition::Goto { target, carry } => {
                    self.emit(Event::Handoff {
                        run_id: run_id.to_string(),
                        from: node_id.to_string(),
                        to: target.clone(),
                    });
                    self.finish(run_id, node_id, turn + 1, started, usage);
                    return Ok(NodeOutcome::HandOff { target, carry });
                }

                // Suspension is journaled like any other awaited thing. On a
                // replay the recorded answer is already there, so resuming
                // needs no separate checkpoint format: the run re-polls from
                // the start, every prior effect is served from the journal,
                // and this point now has its answer.
                Transition::Suspend { reason } => {
                    let site = EffectSite {
                        run_id,
                        node_id,
                        turn,
                        index: 0,
                    };
                    let effect = suspension_effect(&reason);

                    if let Some(answered) = journal.lookup(site, &effect)? {
                        self.emit(Event::Resumed {
                            run_id: run_id.to_string(),
                            node_id: node_id.to_string(),
                            turn,
                        });
                        history.push(vec![answered]);
                        continue;
                    }

                    self.emit(Event::Suspended {
                        run_id: run_id.to_string(),
                        node_id: node_id.to_string(),
                        reason: reason.kind().to_string(),
                    });
                    return Ok(NodeOutcome::Paused { turn, reason });
                }

                Transition::Spawn { specs, join } => {
                    if specs.is_empty() {
                        return Err(SomaError::Execution {
                            node_id: node_id.to_string(),
                            message: format!(
                                "step spawned nothing on turn {turn}; it would spin. \
                                 Return `Done` when there is no work to fan out"
                            ),
                        });
                    }
                    history.push(self.spawn_all(run_id, node_id, turn, &specs, join)?);
                }

                other => {
                    return Err(SomaError::Execution {
                        node_id: node_id.to_string(),
                        message: format!("unsupported transition `{}`", other.label()),
                    });
                }
            }
        }

        Err(SomaError::Execution {
            node_id: node_id.to_string(),
            message: format!(
                "step did not finish within {} turns. Raise `StepMeta::max_turns` if the \
                 work genuinely needs more, or check whether it is looping",
                meta.max_turns
            ),
        })
    }

    fn finish(&self, run_id: &str, node_id: &str, turns: usize, started: Instant, usage: Usage) {
        self.emit(Event::AgentStepCompleted {
            run_id: run_id.to_string(),
            node_id: node_id.to_string(),
            turns,
            duration: started.elapsed(),
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
        });
    }

    /// Deliver the answer a suspended run was waiting for.
    ///
    /// Recorded at the exact site the step suspended, so the next run under
    /// the same id replays to that point and finds it. There is no separate
    /// checkpoint file: the journal *is* the checkpoint.
    ///
    /// `node_id`, `turn` and `reason` come from the
    /// [`NodeOutcome::Paused`] that stopped the run.
    pub fn resume_with(
        &self,
        run_id: &str,
        node_id: &str,
        turn: usize,
        reason: &somatize_core::effect::SuspendReason,
        answer: Value,
    ) -> Result<()> {
        if !self.journal.is_enabled() {
            return Err(SomaError::Execution {
                node_id: node_id.to_string(),
                message: "cannot resume a run whose journal is disabled: there is \
                          nothing to replay up to the suspension point"
                    .into(),
            });
        }
        let site = EffectSite {
            run_id,
            node_id,
            turn,
            index: 0,
        };
        self.journal.record(
            site,
            &suspension_effect(reason),
            &EffectResult::Node(answer),
            0,
        )
    }

    /// Create and run spawned nodes, concurrently, in spec order.
    ///
    /// Each instance gets a hierarchical id, `parent/label`, matching the
    /// convention intra-node auditing already uses. That is not cosmetic:
    /// the id is part of every journal key the instance writes, so two
    /// siblings asking the same question record two answers, and a replay
    /// gives each back its own.
    fn spawn_all(
        &self,
        run_id: &str,
        node_id: &str,
        turn: usize,
        specs: &[somatize_core::effect::NodeSpec],
        join: somatize_core::effect::JoinPolicy,
    ) -> Result<Vec<EffectResult>> {
        use somatize_core::effect::JoinPolicy;

        let catalog = self.catalog.as_ref().ok_or_else(|| SomaError::Execution {
            node_id: node_id.to_string(),
            message: "step spawned work, but the driver has no step library; \
                      build it with `EffectDriver::with_catalog(...)`"
                .into(),
        })?;

        let outcomes: Vec<Result<EffectResult>> = std::thread::scope(|scope| {
            let handles: Vec<_> = specs
                .iter()
                .enumerate()
                .map(|(index, spec)| {
                    let label = spec
                        .label
                        .clone()
                        .unwrap_or_else(|| format!("{turn}.{index}"));
                    let child_id = format!("{node_id}/{label}");
                    scope.spawn(move || {
                        let step = catalog
                            .step(&spec.runs)
                            .ok_or_else(|| SomaError::NodeNotFound(spec.runs.clone()))?;
                        match self.run(step.as_ref(), run_id, &child_id, &spec.input)? {
                            NodeOutcome::Produced(value) => Ok(EffectResult::Node(value)),
                            NodeOutcome::HandOff { target, .. } => Err(SomaError::Execution {
                                node_id: child_id.clone(),
                                message: format!(
                                    "a spawned step handed control to `{target}`; spawned \
                                     work must finish with `Done`, since it has no place \
                                     in the graph to hand control to"
                                ),
                            }),
                            NodeOutcome::Paused { .. } => Err(SomaError::Execution {
                                node_id: child_id.clone(),
                                message: "a spawned step suspended; suspension is only \
                                          supported for nodes in the graph"
                                    .into(),
                            }),
                        }
                    })
                })
                .collect();

            handles
                .into_iter()
                .map(|h| {
                    h.join().unwrap_or_else(|_| {
                        Err(SomaError::Execution {
                            node_id: node_id.to_string(),
                            message: "a spawned step panicked".into(),
                        })
                    })
                })
                .collect()
        });

        match join {
            // Any failure fails the join: the step asked for all of it.
            JoinPolicy::All => outcomes.into_iter().collect(),

            // Keep what worked; a failure becomes a result the step can see
            // and decide about, which is the point of asking for `AllSettled`.
            JoinPolicy::AllSettled => Ok(outcomes
                .into_iter()
                .map(|o| match o {
                    Ok(result) => result,
                    Err(e) => EffectResult::Failed {
                        message: e.to_string(),
                    },
                })
                .collect()),

            // First success wins. Everything ran — these are threads, not
            // cancellable tasks — but only the winner is handed back.
            JoinPolicy::First => {
                let mut last_error = None;
                for outcome in outcomes {
                    match outcome {
                        Ok(result) => return Ok(vec![result]),
                        Err(e) => last_error = Some(e),
                    }
                }
                Err(last_error.unwrap_or_else(|| SomaError::Execution {
                    node_id: node_id.to_string(),
                    message: "no spawned step succeeded".into(),
                }))
            }

            _ => Err(SomaError::Execution {
                node_id: node_id.to_string(),
                message: format!("unsupported join policy {join:?}"),
            }),
        }
    }

    /// Perform a turn's effects, concurrently, returning results in request
    /// order — the order a step relies on to match answers to questions.
    fn perform_all(
        &self,
        journal: &EffectJournal,
        run_id: &str,
        node_id: &str,
        turn: usize,
        effects: &[Effect],
        usage: &mut Usage,
    ) -> Result<Vec<EffectResult>> {
        for (index, effect) in effects.iter().enumerate() {
            let _ = index;
            self.emit(Event::EffectRequested {
                run_id: run_id.to_string(),
                node_id: node_id.to_string(),
                turn,
                effect: effect.label(),
            });
        }

        let outcomes: Vec<Result<(EffectResult, bool, std::time::Duration)>> =
            std::thread::scope(|scope| {
                let handles: Vec<_> = effects
                    .iter()
                    .enumerate()
                    .map(|(index, effect)| {
                        let site = EffectSite {
                            run_id,
                            node_id,
                            turn,
                            index,
                        };
                        scope.spawn(move || self.perform_one(journal, site, effect))
                    })
                    .collect();

                handles
                    .into_iter()
                    .map(|h| {
                        h.join().unwrap_or_else(|_| {
                            Err(SomaError::Execution {
                                node_id: node_id.to_string(),
                                message: "effect handler panicked".into(),
                            })
                        })
                    })
                    .collect()
            });

        let mut results = Vec::with_capacity(effects.len());
        for (effect, outcome) in effects.iter().zip(outcomes) {
            let (result, replayed, elapsed) = outcome?;

            if let EffectResult::Llm(response) = &result {
                *usage += response.usage;
            }
            if let Effect::Tool { name, .. } = effect {
                self.emit(Event::ToolCalled {
                    run_id: run_id.to_string(),
                    node_id: node_id.to_string(),
                    tool: name.clone(),
                    is_error: result.is_error(),
                });
            }

            self.emit(Event::EffectCompleted {
                run_id: run_id.to_string(),
                node_id: node_id.to_string(),
                turn,
                effect: effect.label(),
                duration: elapsed,
                replayed,
                is_error: result.is_error(),
            });

            results.push(result);
        }
        Ok(results)
    }

    /// One effect: journal first, perform only on a miss.
    fn perform_one(
        &self,
        journal: &EffectJournal,
        site: EffectSite<'_>,
        effect: &Effect,
    ) -> Result<(EffectResult, bool, std::time::Duration)> {
        let started = Instant::now();

        if let Some(recorded) = journal.lookup(site, effect)? {
            return Ok((recorded, true, started.elapsed()));
        }

        let handler = self
            .handlers
            .iter()
            .find(|h| h.handles(effect))
            .ok_or_else(|| SomaError::Execution {
                node_id: site.node_id.to_string(),
                message: format!(
                    "no handler for effect `{}`. Register one on the driver",
                    effect.label()
                ),
            })?;

        let result = handler.perform(effect)?;
        let elapsed = started.elapsed();
        journal.record(site, effect, &result, elapsed.as_millis() as u64)?;
        Ok((result, false, elapsed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::fs_store::FsActionStore;
    use somatize_core::cache::CacheKey;
    use somatize_core::effect::{LlmRequest, LlmResponse, StopReason};
    use somatize_core::message::Message;
    use somatize_core::step::StepMeta;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Counts calls, so tests can prove a replay performed none.
    struct CountingLlm {
        calls: AtomicUsize,
        reply: String,
    }

    impl CountingLlm {
        fn new(reply: &str) -> Arc<Self> {
            Arc::new(Self {
                calls: AtomicUsize::new(0),
                reply: reply.to_string(),
            })
        }
    }

    impl EffectHandler for CountingLlm {
        fn handles(&self, effect: &Effect) -> bool {
            matches!(effect, Effect::Llm(_))
        }
        fn perform(&self, _effect: &Effect) -> Result<EffectResult> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(EffectResult::Llm(LlmResponse {
                message: Message::assistant(&self.reply),
                stop_reason: StopReason::EndTurn,
                usage: Usage {
                    input_tokens: 10,
                    output_tokens: 3,
                    ..Default::default()
                },
                model: None,
            }))
        }
    }

    /// Asks the model `rounds` times, then returns the last reply.
    struct MultiTurn {
        rounds: usize,
    }

    impl Step for MultiTurn {
        fn config_hash(&self) -> CacheKey {
            CacheKey::from_parts(&[b"MultiTurn"])
        }
        fn meta(&self) -> StepMeta {
            StepMeta::new("MultiTurn")
        }
        fn poll(&self, ctx: &StepCtx<'_>) -> Result<Transition> {
            if ctx.turn < self.rounds {
                return Ok(Transition::Await(vec![Effect::Llm(LlmRequest::new(
                    "claude-opus-5",
                    vec![Message::user(format!("turn {}", ctx.turn))].into(),
                ))]));
            }
            let text = match ctx.result() {
                Some(EffectResult::Llm(r)) => r.message.text(),
                _ => String::new(),
            };
            Ok(Transition::Done(Value::text(text)))
        }
    }

    fn driver(handler: Arc<dyn EffectHandler>) -> (EffectDriver, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(FsActionStore::new(dir.path()).unwrap());
        let journal = EffectJournal::new(store.clone(), store);
        (EffectDriver::new(journal).with_handler(handler), dir)
    }

    #[test]
    fn runs_a_multi_turn_step() {
        let llm = CountingLlm::new("hello");
        let (d, _dir) = driver(llm.clone());

        let out = d
            .run(&MultiTurn { rounds: 3 }, "r1", "agent", &Value::Empty)
            .unwrap();

        match out {
            NodeOutcome::Produced(v) => assert_eq!(v.as_text(), Some("hello")),
            other => panic!("expected Done, got {other:?}"),
        }
        assert_eq!(llm.calls.load(Ordering::SeqCst), 3);
    }

    /// The durability property: replaying a run performs no effects at all,
    /// and lands on the same answer.
    #[test]
    fn replaying_a_run_performs_nothing() {
        let llm = CountingLlm::new("recorded answer");
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(FsActionStore::new(dir.path()).unwrap());
        let journal = EffectJournal::new(store.clone(), store);

        let d = EffectDriver::new(journal).with_handler(llm.clone());

        let first = d
            .run(&MultiTurn { rounds: 3 }, "run-A", "agent", &Value::Empty)
            .unwrap();
        assert_eq!(llm.calls.load(Ordering::SeqCst), 3);

        // Same run id — this is a replay, not a new run.
        let second = d
            .run(&MultiTurn { rounds: 3 }, "run-A", "agent", &Value::Empty)
            .unwrap();

        assert_eq!(
            llm.calls.load(Ordering::SeqCst),
            3,
            "a replay called the model again"
        );
        match (first, second) {
            (NodeOutcome::Produced(a), NodeOutcome::Produced(b)) => assert_eq!(a, b),
            other => panic!("expected two Done outcomes, got {other:?}"),
        }
    }

    /// A different run must actually ask again.
    #[test]
    fn a_fresh_run_calls_the_model() {
        let llm = CountingLlm::new("x");
        let (d, _dir) = driver(llm.clone());

        d.run(&MultiTurn { rounds: 2 }, "run-A", "agent", &Value::Empty)
            .unwrap();
        d.run(&MultiTurn { rounds: 2 }, "run-B", "agent", &Value::Empty)
            .unwrap();

        assert_eq!(llm.calls.load(Ordering::SeqCst), 4);
    }

    /// Effects requested together are answered in request order, so a step
    /// can line results up against the questions it asked.
    #[test]
    fn concurrent_effects_keep_request_order() {
        struct Echo;
        impl EffectHandler for Echo {
            fn handles(&self, e: &Effect) -> bool {
                matches!(e, Effect::Tool { .. })
            }
            fn perform(&self, e: &Effect) -> Result<EffectResult> {
                let Effect::Tool { args, .. } = e else {
                    unreachable!()
                };
                // Reversed sleeps: if results came back in completion order
                // rather than request order, this test would catch it.
                let n = args.as_text().unwrap_or("0").parse::<u64>().unwrap_or(0);
                std::thread::sleep(std::time::Duration::from_millis(30 - n * 10));
                Ok(EffectResult::Tool {
                    output: args.clone(),
                    is_error: false,
                })
            }
        }

        struct FanOut;
        impl Step for FanOut {
            fn config_hash(&self) -> CacheKey {
                CacheKey::from_parts(&[b"FanOut"])
            }
            fn meta(&self) -> StepMeta {
                StepMeta::new("FanOut")
            }
            fn poll(&self, ctx: &StepCtx<'_>) -> Result<Transition> {
                if ctx.turn == 0 {
                    return Ok(Transition::Await(
                        (0..3)
                            .map(|i| Effect::Tool {
                                name: "echo".into(),
                                args: Value::text(i.to_string()),
                            })
                            .collect(),
                    ));
                }
                let joined: Vec<String> = ctx
                    .results
                    .iter()
                    .filter_map(|r| r.value().and_then(|v| v.as_text()).map(String::from))
                    .collect();
                Ok(Transition::Done(Value::text(joined.join(","))))
            }
        }

        let (d, _dir) = driver(Arc::new(Echo));
        match d.run(&FanOut, "r", "n", &Value::Empty).unwrap() {
            NodeOutcome::Produced(v) => assert_eq!(v.as_text(), Some("0,1,2")),
            other => panic!("{other:?}"),
        }
    }

    /// A step that never finishes is stopped and told why.
    #[test]
    fn a_runaway_step_is_capped() {
        struct Forever;
        impl Step for Forever {
            fn config_hash(&self) -> CacheKey {
                CacheKey::from_parts(&[b"Forever"])
            }
            fn meta(&self) -> StepMeta {
                StepMeta::new("Forever").with_max_turns(3)
            }
            fn poll(&self, ctx: &StepCtx<'_>) -> Result<Transition> {
                Ok(Transition::Await(vec![Effect::Llm(LlmRequest::new(
                    "claude-opus-5",
                    vec![Message::user(format!("{}", ctx.turn))].into(),
                ))]))
            }
        }

        let llm = CountingLlm::new("x");
        let (d, _dir) = driver(llm.clone());
        let err = d.run(&Forever, "r", "n", &Value::Empty).unwrap_err();

        assert!(err.to_string().contains("max_turns"), "{err}");
        assert_eq!(llm.calls.load(Ordering::SeqCst), 3, "ran past the cap");
    }

    /// An unhandled effect names itself, rather than failing obscurely.
    #[test]
    fn an_unhandled_effect_says_so() {
        let (d, _dir) = driver(CountingLlm::new("x"));
        struct WantsTool;
        impl Step for WantsTool {
            fn config_hash(&self) -> CacheKey {
                CacheKey::from_parts(&[b"WantsTool"])
            }
            fn meta(&self) -> StepMeta {
                StepMeta::new("WantsTool")
            }
            fn poll(&self, _ctx: &StepCtx<'_>) -> Result<Transition> {
                Ok(Transition::Await(vec![Effect::Tool {
                    name: "search".into(),
                    args: Value::Empty,
                }]))
            }
        }

        let err = d.run(&WantsTool, "r", "n", &Value::Empty).unwrap_err();
        assert!(err.to_string().contains("tool:search"), "{err}");
        assert!(err.to_string().contains("no handler"), "{err}");
    }

    /// Awaiting nothing would spin forever; say so instead.
    #[test]
    fn awaiting_nothing_is_an_error() {
        struct Empty;
        impl Step for Empty {
            fn config_hash(&self) -> CacheKey {
                CacheKey::from_parts(&[b"Empty"])
            }
            fn meta(&self) -> StepMeta {
                StepMeta::new("Empty")
            }
            fn poll(&self, _ctx: &StepCtx<'_>) -> Result<Transition> {
                Ok(Transition::Await(vec![]))
            }
        }
        let (d, _dir) = driver(CountingLlm::new("x"));
        let err = d.run(&Empty, "r", "n", &Value::Empty).unwrap_err();
        assert!(err.to_string().contains("awaited nothing"), "{err}");
    }

    // ── Dynamic fan-out ──

    /// A worker: uppercases whatever it is given.
    struct Worker;
    impl Step for Worker {
        fn config_hash(&self) -> CacheKey {
            CacheKey::from_parts(&[b"Worker"])
        }
        fn meta(&self) -> StepMeta {
            StepMeta::new("Worker")
        }
        fn poll(&self, ctx: &StepCtx<'_>) -> Result<Transition> {
            Ok(Transition::Done(Value::text(
                ctx.input.as_text().unwrap_or_default().to_uppercase(),
            )))
        }
    }

    /// Splits its input on commas and fans a worker out over the pieces —
    /// the orchestrator-workers shape, where the width is only known once
    /// the input is in hand and so cannot be pre-declared as topology.
    struct Orchestrator {
        join: somatize_core::effect::JoinPolicy,
    }

    impl Step for Orchestrator {
        fn config_hash(&self) -> CacheKey {
            CacheKey::from_parts(&[b"Orchestrator"])
        }
        fn meta(&self) -> StepMeta {
            StepMeta::new("Orchestrator")
        }
        fn poll(&self, ctx: &StepCtx<'_>) -> Result<Transition> {
            if ctx.turn == 0 {
                let specs = ctx
                    .input
                    .as_text()
                    .unwrap_or_default()
                    .split(',')
                    .enumerate()
                    .map(|(i, part)| {
                        somatize_core::effect::NodeSpec::new("worker", Value::text(part))
                            .with_label(format!("w{i}"))
                    })
                    .collect();
                return Ok(Transition::Spawn {
                    specs,
                    join: self.join,
                });
            }
            let joined: Vec<String> = ctx
                .results
                .iter()
                .map(|r| match r {
                    EffectResult::Node(v) => v.as_text().unwrap_or_default().to_string(),
                    EffectResult::Failed { message } => format!("<{message}>"),
                    other => format!("<unexpected {other:?}>"),
                })
                .collect();
            Ok(Transition::Done(Value::text(joined.join("|"))))
        }
    }

    fn spawning_driver(
        join: somatize_core::effect::JoinPolicy,
    ) -> (EffectDriver, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(FsActionStore::new(dir.path()).unwrap());
        let journal = EffectJournal::new(store.clone(), store);

        let mut steps = crate::node_catalog::NodeCatalog::new();
        steps.register_step("worker", Box::new(Worker));
        steps.register_step("orchestrator", Box::new(Orchestrator { join }));

        (
            EffectDriver::new(journal).with_catalog(Arc::new(steps)),
            dir,
        )
    }

    #[test]
    fn spawns_a_worker_per_item_and_joins_in_order() {
        use somatize_core::effect::JoinPolicy;

        let (d, _dir) = spawning_driver(JoinPolicy::All);
        let out = d
            .run(
                &Orchestrator {
                    join: JoinPolicy::All,
                },
                "r",
                "orch",
                &Value::text("alpha,beta,gamma"),
            )
            .unwrap();

        match out {
            NodeOutcome::Produced(v) => assert_eq!(v.as_text(), Some("ALPHA|BETA|GAMMA")),
            other => panic!("{other:?}"),
        }
    }

    /// Siblings get distinct journal keys via their hierarchical ids, so
    /// replaying a fan-out gives each worker back its own answer rather
    /// than the first one's.
    #[test]
    fn spawned_siblings_journal_separately() {
        use somatize_core::effect::JoinPolicy;

        let (d, _dir) = spawning_driver(JoinPolicy::All);
        let orch = Orchestrator {
            join: JoinPolicy::All,
        };

        let first = d.run(&orch, "r", "orch", &Value::text("a,b,c")).unwrap();
        let replay = d.run(&orch, "r", "orch", &Value::text("a,b,c")).unwrap();

        match (first, replay) {
            (NodeOutcome::Produced(a), NodeOutcome::Produced(b)) => {
                assert_eq!(a.as_text(), Some("A|B|C"));
                assert_eq!(a, b, "replay of a fan-out diverged");
            }
            other => panic!("{other:?}"),
        }
    }

    /// A spawner that names a step nobody registered says which one.
    #[test]
    fn spawning_an_unknown_step_names_it() {
        struct BadOrchestrator;
        impl Step for BadOrchestrator {
            fn config_hash(&self) -> CacheKey {
                CacheKey::from_parts(&[b"Bad"])
            }
            fn meta(&self) -> StepMeta {
                StepMeta::new("Bad")
            }
            fn poll(&self, _ctx: &StepCtx<'_>) -> Result<Transition> {
                Ok(Transition::Spawn {
                    specs: vec![somatize_core::effect::NodeSpec::new(
                        "nonexistent",
                        Value::Empty,
                    )],
                    join: somatize_core::effect::JoinPolicy::All,
                })
            }
        }

        let (d, _dir) = spawning_driver(somatize_core::effect::JoinPolicy::All);
        let err = d
            .run(&BadOrchestrator, "r", "orch", &Value::Empty)
            .unwrap_err();
        assert!(err.to_string().contains("nonexistent"), "{err}");
    }

    /// Spawning without a step library explains what is missing.
    #[test]
    fn spawning_without_a_library_explains_itself() {
        use somatize_core::effect::JoinPolicy;

        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(FsActionStore::new(dir.path()).unwrap());
        let d = EffectDriver::new(EffectJournal::new(store.clone(), store));

        let err = d
            .run(
                &Orchestrator {
                    join: JoinPolicy::All,
                },
                "r",
                "orch",
                &Value::text("a"),
            )
            .unwrap_err();
        assert!(err.to_string().contains("with_catalog"), "{err}");
    }

    /// Spawning nothing would spin; say so.
    #[test]
    fn spawning_nothing_is_an_error() {
        use somatize_core::effect::JoinPolicy;

        let (d, _dir) = spawning_driver(JoinPolicy::All);

        struct SpawnsNothing;
        impl Step for SpawnsNothing {
            fn config_hash(&self) -> CacheKey {
                CacheKey::from_parts(&[b"SpawnsNothing"])
            }
            fn meta(&self) -> StepMeta {
                StepMeta::new("SpawnsNothing")
            }
            fn poll(&self, _ctx: &StepCtx<'_>) -> Result<Transition> {
                Ok(Transition::Spawn {
                    specs: vec![],
                    join: JoinPolicy::All,
                })
            }
        }
        let err = d
            .run(&SpawnsNothing, "r", "orch", &Value::Empty)
            .unwrap_err();
        assert!(err.to_string().contains("spawned nothing"), "{err}");
    }

    // ── Suspension and resume ──

    /// Asks a person to approve, then reports what they said.
    struct NeedsApproval;

    impl Step for NeedsApproval {
        fn config_hash(&self) -> CacheKey {
            CacheKey::from_parts(&[b"NeedsApproval"])
        }
        fn meta(&self) -> StepMeta {
            StepMeta::new("NeedsApproval")
        }
        fn poll(&self, ctx: &StepCtx<'_>) -> Result<Transition> {
            match ctx.result() {
                None => Ok(Transition::Suspend {
                    reason: somatize_core::effect::SuspendReason::Human {
                        prompt: "Approve deleting 3 files?".into(),
                        schema: None,
                    },
                }),
                Some(EffectResult::Node(answer)) => Ok(Transition::Done(Value::text(format!(
                    "decision: {}",
                    answer.as_text().unwrap_or("?")
                )))),
                Some(other) => Ok(Transition::Done(Value::text(format!("odd: {other:?}")))),
            }
        }
    }

    fn reason() -> somatize_core::effect::SuspendReason {
        somatize_core::effect::SuspendReason::Human {
            prompt: "Approve deleting 3 files?".into(),
            schema: None,
        }
    }

    /// The full human-in-the-loop cycle: stop, answer out of band, resume,
    /// finish — with no separate checkpoint format, because the journal is
    /// the checkpoint.
    #[test]
    fn suspends_then_resumes_with_the_answer() {
        let (d, _dir) = driver(CountingLlm::new("unused"));

        let first = d
            .run(&NeedsApproval, "run-hitl", "approve", &Value::Empty)
            .unwrap();
        let turn = match first {
            NodeOutcome::Paused { turn, .. } => turn,
            other => panic!("expected a suspension, got {other:?}"),
        };
        assert_eq!(turn, 0);

        d.resume_with("run-hitl", "approve", turn, &reason(), Value::text("yes"))
            .unwrap();

        match d
            .run(&NeedsApproval, "run-hitl", "approve", &Value::Empty)
            .unwrap()
        {
            NodeOutcome::Produced(v) => assert_eq!(v.as_text(), Some("decision: yes")),
            other => panic!("expected Done after resuming, got {other:?}"),
        }
    }

    /// Without an answer it suspends again rather than proceeding on a guess.
    #[test]
    fn re_running_without_an_answer_suspends_again() {
        let (d, _dir) = driver(CountingLlm::new("unused"));

        for _ in 0..2 {
            match d
                .run(&NeedsApproval, "r", "approve", &Value::Empty)
                .unwrap()
            {
                NodeOutcome::Paused { .. } => {}
                other => panic!("expected a suspension, got {other:?}"),
            }
        }
    }

    /// An answer belongs to one run. Another run must still ask.
    #[test]
    fn an_answer_does_not_carry_to_another_run() {
        let (d, _dir) = driver(CountingLlm::new("unused"));

        d.run(&NeedsApproval, "run-A", "approve", &Value::Empty)
            .unwrap();
        d.resume_with("run-A", "approve", 0, &reason(), Value::text("yes"))
            .unwrap();

        match d
            .run(&NeedsApproval, "run-B", "approve", &Value::Empty)
            .unwrap()
        {
            NodeOutcome::Paused { .. } => {}
            other => panic!("run B reused run A's approval: {other:?}"),
        }
    }

    /// Resuming needs a journal; without one there is nothing to replay to
    /// the suspension point, and saying so beats silently restarting.
    #[test]
    fn resuming_without_a_journal_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(FsActionStore::new(dir.path()).unwrap());
        let d = EffectDriver::new(EffectJournal::disabled(store.clone(), store));

        let err = d
            .resume_with("r", "approve", 0, &reason(), Value::text("yes"))
            .unwrap_err();
        assert!(err.to_string().contains("journal is disabled"), "{err}");
    }

    /// A step opting out of the journal is not replayable — the trade its
    /// author accepted.
    #[test]
    fn a_step_can_decline_journaling() {
        struct Private;
        impl Step for Private {
            fn config_hash(&self) -> CacheKey {
                CacheKey::from_parts(&[b"Private"])
            }
            fn meta(&self) -> StepMeta {
                StepMeta::new("Private").without_journal()
            }
            fn poll(&self, ctx: &StepCtx<'_>) -> Result<Transition> {
                if ctx.turn == 0 {
                    return Ok(Transition::Await(vec![Effect::Llm(LlmRequest::new(
                        "claude-opus-5",
                        vec![Message::user("sensitive")].into(),
                    ))]));
                }
                Ok(Transition::Done(Value::Empty))
            }
        }

        let llm = CountingLlm::new("x");
        let (d, _dir) = driver(llm.clone());
        d.run(&Private, "r", "n", &Value::Empty).unwrap();
        d.run(&Private, "r", "n", &Value::Empty).unwrap();

        assert_eq!(
            llm.calls.load(Ordering::SeqCst),
            2,
            "an un-journaled step was replayed from disk"
        );
    }
}
