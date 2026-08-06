//! Runtime lifecycle events — emitted during plan execution.
//!
//! Events track run/node/study/trial state transitions and are
//! broadcast via the runtime's `EventBus` for observability and debugging.

use crate::cache::{CacheKey, CacheTier};
use crate::graph::NodeId;
use crate::graph::filter::FilterKind;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Unique identifier for a pipeline run.
pub type RunId = String;

/// Unique identifier for an optimization study.
pub type StudyId = String;

/// Unique identifier for a trial within a study.
pub type TrialId = String;

/// A metric measurement reported during training.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MetricRecord {
    /// Metric name, e.g. `loss` or `val_f1`.
    pub name: String,
    /// The measured value.
    pub value: f64,
    /// Training step at which the measurement was taken.
    pub step: usize,
    /// When the measurement was recorded.
    pub timestamp: DateTime<Utc>,
}

/// Summary of a compiled plan (for event payloads without the full plan).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanSummary {
    /// Number of nodes in the compiled plan.
    pub total_nodes: usize,
    /// Always 0 from a compiled plan, and not a placeholder: cache keys are
    /// resolved per node at run time (a node's key depends on its input's
    /// content hash), so at `RunStarted` nothing is yet known about what
    /// will be served from cache. The answer arrives as
    /// [`Event::NodeCacheHit`] per node — count those, not this.
    pub cached_nodes: usize,
    /// Number of branches the plan can execute concurrently.
    pub parallel_branches: usize,
}

/// Structured events emitted during execution at three levels.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event_type")]
#[non_exhaustive]
pub enum Event {
    // ── Level 1: Pipeline execution (per run) ──
    /// A pipeline run has started.
    RunStarted {
        /// The run this event belongs to.
        run_id: RunId,
        /// Coarse shape of the plan about to execute — see
        /// [`PlanSummary::cached_nodes`] for what it deliberately cannot say.
        plan_summary: PlanSummary,
    },

    /// A filter node has started execution.
    NodeStarted {
        /// The run this event belongs to.
        run_id: RunId,
        /// The node this event concerns.
        node_id: NodeId,
        /// Structural kind of the node, from its metadata — see [`FilterKind`].
        kind: FilterKind,
        /// Does this node reach outside the graph — a model, a tool, a
        /// person?
        ///
        /// An effectful node has no honest [`FilterKind`], and reporting
        /// it as `Opaque` made every consumer see an agent as a filter it
        /// could not look inside. Defaulted so run logs written before
        /// this field existed still parse.
        #[serde(default)]
        effectful: bool,
    },

    /// A filter node reports progress (0.0 to 1.0).
    NodeProgress {
        /// The run this event belongs to.
        run_id: RunId,
        /// The node this event concerns.
        node_id: NodeId,
        /// Fraction complete, from 0.0 to 1.0.
        progress: f32,
    },

    /// A filter node's result was loaded from cache.
    NodeCacheHit {
        /// The run this event belongs to.
        run_id: RunId,
        /// The node this event concerns.
        node_id: NodeId,
        /// The key the result was found under.
        key: CacheKey,
        /// Which [`CacheTier`] served the hit.
        tier: CacheTier,
        /// Time spent loading the cached result.
        #[serde(with = "duration_millis")]
        load_time: Duration,
    },

    /// A cacheable node's key was computed but not found — the filter
    /// executes and (on success) fills this key.
    NodeCacheMiss {
        /// The run this event belongs to.
        run_id: RunId,
        /// The node this event concerns.
        node_id: NodeId,
        /// The key that was looked up and not found — the same key the
        /// node's output will be stored under on success.
        key: CacheKey,
    },

    /// A filter node completed successfully.
    NodeCompleted {
        /// The run this event belongs to.
        run_id: RunId,
        /// The node this event concerns.
        node_id: NodeId,
        /// Wall time from start to finish. Control-flow constructs (loops,
        /// branches) report zero — their time is in the nodes they ran.
        #[serde(with = "duration_millis")]
        duration: Duration,
        /// Short human-readable rendering of the output, never the payload
        /// itself.
        output_summary: String,
    },

    /// A filter node failed.
    NodeFailed {
        /// The run this event belongs to.
        run_id: RunId,
        /// The node this event concerns.
        node_id: NodeId,
        /// The error, rendered as a string.
        error: String,
    },

    /// The pipeline run completed.
    RunCompleted {
        /// The run this event belongs to.
        run_id: RunId,
        /// Wall time for the whole run.
        #[serde(with = "duration_millis")]
        duration: Duration,
    },

    /// The pipeline run failed.
    RunFailed {
        /// The run this event belongs to.
        run_id: RunId,
        /// The error, rendered as a string.
        error: String,
    },

    // ── Level 2: Trial execution (per hyperparameter set) ──
    /// A new trial has started.
    TrialStarted {
        /// The study this event belongs to.
        study_id: StudyId,
        /// The trial this event concerns.
        trial_id: TrialId,
        /// The hyperparameters sampled for this trial, as JSON.
        params: serde_json::Value,
    },

    /// A trial reports an intermediate metric.
    TrialMetric {
        /// The study this event belongs to.
        study_id: StudyId,
        /// The trial this event concerns.
        trial_id: TrialId,
        /// The intermediate measurement — what pruners decide on.
        metric: MetricRecord,
    },

    /// A trial was pruned (stopped early).
    TrialPruned {
        /// The study this event belongs to.
        study_id: StudyId,
        /// The trial this event concerns.
        trial_id: TrialId,
        /// The training step at which the pruner struck.
        step: usize,
        /// Why the pruner stopped it, human-readable (e.g. `below median`).
        reason: String,
    },

    /// A trial completed successfully.
    TrialCompleted {
        /// The study this event belongs to.
        study_id: StudyId,
        /// The trial this event concerns.
        trial_id: TrialId,
        /// The measurements the trial finished with — the values the
        /// sampler and the study's best-trial bookkeeping consume.
        final_metrics: Vec<MetricRecord>,
    },

    /// A trial failed.
    TrialFailed {
        /// The study this event belongs to.
        study_id: StudyId,
        /// The trial this event concerns.
        trial_id: TrialId,
        /// The error, rendered as a string.
        error: String,
    },

    // ── Level 3: Study execution (optimization session) ──
    /// An optimization study has started.
    StudyStarted {
        /// The study this event belongs to.
        study_id: StudyId,
        /// Human-readable study name.
        name: String,
        /// How many trials the study intends to run.
        total_trials: usize,
    },

    /// Study progress update.
    StudyProgress {
        /// The study this event belongs to.
        study_id: StudyId,
        /// Trials finished so far.
        completed: usize,
        /// Total trials planned.
        total: usize,
        /// Best objective value seen so far; `NaN` until a trial has
        /// completed (consumers render it as "no best yet", they do not
        /// compare it).
        best_value: f64,
    },

    /// The best trial has been updated.
    BestUpdated {
        /// The study this event belongs to.
        study_id: StudyId,
        /// The trial this event concerns.
        trial_id: TrialId,
        /// The new best objective value.
        value: f64,
        /// The hyperparameters that produced it, as JSON.
        params: serde_json::Value,
    },

    /// The Pareto front has changed (multi-objective).
    ParetoUpdated {
        /// The study this event belongs to.
        study_id: StudyId,
        /// Number of non-dominated trials after the update.
        front_size: usize,
    },

    /// The study completed.
    StudyCompleted {
        /// The study this event belongs to.
        study_id: StudyId,
        /// The winning trial.
        best_trial_id: TrialId,
        /// Its objective value (`NaN` when no trial completed).
        best_value: f64,
    },

    // ── Level 4: Population-Based Training ──
    /// A PBT generation started (train → evaluate → exploit/explore).
    GenerationStarted {
        /// The study this event belongs to.
        study_id: StudyId,
        /// The generation index.
        generation: usize,
        /// How many population members train in this generation.
        population_size: usize,
    },

    /// A PBT generation completed.
    GenerationCompleted {
        /// The study this event belongs to.
        study_id: StudyId,
        /// The generation index.
        generation: usize,
        /// Best fitness in the population after evaluation.
        best_fitness: f64,
        /// Mean fitness across the population — tracks whether the whole
        /// population improves, not just its champion.
        mean_fitness: f64,
    },

    /// A population member was replaced during exploit step.
    MemberExploited {
        /// The study this event belongs to.
        study_id: StudyId,
        /// The generation index.
        generation: usize,
        /// The underperforming member whose params and state were
        /// overwritten.
        replaced_id: String,
        /// The better-performing member it copied them from (explore then
        /// perturbs the copy).
        donor_id: String,
    },

    // ── Level 5: Training telemetry (native training loop) ──
    /// A training epoch started.
    EpochStarted {
        /// The run this event belongs to.
        run_id: RunId,
        /// Zero-based epoch index.
        epoch: usize,
        /// Planned epoch count when the loop knows it up front; `None` for
        /// open-ended training, where progress cannot be a percentage.
        total_epochs: Option<usize>,
    },

    /// A training epoch completed with its summary metrics.
    EpochCompleted {
        /// The run this event belongs to.
        run_id: RunId,
        /// Zero-based epoch index.
        epoch: usize,
        /// Summary metrics for the epoch (e.g. mean loss).
        metrics: Vec<MetricRecord>,
    },

    /// One optimizer step completed (coarse liveness marker).
    StepCompleted {
        /// The run this event belongs to.
        run_id: RunId,
        /// The optimizer step index.
        step: usize,
        /// The epoch this step belongs to, when the loop tracks one.
        epoch: Option<usize>,
    },

    /// A user- or node-scoped metric reported outside a trial.
    MetricReported {
        /// The run this event belongs to.
        run_id: RunId,
        /// The measurement.
        metric: MetricRecord,
        /// The node the metric is scoped to, if any.
        node_id: Option<NodeId>,
        /// The trial the metric is scoped to, if any.
        trial_id: Option<TrialId>,
    },

    /// A training-health diagnostic fired for a node (e.g.
    /// `DEAD_CHANNELS`, `IGNORED_CHANNELS`, `LEAKAGE`, `NONFINITE`).
    HealthFlag {
        /// The run this event belongs to.
        run_id: RunId,
        /// The node this event concerns.
        node_id: NodeId,
        /// The training step at which the diagnostic fired.
        step: usize,
        /// The flag family, optionally with a count — e.g.
        /// `DEAD_CHANNELS(3)`.
        flag: String,
        /// Supporting numbers behind the flag, e.g. `zero_frac=0.98`.
        detail: String,
    },

    // ── Level 6: Effectful steps (agentic execution) ──
    //
    // Payloads carry *labels*, never prompts or completions. These events
    // land in `.soma/runs/<id>/events.jsonl` and are rendered into reports;
    // the conversation itself belongs in the journal, which is subject to
    // `StepMeta::journal`, not in a telemetry stream.
    //
    // Naming note: `AgentTurnStarted`/`AgentStepCompleted` are spelled out
    // because `StepCompleted` above already means "one optimizer step".
    /// A step began a turn (one `poll`).
    AgentTurnStarted {
        /// The run this event belongs to.
        run_id: RunId,
        /// The node this event concerns.
        node_id: NodeId,
        /// Zero-based index of the turn being started.
        turn: usize,
    },

    /// A step asked for an effect to be performed.
    EffectRequested {
        /// The run this event belongs to.
        run_id: RunId,
        /// The node this event concerns.
        node_id: NodeId,
        /// The turn (one `poll`) that requested the effect.
        turn: usize,
        /// `Effect::label()` — e.g. `llm:claude-opus-5`, `tool:search`.
        effect: String,
    },

    /// An effect finished.
    EffectCompleted {
        /// The run this event belongs to.
        run_id: RunId,
        /// The node this event concerns.
        node_id: NodeId,
        /// The turn (one `poll`) that requested the effect.
        turn: usize,
        /// `Effect::label()` — matches the [`Event::EffectRequested`] this
        /// answers.
        effect: String,
        /// How long performing (or replaying) the effect took.
        #[serde(with = "duration_millis")]
        duration: Duration,
        /// Served from the journal rather than actually performed.
        /// A replay should be nearly all `true`.
        replayed: bool,
        /// The effect's result was an error. It is still delivered to the
        /// step, which decides whether that is fatal.
        is_error: bool,
    },

    /// A tool ran. Separate from `EffectCompleted` because tool usage is the
    /// thing worth counting per run, and it is what a permission or audit
    /// layer hooks into.
    ToolCalled {
        /// The run this event belongs to.
        run_id: RunId,
        /// The node this event concerns.
        node_id: NodeId,
        /// The tool's name.
        tool: String,
        /// The tool returned an error result.
        is_error: bool,
    },

    /// Control passed from one node to another.
    Handoff {
        /// The run this event belongs to.
        run_id: RunId,
        /// The node handing control off.
        from: NodeId,
        /// The node receiving it.
        to: NodeId,
    },

    /// The run stopped, pending something outside it.
    ///
    /// Carries what the step has cost *so far*: a suspended step has not
    /// finished, so no [`Event::AgentStepCompleted`] fires for it. When the
    /// run resumes and finishes, that final event's totals are cumulative
    /// (replayed effects re-count their recorded usage), superseding these.
    /// The fields default to zero so run dirs written before they existed
    /// still deserialize.
    Suspended {
        /// The run this event belongs to.
        run_id: RunId,
        /// The node this event concerns.
        node_id: NodeId,
        /// What the run is waiting on, as a short label (e.g. `human`).
        reason: String,
        /// Turns taken up to and including the one that suspended.
        #[serde(default)]
        turns: usize,
        /// Wall time spent so far.
        #[serde(default, with = "duration_millis")]
        duration: Duration,
        /// LLM input tokens consumed so far.
        #[serde(default)]
        input_tokens: u64,
        /// LLM output tokens generated so far.
        #[serde(default)]
        output_tokens: u64,
    },

    /// A suspended run picked up again.
    Resumed {
        /// The run this event belongs to.
        run_id: RunId,
        /// The node this event concerns.
        node_id: NodeId,
        /// The turn the step resumes at.
        turn: usize,
    },

    /// A step finished, with what it cost — however it finished.
    ///
    /// `Done`, a handoff, a failed poll or effect, and turn exhaustion all
    /// emit this: the cost was paid either way, and telemetry that loses
    /// the expensive failures undercounts exactly the runs worth studying.
    /// Suspension is the one exit that does not — see [`Event::Suspended`].
    AgentStepCompleted {
        /// The run this event belongs to.
        run_id: RunId,
        /// The node this event concerns.
        node_id: NodeId,
        /// Total turns taken.
        turns: usize,
        /// Total wall time across all turns.
        #[serde(with = "duration_millis")]
        duration: Duration,
        /// Total LLM input tokens consumed. Cumulative across a resume:
        /// replayed effects re-count their recorded usage.
        input_tokens: u64,
        /// Total LLM output tokens generated (same accounting as
        /// `input_tokens`).
        output_tokens: u64,
        /// The step ended in an error (the node will also emit
        /// [`Event::NodeFailed`]). Defaults to `false` so run dirs written
        /// before the field existed still deserialize.
        #[serde(default)]
        failed: bool,
    },

    /// A step fanned work out to spawned instances.
    ///
    /// `children` are the hierarchical ids (`parent/label`) the instances
    /// run under; their own turn and completion events appear under those
    /// ids, which is how a reader ties the sub-tree together.
    AgentSpawned {
        /// The run this event belongs to.
        run_id: RunId,
        /// The node this event concerns.
        node_id: NodeId,
        /// The turn (one `poll`) that spawned the instances.
        turn: usize,
        /// Hierarchical ids (`parent/label`) the spawned instances run
        /// under.
        children: Vec<NodeId>,
        /// The join policy, as a label (`all`, `all-settled`, `first`).
        join: String,
    },
}

/// Serde helper: Duration as milliseconds (u64).
mod duration_millis {
    use serde::{self, Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(duration.as_millis() as u64)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let millis = u64::deserialize(deserializer)?;
        Ok(Duration::from_millis(millis))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_serde_run_started() {
        let event = Event::RunStarted {
            run_id: "run_001".into(),
            plan_summary: PlanSummary {
                total_nodes: 5,
                cached_nodes: 2,
                parallel_branches: 1,
            },
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("RunStarted"));
        let deserialized: Event = serde_json::from_str(&json).unwrap();
        if let Event::RunStarted {
            run_id,
            plan_summary,
        } = deserialized
        {
            assert_eq!(run_id, "run_001");
            assert_eq!(plan_summary.total_nodes, 5);
        } else {
            panic!("wrong variant");
        }
    }

    /// Agent-level events ride the same envelope as everything else — that
    /// is the point of putting them on the existing bus rather than building
    /// a second telemetry path.
    #[test]
    fn agent_events_roundtrip() {
        let events = vec![
            Event::AgentTurnStarted {
                run_id: "r".into(),
                node_id: "researcher".into(),
                turn: 0,
            },
            Event::EffectRequested {
                run_id: "r".into(),
                node_id: "researcher".into(),
                turn: 0,
                effect: "llm:claude-opus-5".into(),
            },
            Event::EffectCompleted {
                run_id: "r".into(),
                node_id: "researcher".into(),
                turn: 0,
                effect: "llm:claude-opus-5".into(),
                duration: Duration::from_millis(1200),
                replayed: false,
                is_error: false,
            },
            Event::ToolCalled {
                run_id: "r".into(),
                node_id: "researcher".into(),
                tool: "search".into(),
                is_error: false,
            },
            Event::Handoff {
                run_id: "r".into(),
                from: "router".into(),
                to: "billing".into(),
            },
            Event::Suspended {
                run_id: "r".into(),
                node_id: "approve".into(),
                reason: "human".into(),
                turns: 2,
                duration: Duration::from_millis(3400),
                input_tokens: 600,
                output_tokens: 120,
            },
            Event::Resumed {
                run_id: "r".into(),
                node_id: "approve".into(),
                turn: 3,
            },
            Event::AgentStepCompleted {
                run_id: "r".into(),
                node_id: "researcher".into(),
                turns: 4,
                duration: Duration::from_millis(8000),
                input_tokens: 1200,
                output_tokens: 340,
                failed: false,
            },
            Event::AgentSpawned {
                run_id: "r".into(),
                node_id: "orchestrator".into(),
                turn: 1,
                children: vec!["orchestrator/web".into(), "orchestrator/code".into()],
                join: "all".into(),
            },
        ];

        for event in events {
            let json = serde_json::to_string(&event).unwrap();
            let back: Event = serde_json::from_str(&json).unwrap();
            assert_eq!(
                serde_json::to_string(&back).unwrap(),
                json,
                "agent event did not survive a round trip"
            );
        }
    }

    /// Run dirs written before `Suspended` carried cost and
    /// `AgentStepCompleted` carried `failed` must still read back — the
    /// fields default rather than fail the whole line.
    #[test]
    fn agent_events_read_back_without_the_newer_fields() {
        let suspended: Event = serde_json::from_str(
            r#"{"event_type":"Suspended","run_id":"r","node_id":"approve","reason":"human"}"#,
        )
        .unwrap();
        let Event::Suspended {
            turns,
            input_tokens,
            ..
        } = suspended
        else {
            panic!("wrong variant");
        };
        assert_eq!((turns, input_tokens), (0, 0));

        let completed: Event = serde_json::from_str(
            r#"{"event_type":"AgentStepCompleted","run_id":"r","node_id":"n","turns":2,
                "duration":100,"input_tokens":10,"output_tokens":5}"#,
        )
        .unwrap();
        let Event::AgentStepCompleted { failed, .. } = completed else {
            panic!("wrong variant");
        };
        assert!(!failed);
    }

    /// Telemetry must never carry the conversation. A prompt belongs in the
    /// journal, which honours `StepMeta::journal`; an event stream does not.
    #[test]
    fn effect_events_carry_labels_not_payloads() {
        let effect = crate::agentic::effect::Effect::Llm(crate::agentic::effect::LlmRequest::new(
            "claude-opus-5",
            vec![crate::agentic::message::Message::user("my secret prompt")].into(),
        ));
        let event = Event::EffectRequested {
            run_id: "r".into(),
            node_id: "n".into(),
            turn: 0,
            effect: effect.label(),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("claude-opus-5"));
        assert!(
            !json.contains("secret"),
            "the prompt leaked into telemetry: {json}"
        );
    }

    #[test]
    fn event_serde_node_cache_hit() {
        let event = Event::NodeCacheHit {
            run_id: "run_001".into(),
            node_id: "scaler".into(),
            key: CacheKey::hash_data(b"test"),
            tier: CacheTier::Memory,
            load_time: Duration::from_micros(200),
        };
        let json = serde_json::to_string(&event).unwrap();
        let deserialized: Event = serde_json::from_str(&json).unwrap();
        if let Event::NodeCacheHit { tier, .. } = deserialized {
            assert_eq!(tier, CacheTier::Memory);
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn event_serde_trial_metric() {
        let event = Event::TrialMetric {
            study_id: "study_001".into(),
            trial_id: "trial_042".into(),
            metric: MetricRecord {
                name: "f1".into(),
                value: 0.847,
                step: 15,
                timestamp: Utc::now(),
            },
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("TrialMetric"));
        assert!(json.contains("0.847"));
    }

    #[test]
    fn event_serde_study_completed() {
        let event = Event::StudyCompleted {
            study_id: "study_001".into(),
            best_trial_id: "trial_042".into(),
            best_value: 0.91,
        };
        let json = serde_json::to_string(&event).unwrap();
        let deserialized: Event = serde_json::from_str(&json).unwrap();
        if let Event::StudyCompleted { best_value, .. } = deserialized {
            assert!((best_value - 0.91).abs() < f64::EPSILON);
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn duration_serialized_as_millis() {
        let event = Event::NodeCompleted {
            run_id: "r".into(),
            node_id: "n".into(),
            duration: Duration::from_millis(1234),
            output_summary: "ok".into(),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("1234"));
    }

    #[test]
    fn all_event_levels_serialize() {
        let events: Vec<Event> = vec![
            // Level 1
            Event::RunStarted {
                run_id: "r".into(),
                plan_summary: PlanSummary {
                    total_nodes: 1,
                    cached_nodes: 0,
                    parallel_branches: 0,
                },
            },
            Event::RunCompleted {
                run_id: "r".into(),
                duration: Duration::from_secs(1),
            },
            // Level 2
            Event::TrialStarted {
                study_id: "s".into(),
                trial_id: "t".into(),
                params: serde_json::json!({"lr": 0.01}),
            },
            Event::TrialPruned {
                study_id: "s".into(),
                trial_id: "t".into(),
                step: 5,
                reason: "below median".into(),
            },
            // Level 3
            Event::StudyStarted {
                study_id: "s".into(),
                name: "test".into(),
                total_trials: 100,
            },
            Event::BestUpdated {
                study_id: "s".into(),
                trial_id: "t".into(),
                value: 0.95,
                params: serde_json::json!({"C": 1.0}),
            },
            // Level 4
            Event::GenerationCompleted {
                study_id: "s".into(),
                generation: 2,
                best_fitness: 0.9,
                mean_fitness: 0.7,
            },
            // Level 5
            Event::EpochStarted {
                run_id: "r".into(),
                epoch: 0,
                total_epochs: Some(30),
            },
            Event::EpochStarted {
                run_id: "r".into(),
                epoch: 1,
                total_epochs: None,
            },
            Event::EpochCompleted {
                run_id: "r".into(),
                epoch: 0,
                metrics: vec![MetricRecord {
                    name: "loss".into(),
                    value: 0.4,
                    step: 12,
                    timestamp: chrono::Utc::now(),
                }],
            },
            Event::StepCompleted {
                run_id: "r".into(),
                step: 7,
                epoch: Some(1),
            },
            Event::StepCompleted {
                run_id: "r".into(),
                step: 8,
                epoch: None,
            },
            Event::MetricReported {
                run_id: "r".into(),
                metric: MetricRecord {
                    name: "val_f1".into(),
                    value: 0.8,
                    step: 3,
                    timestamp: chrono::Utc::now(),
                },
                node_id: Some("encoder".into()),
                trial_id: Some("trial_0001".into()),
            },
            Event::HealthFlag {
                run_id: "r".into(),
                node_id: "encoder".into(),
                step: 50,
                flag: "DEAD_CHANNELS(3)".into(),
                detail: "zero_frac=0.98".into(),
            },
        ];

        for event in events {
            let json = serde_json::to_string(&event).unwrap();
            let back: Event = serde_json::from_str(&json).unwrap();
            // Typed roundtrip must preserve the variant and its Options.
            assert_eq!(
                serde_json::to_value(&back).unwrap(),
                serde_json::from_str::<serde_json::Value>(&json).unwrap()
            );
        }
    }

    #[test]
    fn documented_health_flags_roundtrip() {
        for flag in [
            "DEAD_CHANNELS(2)",
            "IGNORED_CHANNELS(1)",
            "LEAKAGE",
            "NONFINITE",
        ] {
            let event = Event::HealthFlag {
                run_id: "r".into(),
                node_id: "n".into(),
                step: 0,
                flag: flag.into(),
                detail: String::new(),
            };
            let json = serde_json::to_string(&event).unwrap();
            let back: Event = serde_json::from_str(&json).unwrap();
            if let Event::HealthFlag { flag: f, .. } = back {
                assert_eq!(f, flag);
            } else {
                panic!("wrong variant");
            }
        }
    }
}
