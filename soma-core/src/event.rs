//! Runtime lifecycle events — emitted during plan execution.
//!
//! Events track run/node/study/trial state transitions and are
//! broadcast via the runtime's `EventBus` for observability and debugging.

use crate::cache::{CacheKey, CacheTier};
use crate::filter::FilterKind;
use crate::graph::NodeId;
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
    pub name: String,
    pub value: f64,
    pub step: usize,
    pub timestamp: DateTime<Utc>,
}

/// Summary of a compiled plan (for event payloads without the full plan).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanSummary {
    pub total_nodes: usize,
    pub cached_nodes: usize,
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
        run_id: RunId,
        plan_summary: PlanSummary,
    },

    /// A filter node has started execution.
    NodeStarted {
        run_id: RunId,
        node_id: NodeId,
        kind: FilterKind,
    },

    /// A filter node reports progress (0.0 to 1.0).
    NodeProgress {
        run_id: RunId,
        node_id: NodeId,
        progress: f32,
    },

    /// A filter node's result was loaded from cache.
    NodeCacheHit {
        run_id: RunId,
        node_id: NodeId,
        key: CacheKey,
        tier: CacheTier,
        #[serde(with = "duration_millis")]
        load_time: Duration,
    },

    /// A cacheable node's key was computed but not found — the filter
    /// executes and (on success) fills this key.
    NodeCacheMiss {
        run_id: RunId,
        node_id: NodeId,
        key: CacheKey,
    },

    /// A filter node completed successfully.
    NodeCompleted {
        run_id: RunId,
        node_id: NodeId,
        #[serde(with = "duration_millis")]
        duration: Duration,
        output_summary: String,
    },

    /// A filter node failed.
    NodeFailed {
        run_id: RunId,
        node_id: NodeId,
        error: String,
    },

    /// The pipeline run completed.
    RunCompleted {
        run_id: RunId,
        #[serde(with = "duration_millis")]
        duration: Duration,
    },

    /// The pipeline run failed.
    RunFailed { run_id: RunId, error: String },

    // ── Level 2: Trial execution (per hyperparameter set) ──
    /// A new trial has started.
    TrialStarted {
        study_id: StudyId,
        trial_id: TrialId,
        params: serde_json::Value,
    },

    /// A trial reports an intermediate metric.
    TrialMetric {
        study_id: StudyId,
        trial_id: TrialId,
        metric: MetricRecord,
    },

    /// A trial was pruned (stopped early).
    TrialPruned {
        study_id: StudyId,
        trial_id: TrialId,
        step: usize,
        reason: String,
    },

    /// A trial completed successfully.
    TrialCompleted {
        study_id: StudyId,
        trial_id: TrialId,
        final_metrics: Vec<MetricRecord>,
    },

    /// A trial failed.
    TrialFailed {
        study_id: StudyId,
        trial_id: TrialId,
        error: String,
    },

    // ── Level 3: Study execution (optimization session) ──
    /// An optimization study has started.
    StudyStarted {
        study_id: StudyId,
        name: String,
        total_trials: usize,
    },

    /// Study progress update.
    StudyProgress {
        study_id: StudyId,
        completed: usize,
        total: usize,
        best_value: f64,
    },

    /// The best trial has been updated.
    BestUpdated {
        study_id: StudyId,
        trial_id: TrialId,
        value: f64,
        params: serde_json::Value,
    },

    /// The Pareto front has changed (multi-objective).
    ParetoUpdated {
        study_id: StudyId,
        front_size: usize,
    },

    /// The study completed.
    StudyCompleted {
        study_id: StudyId,
        best_trial_id: TrialId,
        best_value: f64,
    },

    // ── Level 4: Population-Based Training ──
    /// A PBT generation started (train → evaluate → exploit/explore).
    GenerationStarted {
        study_id: StudyId,
        generation: usize,
        population_size: usize,
    },

    /// A PBT generation completed.
    GenerationCompleted {
        study_id: StudyId,
        generation: usize,
        best_fitness: f64,
        mean_fitness: f64,
    },

    /// A population member was replaced during exploit step.
    MemberExploited {
        study_id: StudyId,
        generation: usize,
        replaced_id: String,
        donor_id: String,
    },

    // ── Level 5: Training telemetry (native training loop) ──
    /// A training epoch started.
    EpochStarted {
        run_id: RunId,
        epoch: usize,
        total_epochs: Option<usize>,
    },

    /// A training epoch completed with its summary metrics.
    EpochCompleted {
        run_id: RunId,
        epoch: usize,
        metrics: Vec<MetricRecord>,
    },

    /// One optimizer step completed (coarse liveness marker).
    StepCompleted {
        run_id: RunId,
        step: usize,
        epoch: Option<usize>,
    },

    /// A user- or node-scoped metric reported outside a trial.
    MetricReported {
        run_id: RunId,
        metric: MetricRecord,
        node_id: Option<NodeId>,
        trial_id: Option<TrialId>,
    },

    /// A training-health diagnostic fired for a node (e.g.
    /// `DEAD_CHANNELS`, `IGNORED_CHANNELS`, `LEAKAGE`, `NONFINITE`).
    HealthFlag {
        run_id: RunId,
        node_id: NodeId,
        step: usize,
        flag: String,
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
        run_id: RunId,
        node_id: NodeId,
        turn: usize,
    },

    /// A step asked for an effect to be performed.
    EffectRequested {
        run_id: RunId,
        node_id: NodeId,
        turn: usize,
        /// `Effect::label()` — e.g. `llm:claude-opus-5`, `tool:search`.
        effect: String,
    },

    /// An effect finished.
    EffectCompleted {
        run_id: RunId,
        node_id: NodeId,
        turn: usize,
        effect: String,
        #[serde(with = "duration_millis")]
        duration: Duration,
        /// Served from the journal rather than actually performed.
        /// A replay should be nearly all `true`.
        replayed: bool,
        is_error: bool,
    },

    /// A tool ran. Separate from `EffectCompleted` because tool usage is the
    /// thing worth counting per run, and it is what a permission or audit
    /// layer hooks into.
    ToolCalled {
        run_id: RunId,
        node_id: NodeId,
        tool: String,
        is_error: bool,
    },

    /// Control passed from one node to another.
    Handoff {
        run_id: RunId,
        from: NodeId,
        to: NodeId,
    },

    /// The run stopped, pending something outside it.
    Suspended {
        run_id: RunId,
        node_id: NodeId,
        reason: String,
    },

    /// A suspended run picked up again.
    Resumed {
        run_id: RunId,
        node_id: NodeId,
        turn: usize,
    },

    /// A step finished, with what it cost.
    AgentStepCompleted {
        run_id: RunId,
        node_id: NodeId,
        turns: usize,
        #[serde(with = "duration_millis")]
        duration: Duration,
        input_tokens: u64,
        output_tokens: u64,
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

    /// Telemetry must never carry the conversation. A prompt belongs in the
    /// journal, which honours `StepMeta::journal`; an event stream does not.
    #[test]
    fn effect_events_carry_labels_not_payloads() {
        let effect = crate::effect::Effect::Llm(crate::effect::LlmRequest::new(
            "claude-opus-5",
            vec![crate::message::Message::user("my secret prompt")].into(),
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
