//! Runtime lifecycle events — emitted during plan execution.
//!
//! Events track run/node/study/trial state transitions and are
//! broadcast via the [`EventBus`] for observability and debugging.

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
    fn all_three_event_levels_serialize() {
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
        ];

        for event in events {
            let json = serde_json::to_string(&event).unwrap();
            let _: Event = serde_json::from_str(&json).unwrap();
        }
    }
}
