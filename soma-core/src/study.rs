use crate::event::MetricRecord;
use crate::search::SearchSpace;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Direction of optimization for an objective.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    Minimize,
    Maximize,
}

/// An optimization objective (metric + direction).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Objective {
    pub metric: String,
    pub direction: Direction,
}

/// Search strategy for hyperparameter optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "strategy_type")]
pub enum SearchStrategy {
    /// Exhaustive grid search.
    Grid { points_per_dim: usize },

    /// Random sampling.
    Random { n_trials: usize, seed: Option<u64> },

    /// Bayesian optimization (TPE).
    Bayesian {
        n_trials: usize,
        n_startup: usize,
        seed: Option<u64>,
    },

    /// Successive halving with early stopping.
    Hyperband {
        max_resource: usize,
        reduction_factor: usize,
    },

    /// Multi-objective optimization.
    MultiObjective {
        n_trials: usize,
        objectives: Vec<Objective>,
    },
}

impl SearchStrategy {
    /// Planned number of trials (if known).
    pub fn n_trials(&self) -> Option<usize> {
        match self {
            Self::Grid { .. } => None, // depends on search space
            Self::Random { n_trials, .. } => Some(*n_trials),
            Self::Bayesian { n_trials, .. } => Some(*n_trials),
            Self::Hyperband { .. } => None, // depends on brackets
            Self::MultiObjective { n_trials, .. } => Some(*n_trials),
        }
    }
}

/// Pruning strategy for early stopping of unpromising trials.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "pruning_type")]
pub enum PruningStrategy {
    /// No pruning.
    None,

    /// Prune if metric is below median of completed trials at same step.
    Median { n_warmup_steps: usize },

    /// Prune if metric is below given percentile.
    Percentile {
        percentile: f64,
        n_warmup_steps: usize,
    },

    /// Bracket-based pruning (used with Hyperband).
    Hyperband,
}

/// State of a single trial.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "trial_state")]
pub enum TrialState {
    Pending,
    Running,
    Completed,
    Pruned { step: usize, reason: String },
    Failed { error: String },
}

/// A single hyperparameter evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trial {
    pub id: String,
    pub params: HashMap<String, serde_json::Value>,
    pub state: TrialState,
    pub metrics: Vec<MetricRecord>,
    pub duration_ms: Option<u64>,
}

impl Trial {
    pub fn new(id: impl Into<String>, params: HashMap<String, serde_json::Value>) -> Self {
        Self {
            id: id.into(),
            params,
            state: TrialState::Pending,
            metrics: Vec::new(),
            duration_ms: None,
        }
    }

    /// Get the last recorded value for a specific metric.
    pub fn best_metric(&self, name: &str, direction: Direction) -> Option<f64> {
        let values: Vec<f64> = self
            .metrics
            .iter()
            .filter(|m| m.name == name)
            .map(|m| m.value)
            .collect();
        match direction {
            Direction::Maximize => values.into_iter().reduce(f64::max),
            Direction::Minimize => values.into_iter().reduce(f64::min),
        }
    }

    pub fn is_complete(&self) -> bool {
        matches!(self.state, TrialState::Completed)
    }

    pub fn is_terminal(&self) -> bool {
        matches!(
            self.state,
            TrialState::Completed | TrialState::Pruned { .. } | TrialState::Failed { .. }
        )
    }
}

/// An optimization study: orchestrates multiple trials.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Study {
    pub id: String,
    pub name: String,
    pub search_space: SearchSpace,
    pub strategy: SearchStrategy,
    pub pruning: PruningStrategy,
    pub objectives: Vec<Objective>,
    pub trials: Vec<Trial>,
    pub frozen: HashMap<String, serde_json::Value>,
}

impl Study {
    pub fn new(
        name: impl Into<String>,
        search_space: SearchSpace,
        strategy: SearchStrategy,
        objectives: Vec<Objective>,
    ) -> Self {
        Self {
            id: uuid_v4(),
            name: name.into(),
            search_space,
            strategy,
            pruning: PruningStrategy::None,
            objectives,
            trials: Vec::new(),
            frozen: HashMap::new(),
        }
    }

    pub fn with_pruning(mut self, pruning: PruningStrategy) -> Self {
        self.pruning = pruning;
        self
    }

    /// Get completed trials.
    pub fn completed_trials(&self) -> Vec<&Trial> {
        self.trials.iter().filter(|t| t.is_complete()).collect()
    }

    /// Get the best trial for the primary objective.
    pub fn best_trial(&self) -> Option<&Trial> {
        let obj = self.objectives.first()?;
        self.completed_trials()
            .into_iter()
            .filter_map(|t| {
                let val = t.best_metric(&obj.metric, obj.direction)?;
                Some((t, val))
            })
            .reduce(|best, current| match obj.direction {
                Direction::Maximize => {
                    if current.1 > best.1 {
                        current
                    } else {
                        best
                    }
                }
                Direction::Minimize => {
                    if current.1 < best.1 {
                        current
                    } else {
                        best
                    }
                }
            })
            .map(|(t, _)| t)
    }

    /// Number of total planned trials (if known).
    pub fn total_trials(&self) -> Option<usize> {
        self.strategy.n_trials()
    }

    /// Fraction of trials completed.
    pub fn progress(&self) -> f64 {
        let completed = self.trials.iter().filter(|t| t.is_terminal()).count();
        match self.total_trials() {
            Some(total) if total > 0 => completed as f64 / total as f64,
            _ => 0.0,
        }
    }
}

fn uuid_v4() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("study_{nanos:x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::{Scale, SearchDimension};
    use chrono::Utc;
    use serde_json::json;

    fn sample_search_space() -> SearchSpace {
        let mut space = SearchSpace::new();
        space.add(SearchDimension::Float {
            name: "lr".into(),
            low: 0.001,
            high: 0.1,
            scale: Scale::Log,
            default: None,
        });
        space.add(SearchDimension::Categorical {
            name: "kernel".into(),
            choices: vec![json!("rbf"), json!("linear")],
        });
        space
    }

    fn make_trial(id: &str, f1: f64) -> Trial {
        let mut t = Trial::new(id, HashMap::from([("lr".into(), json!(0.01))]));
        t.state = TrialState::Completed;
        t.metrics.push(MetricRecord {
            name: "f1".into(),
            value: f1,
            step: 10,
            timestamp: Utc::now(),
        });
        t
    }

    #[test]
    fn study_best_trial_maximize() {
        let mut study = Study::new(
            "test",
            sample_search_space(),
            SearchStrategy::Random {
                n_trials: 10,
                seed: None,
            },
            vec![Objective {
                metric: "f1".into(),
                direction: Direction::Maximize,
            }],
        );

        study.trials.push(make_trial("t1", 0.75));
        study.trials.push(make_trial("t2", 0.90));
        study.trials.push(make_trial("t3", 0.82));

        let best = study.best_trial().unwrap();
        assert_eq!(best.id, "t2");
    }

    #[test]
    fn study_best_trial_minimize() {
        let mut study = Study::new(
            "test",
            sample_search_space(),
            SearchStrategy::Random {
                n_trials: 10,
                seed: None,
            },
            vec![Objective {
                metric: "loss".into(),
                direction: Direction::Minimize,
            }],
        );

        let mut t1 = Trial::new("t1", HashMap::new());
        t1.state = TrialState::Completed;
        t1.metrics.push(MetricRecord {
            name: "loss".into(),
            value: 0.5,
            step: 10,
            timestamp: Utc::now(),
        });

        let mut t2 = Trial::new("t2", HashMap::new());
        t2.state = TrialState::Completed;
        t2.metrics.push(MetricRecord {
            name: "loss".into(),
            value: 0.3,
            step: 10,
            timestamp: Utc::now(),
        });

        study.trials.push(t1);
        study.trials.push(t2);

        let best = study.best_trial().unwrap();
        assert_eq!(best.id, "t2");
    }

    #[test]
    fn study_progress() {
        let mut study = Study::new(
            "test",
            sample_search_space(),
            SearchStrategy::Random {
                n_trials: 10,
                seed: None,
            },
            vec![],
        );

        assert_eq!(study.progress(), 0.0);

        study.trials.push(make_trial("t1", 0.5));
        study.trials.push(make_trial("t2", 0.6));
        assert!((study.progress() - 0.2).abs() < f64::EPSILON);
    }

    #[test]
    fn trial_terminal_states() {
        let mut t = Trial::new("t1", HashMap::new());
        assert!(!t.is_terminal());

        t.state = TrialState::Running;
        assert!(!t.is_terminal());

        t.state = TrialState::Completed;
        assert!(t.is_terminal());

        t.state = TrialState::Pruned {
            step: 5,
            reason: "bad".into(),
        };
        assert!(t.is_terminal());

        t.state = TrialState::Failed {
            error: "oops".into(),
        };
        assert!(t.is_terminal());
    }

    #[test]
    fn study_serde_roundtrip() {
        let mut study = Study::new(
            "test_study",
            sample_search_space(),
            SearchStrategy::Bayesian {
                n_trials: 100,
                n_startup: 10,
                seed: Some(42),
            },
            vec![Objective {
                metric: "f1".into(),
                direction: Direction::Maximize,
            }],
        );
        study.trials.push(make_trial("t1", 0.85));

        let json = serde_json::to_string(&study).unwrap();
        let deserialized: Study = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.name, "test_study");
        assert_eq!(deserialized.trials.len(), 1);
    }

    #[test]
    fn search_strategy_n_trials() {
        assert_eq!(
            SearchStrategy::Random { n_trials: 50, seed: None }.n_trials(),
            Some(50)
        );
        assert_eq!(
            SearchStrategy::Grid { points_per_dim: 5 }.n_trials(),
            None
        );
        assert_eq!(
            SearchStrategy::Bayesian { n_trials: 100, n_startup: 10, seed: None }.n_trials(),
            Some(100)
        );
    }

    #[test]
    fn no_best_trial_when_empty() {
        let study = Study::new(
            "empty",
            SearchSpace::new(),
            SearchStrategy::Random { n_trials: 10, seed: None },
            vec![Objective { metric: "f1".into(), direction: Direction::Maximize }],
        );
        assert!(study.best_trial().is_none());
    }
}
