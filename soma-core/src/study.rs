//! Study — defines an optimization experiment with objectives and strategy.
//!
//! A [`Study`] holds the search space, strategy (Grid/Random/Bayesian),
//! objectives, and tracks trials. The [`StudyRunner`] in soma-runtime
//! orchestrates execution.

use crate::error::{Result, SomaError};
use crate::event::MetricRecord;
use crate::search::SearchSpace;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Direction of optimization for an objective.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    Minimize,
    Maximize,
}

impl Direction {
    /// Map a value onto a maximize scale: identity for `Maximize`,
    /// negation for `Minimize`. Lets samplers and pruners assume
    /// "higher is better" throughout.
    pub fn normalize(self, value: f64) -> f64 {
        match self {
            Direction::Maximize => value,
            Direction::Minimize => -value,
        }
    }
}

/// An optimization objective (metric + direction).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Objective {
    pub metric: String,
    pub direction: Direction,
}

/// How a [`CompositeObjective`] combines its weighted terms.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
#[serde(tag = "scalarizer_type")]
#[non_exhaustive]
pub enum Scalarizer {
    /// `Σ wᵢ·vᵢ` — the plain weighted sum.
    #[default]
    WeightedSum,
    /// Augmented weighted min/max (Tchebycheff-style, Knowles 2006):
    /// emphasizes the worst-performing term so non-convex trade-offs
    /// aren't missed. For `Maximize`: `minᵢ(wᵢ·vᵢ) + ρ·Σ wᵢ·vᵢ`;
    /// for `Minimize` the `min` becomes a `max`.
    AugmentedTchebycheff { rho: f64 },
}

/// A scalar objective composed from several named metrics.
///
/// The composite is the single value the optimizer sees; the component
/// metrics stay recorded on each trial, so a Pareto/multi-objective
/// layer can be added later without schema migration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositeObjective {
    /// `(metric_name, weight)` pairs. Negative weights penalize.
    pub terms: Vec<(String, f64)>,
    pub direction: Direction,
    #[serde(default)]
    pub scalarizer: Scalarizer,
}

impl CompositeObjective {
    /// Evaluate over a trial's final (last-recorded) metric values.
    /// `None` if any term's metric is missing.
    pub fn evaluate(&self, trial: &Trial) -> Option<f64> {
        let weighted: Vec<f64> = self
            .terms
            .iter()
            .map(|(name, weight)| trial.last_metric(name).map(|v| weight * v))
            .collect::<Option<Vec<f64>>>()?;
        if weighted.is_empty() {
            return None;
        }
        let sum: f64 = weighted.iter().sum();
        Some(match self.scalarizer {
            Scalarizer::WeightedSum => sum,
            Scalarizer::AugmentedTchebycheff { rho } => {
                let worst = match self.direction {
                    Direction::Maximize => weighted.iter().cloned().fold(f64::INFINITY, f64::min),
                    Direction::Minimize => {
                        weighted.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                    }
                };
                worst + rho * sum
            }
        })
    }
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
    #[serde(default)]
    pub started_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub finished_at: Option<DateTime<Utc>>,
}

impl Trial {
    pub fn new(id: impl Into<String>, params: HashMap<String, serde_json::Value>) -> Self {
        Self {
            id: id.into(),
            params,
            state: TrialState::Pending,
            metrics: Vec::new(),
            duration_ms: None,
            started_at: None,
            finished_at: None,
        }
    }

    /// Last recorded value for a metric (its final value).
    pub fn last_metric(&self, name: &str) -> Option<f64> {
        self.metrics
            .iter()
            .filter(|m| m.name == name)
            .map(|m| m.value)
            .next_back()
    }

    /// Get the best recorded value for a specific metric.
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
    /// Scalar objective composed from several metrics; takes precedence
    /// over `objectives` when set.
    #[serde(default)]
    pub composite: Option<CompositeObjective>,
    #[serde(default)]
    pub created_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub updated_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub git_sha: Option<String>,
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
            composite: None,
            created_at: Some(Utc::now()),
            updated_at: None,
            tags: Vec::new(),
            git_sha: None,
        }
    }

    pub fn with_pruning(mut self, pruning: PruningStrategy) -> Self {
        self.pruning = pruning;
        self
    }

    pub fn with_composite(mut self, composite: CompositeObjective) -> Self {
        self.composite = composite.into();
        self
    }

    /// Get completed trials.
    pub fn completed_trials(&self) -> Vec<&Trial> {
        self.trials.iter().filter(|t| t.is_complete()).collect()
    }

    /// Direction of the effective objective (composite if set, else the
    /// first declared objective).
    pub fn primary_direction(&self) -> Option<Direction> {
        self.composite
            .as_ref()
            .map(|c| c.direction)
            .or_else(|| self.objectives.first().map(|o| o.direction))
    }

    /// The single source of truth for scoring a trial: the composite
    /// objective if set, else the best value of the first objective's
    /// metric. `None` for incomplete trials or missing metrics.
    pub fn objective_value(&self, trial: &Trial) -> Option<f64> {
        if !trial.is_complete() {
            return None;
        }
        if let Some(composite) = &self.composite {
            return composite.evaluate(trial);
        }
        let obj = self.objectives.first()?;
        trial.best_metric(&obj.metric, obj.direction)
    }

    /// Get the best trial for the effective objective.
    pub fn best_trial(&self) -> Option<&Trial> {
        let direction = self.primary_direction()?;
        self.trials
            .iter()
            .filter_map(|t| Some((t, self.objective_value(t)?)))
            .reduce(|best, current| {
                if direction.normalize(current.1) > direction.normalize(best.1) {
                    current
                } else {
                    best
                }
            })
            .map(|(t, _)| t)
    }

    /// Objective value of the best trial.
    pub fn best_value(&self) -> Option<f64> {
        self.best_trial().and_then(|t| self.objective_value(t))
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

    /// Serialize to pretty JSON at `path`.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let json =
            serde_json::to_vec_pretty(self).map_err(|e| SomaError::Serialization(e.to_string()))?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load a study previously written by [`save`](Self::save) (or by a
    /// tracker's `study.json`).
    pub fn load(path: impl AsRef<Path>) -> Result<Study> {
        let bytes = std::fs::read(path)?;
        serde_json::from_slice(&bytes).map_err(|e| SomaError::Serialization(e.to_string()))
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
            SearchStrategy::Random {
                n_trials: 50,
                seed: None
            }
            .n_trials(),
            Some(50)
        );
        assert_eq!(SearchStrategy::Grid { points_per_dim: 5 }.n_trials(), None);
        assert_eq!(
            SearchStrategy::Bayesian {
                n_trials: 100,
                n_startup: 10,
                seed: None
            }
            .n_trials(),
            Some(100)
        );
    }

    fn multi_metric_trial(id: &str, f1: f64, gap: f64) -> Trial {
        let mut t = make_trial(id, f1);
        t.metrics.push(MetricRecord {
            name: "gap".into(),
            value: gap,
            step: 10,
            timestamp: Utc::now(),
        });
        t
    }

    #[test]
    fn composite_weighted_sum_picks_best() {
        let mut study = Study::new(
            "composite",
            sample_search_space(),
            SearchStrategy::Random {
                n_trials: 3,
                seed: None,
            },
            vec![],
        )
        .with_composite(CompositeObjective {
            terms: vec![("f1".into(), 0.7), ("gap".into(), -0.3)],
            direction: Direction::Maximize,
            scalarizer: Scalarizer::WeightedSum,
        });

        // t1: 0.7*0.9 - 0.3*0.5 = 0.48   t2: 0.7*0.8 - 0.3*0.05 = 0.545
        study.trials.push(multi_metric_trial("t1", 0.9, 0.5));
        study.trials.push(multi_metric_trial("t2", 0.8, 0.05));

        assert_eq!(study.best_trial().unwrap().id, "t2");
        let v = study.objective_value(&study.trials[1]).unwrap();
        assert!((v - 0.545).abs() < 1e-9);
    }

    #[test]
    fn composite_missing_metric_is_none() {
        let study = Study::new(
            "c",
            SearchSpace::new(),
            SearchStrategy::Random {
                n_trials: 1,
                seed: None,
            },
            vec![],
        )
        .with_composite(CompositeObjective {
            terms: vec![("f1".into(), 1.0), ("missing".into(), 1.0)],
            direction: Direction::Maximize,
            scalarizer: Scalarizer::WeightedSum,
        });
        let t = make_trial("t1", 0.9);
        assert!(study.objective_value(&t).is_none());
    }

    #[test]
    fn composite_tchebycheff_penalizes_worst_term() {
        let composite = CompositeObjective {
            terms: vec![("f1".into(), 1.0), ("gap".into(), 1.0)],
            direction: Direction::Maximize,
            scalarizer: Scalarizer::AugmentedTchebycheff { rho: 0.1 },
        };
        // Balanced (0.5, 0.5) should beat lopsided (0.9, 0.1):
        // balanced: min=0.5 + 0.1*1.0 = 0.6; lopsided: min=0.1 + 0.1*1.0 = 0.2
        let balanced = multi_metric_trial("b", 0.5, 0.5);
        let lopsided = multi_metric_trial("l", 0.9, 0.1);
        assert!(composite.evaluate(&balanced).unwrap() > composite.evaluate(&lopsided).unwrap());
    }

    #[test]
    fn study_save_load_roundtrip() {
        let dir = std::env::temp_dir().join(format!("soma_study_test_{}", uuid_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("study.json");

        let mut study = Study::new(
            "persisted",
            sample_search_space(),
            SearchStrategy::Grid { points_per_dim: 3 },
            vec![Objective {
                metric: "f1".into(),
                direction: Direction::Maximize,
            }],
        );
        study.tags = vec!["mos".into()];
        study.trials.push(make_trial("t1", 0.8));
        study.save(&path).unwrap();

        let back = Study::load(&path).unwrap();
        assert_eq!(back.name, "persisted");
        assert_eq!(back.trials.len(), 1);
        assert_eq!(back.tags, vec!["mos"]);
        assert!(back.created_at.is_some());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn pre_composite_study_json_still_loads() {
        // A study serialized before composite/timestamps/tags existed.
        let old = serde_json::json!({
            "id": "study_abc",
            "name": "legacy",
            "search_space": {"dimensions": [], "frozen": {}},
            "strategy": {"strategy_type": "Random", "n_trials": 5, "seed": null},
            "pruning": {"pruning_type": "None"},
            "objectives": [{"metric": "f1", "direction": "Maximize"}],
            "trials": [{
                "id": "t1",
                "params": {"lr": 0.01},
                "state": {"trial_state": "Completed"},
                "metrics": [],
                "duration_ms": 12
            }],
            "frozen": {}
        });
        let study: Study = serde_json::from_value(old).unwrap();
        assert_eq!(study.name, "legacy");
        assert!(study.composite.is_none());
        assert!(study.created_at.is_none());
        assert!(study.trials[0].started_at.is_none());
    }

    #[test]
    fn direction_normalize() {
        assert_eq!(Direction::Maximize.normalize(0.5), 0.5);
        assert_eq!(Direction::Minimize.normalize(0.5), -0.5);
    }

    #[test]
    fn no_best_trial_when_empty() {
        let study = Study::new(
            "empty",
            SearchSpace::new(),
            SearchStrategy::Random {
                n_trials: 10,
                seed: None,
            },
            vec![Objective {
                metric: "f1".into(),
                direction: Direction::Maximize,
            }],
        );
        assert!(study.best_trial().is_none());
    }
}
