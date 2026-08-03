//! Study — defines an optimization experiment with objectives and strategy.
//!
//! A [`Study`] holds the search space, strategy (Grid/Random/Bayesian),
//! objectives, and tracks trials. The `StudyRunner` in soma-runtime
//! orchestrates execution.

use crate::event::MetricRecord;
use crate::search::SearchSpace;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
    /// Experiment seeds: when non-empty, every sampled configuration is
    /// evaluated once per seed (trial params carry `"seed"`), giving
    /// each seed an independent cache line and resumable trial.
    #[serde(default)]
    pub seeds: Vec<i64>,
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
    /// Total trials resolved by the sampler at run start (grid sizes
    /// are unknown until the search space is prepared).
    #[serde(default)]
    pub planned_trials: Option<usize>,
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
            seeds: Vec::new(),
            composite: None,
            created_at: Some(Utc::now()),
            updated_at: None,
            tags: Vec::new(),
            git_sha: None,
            planned_trials: None,
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

    /// Number of total planned trials (if known). Prefers the count
    /// the sampler resolved at run start (covers grid strategies).
    pub fn total_trials(&self) -> Option<usize> {
        self.planned_trials.or_else(|| self.strategy.n_trials())
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

// Reading and writing a study to disk is I/O, so it lives in the runtime
// as `somatize_runtime::study_io::StudyIo`. Import that trait and
// `study.save(path)` / `Study::load(path)` read the same as they always
// did. See design/decisions.

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
        assert!(study.planned_trials.is_none());
    }
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
    fn direction_normalize() {
        assert_eq!(Direction::Maximize.normalize(0.5), 0.5);
        assert_eq!(Direction::Minimize.normalize(0.5), -0.5);
    }

    fn rising_falling_trial(id: &str, name: &str, values: &[f64]) -> Trial {
        let mut t = Trial::new(id, HashMap::new());
        t.state = TrialState::Completed;
        for (step, v) in values.iter().enumerate() {
            t.metrics.push(MetricRecord {
                name: name.into(),
                value: *v,
                step,
                timestamp: Utc::now(),
            });
        }
        t
    }

    #[test]
    fn last_metric_is_last_not_best() {
        let t = rising_falling_trial("t", "f1", &[0.5, 0.9, 0.4]);
        assert_eq!(t.last_metric("f1"), Some(0.4));
        assert_eq!(t.best_metric("f1", Direction::Maximize), Some(0.9));
        assert_eq!(t.best_metric("f1", Direction::Minimize), Some(0.4));
        assert_eq!(t.last_metric("missing"), None);
    }

    /// CONTRACT: `objective_value` scores single-objective studies on
    /// the BEST value across steps, but composite studies on the LAST
    /// (final) value of each term. The same rising-then-falling curve
    /// therefore scores differently depending on which mode is active.
    #[test]
    fn objective_value_best_vs_last_divergence_is_pinned() {
        let t = rising_falling_trial("t", "f1", &[0.5, 0.9, 0.4]);

        let single = Study::new(
            "single",
            SearchSpace::new(),
            SearchStrategy::Random {
                n_trials: 1,
                seed: None,
            },
            vec![Objective {
                metric: "f1".into(),
                direction: Direction::Maximize,
            }],
        );
        assert_eq!(single.objective_value(&t), Some(0.9)); // best

        let composite = Study::new(
            "composite",
            SearchSpace::new(),
            SearchStrategy::Random {
                n_trials: 1,
                seed: None,
            },
            vec![],
        )
        .with_composite(CompositeObjective {
            terms: vec![("f1".into(), 1.0)],
            direction: Direction::Maximize,
            scalarizer: Scalarizer::WeightedSum,
        });
        assert_eq!(composite.objective_value(&t), Some(0.4)); // last
    }

    #[test]
    fn objective_value_none_for_non_completed_trials() {
        let study = Study::new(
            "s",
            SearchSpace::new(),
            SearchStrategy::Random {
                n_trials: 1,
                seed: None,
            },
            vec![Objective {
                metric: "f1".into(),
                direction: Direction::Maximize,
            }],
        );
        for state in [
            TrialState::Pending,
            TrialState::Running,
            TrialState::Pruned {
                step: 1,
                reason: "bad".into(),
            },
            TrialState::Failed {
                error: "boom".into(),
            },
        ] {
            let mut t = make_trial("t", 0.9);
            t.state = state;
            assert!(study.objective_value(&t).is_none());
        }
    }

    #[test]
    fn composite_empty_terms_is_none() {
        for scalarizer in [
            Scalarizer::WeightedSum,
            Scalarizer::AugmentedTchebycheff { rho: 0.1 },
        ] {
            let composite = CompositeObjective {
                terms: vec![],
                direction: Direction::Maximize,
                scalarizer,
            };
            assert!(composite.evaluate(&make_trial("t", 0.9)).is_none());
        }
    }

    #[test]
    fn composite_tchebycheff_minimize_penalizes_worst_loss() {
        // On a minimize scale the WORST term is the largest one.
        let composite = CompositeObjective {
            terms: vec![("loss_a".into(), 1.0), ("loss_b".into(), 1.0)],
            direction: Direction::Minimize,
            scalarizer: Scalarizer::AugmentedTchebycheff { rho: 0.1 },
        };
        let balanced = {
            let mut t = rising_falling_trial("b", "loss_a", &[0.5]);
            t.metrics.push(MetricRecord {
                name: "loss_b".into(),
                value: 0.5,
                step: 0,
                timestamp: Utc::now(),
            });
            t
        };
        let lopsided = {
            let mut t = rising_falling_trial("l", "loss_a", &[0.1]);
            t.metrics.push(MetricRecord {
                name: "loss_b".into(),
                value: 0.9,
                step: 0,
                timestamp: Utc::now(),
            });
            t
        };
        // balanced: max=0.5 + 0.1*1.0 = 0.6; lopsided: max=0.9 + 0.1*1.0 = 1.0.
        // Lower is better under Minimize → balanced wins.
        let b = composite.evaluate(&balanced).unwrap();
        let l = composite.evaluate(&lopsided).unwrap();
        assert!(
            b < l,
            "balanced {b} must beat lopsided {l} on a minimize scale"
        );
    }

    #[test]
    fn composite_tchebycheff_rho_zero_is_pure_worst_case() {
        let composite = CompositeObjective {
            terms: vec![("a".into(), 1.0), ("b".into(), 1.0)],
            direction: Direction::Maximize,
            scalarizer: Scalarizer::AugmentedTchebycheff { rho: 0.0 },
        };
        let t = multi_metric_trial("t", 0.9, 0.2); // f1=0.9, gap=0.2 — wrong names
        let mut t2 = Trial::new("t2", HashMap::new());
        t2.state = TrialState::Completed;
        for (name, v) in [("a", 0.9), ("b", 0.2)] {
            t2.metrics.push(MetricRecord {
                name: name.into(),
                value: v,
                step: 0,
                timestamp: Utc::now(),
            });
        }
        let _ = t;
        assert_eq!(composite.evaluate(&t2), Some(0.2)); // min of the terms
    }

    #[test]
    fn scalarizer_serde_roundtrip_and_default() {
        let study = Study::new(
            "s",
            SearchSpace::new(),
            SearchStrategy::Random {
                n_trials: 1,
                seed: None,
            },
            vec![],
        )
        .with_composite(CompositeObjective {
            terms: vec![("f1".into(), 0.7)],
            direction: Direction::Maximize,
            scalarizer: Scalarizer::AugmentedTchebycheff { rho: 0.25 },
        });
        let json = serde_json::to_string(&study).unwrap();
        let back: Study = serde_json::from_str(&json).unwrap();
        match back.composite.unwrap().scalarizer {
            Scalarizer::AugmentedTchebycheff { rho } => assert_eq!(rho, 0.25),
            other => panic!("wrong scalarizer: {other:?}"),
        }

        // Explicit WeightedSum tag and a missing scalarizer field both
        // resolve to the default.
        let explicit: Scalarizer =
            serde_json::from_value(serde_json::json!({"scalarizer_type": "WeightedSum"})).unwrap();
        assert_eq!(explicit, Scalarizer::WeightedSum);
        let composite: CompositeObjective = serde_json::from_value(serde_json::json!({
            "terms": [["f1", 1.0]],
            "direction": "Maximize",
        }))
        .unwrap();
        assert_eq!(composite.scalarizer, Scalarizer::default());
    }

    #[test]
    fn primary_direction_composite_wins_over_objectives() {
        let study = Study::new(
            "s",
            SearchSpace::new(),
            SearchStrategy::Random {
                n_trials: 1,
                seed: None,
            },
            vec![Objective {
                metric: "loss".into(),
                direction: Direction::Minimize,
            }],
        )
        .with_composite(CompositeObjective {
            terms: vec![("f1".into(), 1.0)],
            direction: Direction::Maximize,
            scalarizer: Scalarizer::WeightedSum,
        });
        assert_eq!(study.primary_direction(), Some(Direction::Maximize));
    }

    #[test]
    fn best_value_matches_best_trial_and_handles_empty() {
        let mut study = Study::new(
            "s",
            sample_search_space(),
            SearchStrategy::Random {
                n_trials: 2,
                seed: None,
            },
            vec![Objective {
                metric: "f1".into(),
                direction: Direction::Maximize,
            }],
        );
        assert!(study.best_value().is_none());
        study.trials.push(make_trial("t1", 0.7));
        study.trials.push(make_trial("t2", 0.9));
        assert_eq!(study.best_value(), Some(0.9));

        // All-failed study: no best.
        let mut failed = make_trial("t3", 1.0);
        failed.state = TrialState::Failed { error: "x".into() };
        let mut all_failed = study.clone();
        all_failed.trials = vec![failed];
        assert!(all_failed.best_trial().is_none());
        assert!(all_failed.best_value().is_none());
    }

    #[test]
    fn best_trial_tie_keeps_first() {
        let mut study = Study::new(
            "s",
            sample_search_space(),
            SearchStrategy::Random {
                n_trials: 2,
                seed: None,
            },
            vec![Objective {
                metric: "f1".into(),
                direction: Direction::Maximize,
            }],
        );
        study.trials.push(make_trial("first", 0.8));
        study.trials.push(make_trial("second", 0.8));
        assert_eq!(study.best_trial().unwrap().id, "first");
    }

    #[test]
    fn planned_trials_governs_total_and_progress() {
        let mut study = Study::new(
            "grid",
            sample_search_space(),
            SearchStrategy::Grid { points_per_dim: 3 },
            vec![],
        );
        // Grid size is unknown until a sampler resolves it.
        assert_eq!(study.total_trials(), None);
        assert_eq!(study.progress(), 0.0);

        study.planned_trials = Some(6);
        study.trials.push(make_trial("t1", 0.5));
        study.trials.push(make_trial("t2", 0.5));
        study.trials.push(make_trial("t3", 0.5));
        assert_eq!(study.total_trials(), Some(6));
        assert!((study.progress() - 0.5).abs() < f64::EPSILON);
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
