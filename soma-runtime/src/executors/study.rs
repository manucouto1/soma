//! Study runner — orchestrates hyperparameter optimization.
//!
//! Iterates over trials: samples parameters, calls the executor with a
//! [`TrialContext`] handle, records metrics, feeds results back to the
//! sampler (ask/tell), consults the pruner on intermediate metrics,
//! and persists the study through an optional tracker after every
//! trial. Resume-aware: a study with N recorded trials continues at
//! trial N.

use crate::event_bus::EventBus;
use crate::pruner::{MedianPruner, PercentilePruner, Pruner, TrialMetricHistory};
use crate::sampler::Sampler;
use chrono::Utc;
use somatize_core::error::Result;
use somatize_core::event::{Event, MetricRecord};
use somatize_core::study::{Direction, PruningStrategy, Study, Trial, TrialState};
use somatize_core::tracking::Tracker;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// Result of executing a trial. Separates control flow (pruning) from errors.
#[derive(Debug, Clone)]
pub enum TrialOutcome {
    /// Trial completed successfully with final metrics.
    Completed(Vec<MetricRecord>),
    /// Trial was pruned (stopped early) at the given step.
    Pruned { step: usize, reason: String },
}

/// Handle passed to a trial: reports intermediate metrics and asks
/// whether the pruner wants the trial stopped.
///
/// [`report`](Self::report) is the single channel for per-step metrics:
/// it records the value on the trial, emits [`Event::TrialMetric`], and
/// — when the metric is the study's objective — consults the pruner
/// against the completed trials' histories. Values are compared on a
/// maximize scale (the runner pre-normalizes for `Minimize`).
pub struct TrialContext<'a> {
    study_id: String,
    trial_id: String,
    /// Objective metric name + direction used for pruning decisions.
    objective: Option<(String, Direction)>,
    pruner: Option<&'a dyn Pruner>,
    /// Completed trials' metric histories, direction-normalized.
    history: &'a [TrialMetricHistory],
    event_bus: &'a EventBus,
    metrics: Vec<MetricRecord>,
    pruned: Option<(usize, String)>,
}

impl TrialContext<'_> {
    /// Record an intermediate metric at `step`. Returns `true` when the
    /// trial should stop (pruned) — the executor should then return
    /// early; the runner marks the trial pruned regardless.
    pub fn report(&mut self, name: &str, value: f64, step: usize) -> bool {
        let record = MetricRecord {
            name: name.to_string(),
            value,
            step,
            timestamp: Utc::now(),
        };
        self.metrics.push(record.clone());
        self.event_bus.emit(Event::TrialMetric {
            study_id: self.study_id.clone(),
            trial_id: self.trial_id.clone(),
            metric: record,
        });

        if self.pruned.is_some() {
            return true;
        }
        if let (Some(pruner), Some((obj_name, direction))) = (self.pruner, &self.objective)
            && name == obj_name
            && let Some(reason) =
                pruner.should_prune(obj_name, direction.normalize(value), step, self.history)
        {
            self.pruned = Some((step, reason));
            return true;
        }
        false
    }

    /// Whether the pruner has decided to stop this trial.
    pub fn should_prune(&self) -> bool {
        self.pruned.is_some()
    }

    /// Metrics reported so far.
    pub fn metrics(&self) -> &[MetricRecord] {
        &self.metrics
    }
}

/// Callback that executes a trial given sampled parameters.
///
/// Returns `Ok(TrialOutcome)` for normal completion or pruning,
/// `Err(SomaError)` only for unexpected failures.
pub trait TrialExecutor: Send + Sync {
    fn execute_trial(
        &self,
        params: &HashMap<String, serde_json::Value>,
        ctx: &mut TrialContext<'_>,
    ) -> Result<TrialOutcome>;
}

/// Function-based trial executor for convenience.
pub struct FnTrialExecutor<F>(pub F);

impl<F> TrialExecutor for FnTrialExecutor<F>
where
    F: Fn(&HashMap<String, serde_json::Value>, &mut TrialContext<'_>) -> Result<TrialOutcome>
        + Send
        + Sync,
{
    fn execute_trial(
        &self,
        params: &HashMap<String, serde_json::Value>,
        ctx: &mut TrialContext<'_>,
    ) -> Result<TrialOutcome> {
        (self.0)(params, ctx)
    }
}

/// Runs a Study: samples parameters, executes trials, records results.
pub struct StudyRunner {
    event_bus: Arc<EventBus>,
    tracker: Option<Arc<dyn Tracker>>,
}

impl StudyRunner {
    pub fn new(event_bus: Arc<EventBus>) -> Self {
        Self {
            event_bus,
            tracker: None,
        }
    }

    /// Persist the study (`study.json`, atomic) after every trial.
    pub fn with_tracker(mut self, tracker: Arc<dyn Tracker>) -> Self {
        self.tracker = Some(tracker);
        self
    }

    /// Run the study to completion.
    ///
    /// Resume-aware: starts at `trial_index = study.trials.len()` and
    /// replays already-completed trials into the sampler's history
    /// first, so model-based samplers continue informed and grids don't
    /// repeat configurations.
    pub fn run(
        &self,
        study: &mut Study,
        sampler: &mut dyn Sampler,
        executor: &dyn TrialExecutor,
    ) -> Result<()> {
        sampler.prepare(&study.search_space);
        let total = sampler.n_trials().unwrap_or(0);
        let direction = study.primary_direction().unwrap_or(Direction::Maximize);
        if study.created_at.is_none() {
            study.created_at = Some(Utc::now());
        }

        // Replay prior trials (resume) into the sampler's history.
        for trial in &study.trials {
            if let Some(value) = study.objective_value(trial) {
                sampler.record_result(&trial.params, direction.normalize(value));
            }
        }

        self.event_bus.emit(Event::StudyStarted {
            study_id: study.id.clone(),
            name: study.name.clone(),
            total_trials: total,
        });

        let pruner = build_pruner(&study.pruning);
        let mut trial_index = study.trials.len();

        while let Some(mut params) = sampler.sample(&study.search_space, trial_index)? {
            // Frozen parameters are fixed values excluded from the
            // search space — inject them into every configuration.
            for (name, value) in &study.frozen {
                params.insert(name.clone(), value.clone());
            }

            let trial_id = format!("trial_{trial_index:04}");
            let mut trial = Trial::new(trial_id.clone(), params.clone());
            trial.state = TrialState::Running;
            trial.started_at = Some(Utc::now());

            self.event_bus.emit(Event::TrialStarted {
                study_id: study.id.clone(),
                trial_id: trial_id.clone(),
                params: serde_json::json!(params),
            });

            // Histories the pruner compares against, on a maximize scale.
            let history = normalized_histories(study, direction);
            let mut ctx = TrialContext {
                study_id: study.id.clone(),
                trial_id: trial_id.clone(),
                objective: objective_metric(study).map(|name| (name, direction)),
                pruner: pruner.as_deref(),
                history: &history,
                event_bus: &self.event_bus,
                metrics: Vec::new(),
                pruned: None,
            };

            let start = Instant::now();
            let outcome = executor.execute_trial(&params, &mut ctx);
            let TrialContext {
                metrics: reported,
                pruned,
                ..
            } = ctx;
            trial.duration_ms = Some(start.elapsed().as_millis() as u64);
            trial.finished_at = Some(Utc::now());
            trial.metrics = reported;

            match (outcome, pruned) {
                // The pruner's verdict wins over a Completed return —
                // an executor may not notice report() returned true.
                (Ok(_), Some((step, reason)))
                | (Ok(TrialOutcome::Pruned { step, reason }), None) => {
                    trial.state = TrialState::Pruned {
                        step,
                        reason: reason.clone(),
                    };
                    self.event_bus.emit(Event::TrialPruned {
                        study_id: study.id.clone(),
                        trial_id: trial_id.clone(),
                        step,
                        reason,
                    });
                }
                (Ok(TrialOutcome::Completed(final_metrics)), None) => {
                    for metric in &final_metrics {
                        self.event_bus.emit(Event::TrialMetric {
                            study_id: study.id.clone(),
                            trial_id: trial_id.clone(),
                            metric: metric.clone(),
                        });
                    }
                    trial.metrics.extend(final_metrics.clone());
                    trial.state = TrialState::Completed;

                    self.event_bus.emit(Event::TrialCompleted {
                        study_id: study.id.clone(),
                        trial_id: trial_id.clone(),
                        final_metrics,
                    });
                }
                (Err(e), _) => {
                    trial.state = TrialState::Failed {
                        error: e.to_string(),
                    };
                    self.event_bus.emit(Event::TrialFailed {
                        study_id: study.id.clone(),
                        trial_id: trial_id.clone(),
                        error: e.to_string(),
                    });
                }
            }

            study.trials.push(trial);

            // Tell the sampler (ask/tell feedback for TPE and future BO).
            if let Some(value) = study.objective_value(study.trials.last().unwrap()) {
                sampler.record_result(&params, direction.normalize(value));
            }

            // Check if we have a new best
            if let Some(best) = study.best_trial()
                && best.id == trial_id
                && let Some(val) = study.best_value()
            {
                self.event_bus.emit(Event::BestUpdated {
                    study_id: study.id.clone(),
                    trial_id: trial_id.clone(),
                    value: val,
                    params: serde_json::json!(params),
                });
            }

            let completed = study.trials.iter().filter(|t| t.is_terminal()).count();
            self.event_bus.emit(Event::StudyProgress {
                study_id: study.id.clone(),
                completed,
                total,
                best_value: study.best_value().unwrap_or(f64::NAN),
            });

            study.updated_at = Some(Utc::now());
            self.save_study(study);
            trial_index += 1;
        }

        let best_trial_id = study.best_trial().map(|t| t.id.clone()).unwrap_or_default();
        let best_value = study.best_value().unwrap_or(f64::NAN);

        self.event_bus.emit(Event::StudyCompleted {
            study_id: study.id.clone(),
            best_trial_id,
            best_value,
        });
        study.updated_at = Some(Utc::now());
        self.save_study(study);
        // Make the event log durable before returning — a caller may
        // read the run directory immediately after run() completes.
        self.event_bus.flush_sinks();

        Ok(())
    }

    fn save_study(&self, study: &Study) {
        if let Some(tracker) = &self.tracker
            && let Err(e) = tracker.save_study(study)
        {
            tracing::warn!("tracking: failed to persist study: {e}");
        }
    }
}

/// Metric name the pruner watches: the composite's recorded name is not
/// a raw metric, so pruning tracks the first declared objective (or the
/// first composite term as a fallback).
fn objective_metric(study: &Study) -> Option<String> {
    study
        .objectives
        .first()
        .map(|o| o.metric.clone())
        .or_else(|| {
            study
                .composite
                .as_ref()
                .and_then(|c| c.terms.first().map(|(name, _)| name.clone()))
        })
}

fn build_pruner(strategy: &PruningStrategy) -> Option<Box<dyn Pruner>> {
    match strategy {
        PruningStrategy::None | PruningStrategy::Hyperband => None,
        PruningStrategy::Median { n_warmup_steps } => {
            Some(Box::new(MedianPruner::new(*n_warmup_steps)))
        }
        PruningStrategy::Percentile {
            percentile,
            n_warmup_steps,
        } => Some(Box::new(PercentilePruner::new(
            *percentile,
            *n_warmup_steps,
        ))),
    }
}

/// Completed trials' metric histories with values mapped onto a
/// maximize scale, so pruners can always assume higher-is-better.
fn normalized_histories(study: &Study, direction: Direction) -> Vec<TrialMetricHistory> {
    study
        .trials
        .iter()
        .filter(|t| t.is_complete())
        .map(|t| TrialMetricHistory {
            trial_id: t.id.clone(),
            metrics: t
                .metrics
                .iter()
                .map(|m| MetricRecord {
                    name: m.name.clone(),
                    value: direction.normalize(m.value),
                    step: m.step,
                    timestamp: m.timestamp,
                })
                .collect(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::{GridSampler, RandomSampler};
    use chrono::Utc;
    use somatize_core::error::SomaError;
    use somatize_core::search::{Scale, SearchDimension, SearchSpace};
    use somatize_core::study::{Direction, Objective, SearchStrategy};

    fn sample_space() -> SearchSpace {
        let mut space = SearchSpace::new();
        space.add(SearchDimension::Float {
            name: "lr".into(),
            low: 0.001,
            high: 0.1,
            scale: Scale::Log,
            default: None,
        });
        space.add(SearchDimension::Categorical {
            name: "activation".into(),
            choices: vec![serde_json::json!("relu"), serde_json::json!("tanh")],
        });
        space
    }

    /// Simple executor: f1 = 1.0 - |lr - 0.01| * 10
    fn make_executor() -> FnTrialExecutor<
        impl Fn(&HashMap<String, serde_json::Value>, &mut TrialContext<'_>) -> Result<TrialOutcome>,
    > {
        FnTrialExecutor(
            |params: &HashMap<String, serde_json::Value>, _ctx: &mut TrialContext<'_>| {
                let lr = params["lr"].as_f64().unwrap();
                let f1 = (1.0 - (lr - 0.01).abs() * 10.0).max(0.0);
                Ok(TrialOutcome::Completed(vec![MetricRecord {
                    name: "f1".into(),
                    value: f1,
                    step: 0,
                    timestamp: Utc::now(),
                }]))
            },
        )
    }

    #[test]
    fn study_runner_grid_search() {
        let bus = Arc::new(EventBus::new(256));
        let mut rx = bus.subscribe();
        let runner = StudyRunner::new(bus);

        let space = sample_space();
        let mut study = Study::new(
            "grid_test",
            space,
            SearchStrategy::Grid { points_per_dim: 3 },
            vec![Objective {
                metric: "f1".into(),
                direction: Direction::Maximize,
            }],
        );

        let mut sampler = GridSampler::new(3);
        let executor = make_executor();

        runner.run(&mut study, &mut sampler, &executor).unwrap();

        // 3 lr points * 2 activations = 6 trials
        assert_eq!(study.trials.len(), 6);
        assert!(study.trials.iter().all(|t| t.is_complete()));

        // Best trial should have lr closest to 0.01
        let best = study.best_trial().unwrap();
        let best_lr = best.params["lr"].as_f64().unwrap();
        assert!(
            (best_lr - 0.01).abs() < 0.05,
            "best lr should be near 0.01, got {best_lr}"
        );

        // Check events were emitted
        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }
        assert!(
            events
                .iter()
                .any(|e| matches!(e, Event::StudyStarted { .. }))
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, Event::TrialStarted { .. }))
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, Event::TrialCompleted { .. }))
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, Event::BestUpdated { .. }))
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, Event::StudyCompleted { .. }))
        );
    }

    #[test]
    fn study_runner_random_search() {
        let bus = Arc::new(EventBus::new(256));
        let runner = StudyRunner::new(bus);

        let space = sample_space();
        let mut study = Study::new(
            "random_test",
            space,
            SearchStrategy::Random {
                n_trials: 20,
                seed: Some(42),
            },
            vec![Objective {
                metric: "f1".into(),
                direction: Direction::Maximize,
            }],
        );

        let mut sampler = RandomSampler::new(20, Some(42));
        let executor = make_executor();

        runner.run(&mut study, &mut sampler, &executor).unwrap();

        assert_eq!(study.trials.len(), 20);
        assert!(study.best_trial().is_some());
    }

    #[test]
    fn study_runner_handles_failed_trials() {
        let bus = Arc::new(EventBus::new(256));
        let runner = StudyRunner::new(bus);

        let mut space = SearchSpace::new();
        space.add(SearchDimension::Float {
            name: "x".into(),
            low: 0.0,
            high: 1.0,
            scale: Scale::Linear,
            default: None,
        });

        let mut study = Study::new(
            "fail_test",
            space,
            SearchStrategy::Random {
                n_trials: 5,
                seed: None,
            },
            vec![Objective {
                metric: "f1".into(),
                direction: Direction::Maximize,
            }],
        );

        // Executor that fails on even trials
        let executor = FnTrialExecutor(
            |params: &HashMap<String, serde_json::Value>, _ctx: &mut TrialContext<'_>| {
                let x = params["x"].as_f64().unwrap();
                if x > 0.5 {
                    Err(SomaError::Other("too high".into()))
                } else {
                    Ok(TrialOutcome::Completed(vec![MetricRecord {
                        name: "f1".into(),
                        value: x,
                        step: 0,
                        timestamp: Utc::now(),
                    }]))
                }
            },
        );

        let mut sampler = RandomSampler::new(5, Some(42));
        runner.run(&mut study, &mut sampler, &executor).unwrap();

        assert_eq!(study.trials.len(), 5);
        // Some should be Failed
        let failed = study
            .trials
            .iter()
            .filter(|t| matches!(t.state, TrialState::Failed { .. }))
            .count();
        assert!(failed > 0, "should have some failed trials");
    }

    #[test]
    fn study_runner_handles_pruned_trials() {
        let bus = Arc::new(EventBus::new(256));
        let runner = StudyRunner::new(bus);

        let mut space = SearchSpace::new();
        space.add(SearchDimension::Float {
            name: "x".into(),
            low: 0.0,
            high: 1.0,
            scale: Scale::Linear,
            default: None,
        });

        let mut study = Study::new(
            "prune_test",
            space,
            SearchStrategy::Random {
                n_trials: 3,
                seed: None,
            },
            vec![Objective {
                metric: "f1".into(),
                direction: Direction::Maximize,
            }],
        );

        // Executor that prunes every trial
        let executor = FnTrialExecutor(
            |_params: &HashMap<String, serde_json::Value>, _ctx: &mut TrialContext<'_>| {
                Ok(TrialOutcome::Pruned {
                    step: 5,
                    reason: "below median".into(),
                })
            },
        );

        let mut sampler = RandomSampler::new(3, Some(42));
        runner.run(&mut study, &mut sampler, &executor).unwrap();

        assert!(
            study
                .trials
                .iter()
                .all(|t| matches!(t.state, TrialState::Pruned { .. }))
        );
    }

    fn one_dim_space() -> SearchSpace {
        let mut space = SearchSpace::new();
        space.add(SearchDimension::Float {
            name: "x".into(),
            low: 0.0,
            high: 1.0,
            scale: Scale::Linear,
            default: None,
        });
        space
    }

    #[test]
    fn grid_study_started_reports_real_total() {
        let bus = Arc::new(EventBus::new(256));
        let mut rx = bus.subscribe();
        let runner = StudyRunner::new(bus);

        let mut study = Study::new(
            "grid_total",
            sample_space(),
            SearchStrategy::Grid { points_per_dim: 3 },
            vec![Objective {
                metric: "f1".into(),
                direction: Direction::Maximize,
            }],
        );
        let mut sampler = GridSampler::new(3);
        runner
            .run(&mut study, &mut sampler, &make_executor())
            .unwrap();

        let mut started_total = None;
        while let Ok(e) = rx.try_recv() {
            if let Event::StudyStarted { total_trials, .. } = e {
                started_total = Some(total_trials);
            }
        }
        // 3 lr points × 2 activations — known BEFORE the first sample.
        assert_eq!(started_total, Some(6));
    }

    /// Sampler spy that counts record_result calls.
    struct SpySampler {
        inner: RandomSampler,
        recorded: Vec<f64>,
    }

    impl Sampler for SpySampler {
        fn sample(
            &mut self,
            space: &SearchSpace,
            trial_index: usize,
        ) -> Result<Option<HashMap<String, serde_json::Value>>> {
            self.inner.sample(space, trial_index)
        }
        fn n_trials(&self) -> Option<usize> {
            self.inner.n_trials()
        }
        fn record_result(&mut self, _params: &HashMap<String, serde_json::Value>, value: f64) {
            self.recorded.push(value);
        }
    }

    #[test]
    fn sampler_receives_feedback_per_completed_trial() {
        let bus = Arc::new(EventBus::new(256));
        let runner = StudyRunner::new(bus);

        let mut study = Study::new(
            "feedback",
            one_dim_space(),
            SearchStrategy::Random {
                n_trials: 4,
                seed: Some(1),
            },
            vec![Objective {
                metric: "loss".into(),
                direction: Direction::Minimize,
            }],
        );
        let executor = FnTrialExecutor(
            |params: &HashMap<String, serde_json::Value>, _ctx: &mut TrialContext<'_>| {
                Ok(TrialOutcome::Completed(vec![MetricRecord {
                    name: "loss".into(),
                    value: params["x"].as_f64().unwrap(),
                    step: 0,
                    timestamp: Utc::now(),
                }]))
            },
        );

        let mut sampler = SpySampler {
            inner: RandomSampler::new(4, Some(1)),
            recorded: Vec::new(),
        };
        runner.run(&mut study, &mut sampler, &executor).unwrap();

        assert_eq!(
            sampler.recorded.len(),
            4,
            "one feedback per completed trial"
        );
        // Minimize → values arrive negated (maximize scale).
        assert!(sampler.recorded.iter().all(|v| *v <= 0.0));
    }

    fn pruning_study(direction: Direction) -> Study {
        let metric = match direction {
            Direction::Maximize => "f1",
            Direction::Minimize => "loss",
        };
        Study::new(
            "pruning",
            one_dim_space(),
            SearchStrategy::Random {
                n_trials: 4,
                seed: Some(7),
            },
            vec![Objective {
                metric: metric.into(),
                direction,
            }],
        )
        .with_pruning(PruningStrategy::Median { n_warmup_steps: 2 })
    }

    /// First trial completes with a good curve; every later (bad)
    /// trial must get pruned mid-way once it reports below the median.
    fn run_pruning_case(direction: Direction) -> Study {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let bus = Arc::new(EventBus::new(256));
        let runner = StudyRunner::new(bus);
        let mut study = pruning_study(direction);
        let counter = AtomicUsize::new(0);

        let executor = FnTrialExecutor(
            move |_params: &HashMap<String, serde_json::Value>, ctx: &mut TrialContext<'_>| {
                let good = counter.fetch_add(1, Ordering::SeqCst) == 0;
                let metric = match direction {
                    Direction::Maximize => "f1",
                    Direction::Minimize => "loss",
                };
                for step in 0..10 {
                    let value = match (direction, good) {
                        (Direction::Maximize, true) => 0.5 + step as f64 * 0.05,
                        (Direction::Maximize, false) => 0.01,
                        (Direction::Minimize, true) => 1.0 - step as f64 * 0.05,
                        (Direction::Minimize, false) => 10.0,
                    };
                    if ctx.report(metric, value, step) {
                        return Ok(TrialOutcome::Pruned {
                            step,
                            reason: "stopped by pruner".into(),
                        });
                    }
                }
                Ok(TrialOutcome::Completed(vec![]))
            },
        );

        let mut sampler = RandomSampler::new(4, Some(7));
        runner.run(&mut study, &mut sampler, &executor).unwrap();
        study
    }

    fn assert_bad_trials_pruned(study: &Study) {
        let pruned: Vec<_> = study
            .trials
            .iter()
            .filter(|t| matches!(t.state, TrialState::Pruned { .. }))
            .collect();
        assert_eq!(pruned.len(), 3, "all trials after the first get pruned");
        for t in pruned {
            if let TrialState::Pruned { step, .. } = &t.state {
                assert_eq!(*step, 2, "pruned right after warmup, not at the end");
            }
        }
        assert!(study.trials[0].is_complete());
    }

    #[test]
    fn median_pruner_stops_bad_trials_maximize() {
        assert_bad_trials_pruned(&run_pruning_case(Direction::Maximize));
    }

    #[test]
    fn median_pruner_stops_bad_trials_minimize() {
        // Direction normalization: a HIGH loss must read as "bad".
        assert_bad_trials_pruned(&run_pruning_case(Direction::Minimize));
    }

    #[test]
    fn resume_continues_without_repeating_grid_params() {
        let executor = make_executor();

        // Full reference run: 6 grid trials.
        let bus = Arc::new(EventBus::new(256));
        let runner = StudyRunner::new(bus);
        let mut full = Study::new(
            "full",
            sample_space(),
            SearchStrategy::Grid { points_per_dim: 3 },
            vec![Objective {
                metric: "f1".into(),
                direction: Direction::Maximize,
            }],
        );
        runner
            .run(&mut full, &mut GridSampler::new(3), &executor)
            .unwrap();
        assert_eq!(full.trials.len(), 6);

        // Interrupted run: only the first 3 trials happened.
        let mut resumed = full.clone();
        resumed.trials.truncate(3);

        let bus = Arc::new(EventBus::new(256));
        let mut rx = bus.subscribe();
        let runner = StudyRunner::new(bus);
        runner
            .run(&mut resumed, &mut GridSampler::new(3), &executor)
            .unwrap();

        assert_eq!(resumed.trials.len(), 6);
        // The resumed half reproduces exactly the reference tail — no
        // repeated or skipped configurations, ids continue.
        for i in 3..6 {
            assert_eq!(resumed.trials[i].params, full.trials[i].params);
            assert_eq!(resumed.trials[i].id, format!("trial_{i:04}"));
        }
        // Only 3 TrialStarted events in the resumed run.
        let mut started = 0;
        while let Ok(e) = rx.try_recv() {
            if matches!(e, Event::TrialStarted { .. }) {
                started += 1;
            }
        }
        assert_eq!(started, 3);
    }

    #[test]
    fn frozen_params_reach_every_trial() {
        let bus = Arc::new(EventBus::new(256));
        let runner = StudyRunner::new(bus);

        let mut study = Study::new(
            "frozen",
            one_dim_space(),
            SearchStrategy::Random {
                n_trials: 3,
                seed: Some(2),
            },
            vec![Objective {
                metric: "f1".into(),
                direction: Direction::Maximize,
            }],
        );
        study
            .frozen
            .insert("batch_size".into(), serde_json::json!(64));

        let executor = FnTrialExecutor(
            |params: &HashMap<String, serde_json::Value>, _ctx: &mut TrialContext<'_>| {
                assert_eq!(params["batch_size"], serde_json::json!(64));
                Ok(TrialOutcome::Completed(vec![MetricRecord {
                    name: "f1".into(),
                    value: 0.5,
                    step: 0,
                    timestamp: Utc::now(),
                }]))
            },
        );
        let mut sampler = RandomSampler::new(3, Some(2));
        runner.run(&mut study, &mut sampler, &executor).unwrap();

        for t in &study.trials {
            assert_eq!(t.params["batch_size"], serde_json::json!(64));
        }
    }

    #[test]
    fn composite_objective_selects_best_trial() {
        use somatize_core::study::{CompositeObjective, Scalarizer};

        let bus = Arc::new(EventBus::new(256));
        let runner = StudyRunner::new(bus);

        let mut study = Study::new(
            "composite",
            one_dim_space(),
            SearchStrategy::Grid { points_per_dim: 5 },
            vec![],
        )
        .with_composite(CompositeObjective {
            // Maximize x while penalizing x² → optimum at x = 0.5.
            terms: vec![("x".into(), 1.0), ("x_sq".into(), -1.0)],
            direction: Direction::Maximize,
            scalarizer: Scalarizer::WeightedSum,
        });

        let executor = FnTrialExecutor(
            |params: &HashMap<String, serde_json::Value>, _ctx: &mut TrialContext<'_>| {
                let x = params["x"].as_f64().unwrap();
                let now = Utc::now();
                Ok(TrialOutcome::Completed(vec![
                    MetricRecord {
                        name: "x".into(),
                        value: x,
                        step: 0,
                        timestamp: now,
                    },
                    MetricRecord {
                        name: "x_sq".into(),
                        value: x * x,
                        step: 0,
                        timestamp: now,
                    },
                ]))
            },
        );
        let mut sampler = GridSampler::new(5);
        runner.run(&mut study, &mut sampler, &executor).unwrap();

        let best_x = study.best_trial().unwrap().params["x"].as_f64().unwrap();
        assert!(
            (best_x - 0.5).abs() < 1e-9,
            "optimum of x - x² is 0.5, got {best_x}"
        );
    }

    #[test]
    fn tracker_persists_study_after_every_trial() {
        use somatize_core::tracking::{RunKind, Tracker as _};

        let root = tempfile::tempdir().unwrap();
        let tracker = Arc::new(
            crate::tracking::LocalTracker::create(root.path(), RunKind::Study, "t").unwrap(),
        );
        let run_dir = tracker.run_dir().to_path_buf();

        let bus = Arc::new(EventBus::new(256));
        bus.add_sink(tracker.sink());
        let runner = StudyRunner::new(bus).with_tracker(tracker);

        let mut study = Study::new(
            "persisted",
            one_dim_space(),
            SearchStrategy::Random {
                n_trials: 3,
                seed: Some(3),
            },
            vec![Objective {
                metric: "f1".into(),
                direction: Direction::Maximize,
            }],
        );
        let mut sampler = RandomSampler::new(3, Some(3));
        runner
            .run(&mut study, &mut sampler, &make_simple_executor())
            .unwrap();

        // study.json is a complete, loadable study (crash-safe resume).
        let loaded = Study::load(run_dir.join("study.json")).unwrap();
        assert_eq!(loaded.trials.len(), 3);
        assert!(loaded.updated_at.is_some());
        // Trial events reached the run's events.jsonl through the sink.
        let events = std::fs::read_to_string(run_dir.join("events.jsonl")).unwrap();
        assert!(events.contains("TrialCompleted"));
    }

    fn make_simple_executor() -> FnTrialExecutor<
        impl Fn(&HashMap<String, serde_json::Value>, &mut TrialContext<'_>) -> Result<TrialOutcome>,
    > {
        FnTrialExecutor(
            |params: &HashMap<String, serde_json::Value>, _ctx: &mut TrialContext<'_>| {
                Ok(TrialOutcome::Completed(vec![MetricRecord {
                    name: "f1".into(),
                    value: params["x"].as_f64().unwrap_or(0.5),
                    step: 0,
                    timestamp: Utc::now(),
                }]))
            },
        )
    }

    #[test]
    fn study_progress_tracking() {
        let bus = Arc::new(EventBus::new(256));
        let mut rx = bus.subscribe();
        let runner = StudyRunner::new(bus);

        let mut space = SearchSpace::new();
        space.add(SearchDimension::Float {
            name: "x".into(),
            low: 0.0,
            high: 1.0,
            scale: Scale::Linear,
            default: None,
        });

        let mut study = Study::new(
            "progress_test",
            space,
            SearchStrategy::Random {
                n_trials: 3,
                seed: None,
            },
            vec![Objective {
                metric: "f1".into(),
                direction: Direction::Maximize,
            }],
        );

        let executor = FnTrialExecutor(
            |_params: &HashMap<String, serde_json::Value>, _ctx: &mut TrialContext<'_>| {
                Ok(TrialOutcome::Completed(vec![MetricRecord {
                    name: "f1".into(),
                    value: 0.5,
                    step: 0,
                    timestamp: Utc::now(),
                }]))
            },
        );

        let mut sampler = RandomSampler::new(3, Some(42));
        runner.run(&mut study, &mut sampler, &executor).unwrap();

        // Collect progress events
        let mut progress_events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            if let Event::StudyProgress {
                completed, total, ..
            } = e
            {
                progress_events.push((completed, total));
            }
        }

        assert_eq!(progress_events.len(), 3);
        assert_eq!(progress_events[0], (1, 3));
        assert_eq!(progress_events[1], (2, 3));
        assert_eq!(progress_events[2], (3, 3));
    }
}
