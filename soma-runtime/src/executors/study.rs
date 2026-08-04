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
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Result of executing a trial. Separates control flow (pruning) from errors.
#[derive(Debug, Clone)]
pub enum TrialOutcome {
    /// Trial completed successfully with final metrics.
    Completed(Vec<MetricRecord>),
    /// Trial was pruned (stopped early) at the given step.
    Pruned {
        /// Step at which the pruner stopped the trial.
        step: usize,
        /// The pruner's explanation, recorded on the trial.
        reason: String,
    },
}

/// Handle passed to a trial: reports intermediate metrics and asks
/// whether the pruner wants the trial stopped.
///
/// [`report`](Self::report) is the single channel for per-step metrics:
/// it records the value on the trial, emits [`Event::TrialMetric`], and
/// — when the metric is the study's objective — consults the pruner
/// against the completed trials' histories. Values are compared on a
/// maximize scale (the runner pre-normalizes for `Minimize`).
///
/// The handle is cheaply cloneable (`Arc`-backed shared state) so it
/// can outlive the borrow stack — e.g. cross into a Python callback.
#[derive(Clone)]
pub struct TrialContext {
    study_id: String,
    trial_id: String,
    /// Objective metric name + direction used for pruning decisions.
    objective: Option<(String, Direction)>,
    pruner: Option<Arc<dyn Pruner>>,
    /// Completed trials' metric histories, direction-normalized.
    history: Arc<Vec<TrialMetricHistory>>,
    event_bus: Arc<EventBus>,
    shared: Arc<Mutex<TrialShared>>,
}

#[derive(Default)]
struct TrialShared {
    metrics: Vec<MetricRecord>,
    pruned: Option<(usize, String)>,
}

impl TrialContext {
    /// Record an intermediate metric at `step`. Returns `true` when the
    /// trial should stop (pruned) — the executor should then return
    /// early; the runner marks the trial pruned regardless.
    pub fn report(&self, name: &str, value: f64, step: usize) -> bool {
        let record = MetricRecord {
            name: name.to_string(),
            value,
            step,
            timestamp: Utc::now(),
        };
        {
            let mut shared = self.lock_shared();
            shared.metrics.push(record.clone());
        }
        self.event_bus.emit(Event::TrialMetric {
            study_id: self.study_id.clone(),
            trial_id: self.trial_id.clone(),
            metric: record,
        });

        if self.should_prune() {
            return true;
        }
        if let (Some(pruner), Some((obj_name, direction))) = (&self.pruner, &self.objective)
            && name == obj_name
            && let Some(reason) =
                pruner.should_prune(obj_name, direction.normalize(value), step, &self.history)
        {
            self.lock_shared().pruned = Some((step, reason));
            return true;
        }
        false
    }

    /// Whether the pruner has decided to stop this trial.
    pub fn should_prune(&self) -> bool {
        self.lock_shared().pruned.is_some()
    }

    /// Metrics reported so far.
    pub fn metrics(&self) -> Vec<MetricRecord> {
        self.lock_shared().metrics.clone()
    }

    /// Id of the trial this handle belongs to.
    pub fn trial_id(&self) -> &str {
        &self.trial_id
    }

    fn lock_shared(&self) -> std::sync::MutexGuard<'_, TrialShared> {
        match self.shared.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    fn take_results(&self) -> (Vec<MetricRecord>, Option<(usize, String)>) {
        let mut shared = self.lock_shared();
        (std::mem::take(&mut shared.metrics), shared.pruned.take())
    }
}

/// Callback that executes a trial given sampled parameters.
///
/// Returns `Ok(TrialOutcome)` for normal completion or pruning,
/// `Err(SomaError)` only for unexpected failures.
pub trait TrialExecutor: Send + Sync {
    /// Run one trial with the sampled `params`, reporting intermediate
    /// metrics through `ctx` and honouring its pruning verdicts.
    fn execute_trial(
        &self,
        params: &HashMap<String, serde_json::Value>,
        ctx: &TrialContext,
    ) -> Result<TrialOutcome>;
}

/// Function-based trial executor for convenience.
pub struct FnTrialExecutor<F>(pub F);

impl<F> TrialExecutor for FnTrialExecutor<F>
where
    F: Fn(&HashMap<String, serde_json::Value>, &TrialContext) -> Result<TrialOutcome> + Send + Sync,
{
    fn execute_trial(
        &self,
        params: &HashMap<String, serde_json::Value>,
        ctx: &TrialContext,
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
    /// A runner emitting trial events on `event_bus`, with no persistence
    /// until [`Self::with_tracker`] adds it.
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
        // With experiment seeds, every sampled config runs once per seed.
        let total = sampler.n_trials().unwrap_or(0) * study.seeds.len().max(1);
        if total > 0 {
            study.planned_trials = Some(total);
        }
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

        // Experiment seeds: each sampled configuration runs once per
        // seed (params carry "seed"), so every seed is an independent,
        // resumable trial with its own cache line. trial_index enumerates
        // config-major: config 0 × all seeds, config 1 × all seeds, …
        let seeds = study.seeds.clone();
        let n_seeds = seeds.len().max(1);
        let mut current_config: Option<HashMap<String, serde_json::Value>> = None;

        loop {
            let config_index = trial_index / n_seeds;
            let seed_slot = trial_index % n_seeds;

            let base = if seed_slot == 0 || current_config.is_none() {
                if seed_slot > 0 {
                    // Resuming mid-seed-block: recover the block's config
                    // from the previous (persisted) trial.
                    let mut prev = study.trials[trial_index - 1].params.clone();
                    prev.remove("seed");
                    Some(prev)
                } else {
                    sampler.sample(&study.search_space, config_index)?
                }
            } else {
                current_config.clone()
            };
            let Some(base) = base else { break };
            current_config = Some(base.clone());

            let mut params = base;
            if !seeds.is_empty() {
                params.insert("seed".to_string(), serde_json::json!(seeds[seed_slot]));
            }
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
            let ctx = TrialContext {
                study_id: study.id.clone(),
                trial_id: trial_id.clone(),
                objective: objective_metric(study).map(|name| (name, direction)),
                pruner: pruner.clone(),
                history: Arc::new(normalized_histories(study, direction)),
                event_bus: self.event_bus.clone(),
                shared: Arc::new(Mutex::new(TrialShared::default())),
            };

            let start = Instant::now();
            let outcome = executor.execute_trial(&params, &ctx);
            let (reported, pruned) = ctx.take_results();
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

fn build_pruner(strategy: &PruningStrategy) -> Option<Arc<dyn Pruner>> {
    match strategy {
        PruningStrategy::None | PruningStrategy::Hyperband => None,
        PruningStrategy::Median { n_warmup_steps } => {
            Some(Arc::new(MedianPruner::new(*n_warmup_steps)))
        }
        PruningStrategy::Percentile {
            percentile,
            n_warmup_steps,
        } => Some(Arc::new(PercentilePruner::new(
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
    use crate::sampler::{BayesianSampler, GridSampler, RandomSampler};
    use crate::study_io::StudyIo;
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
        impl Fn(&HashMap<String, serde_json::Value>, &TrialContext) -> Result<TrialOutcome>,
    > {
        FnTrialExecutor(
            |params: &HashMap<String, serde_json::Value>, _ctx: &TrialContext| {
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
            |params: &HashMap<String, serde_json::Value>, _ctx: &TrialContext| {
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
            |_params: &HashMap<String, serde_json::Value>, _ctx: &TrialContext| {
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
            |params: &HashMap<String, serde_json::Value>, _ctx: &TrialContext| {
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
            move |_params: &HashMap<String, serde_json::Value>, ctx: &TrialContext| {
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
            |params: &HashMap<String, serde_json::Value>, _ctx: &TrialContext| {
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
            |params: &HashMap<String, serde_json::Value>, _ctx: &TrialContext| {
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
        impl Fn(&HashMap<String, serde_json::Value>, &TrialContext) -> Result<TrialOutcome>,
    > {
        FnTrialExecutor(
            |params: &HashMap<String, serde_json::Value>, _ctx: &TrialContext| {
                Ok(TrialOutcome::Completed(vec![MetricRecord {
                    name: "f1".into(),
                    value: params["x"].as_f64().unwrap_or(0.5),
                    step: 0,
                    timestamp: Utc::now(),
                }]))
            },
        )
    }

    // ── TrialContext unit tests (direct construction) ──

    use crate::pruner::TrialMetricHistory as History;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Spy pruner: counts consultations, prunes everything after warmup.
    struct SpyPruner {
        calls: AtomicUsize,
        prune_from_step: usize,
    }

    impl Pruner for SpyPruner {
        fn should_prune(
            &self,
            _metric: &str,
            _value: f64,
            step: usize,
            _history: &[History],
        ) -> Option<String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            (step >= self.prune_from_step).then(|| "spy".to_string())
        }
    }

    fn make_ctx(pruner: Option<Arc<SpyPruner>>) -> (TrialContext, Arc<EventBus>) {
        let bus = Arc::new(EventBus::new(64));
        let ctx = TrialContext {
            study_id: "s".into(),
            trial_id: "trial_0000".into(),
            objective: Some(("f1".to_string(), Direction::Maximize)),
            pruner: pruner.map(|p| p as Arc<dyn Pruner>),
            history: Arc::new(Vec::new()),
            event_bus: bus.clone(),
            shared: Arc::new(Mutex::new(TrialShared::default())),
        };
        (ctx, bus)
    }

    #[test]
    fn report_after_prune_is_sticky_and_skips_the_pruner() {
        let pruner = Arc::new(SpyPruner {
            calls: AtomicUsize::new(0),
            prune_from_step: 0,
        });
        let (ctx, _bus) = make_ctx(Some(pruner.clone()));

        assert!(!ctx.should_prune());
        assert!(ctx.report("f1", 0.1, 0), "pruned immediately");
        assert!(ctx.should_prune());
        let calls_after_prune = pruner.calls.load(Ordering::SeqCst);

        // Later reports return true WITHOUT consulting the pruner…
        assert!(ctx.report("f1", 0.9, 1));
        assert!(ctx.report("f1", 0.9, 2));
        assert_eq!(pruner.calls.load(Ordering::SeqCst), calls_after_prune);
        // …but the metrics are still recorded (pinned: push happens
        // before the prune check).
        assert_eq!(ctx.metrics().len(), 3);
    }

    #[test]
    fn non_objective_metric_never_consults_the_pruner() {
        let pruner = Arc::new(SpyPruner {
            calls: AtomicUsize::new(0),
            prune_from_step: 0,
        });
        let (ctx, _bus) = make_ctx(Some(pruner.clone()));

        assert!(!ctx.report("train_loss", 99.0, 0));
        assert!(!ctx.report("lr", 0.001, 0));
        assert_eq!(pruner.calls.load(Ordering::SeqCst), 0);
        assert!(ctx.report("f1", 0.0, 0), "objective metric does consult");
        assert_eq!(pruner.calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn report_emits_trial_metric_events() {
        let (ctx, bus) = make_ctx(None);
        let mut rx = bus.subscribe();
        ctx.report("f1", 0.5, 3);
        ctx.report("aux", 1.5, 3);

        let mut got = Vec::new();
        while let Ok(e) = rx.try_recv() {
            if let Event::TrialMetric {
                study_id,
                trial_id,
                metric,
            } = e
            {
                assert_eq!(study_id, "s");
                assert_eq!(trial_id, "trial_0000");
                got.push((metric.name, metric.value, metric.step));
            }
        }
        assert_eq!(
            got,
            vec![("f1".to_string(), 0.5, 3), ("aux".to_string(), 1.5, 3)]
        );
    }

    #[test]
    fn trial_context_accessors_and_cross_thread_clone() {
        let (ctx, _bus) = make_ctx(None);
        assert_eq!(ctx.trial_id(), "trial_0000");

        // The documented property: the handle can cross threads (e.g.
        // into a Python callback) and all reports land on the trial.
        let clone = ctx.clone();
        let handle = std::thread::spawn(move || {
            clone.report("from_thread", 1.0, 0);
        });
        ctx.report("from_main", 2.0, 0);
        handle.join().unwrap();

        let names: Vec<String> = ctx.metrics().into_iter().map(|m| m.name).collect();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"from_thread".to_string()));
        assert!(names.contains(&"from_main".to_string()));
    }

    // ── Runner behavior ──

    #[test]
    fn percentile_pruning_works_through_the_runner() {
        // Same shape as the median cases but exercising the Percentile
        // arm of build_pruner (dead code until now).
        use std::sync::atomic::AtomicUsize;
        let bus = Arc::new(EventBus::new(256));
        let runner = StudyRunner::new(bus);
        let mut study = Study::new(
            "percentile",
            one_dim_space(),
            SearchStrategy::Random {
                n_trials: 4,
                seed: Some(7),
            },
            vec![Objective {
                metric: "f1".into(),
                direction: Direction::Maximize,
            }],
        )
        .with_pruning(PruningStrategy::Percentile {
            percentile: 50.0,
            n_warmup_steps: 2,
        });
        let counter = AtomicUsize::new(0);
        let executor = FnTrialExecutor(
            move |_p: &HashMap<String, serde_json::Value>, ctx: &TrialContext| {
                let good = counter.fetch_add(1, Ordering::SeqCst) == 0;
                for step in 0..10 {
                    let value = if good { 0.5 + step as f64 * 0.05 } else { 0.01 };
                    if ctx.report("f1", value, step) {
                        return Ok(TrialOutcome::Pruned {
                            step,
                            reason: "stopped".into(),
                        });
                    }
                }
                Ok(TrialOutcome::Completed(vec![]))
            },
        );
        let mut sampler = RandomSampler::new(4, Some(7));
        runner.run(&mut study, &mut sampler, &executor).unwrap();
        assert_bad_trials_pruned(&study);
    }

    #[test]
    fn pruner_verdict_wins_over_completed_outcome() {
        // An executor that IGNORES report()'s stop signal and returns
        // Completed anyway: the runner must mark the trial pruned with
        // the pruner's step/reason and drop the returned metrics.
        use std::sync::atomic::AtomicUsize;
        let bus = Arc::new(EventBus::new(256));
        let mut rx = bus.subscribe();
        let runner = StudyRunner::new(bus);
        let mut study = pruning_study(Direction::Maximize);

        let counter = AtomicUsize::new(0);
        let executor = FnTrialExecutor(
            move |_p: &HashMap<String, serde_json::Value>, ctx: &TrialContext| {
                let good = counter.fetch_add(1, Ordering::SeqCst) == 0;
                for step in 0..10 {
                    let value = if good { 0.5 + step as f64 * 0.05 } else { 0.01 };
                    ctx.report("f1", value, step); // return value ignored!
                }
                Ok(TrialOutcome::Completed(vec![MetricRecord {
                    name: "sneaky".into(),
                    value: 1.0,
                    step: 0,
                    timestamp: Utc::now(),
                }]))
            },
        );
        let mut sampler = RandomSampler::new(4, Some(7));
        runner.run(&mut study, &mut sampler, &executor).unwrap();

        let pruned: Vec<_> = study
            .trials
            .iter()
            .filter(|t| matches!(t.state, TrialState::Pruned { .. }))
            .collect();
        assert_eq!(pruned.len(), 3);
        for t in &pruned {
            if let TrialState::Pruned { step, reason } = &t.state {
                assert_eq!(*step, 2, "pruner's step, not the executor's");
                assert!(reason.contains("median"), "pruner's reason: {reason}");
            }
            assert!(
                !t.metrics.iter().any(|m| m.name == "sneaky"),
                "final metrics of an overridden Completed are dropped"
            );
        }
        // TrialPruned events emitted, and no TrialCompleted for them.
        let mut completed = 0;
        let mut pruned_events = 0;
        while let Ok(e) = rx.try_recv() {
            match e {
                Event::TrialCompleted { .. } => completed += 1,
                Event::TrialPruned { .. } => pruned_events += 1,
                _ => {}
            }
        }
        assert_eq!(completed, 1);
        assert_eq!(pruned_events, 3);
    }

    /// Spy tracker recording every save_study call.
    #[derive(Default)]
    struct SpyTracker {
        saves: Mutex<Vec<usize>>, // trials.len() at each save
        fail: bool,
    }

    impl somatize_core::tracking::Tracker for SpyTracker {
        fn run_id(&self) -> &str {
            "spy"
        }
        fn run_dir(&self) -> &std::path::Path {
            std::path::Path::new("/nonexistent")
        }
        fn sink(&self) -> Arc<dyn somatize_core::tracking::EventSink> {
            struct Null;
            impl somatize_core::tracking::EventSink for Null {
                fn record(&self, _event: &Event) {}
            }
            Arc::new(Null)
        }
        fn save_manifest(&self, _m: &somatize_core::tracking::RunManifest) -> Result<()> {
            Ok(())
        }
        fn save_artifact(&self, _p: &str, _b: &[u8]) -> Result<()> {
            Ok(())
        }
        fn save_study(&self, study: &Study) -> Result<()> {
            if self.fail {
                return Err(somatize_core::SomaError::Other("disk gone".into()));
            }
            self.saves.lock().unwrap().push(study.trials.len());
            Ok(())
        }
        fn heartbeat(&self) -> Result<()> {
            Ok(())
        }
        fn finalize(&self, _s: somatize_core::tracking::RunState) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn study_is_saved_after_every_trial_with_monotonic_growth() {
        let bus = Arc::new(EventBus::new(256));
        let tracker = Arc::new(SpyTracker::default());
        let runner = StudyRunner::new(bus).with_tracker(tracker.clone());

        let mut study = Study::new(
            "persist",
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

        // One save per trial plus the final save: a runner that saved
        // only once at the end would fail this.
        let saves = tracker.saves.lock().unwrap().clone();
        assert_eq!(saves, vec![1, 2, 3, 3]);
    }

    #[test]
    fn failing_tracker_never_fails_the_study() {
        let bus = Arc::new(EventBus::new(256));
        let tracker = Arc::new(SpyTracker {
            fail: true,
            ..Default::default()
        });
        let runner = StudyRunner::new(bus).with_tracker(tracker);

        let mut study = Study::new(
            "resilient",
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
        assert_eq!(study.trials.len(), 3);
    }

    /// Sampler spy recording the ORDER of record_result vs sample calls.
    struct OrderSpySampler {
        inner: GridSampler,
        log: Vec<String>,
    }

    impl Sampler for OrderSpySampler {
        fn prepare(&mut self, space: &SearchSpace) {
            self.inner.prepare(space);
        }
        fn sample(
            &mut self,
            space: &SearchSpace,
            trial_index: usize,
        ) -> Result<Option<HashMap<String, serde_json::Value>>> {
            self.log.push(format!("sample:{trial_index}"));
            self.inner.sample(space, trial_index)
        }
        fn n_trials(&self) -> Option<usize> {
            self.inner.n_trials()
        }
        fn record_result(&mut self, _params: &HashMap<String, serde_json::Value>, value: f64) {
            self.log.push(format!("record:{value:.2}"));
        }
    }

    #[test]
    fn resume_replays_history_into_the_sampler_before_sampling() {
        let executor = make_executor();

        // Reference run to harvest 3 completed trials.
        let bus = Arc::new(EventBus::new(256));
        let runner = StudyRunner::new(bus);
        let mut study = Study::new(
            "replay",
            sample_space(),
            SearchStrategy::Grid { points_per_dim: 3 },
            vec![Objective {
                metric: "f1".into(),
                direction: Direction::Maximize,
            }],
        );
        runner
            .run(&mut study, &mut GridSampler::new(3), &executor)
            .unwrap();
        study.trials.truncate(3);

        let bus = Arc::new(EventBus::new(256));
        let runner = StudyRunner::new(bus);
        let mut spy = OrderSpySampler {
            inner: GridSampler::new(3),
            log: Vec::new(),
        };
        runner.run(&mut study, &mut spy, &executor).unwrap();

        // The first 3 log entries are replayed history; sampling only
        // starts afterwards, at the resume index.
        assert_eq!(
            spy.log.iter().filter(|e| e.starts_with("record:")).count(),
            3 + 3
        );
        assert!(
            spy.log[..3].iter().all(|e| e.starts_with("record:")),
            "history replay must precede sampling: {:?}",
            &spy.log[..4]
        );
        assert_eq!(spy.log[3], "sample:3", "sampling resumes at the next index");
    }

    #[test]
    fn frozen_param_overrides_a_sampled_dimension() {
        let bus = Arc::new(EventBus::new(256));
        let runner = StudyRunner::new(bus);

        // "x" is IN the search space and also frozen: frozen wins.
        let mut study = Study::new(
            "frozen-collision",
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
        study.frozen.insert("x".into(), serde_json::json!(0.75));

        let executor = FnTrialExecutor(
            |params: &HashMap<String, serde_json::Value>, _ctx: &TrialContext| {
                assert_eq!(params["x"], serde_json::json!(0.75));
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
            assert_eq!(t.params["x"], serde_json::json!(0.75));
        }
    }

    #[test]
    fn pruning_watches_first_composite_term_when_no_objectives() {
        use somatize_core::study::{CompositeObjective, Scalarizer};
        use std::sync::atomic::AtomicUsize;

        let bus = Arc::new(EventBus::new(256));
        let runner = StudyRunner::new(bus);
        let mut study = Study::new(
            "composite-pruning",
            one_dim_space(),
            SearchStrategy::Random {
                n_trials: 4,
                seed: Some(7),
            },
            vec![], // no objectives — pruning falls back to terms[0]
        )
        .with_composite(CompositeObjective {
            terms: vec![("f1".into(), 1.0), ("aux".into(), 0.1)],
            direction: Direction::Maximize,
            scalarizer: Scalarizer::WeightedSum,
        })
        .with_pruning(PruningStrategy::Median { n_warmup_steps: 2 });

        let counter = AtomicUsize::new(0);
        let executor = FnTrialExecutor(
            move |_p: &HashMap<String, serde_json::Value>, ctx: &TrialContext| {
                let good = counter.fetch_add(1, Ordering::SeqCst) == 0;
                for step in 0..10 {
                    let value = if good { 0.5 + step as f64 * 0.05 } else { 0.01 };
                    if ctx.report("f1", value, step) {
                        return Ok(TrialOutcome::Pruned {
                            step,
                            reason: "stopped".into(),
                        });
                    }
                }
                Ok(TrialOutcome::Completed(vec![MetricRecord {
                    name: "aux".into(),
                    value: 0.0,
                    step: 0,
                    timestamp: Utc::now(),
                }]))
            },
        );
        let mut sampler = RandomSampler::new(4, Some(7));
        runner.run(&mut study, &mut sampler, &executor).unwrap();
        assert_bad_trials_pruned(&study);
    }

    #[test]
    fn planned_trials_is_stamped_and_progress_completes() {
        let bus = Arc::new(EventBus::new(256));
        let runner = StudyRunner::new(bus);
        let mut study = Study::new(
            "stamped",
            sample_space(),
            SearchStrategy::Grid { points_per_dim: 3 },
            vec![Objective {
                metric: "f1".into(),
                direction: Direction::Maximize,
            }],
        );
        assert_eq!(study.total_trials(), None);
        runner
            .run(&mut study, &mut GridSampler::new(3), &make_executor())
            .unwrap();
        assert_eq!(study.planned_trials, Some(6));
        assert_eq!(study.total_trials(), Some(6));
        assert!((study.progress() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn all_failed_study_completes_with_nan_best() {
        let bus = Arc::new(EventBus::new(256));
        let mut rx = bus.subscribe();
        let runner = StudyRunner::new(bus);
        let mut study = Study::new(
            "doomed",
            one_dim_space(),
            SearchStrategy::Random {
                n_trials: 3,
                seed: Some(1),
            },
            vec![Objective {
                metric: "f1".into(),
                direction: Direction::Maximize,
            }],
        );
        let executor = FnTrialExecutor(
            |_p: &HashMap<String, serde_json::Value>, _ctx: &TrialContext| {
                Err(SomaError::Other("cuda out of memory".into()))
            },
        );
        let mut sampler = RandomSampler::new(3, Some(1));
        runner.run(&mut study, &mut sampler, &executor).unwrap();

        // Error strings preserved, ids contiguous, best is empty/NaN.
        for (i, t) in study.trials.iter().enumerate() {
            assert_eq!(t.id, format!("trial_{i:04}"));
            match &t.state {
                TrialState::Failed { error } => assert!(error.contains("cuda out of memory")),
                other => panic!("expected Failed, got {other:?}"),
            }
        }
        assert!(study.best_trial().is_none());

        let mut failed_events = 0;
        let mut completed_nan = false;
        while let Ok(e) = rx.try_recv() {
            match e {
                Event::TrialFailed { error, .. } => {
                    assert!(error.contains("cuda out of memory"));
                    failed_events += 1;
                }
                Event::StudyCompleted {
                    best_trial_id,
                    best_value,
                    ..
                } => {
                    assert!(best_trial_id.is_empty());
                    assert!(best_value.is_nan());
                    completed_nan = true;
                }
                _ => {}
            }
        }
        assert_eq!(failed_events, 3);
        assert!(completed_nan);
    }

    #[test]
    fn best_updated_fires_once_when_trials_worsen() {
        use std::sync::atomic::AtomicUsize;
        let bus = Arc::new(EventBus::new(256));
        let mut rx = bus.subscribe();
        let runner = StudyRunner::new(bus);
        let mut study = Study::new(
            "worsening",
            one_dim_space(),
            SearchStrategy::Random {
                n_trials: 3,
                seed: Some(1),
            },
            vec![Objective {
                metric: "f1".into(),
                direction: Direction::Maximize,
            }],
        );
        let counter = AtomicUsize::new(0);
        let executor = FnTrialExecutor(
            move |_p: &HashMap<String, serde_json::Value>, _ctx: &TrialContext| {
                let i = counter.fetch_add(1, Ordering::SeqCst);
                Ok(TrialOutcome::Completed(vec![MetricRecord {
                    name: "f1".into(),
                    value: 0.9 - i as f64 * 0.2, // 0.9, 0.7, 0.5
                    step: 0,
                    timestamp: Utc::now(),
                }]))
            },
        );
        let mut sampler = RandomSampler::new(3, Some(1));
        runner.run(&mut study, &mut sampler, &executor).unwrap();

        let mut best_events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            if let Event::BestUpdated { value, .. } = e {
                best_events.push(value);
            }
        }
        assert_eq!(best_events, vec![0.9], "only the first trial is ever best");
    }

    /// CONTRACT (pinned): metrics reported via ctx AND returned in
    /// Completed(final_metrics) are BOTH kept — the same name appears
    /// twice and the TrialMetric event fires twice. De-dup is the
    /// executor's responsibility.
    #[test]
    fn reported_and_final_metrics_are_concatenated_not_deduped() {
        let bus = Arc::new(EventBus::new(256));
        let mut rx = bus.subscribe();
        let runner = StudyRunner::new(bus);
        let mut study = Study::new(
            "dup",
            one_dim_space(),
            SearchStrategy::Random {
                n_trials: 1,
                seed: Some(1),
            },
            vec![Objective {
                metric: "f1".into(),
                direction: Direction::Maximize,
            }],
        );
        let executor = FnTrialExecutor(
            |_p: &HashMap<String, serde_json::Value>, ctx: &TrialContext| {
                ctx.report("f1", 0.4, 0);
                Ok(TrialOutcome::Completed(vec![MetricRecord {
                    name: "f1".into(),
                    value: 0.6,
                    step: 1,
                    timestamp: Utc::now(),
                }]))
            },
        );
        let mut sampler = RandomSampler::new(1, Some(1));
        runner.run(&mut study, &mut sampler, &executor).unwrap();

        let f1_records: Vec<f64> = study.trials[0]
            .metrics
            .iter()
            .filter(|m| m.name == "f1")
            .map(|m| m.value)
            .collect();
        assert_eq!(f1_records, vec![0.4, 0.6]);

        let mut metric_events = 0;
        while let Ok(e) = rx.try_recv() {
            if matches!(e, Event::TrialMetric { .. }) {
                metric_events += 1;
            }
        }
        assert_eq!(metric_events, 2);
    }

    #[test]
    fn sampler_feedback_values_are_exact_and_skip_non_completed() {
        let bus = Arc::new(EventBus::new(256));
        let runner = StudyRunner::new(bus);
        let mut study = Study::new(
            "feedback-exact",
            one_dim_space(),
            SearchStrategy::Random {
                n_trials: 3,
                seed: Some(1),
            },
            vec![Objective {
                metric: "loss".into(),
                direction: Direction::Minimize,
            }],
        );
        use std::sync::atomic::AtomicUsize;
        let counter = AtomicUsize::new(0);
        let executor = FnTrialExecutor(
            move |_p: &HashMap<String, serde_json::Value>, _ctx: &TrialContext| match counter
                .fetch_add(1, Ordering::SeqCst)
            {
                0 => Ok(TrialOutcome::Completed(vec![MetricRecord {
                    name: "loss".into(),
                    value: 0.25,
                    step: 0,
                    timestamp: Utc::now(),
                }])),
                1 => Ok(TrialOutcome::Pruned {
                    step: 1,
                    reason: "bad".into(),
                }),
                _ => Err(SomaError::Other("boom".into())),
            },
        );
        let mut spy = SpySampler {
            inner: RandomSampler::new(3, Some(1)),
            recorded: Vec::new(),
        };
        runner.run(&mut study, &mut spy, &executor).unwrap();

        // Only the completed trial fed back, negated for Minimize.
        assert_eq!(spy.recorded, vec![-0.25]);
    }

    #[test]
    fn timestamps_backfilled_and_monotonic() {
        let bus = Arc::new(EventBus::new(256));
        let tracker = Arc::new(SpyTracker::default());
        let runner = StudyRunner::new(bus).with_tracker(tracker);
        let mut study = Study::new(
            "ts",
            one_dim_space(),
            SearchStrategy::Random {
                n_trials: 2,
                seed: Some(1),
            },
            vec![Objective {
                metric: "f1".into(),
                direction: Direction::Maximize,
            }],
        );
        study.created_at = None; // e.g. loaded from a pre-timestamp JSON
        let mut sampler = RandomSampler::new(2, Some(1));
        runner
            .run(&mut study, &mut sampler, &make_simple_executor())
            .unwrap();

        assert!(study.created_at.is_some(), "backfilled");
        assert!(study.updated_at.is_some());
        for t in &study.trials {
            assert!(t.started_at.is_some());
            assert!(t.finished_at.unwrap() >= t.started_at.unwrap());
        }
    }

    #[test]
    fn bayesian_through_runner_improves_over_time() {
        // End-to-end proof that the ask/tell wiring feeds TPE: on a
        // unimodal objective the later trials must beat the early ones.
        // Deterministic (fixed seed) — not statistical.
        let bus = Arc::new(EventBus::new(1024));
        let runner = StudyRunner::new(bus);
        let mut study = Study::new(
            "tpe",
            one_dim_space(),
            SearchStrategy::Bayesian {
                n_trials: 30,
                n_startup: 8,
                seed: Some(42),
            },
            vec![Objective {
                metric: "score".into(),
                direction: Direction::Maximize,
            }],
        );
        let executor = FnTrialExecutor(
            |params: &HashMap<String, serde_json::Value>, _ctx: &TrialContext| {
                let x = params["x"].as_f64().unwrap();
                Ok(TrialOutcome::Completed(vec![MetricRecord {
                    name: "score".into(),
                    value: 1.0 - (x - 0.7).abs(), // peak at x = 0.7
                    step: 0,
                    timestamp: Utc::now(),
                }]))
            },
        );
        let mut sampler = BayesianSampler::new(30, 8, Some(42));
        runner.run(&mut study, &mut sampler, &executor).unwrap();

        let scores: Vec<f64> = study
            .trials
            .iter()
            .map(|t| t.last_metric("score").unwrap())
            .collect();
        let head: f64 = scores[..8].iter().sum::<f64>() / 8.0; // startup = random
        let tail: f64 = scores[20..].iter().sum::<f64>() / 10.0;
        assert!(
            tail > head,
            "TPE with feedback must beat its random startup: head={head:.3} tail={tail:.3}"
        );
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
            |_params: &HashMap<String, serde_json::Value>, _ctx: &TrialContext| {
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
