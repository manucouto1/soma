use crate::event_bus::EventBus;
use crate::sampler::Sampler;
use soma_core::error::{Result, SomaError};
use soma_core::event::{Event, MetricRecord};
use soma_core::study::{Study, Trial, TrialState};
use std::sync::Arc;
use std::time::Instant;

/// Callback that executes a trial given sampled parameters.
/// Returns the final metrics for the trial.
pub trait TrialExecutor: Send + Sync {
    fn execute_trial(
        &self,
        params: &std::collections::HashMap<String, serde_json::Value>,
    ) -> Result<Vec<MetricRecord>>;
}

/// Function-based trial executor for convenience.
pub struct FnTrialExecutor<F>(pub F);

impl<F> TrialExecutor for FnTrialExecutor<F>
where
    F: Fn(&std::collections::HashMap<String, serde_json::Value>) -> Result<Vec<MetricRecord>>
        + Send
        + Sync,
{
    fn execute_trial(
        &self,
        params: &std::collections::HashMap<String, serde_json::Value>,
    ) -> Result<Vec<MetricRecord>> {
        (self.0)(params)
    }
}

/// Runs a Study: samples parameters, executes trials, records results.
pub struct StudyRunner {
    event_bus: Arc<EventBus>,
}

impl StudyRunner {
    pub fn new(event_bus: Arc<EventBus>) -> Self {
        Self { event_bus }
    }

    /// Run the study to completion.
    pub fn run(
        &self,
        study: &mut Study,
        sampler: &mut dyn Sampler,
        executor: &dyn TrialExecutor,
    ) -> Result<()> {
        let total = sampler.n_trials().unwrap_or(0);

        self.event_bus.emit(Event::StudyStarted {
            study_id: study.id.clone(),
            name: study.name.clone(),
            total_trials: total,
        });

        let mut trial_index = 0;

        while let Some(params) = sampler.sample(&study.search_space, trial_index)? {

            let trial_id = format!("trial_{trial_index:04}");
            let mut trial = Trial::new(trial_id.clone(), params.clone());
            trial.state = TrialState::Running;

            self.event_bus.emit(Event::TrialStarted {
                study_id: study.id.clone(),
                trial_id: trial_id.clone(),
                params: serde_json::json!(params),
            });

            let start = Instant::now();

            match executor.execute_trial(&params) {
                Ok(metrics) => {
                    trial.duration_ms = Some(start.elapsed().as_millis() as u64);
                    trial.metrics = metrics.clone();
                    trial.state = TrialState::Completed;

                    // Emit metric events
                    for metric in &metrics {
                        self.event_bus.emit(Event::TrialMetric {
                            study_id: study.id.clone(),
                            trial_id: trial_id.clone(),
                            metric: metric.clone(),
                        });
                    }

                    self.event_bus.emit(Event::TrialCompleted {
                        study_id: study.id.clone(),
                        trial_id: trial_id.clone(),
                        final_metrics: metrics,
                    });
                }
                Err(SomaError::Pruned { step, reason }) => {
                    trial.duration_ms = Some(start.elapsed().as_millis() as u64);
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
                Err(e) => {
                    trial.duration_ms = Some(start.elapsed().as_millis() as u64);
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

            // Check if we have a new best
            if let Some(best) = study.best_trial()
                && best.id == trial_id
                && let Some(obj) = study.objectives.first()
                && let Some(val) = best.best_metric(&obj.metric, obj.direction)
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
                best_value: study
                    .best_trial()
                    .and_then(|t| {
                        study
                            .objectives
                            .first()
                            .and_then(|o| t.best_metric(&o.metric, o.direction))
                    })
                    .unwrap_or(f64::NAN),
            });

            trial_index += 1;
        }

        let best_trial_id = study
            .best_trial()
            .map(|t| t.id.clone())
            .unwrap_or_default();
        let best_value = study
            .best_trial()
            .and_then(|t| {
                study
                    .objectives
                    .first()
                    .and_then(|o| t.best_metric(&o.metric, o.direction))
            })
            .unwrap_or(f64::NAN);

        self.event_bus.emit(Event::StudyCompleted {
            study_id: study.id.clone(),
            best_trial_id,
            best_value,
        });

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::{GridSampler, RandomSampler};
    use chrono::Utc;
    use soma_core::search::{Scale, SearchDimension, SearchSpace};
    use soma_core::study::{Direction, Objective, SearchStrategy};

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
            choices: vec![
                serde_json::json!("relu"),
                serde_json::json!("tanh"),
            ],
        });
        space
    }

    /// Simple executor: f1 = 1.0 - |lr - 0.01| * 10
    fn make_executor() -> FnTrialExecutor<
        impl Fn(
            &std::collections::HashMap<String, serde_json::Value>,
        ) -> Result<Vec<MetricRecord>>,
    > {
        FnTrialExecutor(|params: &std::collections::HashMap<String, serde_json::Value>| {
            let lr = params["lr"].as_f64().unwrap();
            let f1 = (1.0 - (lr - 0.01).abs() * 10.0).max(0.0);
            Ok(vec![MetricRecord {
                name: "f1".into(),
                value: f1,
                step: 0,
                timestamp: Utc::now(),
            }])
        })
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
        assert!(events.iter().any(|e| matches!(e, Event::StudyStarted { .. })));
        assert!(events.iter().any(|e| matches!(e, Event::TrialStarted { .. })));
        assert!(events.iter().any(|e| matches!(e, Event::TrialCompleted { .. })));
        assert!(events.iter().any(|e| matches!(e, Event::BestUpdated { .. })));
        assert!(events.iter().any(|e| matches!(e, Event::StudyCompleted { .. })));
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
            |params: &std::collections::HashMap<String, serde_json::Value>| {
                let x = params["x"].as_f64().unwrap();
                if x > 0.5 {
                    Err(SomaError::Other("too high".into()))
                } else {
                    Ok(vec![MetricRecord {
                        name: "f1".into(),
                        value: x,
                        step: 0,
                        timestamp: Utc::now(),
                    }])
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
            |_params: &std::collections::HashMap<String, serde_json::Value>| {
                Err(SomaError::Pruned {
                    step: 5,
                    reason: "below median".into(),
                })
            },
        );

        let mut sampler = RandomSampler::new(3, Some(42));
        runner.run(&mut study, &mut sampler, &executor).unwrap();

        assert!(study.trials.iter().all(|t| matches!(
            t.state,
            TrialState::Pruned { .. }
        )));
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
            |_params: &std::collections::HashMap<String, serde_json::Value>| {
                Ok(vec![MetricRecord {
                    name: "f1".into(),
                    value: 0.5,
                    step: 0,
                    timestamp: Utc::now(),
                }])
            },
        );

        let mut sampler = RandomSampler::new(3, Some(42));
        runner.run(&mut study, &mut sampler, &executor).unwrap();

        // Collect progress events
        let mut progress_events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            if let Event::StudyProgress { completed, total, .. } = e {
                progress_events.push((completed, total));
            }
        }

        assert_eq!(progress_events.len(), 3);
        assert_eq!(progress_events[0], (1, 3));
        assert_eq!(progress_events[1], (2, 3));
        assert_eq!(progress_events[2], (3, 3));
    }
}
