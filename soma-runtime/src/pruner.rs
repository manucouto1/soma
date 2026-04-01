//! Early stopping strategies for optimization studies.
//!
//! A [`Pruner`] decides whether a trial should be stopped based on
//! intermediate metric values. Implementations: [`MedianPruner`],
//! [`PercentilePruner`].

use soma_core::event::MetricRecord;

/// A pruner decides whether to stop a trial early based on intermediate metrics.
pub trait Pruner: Send + Sync {
    /// Decide whether to prune given the current trial's metrics and
    /// the history of completed trials' metrics at the same step.
    ///
    /// Returns `Some(reason)` if the trial should be pruned.
    fn should_prune(
        &self,
        metric_name: &str,
        current_value: f64,
        step: usize,
        history: &[TrialMetricHistory],
    ) -> Option<String>;
}

/// A completed trial's metric history (for comparing against).
pub struct TrialMetricHistory {
    pub trial_id: String,
    pub metrics: Vec<MetricRecord>,
}

/// Prune if current value is below the median of completed trials at the same step.
pub struct MedianPruner {
    /// Don't prune before this many steps.
    pub n_warmup_steps: usize,
    /// Minimum completed trials needed before pruning starts.
    pub min_trials: usize,
}

impl MedianPruner {
    pub fn new(n_warmup_steps: usize) -> Self {
        Self {
            n_warmup_steps,
            min_trials: 1,
        }
    }

    pub fn with_min_trials(mut self, min_trials: usize) -> Self {
        self.min_trials = min_trials;
        self
    }
}

impl Pruner for MedianPruner {
    fn should_prune(
        &self,
        metric_name: &str,
        current_value: f64,
        step: usize,
        history: &[TrialMetricHistory],
    ) -> Option<String> {
        if step < self.n_warmup_steps {
            return None;
        }

        // Collect values at this step from completed trials
        let mut values_at_step: Vec<f64> = history
            .iter()
            .filter_map(|h| {
                h.metrics
                    .iter()
                    .filter(|m| m.name == metric_name && m.step == step)
                    .map(|m| m.value)
                    .next_back()
            })
            .collect();

        if values_at_step.len() < self.min_trials {
            return None;
        }

        values_at_step.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if values_at_step.len().is_multiple_of(2) {
            let mid = values_at_step.len() / 2;
            (values_at_step[mid - 1] + values_at_step[mid]) / 2.0
        } else {
            values_at_step[values_at_step.len() / 2]
        };

        // Prune if below median (assuming maximize; for minimize, caller inverts)
        if current_value < median {
            Some(format!(
                "value {current_value:.4} below median {median:.4} at step {step}"
            ))
        } else {
            None
        }
    }
}

/// Prune if current value is below the given percentile.
pub struct PercentilePruner {
    pub percentile: f64,
    pub n_warmup_steps: usize,
    pub min_trials: usize,
}

impl PercentilePruner {
    pub fn new(percentile: f64, n_warmup_steps: usize) -> Self {
        Self {
            percentile,
            n_warmup_steps,
            min_trials: 1,
        }
    }
}

impl Pruner for PercentilePruner {
    fn should_prune(
        &self,
        metric_name: &str,
        current_value: f64,
        step: usize,
        history: &[TrialMetricHistory],
    ) -> Option<String> {
        if step < self.n_warmup_steps {
            return None;
        }

        let mut values_at_step: Vec<f64> = history
            .iter()
            .filter_map(|h| {
                h.metrics
                    .iter()
                    .filter(|m| m.name == metric_name && m.step == step)
                    .map(|m| m.value)
                    .next_back()
            })
            .collect();

        if values_at_step.len() < self.min_trials {
            return None;
        }

        values_at_step.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((self.percentile / 100.0) * values_at_step.len() as f64).floor() as usize;
        let idx = idx.min(values_at_step.len() - 1);
        let threshold = values_at_step[idx];

        if current_value < threshold {
            Some(format!(
                "value {current_value:.4} below p{:.0} threshold {threshold:.4} at step {step}",
                self.percentile
            ))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_history(values_per_step: &[Vec<f64>]) -> Vec<TrialMetricHistory> {
        values_per_step[0]
            .iter()
            .enumerate()
            .map(|(trial_idx, _)| {
                let metrics: Vec<MetricRecord> = values_per_step
                    .iter()
                    .enumerate()
                    .filter_map(|(step, vals)| {
                        vals.get(trial_idx).map(|&v| MetricRecord {
                            name: "f1".into(),
                            value: v,
                            step,
                            timestamp: Utc::now(),
                        })
                    })
                    .collect();
                TrialMetricHistory {
                    trial_id: format!("t{trial_idx}"),
                    metrics,
                }
            })
            .collect()
    }

    // ── Median pruner ──

    #[test]
    fn median_no_prune_during_warmup() {
        let pruner = MedianPruner::new(5);
        let history = make_history(&[vec![0.9, 0.8, 0.7]]);
        assert!(pruner.should_prune("f1", 0.1, 3, &history).is_none());
    }

    #[test]
    fn median_prunes_below_median() {
        let pruner = MedianPruner::new(0);
        // At step 0: values are [0.7, 0.8, 0.9]. Median = 0.8
        let history = make_history(&[vec![0.7, 0.8, 0.9]]);
        // Current = 0.5, below median 0.8
        assert!(pruner.should_prune("f1", 0.5, 0, &history).is_some());
    }

    #[test]
    fn median_keeps_above_median() {
        let pruner = MedianPruner::new(0);
        let history = make_history(&[vec![0.7, 0.8, 0.9]]);
        // Current = 0.85, above median 0.8
        assert!(pruner.should_prune("f1", 0.85, 0, &history).is_none());
    }

    #[test]
    fn median_no_prune_insufficient_history() {
        let pruner = MedianPruner::new(0).with_min_trials(5);
        let history = make_history(&[vec![0.7, 0.8]]);
        // Only 2 trials, need 5
        assert!(pruner.should_prune("f1", 0.1, 0, &history).is_none());
    }

    #[test]
    fn median_empty_history() {
        let pruner = MedianPruner::new(0);
        assert!(pruner.should_prune("f1", 0.5, 0, &[]).is_none());
    }

    // ── Percentile pruner ──

    #[test]
    fn percentile_prunes_below_threshold() {
        let pruner = PercentilePruner::new(25.0, 0);
        // At step 0: sorted = [0.3, 0.5, 0.7, 0.9]. p25 idx=1 → threshold=0.5
        let history = make_history(&[vec![0.5, 0.9, 0.3, 0.7]]);
        // Current = 0.2, below p25 threshold
        assert!(pruner.should_prune("f1", 0.2, 0, &history).is_some());
    }

    #[test]
    fn percentile_keeps_above_threshold() {
        let pruner = PercentilePruner::new(25.0, 0);
        let history = make_history(&[vec![0.5, 0.9, 0.3, 0.7]]);
        assert!(pruner.should_prune("f1", 0.6, 0, &history).is_none());
    }

    #[test]
    fn percentile_warmup_respected() {
        let pruner = PercentilePruner::new(50.0, 10);
        let history = make_history(&[vec![0.9]]);
        assert!(pruner.should_prune("f1", 0.1, 5, &history).is_none());
    }
}
