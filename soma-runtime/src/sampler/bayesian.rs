use crate::sampler::{Sampler, hash_u64, pseudo_random, sample_float};
use somatize_core::error::Result;
use somatize_core::search::{SearchDimension, SearchSpace};
use std::collections::HashMap;

/// Bayesian optimization sampler using Tree-Parzen Estimator (TPE).
///
/// For the first `n_startup` trials, samples randomly. After that,
/// uses the history of (params, metric) to model "good" vs "bad"
/// parameter distributions and samples from the "good" distribution.
///
/// This is a simplified TPE: it splits trials into top/bottom quantiles
/// and samples from the top quantile's parameter distributions.
pub struct BayesianSampler {
    n_trials: usize,
    n_startup: usize,
    seed: u64,
    /// History: (params, metric_value) for completed trials.
    history: Vec<(HashMap<String, serde_json::Value>, f64)>,
    /// Quantile split: top gamma fraction is "good".
    gamma: f64,
}

impl BayesianSampler {
    pub fn new(n_trials: usize, n_startup: usize, seed: Option<u64>) -> Self {
        Self {
            n_trials,
            n_startup: n_startup.max(2),
            seed: seed.unwrap_or(42),
            history: Vec::new(),
            gamma: 0.25, // top 25% are "good"
        }
    }

    /// Record a completed trial's result (for informing future samples).
    pub fn record(&mut self, params: HashMap<String, serde_json::Value>, metric: f64) {
        self.history.push((params, metric));
    }

    /// Sample using TPE: bias towards parameters seen in "good" trials.
    fn sample_tpe(
        &self,
        space: &SearchSpace,
        trial_index: usize,
    ) -> HashMap<String, serde_json::Value> {
        // Split history into good/bad by quantile
        let mut sorted_history: Vec<(usize, f64)> = self
            .history
            .iter()
            .enumerate()
            .map(|(i, (_, v))| (i, *v))
            .collect();
        sorted_history.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let n_good = (self.history.len() as f64 * self.gamma).ceil() as usize;
        let n_good = n_good.max(1).min(self.history.len());
        let good_indices: Vec<usize> = sorted_history[..n_good].iter().map(|(i, _)| *i).collect();

        let mut params = HashMap::new();
        for (dim_idx, dim) in space.active_dimensions().iter().enumerate() {
            let rng_state = hash_u64(self.seed, trial_index as u64, dim_idx as u64);
            let t = pseudo_random(rng_state);

            // With 80% probability, sample near good trials' values for this dim.
            // With 20% probability, sample uniformly (exploration).
            let explore_prob = pseudo_random(hash_u64(
                self.seed,
                trial_index as u64,
                dim_idx as u64 + 1000,
            ));

            let value = if explore_prob < 0.2 || good_indices.is_empty() {
                // Explore: sample uniformly
                self.sample_uniform(dim, t)
            } else {
                // Exploit: sample near a good trial's value
                let good_idx = good_indices
                    [((t * good_indices.len() as f64) as usize).min(good_indices.len() - 1)];
                let good_params = &self.history[good_idx].0;

                if let Some(good_val) = good_params.get(dim.name()) {
                    self.sample_near(dim, good_val, rng_state)
                } else {
                    self.sample_uniform(dim, t)
                }
            };

            params.insert(dim.name().to_string(), value);
        }

        params
    }

    fn sample_uniform(&self, dim: &SearchDimension, t: f64) -> serde_json::Value {
        match dim {
            SearchDimension::Float {
                low, high, scale, ..
            } => {
                serde_json::json!(sample_float(*low, *high, *scale, t))
            }
            SearchDimension::Int { low, high, .. } => {
                let range = (*high - *low + 1) as f64;
                let val = *low + (t * range).floor() as i64;
                serde_json::json!(val.min(*high))
            }
            SearchDimension::Categorical { choices, .. } => {
                let idx = (t * choices.len() as f64).floor() as usize;
                choices[idx.min(choices.len() - 1)].clone()
            }
            _ => serde_json::Value::Null,
        }
    }

    /// Sample near a "good" value with gaussian-like perturbation.
    fn sample_near(
        &self,
        dim: &SearchDimension,
        center: &serde_json::Value,
        rng_state: u64,
    ) -> serde_json::Value {
        let t = pseudo_random(hash_u64(rng_state, 777, 0));
        let perturbation = (pseudo_random(hash_u64(rng_state, 888, 0)) - 0.5) * 0.3;

        match dim {
            SearchDimension::Float { low, high, .. } => {
                if let Some(center_val) = center.as_f64() {
                    let range = *high - *low;
                    let new_val = (center_val + perturbation * range).clamp(*low, *high);
                    serde_json::json!(new_val)
                } else {
                    self.sample_uniform(dim, t)
                }
            }
            SearchDimension::Int { low, high, .. } => {
                if let Some(center_val) = center.as_i64() {
                    let range = (*high - *low) as f64;
                    let new_val = (center_val as f64 + perturbation * range).round() as i64;
                    serde_json::json!(new_val.clamp(*low, *high))
                } else {
                    self.sample_uniform(dim, t)
                }
            }
            SearchDimension::Categorical { choices, .. } => {
                // For categorical: mostly keep the good value, sometimes explore
                if perturbation.abs() < 0.1 {
                    center.clone()
                } else {
                    let idx = (t * choices.len() as f64).floor() as usize;
                    choices[idx.min(choices.len() - 1)].clone()
                }
            }
            _ => serde_json::Value::Null,
        }
    }
}

impl Sampler for BayesianSampler {
    fn sample(
        &mut self,
        space: &SearchSpace,
        trial_index: usize,
    ) -> Result<Option<HashMap<String, serde_json::Value>>> {
        if trial_index >= self.n_trials {
            return Ok(None);
        }

        if trial_index < self.n_startup || self.history.is_empty() {
            // Random startup phase
            let mut params = HashMap::new();
            for (i, dim) in space.active_dimensions().iter().enumerate() {
                let rng_state = hash_u64(self.seed, trial_index as u64, i as u64);
                let t = pseudo_random(rng_state);
                params.insert(dim.name().to_string(), self.sample_uniform(dim, t));
            }
            Ok(Some(params))
        } else {
            Ok(Some(self.sample_tpe(space, trial_index)))
        }
    }

    fn n_trials(&self) -> Option<usize> {
        Some(self.n_trials)
    }

    /// Completed-trial feedback — this is what makes TPE model-based
    /// instead of degenerating to random search.
    fn record_result(&mut self, params: &HashMap<String, serde_json::Value>, value: f64) {
        self.record(params.clone(), value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use somatize_core::search::Scale;

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
            name: "kernel".into(),
            choices: vec![serde_json::json!("rbf"), serde_json::json!("linear")],
        });
        space
    }

    #[test]
    fn startup_phase_is_random() {
        let mut sampler = BayesianSampler::new(20, 5, Some(42));
        let space = sample_space();

        // First 5 trials should all produce different params (random)
        let mut samples = Vec::new();
        for i in 0..5 {
            let params = sampler.sample(&space, i).unwrap().unwrap();
            assert!(params.contains_key("lr"));
            assert!(params.contains_key("kernel"));
            samples.push(params);
        }

        // Check they're not all identical
        let lrs: Vec<f64> = samples.iter().map(|p| p["lr"].as_f64().unwrap()).collect();
        assert!(lrs.windows(2).any(|w| (w[0] - w[1]).abs() > 1e-10));
    }

    #[test]
    fn tpe_phase_after_recording_history() {
        let mut sampler = BayesianSampler::new(20, 3, Some(42));
        let space = sample_space();

        // Record some history
        for i in 0..5 {
            let params = sampler.sample(&space, i).unwrap().unwrap();
            let lr = params["lr"].as_f64().unwrap();
            let metric = 1.0 - (lr - 0.01).abs() * 10.0; // best at lr=0.01
            sampler.record(params, metric);
        }

        // Now sample in TPE mode (trial_index >= n_startup)
        let params = sampler.sample(&space, 5).unwrap().unwrap();
        assert!(params.contains_key("lr"));
        let lr = params["lr"].as_f64().unwrap();
        assert!((0.001..=0.1).contains(&lr));
    }

    #[test]
    fn record_result_trait_method_feeds_the_model() {
        // Through &mut dyn Sampler — the exact call path StudyRunner
        // uses. A sampler whose history grew must sample differently
        // from an identical one with no history at the same index.
        use crate::sampler::Sampler as _;

        let space = sample_space();
        let mut fed = BayesianSampler::new(40, 3, Some(42));
        let mut unfed = BayesianSampler::new(40, 3, Some(42));

        {
            let as_dyn: &mut dyn crate::sampler::Sampler = &mut fed;
            for i in 0..10 {
                let params = as_dyn.sample(&space, i).unwrap().unwrap();
                let lr = params["lr"].as_f64().unwrap();
                as_dyn.record_result(&params, 1.0 - (lr - 0.01).abs() * 10.0);
            }
        }
        let with_history = fed.sample(&space, 15).unwrap().unwrap();
        let without_history = unfed.sample(&space, 15).unwrap().unwrap();
        assert_ne!(
            with_history["lr"], without_history["lr"],
            "history received via the trait method must change sampling"
        );
    }

    #[test]
    fn tpe_actually_biases_towards_good_regions() {
        // Feed 20 observations peaked at lr = 0.01, then draw 20 TPE
        // samples and compare against a no-history control: the median
        // distance to the optimum must shrink. Deterministic (seeded).
        use crate::sampler::Sampler as _;

        let space = sample_space();
        let mut tpe = BayesianSampler::new(200, 3, Some(7));
        let mut control = BayesianSampler::new(200, 3, Some(7));

        for i in 0..20 {
            let params = tpe.sample(&space, i).unwrap().unwrap();
            let lr = params["lr"].as_f64().unwrap();
            tpe.record_result(&params, 1.0 - (lr - 0.01).abs() * 10.0);
        }

        let mut median_dist = |s: &mut BayesianSampler| -> f64 {
            let mut dists: Vec<f64> = (100..120)
                .map(|i| {
                    let p = s.sample(&space, i).unwrap().unwrap();
                    (p["lr"].as_f64().unwrap() - 0.01).abs()
                })
                .collect();
            dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
            dists[dists.len() / 2]
        };

        let tpe_median = median_dist(&mut tpe);
        // Control has an empty history → falls back to random sampling.
        let control_median = median_dist(&mut control);
        assert!(
            tpe_median < control_median,
            "TPE median distance {tpe_median:.4} must beat random {control_median:.4}"
        );
    }

    #[test]
    fn respects_n_trials_limit() {
        let mut sampler = BayesianSampler::new(10, 3, Some(42));
        let space = sample_space();

        for i in 0..15 {
            let result = sampler.sample(&space, i).unwrap();
            if i < 10 {
                assert!(result.is_some());
            } else {
                assert!(result.is_none());
            }
        }
    }

    #[test]
    fn deterministic_with_seed() {
        let space = sample_space();

        let mut s1 = BayesianSampler::new(10, 3, Some(42));
        let mut s2 = BayesianSampler::new(10, 3, Some(42));

        for i in 0..5 {
            let p1 = s1.sample(&space, i).unwrap().unwrap();
            let p2 = s2.sample(&space, i).unwrap().unwrap();
            assert_eq!(p1, p2);
        }
    }

    #[test]
    fn different_seeds_differ() {
        let space = sample_space();

        let mut s1 = BayesianSampler::new(10, 3, Some(42));
        let mut s2 = BayesianSampler::new(10, 3, Some(99));

        let p1 = s1.sample(&space, 0).unwrap().unwrap();
        let p2 = s2.sample(&space, 0).unwrap().unwrap();
        assert_ne!(p1["lr"], p2["lr"]);
    }
}
