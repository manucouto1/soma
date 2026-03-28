use soma_core::error::Result;
use soma_core::search::{Scale, SearchDimension, SearchSpace};
use std::collections::HashMap;

/// A sampler produces hyperparameter configurations from a search space.
pub trait Sampler: Send + Sync {
    /// Sample the next set of parameters. Returns None when exhausted.
    fn sample(
        &mut self,
        space: &SearchSpace,
        trial_index: usize,
    ) -> Result<Option<HashMap<String, serde_json::Value>>>;

    /// Total number of trials this sampler will produce (if known).
    fn n_trials(&self) -> Option<usize>;
}

// ──────────────────────────────────────────────
// Grid Sampler
// ──────────────────────────────────────────────

/// Exhaustive grid search over all combinations.
///
/// Uses lazy index-based generation: instead of building the full cartesian
/// product in memory, it computes the parameter set for a given trial index
/// on the fly. Safe for large search spaces.
pub struct GridSampler {
    points_per_dim: usize,
    /// Cached per-dimension discrete values (computed once, not the full grid).
    dim_values: Option<Vec<(String, Vec<serde_json::Value>)>>,
    /// Total number of combinations.
    total: Option<usize>,
}

impl GridSampler {
    pub fn new(points_per_dim: usize) -> Self {
        Self {
            points_per_dim,
            dim_values: None,
            total: None,
        }
    }

    /// Compute discrete values for each dimension (once).
    fn ensure_dims(&mut self, space: &SearchSpace) {
        if self.dim_values.is_some() {
            return;
        }
        let dims: Vec<(String, Vec<serde_json::Value>)> = space
            .active_dimensions()
            .iter()
            .map(|dim| {
                let name = dim.name().to_string();
                let values = self.discretize(dim);
                (name, values)
            })
            .collect();

        let total = if dims.is_empty() {
            1 // one combo with empty params
        } else {
            dims.iter().map(|(_, v)| v.len()).product()
        };

        self.dim_values = Some(dims);
        self.total = Some(total);
    }

    /// Convert a flat trial index into a multi-dimensional index
    /// and look up the parameter values. O(n_dims) per call.
    fn sample_at(&self, trial_index: usize) -> Option<HashMap<String, serde_json::Value>> {
        let dims = self.dim_values.as_ref()?;
        let total = self.total?;

        if trial_index >= total {
            return None;
        }

        if dims.is_empty() {
            return Some(HashMap::new());
        }

        let mut params = HashMap::new();
        let mut remaining = trial_index;

        // Decompose flat index into per-dimension indices
        // like converting a number to mixed-radix representation
        for (name, values) in dims.iter().rev() {
            let dim_size = values.len();
            let dim_idx = remaining % dim_size;
            remaining /= dim_size;
            params.insert(name.clone(), values[dim_idx].clone());
        }

        Some(params)
    }

    fn discretize(&self, dim: &SearchDimension) -> Vec<serde_json::Value> {
        match dim {
            SearchDimension::Float {
                low, high, scale, ..
            } => linspace(*low, *high, self.points_per_dim, *scale)
                .into_iter()
                .map(|v| serde_json::json!(v))
                .collect(),
            SearchDimension::Int {
                low, high, scale, ..
            } => {
                let n = self.points_per_dim.min((*high - *low + 1) as usize);
                linspace(*low as f64, *high as f64, n, *scale)
                    .into_iter()
                    .map(|v| serde_json::json!(v.round() as i64))
                    .collect()
            }
            SearchDimension::Categorical { choices, .. } => choices.clone(),
            SearchDimension::Conditional { dimension, .. } => self.discretize(dimension),
            _ => vec![serde_json::Value::Null],
        }
    }
}

impl Sampler for GridSampler {
    fn sample(
        &mut self,
        space: &SearchSpace,
        trial_index: usize,
    ) -> Result<Option<HashMap<String, serde_json::Value>>> {
        self.ensure_dims(space);
        Ok(self.sample_at(trial_index))
    }

    fn n_trials(&self) -> Option<usize> {
        self.total
    }
}

// ──────────────────────────────────────────────
// Random Sampler
// ──────────────────────────────────────────────

/// Random search: sample uniformly from each dimension.
pub struct RandomSampler {
    n_trials: usize,
    seed: u64,
}

impl RandomSampler {
    pub fn new(n_trials: usize, seed: Option<u64>) -> Self {
        Self {
            n_trials,
            seed: seed.unwrap_or(42),
        }
    }

    fn sample_dim(&self, dim: &SearchDimension, rng_state: u64) -> serde_json::Value {
        let t = pseudo_random(rng_state); // [0.0, 1.0)
        match dim {
            SearchDimension::Float {
                low, high, scale, ..
            } => {
                let val = sample_float(*low, *high, *scale, t);
                serde_json::json!(val)
            }
            SearchDimension::Int { low, high, .. } => {
                let range = (*high - *low + 1) as f64;
                let val = *low + (t * range).floor() as i64;
                let val = val.min(*high);
                serde_json::json!(val)
            }
            SearchDimension::Categorical { choices, .. } => {
                let idx = (t * choices.len() as f64).floor() as usize;
                let idx = idx.min(choices.len() - 1);
                choices[idx].clone()
            }
            SearchDimension::Conditional { dimension, .. } => {
                self.sample_dim(dimension, rng_state)
            }
            _ => serde_json::Value::Null,
        }
    }
}

impl Sampler for RandomSampler {
    fn sample(
        &mut self,
        space: &SearchSpace,
        trial_index: usize,
    ) -> Result<Option<HashMap<String, serde_json::Value>>> {
        if trial_index >= self.n_trials {
            return Ok(None);
        }

        let mut params = HashMap::new();
        for (i, dim) in space.active_dimensions().iter().enumerate() {
            // Different rng state per dimension per trial
            let rng_state = hash_u64(self.seed, trial_index as u64, i as u64);
            let value = self.sample_dim(dim, rng_state);
            params.insert(dim.name().to_string(), value);
        }

        Ok(Some(params))
    }

    fn n_trials(&self) -> Option<usize> {
        Some(self.n_trials)
    }
}

// ──────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────

/// Generate evenly spaced values in a range, respecting scale.
fn linspace(low: f64, high: f64, n: usize, scale: Scale) -> Vec<f64> {
    if n <= 1 {
        return vec![(low + high) / 2.0];
    }
    match scale {
        Scale::Linear => (0..n)
            .map(|i| low + (high - low) * (i as f64 / (n - 1) as f64))
            .collect(),
        Scale::Log => {
            let log_low = low.max(1e-12).ln();
            let log_high = high.max(1e-12).ln();
            (0..n)
                .map(|i| (log_low + (log_high - log_low) * (i as f64 / (n - 1) as f64)).exp())
                .collect()
        }
        Scale::ReverseLog => {
            // Reverse: denser at high end
            linspace(low, high, n, Scale::Log)
                .into_iter()
                .rev()
                .collect()
        }
    }
}

/// Sample a float from [low, high] given t in [0, 1), respecting scale.
fn sample_float(low: f64, high: f64, scale: Scale, t: f64) -> f64 {
    match scale {
        Scale::Linear => low + (high - low) * t,
        Scale::Log => {
            let log_low = low.max(1e-12).ln();
            let log_high = high.max(1e-12).ln();
            (log_low + (log_high - log_low) * t).exp()
        }
        Scale::ReverseLog => {
            let val = sample_float(low, high, Scale::Log, 1.0 - t);
            low + high - val
        }
    }
}

/// Simple deterministic pseudo-random: hash-based, returns [0.0, 1.0).
fn pseudo_random(state: u64) -> f64 {
    let h = splitmix64(state);
    (h >> 11) as f64 / (1u64 << 53) as f64
}

/// Simple hash combiner for generating unique RNG states.
fn hash_u64(seed: u64, a: u64, b: u64) -> u64 {
    splitmix64(seed.wrapping_add(a.wrapping_mul(6364136223846793005)).wrapping_add(b))
}

/// SplitMix64 hash function.
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

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
            choices: vec![json!("rbf"), json!("linear"), json!("poly")],
        });
        space
    }

    // ── Grid tests ──

    #[test]
    fn grid_sampler_generates_all_combinations() {
        let mut sampler = GridSampler::new(3);
        let space = sample_space();

        // 3 points for lr * 3 choices for kernel = 9 combinations
        let mut trials = Vec::new();
        for i in 0.. {
            match sampler.sample(&space, i).unwrap() {
                Some(params) => trials.push(params),
                None => break,
            }
        }

        assert_eq!(trials.len(), 9);

        // All should have both params
        for t in &trials {
            assert!(t.contains_key("lr"));
            assert!(t.contains_key("kernel"));
        }

        // All kernels should appear
        let kernels: Vec<&serde_json::Value> = trials.iter().map(|t| &t["kernel"]).collect();
        assert!(kernels.contains(&&json!("rbf")));
        assert!(kernels.contains(&&json!("linear")));
        assert!(kernels.contains(&&json!("poly")));
    }

    #[test]
    fn grid_sampler_respects_log_scale() {
        let mut space = SearchSpace::new();
        space.add(SearchDimension::Float {
            name: "lr".into(),
            low: 0.001,
            high: 1.0,
            scale: Scale::Log,
            default: None,
        });

        let mut sampler = GridSampler::new(3);
        let t0 = sampler.sample(&space, 0).unwrap().unwrap();
        let t1 = sampler.sample(&space, 1).unwrap().unwrap();
        let t2 = sampler.sample(&space, 2).unwrap().unwrap();

        let v0 = t0["lr"].as_f64().unwrap();
        let v1 = t1["lr"].as_f64().unwrap();
        let v2 = t2["lr"].as_f64().unwrap();

        // Log scale: gap between v0-v1 should be smaller than v1-v2
        assert!(v0 < v1 && v1 < v2);
        assert!((v1 - v0) < (v2 - v1));
    }

    #[test]
    fn grid_sampler_int_dimension() {
        let mut space = SearchSpace::new();
        space.add(SearchDimension::Int {
            name: "n".into(),
            low: 1,
            high: 5,
            scale: Scale::Linear,
        });

        let mut sampler = GridSampler::new(5);
        let mut values = Vec::new();
        for i in 0.. {
            match sampler.sample(&space, i).unwrap() {
                Some(p) => values.push(p["n"].as_i64().unwrap()),
                None => break,
            }
        }
        assert_eq!(values, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn grid_empty_space() {
        let mut sampler = GridSampler::new(3);
        let space = SearchSpace::new();
        let result = sampler.sample(&space, 0).unwrap();
        assert!(result.is_some()); // one combo with empty params
        assert!(result.unwrap().is_empty());
        assert!(sampler.sample(&space, 1).unwrap().is_none());
    }

    // ── Random tests ──

    #[test]
    fn random_sampler_generates_n_trials() {
        let mut sampler = RandomSampler::new(10, Some(42));
        let space = sample_space();

        let mut trials = Vec::new();
        for i in 0..20 {
            match sampler.sample(&space, i).unwrap() {
                Some(params) => trials.push(params),
                None => break,
            }
        }

        assert_eq!(trials.len(), 10);
    }

    #[test]
    fn random_sampler_respects_bounds() {
        let mut space = SearchSpace::new();
        space.add(SearchDimension::Float {
            name: "x".into(),
            low: 0.0,
            high: 1.0,
            scale: Scale::Linear,
            default: None,
        });
        space.add(SearchDimension::Int {
            name: "n".into(),
            low: 5,
            high: 10,
            scale: Scale::Linear,
        });

        let mut sampler = RandomSampler::new(100, Some(123));

        for i in 0..100 {
            let params = sampler.sample(&space, i).unwrap().unwrap();
            let x = params["x"].as_f64().unwrap();
            let n = params["n"].as_i64().unwrap();
            assert!((0.0..=1.0).contains(&x), "x={x} out of bounds");
            assert!((5..=10).contains(&n), "n={n} out of bounds");
        }
    }

    #[test]
    fn random_sampler_deterministic_with_seed() {
        let space = sample_space();

        let mut s1 = RandomSampler::new(5, Some(42));
        let mut s2 = RandomSampler::new(5, Some(42));

        for i in 0..5 {
            let p1 = s1.sample(&space, i).unwrap().unwrap();
            let p2 = s2.sample(&space, i).unwrap().unwrap();
            assert_eq!(p1, p2);
        }
    }

    #[test]
    fn random_sampler_different_seeds_differ() {
        let space = sample_space();

        let mut s1 = RandomSampler::new(5, Some(42));
        let mut s2 = RandomSampler::new(5, Some(99));

        let p1 = s1.sample(&space, 0).unwrap().unwrap();
        let p2 = s2.sample(&space, 0).unwrap().unwrap();
        // Very unlikely to be equal with different seeds
        assert_ne!(p1["lr"], p2["lr"]);
    }

    // ── Linspace tests ──

    #[test]
    fn linspace_linear() {
        let vals = linspace(0.0, 10.0, 5, Scale::Linear);
        assert_eq!(vals, vec![0.0, 2.5, 5.0, 7.5, 10.0]);
    }

    #[test]
    fn linspace_single_point() {
        let vals = linspace(0.0, 10.0, 1, Scale::Linear);
        assert_eq!(vals, vec![5.0]);
    }

    #[test]
    fn linspace_log_denser_at_low_end() {
        let vals = linspace(0.001, 1.0, 5, Scale::Log);
        // Log scale: gaps should increase
        let gaps: Vec<f64> = vals.windows(2).map(|w| w[1] - w[0]).collect();
        for i in 1..gaps.len() {
            assert!(gaps[i] > gaps[i - 1], "gap[{i}] should be > gap[{}]", i - 1);
        }
    }
}
