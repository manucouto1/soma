//! Population-Based Training runner.
//!
//! PBT is a cyclic evolutionary process where each generation:
//! 1. **Train**: each population member trains for N steps
//! 2. **Evaluate**: each member is evaluated to produce a fitness score
//! 3. **Exploit/Explore**: underperformers copy top performers, then mutate hyperparameters
//!
//! Each generation's training phase uses the existing sampler infrastructure.

use crate::optimizer::sampler::{hash_u64, pseudo_random};
use crate::tracking::event_bus::EventBus;
use somatize_core::data::value::Value;
use somatize_core::distributed::{ExploitStrategy, ExploreStrategy};
use somatize_core::error::Result;
use somatize_core::optimizer::search::{SearchDimension, SearchSpace};
use somatize_core::tracking::event::Event;
use std::collections::HashMap;
use std::sync::Arc;

/// Configuration for a PBT run.
#[derive(Debug, Clone)]
pub struct PbtConfig {
    /// Number of members evolved together.
    pub population_size: usize,
    /// Train → evaluate → exploit/explore cycles to run.
    pub generations: usize,
    /// How underperformers copy top performers (truncation, binary tournament).
    pub exploit: ExploitStrategy,
    /// How copied hyperparameters are mutated afterwards.
    pub explore: ExploreStrategy,
    /// Dimensions the initial population and `Resample` mutations draw from.
    pub search_space: SearchSpace,
    /// Advisory length of one generation's training phase. The runner calls
    /// [`PbtExecutor::train`] once per member per generation; what a "step"
    /// means is the executor's to interpret.
    pub train_steps_per_generation: usize,
}

/// A single member of the population.
#[derive(Debug, Clone)]
pub struct PopulationMember {
    /// Stable member identifier (`member_<i>`).
    pub id: String,
    /// Current hyperparameters — copied on exploit, mutated on explore.
    pub params: HashMap<String, serde_json::Value>,
    /// Trained state, updated after each generation's training phase and
    /// copied along with `params` when a top performer is exploited.
    pub state: Value,
    /// Latest evaluation score, higher is better; `None` before the first
    /// evaluation (a failed evaluation records `f64::NEG_INFINITY`).
    pub fitness: Option<f64>,
}

/// Trait for the training + evaluation callback.
pub trait PbtExecutor: Send + Sync {
    /// Train a member for one generation. Returns updated state.
    fn train(&self, member: &PopulationMember) -> Result<Value>;
    /// Evaluate a member. Returns fitness score (higher = better).
    fn evaluate(&self, member: &PopulationMember) -> Result<f64>;
}

/// Function-based PBT executor for convenience.
pub struct FnPbtExecutor<T, E> {
    /// Closure backing [`PbtExecutor::train`].
    pub train_fn: T,
    /// Closure backing [`PbtExecutor::evaluate`].
    pub eval_fn: E,
}

impl<T, E> PbtExecutor for FnPbtExecutor<T, E>
where
    T: Fn(&PopulationMember) -> Result<Value> + Send + Sync,
    E: Fn(&PopulationMember) -> Result<f64> + Send + Sync,
{
    fn train(&self, member: &PopulationMember) -> Result<Value> {
        (self.train_fn)(member)
    }
    fn evaluate(&self, member: &PopulationMember) -> Result<f64> {
        (self.eval_fn)(member)
    }
}

/// Orchestrates the PBT evolutionary cycle.
pub struct PbtRunner {
    event_bus: Arc<EventBus>,
}

impl PbtRunner {
    /// A runner emitting generation and exploit events on `event_bus`.
    pub fn new(event_bus: Arc<EventBus>) -> Self {
        Self { event_bus }
    }

    /// Run the full PBT evolutionary process.
    ///
    /// Returns the final population sorted by fitness (best first).
    pub fn run(
        &self,
        config: &PbtConfig,
        executor: &dyn PbtExecutor,
    ) -> Result<Vec<PopulationMember>> {
        let study_id = somatize_core::util::timestamp_id("pbt");
        let mut rng_state: u64 = 42;

        // Initialize population with random params
        let mut population = self.initialize_population(config, &mut rng_state);

        for generation in 0..config.generations {
            self.event_bus.emit(Event::GenerationStarted {
                study_id: study_id.clone(),
                generation,
                population_size: population.len(),
            });

            // Stage 1: Train
            for member in &mut population {
                match executor.train(member) {
                    Ok(new_state) => member.state = new_state,
                    Err(e) => {
                        tracing::warn!("PBT train failed for {}: {e}", member.id);
                    }
                }
            }

            // Stage 2: Evaluate
            //
            // One member that cannot be evaluated is scored at negative
            // infinity and exploited away — a flaky run should not cost
            // the population. Every member failing is a different thing:
            // the callback is wrong, and a generation where nothing could
            // be scored has no signal to evolve on. Reporting success
            // there hands back a population of -inf that looks ranked.
            let mut first_failure: Option<String> = None;
            let mut failures = 0usize;
            for member in &mut population {
                match executor.evaluate(member) {
                    Ok(fitness) => member.fitness = Some(fitness),
                    Err(e) => {
                        tracing::warn!("PBT evaluate failed for {}: {e}", member.id);
                        first_failure.get_or_insert_with(|| e.to_string());
                        failures += 1;
                        member.fitness = Some(f64::NEG_INFINITY);
                    }
                }
            }
            if failures == population.len() {
                return Err(somatize_core::error::SomaError::Other(format!(
                    "PBT generation {generation}: no member could be evaluated, \
                     so there is no fitness to evolve on. The first failure was: \
                     {}",
                    first_failure.unwrap_or_else(|| "unreported".into())
                )));
            }

            // Sort by fitness (descending)
            population.sort_by(|a, b| {
                b.fitness
                    .unwrap_or(f64::NEG_INFINITY)
                    .partial_cmp(&a.fitness.unwrap_or(f64::NEG_INFINITY))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let best_fitness = population[0].fitness.unwrap_or(0.0);
            let mean_fitness =
                population.iter().filter_map(|m| m.fitness).sum::<f64>() / population.len() as f64;

            // Stage 3: Exploit/Explore
            self.evolve(
                &mut population,
                config,
                generation,
                &study_id,
                &mut rng_state,
            );

            self.event_bus.emit(Event::GenerationCompleted {
                study_id: study_id.clone(),
                generation,
                best_fitness,
                mean_fitness,
            });
        }

        // Final sort
        population.sort_by(|a, b| {
            b.fitness
                .unwrap_or(f64::NEG_INFINITY)
                .partial_cmp(&a.fitness.unwrap_or(f64::NEG_INFINITY))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(population)
    }

    fn initialize_population(
        &self,
        config: &PbtConfig,
        rng_state: &mut u64,
    ) -> Vec<PopulationMember> {
        let mut population = Vec::with_capacity(config.population_size);

        for i in 0..config.population_size {
            let params = sample_params(&config.search_space, rng_state);
            population.push(PopulationMember {
                id: format!("member_{i}"),
                params,
                state: Value::Empty,
                fitness: None,
            });
        }

        population
    }

    fn evolve(
        &self,
        population: &mut [PopulationMember],
        config: &PbtConfig,
        generation: usize,
        study_id: &str,
        rng_state: &mut u64,
    ) {
        let n = population.len();
        if n < 2 {
            return;
        }

        let cutoff = match &config.exploit {
            ExploitStrategy::Truncation { fraction } => {
                let c = ((n as f64) * fraction).ceil() as usize;
                c.max(1).min(n / 2)
            }
            ExploitStrategy::Binary => n / 2,
            _ => n / 2,
        };

        // Exploit: bottom performers copy from top
        match &config.exploit {
            ExploitStrategy::Truncation { .. } => {
                for i in 0..cutoff {
                    let bottom_idx = n - 1 - i;
                    let top_idx = i;
                    if bottom_idx <= top_idx {
                        break;
                    }

                    let donor_id = population[top_idx].id.clone();
                    let replaced_id = population[bottom_idx].id.clone();

                    population[bottom_idx].params = population[top_idx].params.clone();
                    population[bottom_idx].state = population[top_idx].state.clone();

                    self.event_bus.emit(Event::MemberExploited {
                        study_id: study_id.to_string(),
                        generation,
                        replaced_id,
                        donor_id,
                    });
                }
            }
            ExploitStrategy::Binary => {
                for i in cutoff..n {
                    *rng_state = hash_u64(*rng_state, i as u64, generation as u64);
                    let opponent = (*rng_state as usize) % cutoff;
                    let my_fitness = population[i].fitness.unwrap_or(f64::NEG_INFINITY);
                    let opp_fitness = population[opponent].fitness.unwrap_or(f64::NEG_INFINITY);
                    if my_fitness < opp_fitness {
                        let donor_id = population[opponent].id.clone();
                        let replaced_id = population[i].id.clone();
                        population[i].params = population[opponent].params.clone();
                        population[i].state = population[opponent].state.clone();

                        self.event_bus.emit(Event::MemberExploited {
                            study_id: study_id.to_string(),
                            generation,
                            replaced_id,
                            donor_id,
                        });
                    }
                }
            }
            _ => {}
        }

        // Explore: mutate exploited members' hyperparameters
        match &config.explore {
            ExploreStrategy::Perturbation { factor } => {
                for member in population[(n - cutoff)..].iter_mut() {
                    perturb_params(&mut member.params, *factor, rng_state);
                }
            }
            ExploreStrategy::Resample => {
                for member in population[(n - cutoff)..].iter_mut() {
                    member.params = sample_params(&config.search_space, rng_state);
                }
            }
            _ => {}
        }
    }
}

/// Sample random parameters from a search space.
fn sample_params(space: &SearchSpace, rng_state: &mut u64) -> HashMap<String, serde_json::Value> {
    let mut params = HashMap::new();

    for (dim_idx, dim) in space.dimensions.iter().enumerate() {
        *rng_state = hash_u64(*rng_state, dim_idx as u64, 0);
        let value = match dim {
            SearchDimension::Float { low, high, .. } => {
                let t = pseudo_random(*rng_state);
                let v = low + t * (high - low);
                serde_json::Value::from(v)
            }
            SearchDimension::Int { low, high, .. } => {
                let t = pseudo_random(*rng_state);
                let range = (*high - *low + 1) as f64;
                let v = *low + (t * range) as i64;
                serde_json::Value::from(v.min(*high))
            }
            SearchDimension::Categorical { choices, .. } => {
                let t = pseudo_random(*rng_state);
                let idx = (t * choices.len() as f64) as usize;
                let idx = idx.min(choices.len() - 1);
                choices[idx].clone()
            }
            _ => continue,
        };
        params.insert(dim.name().to_string(), value);
    }

    params
}

/// Perturb numeric parameters by a random factor in [1-factor, 1+factor].
fn perturb_params(
    params: &mut HashMap<String, serde_json::Value>,
    factor: f64,
    rng_state: &mut u64,
) {
    for (i, value) in params.values_mut().enumerate() {
        if let Some(v) = value.as_f64() {
            *rng_state = hash_u64(*rng_state, i as u64, 999);
            let t = pseudo_random(*rng_state);
            let perturbation = 1.0 + (t * 2.0 - 1.0) * factor;
            *value = serde_json::Value::from(v * perturbation);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use somatize_core::optimizer::search::Scale;

    fn test_config() -> PbtConfig {
        let mut space = SearchSpace::new();
        space.add(SearchDimension::Float {
            name: "lr".into(),
            low: 0.001,
            high: 1.0,
            scale: Scale::Log,
            default: None,
        });

        PbtConfig {
            population_size: 6,
            generations: 3,
            exploit: ExploitStrategy::Truncation { fraction: 0.33 },
            explore: ExploreStrategy::Perturbation { factor: 0.2 },
            search_space: space,
            train_steps_per_generation: 10,
        }
    }

    #[test]
    fn pbt_basic_run() {
        let bus = Arc::new(EventBus::new(256));
        let runner = PbtRunner::new(bus);

        let executor = FnPbtExecutor {
            train_fn: |member: &PopulationMember| {
                let lr = member
                    .params
                    .get("lr")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.01);
                Ok(Value::json(serde_json::json!({"lr": lr})))
            },
            eval_fn: |member: &PopulationMember| {
                let lr = member
                    .params
                    .get("lr")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.01);
                Ok(-(lr - 0.1).abs())
            },
        };

        let config = test_config();
        let result = runner.run(&config, &executor).unwrap();

        assert_eq!(result.len(), 6);
        assert!(result.iter().all(|m| m.fitness.is_some()));
        // Sorted by fitness descending
        assert!(result[0].fitness.unwrap() >= result.last().unwrap().fitness.unwrap());
    }

    #[test]
    fn pbt_emits_events() {
        let bus = Arc::new(EventBus::new(256));
        let mut rx = bus.subscribe();
        let runner = PbtRunner::new(bus);

        let executor = FnPbtExecutor {
            train_fn: |_: &PopulationMember| Ok(Value::Empty),
            eval_fn: |_: &PopulationMember| Ok(1.0),
        };

        let config = test_config();
        runner.run(&config, &executor).unwrap();

        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }

        let gen_started = events
            .iter()
            .filter(|e| matches!(e, Event::GenerationStarted { .. }))
            .count();
        let gen_completed = events
            .iter()
            .filter(|e| matches!(e, Event::GenerationCompleted { .. }))
            .count();
        assert_eq!(gen_started, 3);
        assert_eq!(gen_completed, 3);
    }

    #[test]
    fn pbt_population_evolves() {
        let bus = Arc::new(EventBus::new(64));
        let runner = PbtRunner::new(bus);

        let executor = FnPbtExecutor {
            train_fn: |_: &PopulationMember| Ok(Value::Empty),
            eval_fn: |member: &PopulationMember| {
                let lr = member
                    .params
                    .get("lr")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.5);
                // Fitness = -|lr - 0.1| (best at lr=0.1)
                Ok(-(lr - 0.1).abs())
            },
        };

        let mut config = test_config();
        config.generations = 10;
        let result = runner.run(&config, &executor).unwrap();

        assert_eq!(result.len(), 6);
        // All should have fitness
        assert!(result.iter().all(|m| m.fitness.is_some()));
    }
}
