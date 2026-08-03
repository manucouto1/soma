//! Training strategies for distributed execution.
//!
//! A [`TrainingStrategy`] is a graph-level attribute that controls HOW the
//! Scheduler distributes work across workers and HOW workers coordinate
//! during training (gradient aggregation, state sync, communication).
//!
//! Subgraphs inherit the parent's strategy unless overridden.

use crate::error::Result;
use crate::filter::RemoteTarget;
use crate::graph::NodeId;
use crate::value::Value;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Training strategy — graph-level attribute, inherited by subgraphs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum TrainingStrategy {
    /// All nodes execute locally (default).
    #[default]
    Local,

    /// Replicate the entire graph on N workers, each sees a data shard.
    /// Gradients are aggregated after each step.
    DataParallel {
        num_replicas: usize,
        aggregation: GradientAggregation,
    },

    /// Arbitrary model partitioning: each Partition maps a set of
    /// node IDs to a worker target. Any topology is supported.
    ModelParallel {
        partitions: Vec<Partition>,
        communication: CommunicationProtocol,
    },

    /// Federated learning: data stays on workers, only model updates
    /// are shared. The coordinator aggregates after each round.
    Federated {
        num_clients: usize,
        rounds: usize,
        aggregation: FederatedAggregation,
        client_selection: ClientSelection,
    },

    /// Population-Based Training: evolutionary hyperparameter optimization.
    /// Each generation trains a population, evaluates, then evolves.
    PopulationBased {
        population_size: usize,
        generations: usize,
        exploit: ExploitStrategy,
        explore: ExploreStrategy,
    },

    /// User-defined strategy with a registered coordinator.
    Custom {
        coordinator: String,
        config: serde_json::Value,
    },
}

/// How gradients are aggregated across workers in data-parallel training.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "method")]
#[non_exhaustive]
pub enum GradientAggregation {
    /// All workers exchange gradients (ring or tree reduction).
    AllReduce,
    /// A central parameter server collects and distributes updates.
    ParameterServer,
    /// Decentralized gossip-based aggregation.
    Decentralized { topology: String },
}

/// A partition maps a set of node IDs to a worker target.
///
/// Used in `ModelParallel` to define which nodes run on which worker.
/// The user has full control over the partitioning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Partition {
    pub node_ids: Vec<NodeId>,
    pub target: RemoteTarget,
}

/// How model-parallel partitions communicate activations and gradients.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "protocol")]
#[non_exhaustive]
pub enum CommunicationProtocol {
    /// Intermediate values flow via DataStore (S3, shared disk).
    DataStore,
    /// Direct point-to-point streaming between workers.
    Direct,
    /// Pipeline parallelism with micro-batching for overlap.
    Pipeline { micro_batch_size: usize },
}

/// Aggregation method for federated learning rounds.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "method")]
#[non_exhaustive]
pub enum FederatedAggregation {
    /// Federated Averaging: weighted mean of client updates.
    FedAvg,
    /// FedProx: adds proximal term to prevent client drift.
    FedProx { mu: f64 },
    /// FedYogi: adaptive federated optimization.
    FedYogi { beta1: f64, beta2: f64, tau: f64 },
}

/// How clients are selected per federated round.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "method")]
#[non_exhaustive]
pub enum ClientSelection {
    /// All available clients participate.
    All,
    /// Random subset of clients.
    Random { fraction: f64 },
    /// Only clients matching specific tags.
    ByCapability { required_tags: Vec<String> },
}

/// PBT exploit strategy: how underperformers learn from top performers.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "method")]
#[non_exhaustive]
pub enum ExploitStrategy {
    /// Bottom fraction copies weights+hyperparams from top fraction.
    Truncation { fraction: f64 },
    /// Each member is compared to a random other; loser copies winner.
    Binary { threshold: f64 },
}

/// PBT explore strategy: how hyperparameters are mutated after exploit.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "method")]
#[non_exhaustive]
pub enum ExploreStrategy {
    /// Multiply each hyperparameter by a random factor in [1-factor, 1+factor].
    Perturbation { factor: f64 },
    /// Resample hyperparameters from the original search space.
    Resample,
}

// ── Traits: execution contracts for strategies and aggregation ──

/// Context provided to strategy executors.
/// Abstracts worker communication — the strategy doesn't know about WS/HTTP.
pub trait StrategyContext {
    /// Number of available workers.
    fn num_workers(&self) -> usize;

    /// Execute a plan on a specific worker (by index). Returns trained states.
    fn execute_on_worker(
        &self,
        worker_idx: usize,
        plan: &serde_json::Value,
        input: &Value,
        y: Option<&Value>,
    ) -> Result<HashMap<String, Value>>;

    /// Get trained states from a worker.
    fn get_state(&self, worker_idx: usize, node_ids: &[String]) -> Result<HashMap<String, Value>>;

    /// Set states on a worker (e.g. after aggregation).
    fn set_state(&self, worker_idx: usize, states: &HashMap<String, Value>) -> Result<()>;

    /// Get gradients from a worker.
    fn get_gradients(
        &self,
        worker_idx: usize,
        node_ids: &[String],
    ) -> Result<HashMap<String, Value>>;

    /// Apply gradients on a worker.
    fn apply_gradients(&self, worker_idx: usize, gradients: &HashMap<String, Value>) -> Result<()>;
}

/// Contract for training strategy execution.
/// Every TrainingStrategy variant implements this — including Local.
pub trait StrategyExecutor {
    /// Train the model according to this strategy.
    fn fit(
        &self,
        ctx: &dyn StrategyContext,
        input: &Value,
        y: Option<&Value>,
        node_ids: &[String],
    ) -> Result<HashMap<String, Value>>;
}

/// Contract for gradient aggregation across workers.
pub trait GradientAggregator {
    fn aggregate(&self, gradients: &[HashMap<String, Value>]) -> Result<HashMap<String, Value>>;
}

/// Contract for federated state aggregation.
pub trait StateAggregator {
    fn aggregate(&self, states: &[HashMap<String, Value>]) -> Result<HashMap<String, Value>>;
}

// ── Trait implementations ──

impl StrategyExecutor for TrainingStrategy {
    fn fit(
        &self,
        ctx: &dyn StrategyContext,
        input: &Value,
        y: Option<&Value>,
        node_ids: &[String],
    ) -> Result<HashMap<String, Value>> {
        match self {
            TrainingStrategy::Local => {
                // Single worker, full dataset
                ctx.execute_on_worker(0, &serde_json::json!({}), input, y)
            }

            TrainingStrategy::DataParallel {
                num_replicas,
                aggregation,
            } => {
                let n = (*num_replicas).min(ctx.num_workers());
                let shards = shard_value(input, n);

                // Fit on each worker with its shard
                for (i, shard) in shards.iter().enumerate() {
                    ctx.execute_on_worker(i, &serde_json::json!({}), shard, y)?;
                }

                // Collect and aggregate gradients
                let mut all_grads = Vec::new();
                for i in 0..n {
                    all_grads.push(ctx.get_gradients(i, node_ids)?);
                }
                let averaged = aggregation.aggregate(&all_grads)?;

                // Apply to all workers
                for i in 0..n {
                    ctx.apply_gradients(i, &averaged)?;
                }

                // Return states from first worker
                ctx.get_state(0, node_ids)
            }

            TrainingStrategy::Federated {
                num_clients,
                rounds,
                aggregation,
                ..
            } => {
                let n = (*num_clients).min(ctx.num_workers());
                let shards = shard_value(input, n);

                for _round in 0..*rounds {
                    // Each client trains on its shard
                    for (i, shard) in shards.iter().enumerate().take(n) {
                        ctx.execute_on_worker(i, &serde_json::json!({}), shard, y)?;
                    }

                    // Collect and aggregate states
                    let mut all_states = Vec::new();
                    for i in 0..n {
                        all_states.push(ctx.get_state(i, node_ids)?);
                    }
                    let aggregated = aggregation.aggregate(&all_states)?;

                    // Distribute back
                    for i in 0..n {
                        ctx.set_state(i, &aggregated)?;
                    }
                }

                ctx.get_state(0, node_ids)
            }

            TrainingStrategy::ModelParallel { .. } => {
                // TODO: forward/backward across partitions
                Err(crate::error::SomaError::Other(
                    "ModelParallel strategy execution not yet implemented".into(),
                ))
            }

            TrainingStrategy::PopulationBased { .. } => {
                // TODO: PBT cycle
                Err(crate::error::SomaError::Other(
                    "PopulationBased strategy execution not yet implemented".into(),
                ))
            }

            TrainingStrategy::Custom { .. } => Err(crate::error::SomaError::Other(
                "Custom strategy requires a user-provided coordinator".into(),
            )),
        }
    }
}

// Both aggregators used to answer with `first()` — one worker's gradients
// presented as the average of all of them. That is not an unfinished
// feature, it is a wrong number that trains a model and reports success.
// Until the tensor arithmetic exists, refusing is the only honest answer,
// which is what the unimplemented strategy arms above already do.

impl GradientAggregator for GradientAggregation {
    fn aggregate(&self, gradients: &[HashMap<String, Value>]) -> Result<HashMap<String, Value>> {
        // A single worker needs no aggregation: its gradients *are* the
        // result, so this case is exact rather than a stand-in.
        if gradients.len() == 1 {
            return Ok(gradients[0].clone());
        }
        Err(crate::error::SomaError::Other(format!(
            "{self:?} gradient aggregation over {} workers is not implemented yet; \
             it would need element-wise tensor averaging",
            gradients.len()
        )))
    }
}

impl StateAggregator for FederatedAggregation {
    fn aggregate(&self, states: &[HashMap<String, Value>]) -> Result<HashMap<String, Value>> {
        if states.len() == 1 {
            return Ok(states[0].clone());
        }
        Err(crate::error::SomaError::Other(format!(
            "{self:?} state aggregation over {} clients is not implemented yet; \
             it would need element-wise tensor averaging",
            states.len()
        )))
    }
}

/// Split a Value::Tensor along the first dimension into N shards.
fn shard_value(value: &Value, n: usize) -> Vec<Value> {
    match value {
        Value::Tensor { values, shape } if !shape.is_empty() && shape[0] >= n => {
            let rows = shape[0];
            let row_size: usize = shape[1..].iter().product::<usize>().max(1);
            let shard_rows = rows / n;
            let mut shards = Vec::new();
            for i in 0..n {
                let start = i * shard_rows;
                let end = if i == n - 1 { rows } else { start + shard_rows };
                let flat_start = start * row_size;
                let flat_end = end * row_size;
                let shard_vals = values[flat_start..flat_end].to_vec();
                let mut shard_shape = shape.clone();
                shard_shape[0] = end - start;
                shards.push(Value::tensor(shard_vals, shard_shape));
            }
            shards
        }
        _ => (0..n).map(|_| value.clone()).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_local() {
        assert!(matches!(
            TrainingStrategy::default(),
            TrainingStrategy::Local
        ));
    }

    #[test]
    fn serde_roundtrip_data_parallel() {
        let strategy = TrainingStrategy::DataParallel {
            num_replicas: 4,
            aggregation: GradientAggregation::AllReduce,
        };
        let json = serde_json::to_string(&strategy).unwrap();
        let parsed: TrainingStrategy = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            parsed,
            TrainingStrategy::DataParallel {
                num_replicas: 4,
                ..
            }
        ));
    }

    #[test]
    fn serde_roundtrip_model_parallel() {
        let strategy = TrainingStrategy::ModelParallel {
            partitions: vec![
                Partition {
                    node_ids: vec!["embed".into(), "backbone".into()],
                    target: RemoteTarget::Tag("gpu-0".into()),
                },
                Partition {
                    node_ids: vec!["head_a".into()],
                    target: RemoteTarget::Tag("gpu-1".into()),
                },
            ],
            communication: CommunicationProtocol::Pipeline {
                micro_batch_size: 4,
            },
        };
        let json = serde_json::to_string(&strategy).unwrap();
        let parsed: TrainingStrategy = serde_json::from_str(&json).unwrap();
        assert!(matches!(parsed, TrainingStrategy::ModelParallel { .. }));
    }

    #[test]
    fn serde_roundtrip_federated() {
        let strategy = TrainingStrategy::Federated {
            num_clients: 10,
            rounds: 50,
            aggregation: FederatedAggregation::FedProx { mu: 0.01 },
            client_selection: ClientSelection::Random { fraction: 0.3 },
        };
        let json = serde_json::to_string(&strategy).unwrap();
        let parsed: TrainingStrategy = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            parsed,
            TrainingStrategy::Federated {
                num_clients: 10,
                rounds: 50,
                ..
            }
        ));
    }

    /// Aggregating several workers is not implemented. It must say so —
    /// answering with the first worker's gradients looks like a trained
    /// model and is arithmetically wrong.
    #[test]
    fn multi_worker_aggregation_refuses_instead_of_guessing() {
        let grads = |v: f64| HashMap::from([("w".to_string(), Value::tensor(vec![v], vec![1]))]);

        let err = GradientAggregation::AllReduce
            .aggregate(&[grads(1.0), grads(3.0)])
            .expect_err("aggregating two workers must not silently succeed");
        assert!(err.to_string().contains("not implemented"), "{err}");

        let err = FederatedAggregation::FedAvg
            .aggregate(&[grads(1.0), grads(3.0)])
            .expect_err("aggregating two clients must not silently succeed");
        assert!(err.to_string().contains("not implemented"), "{err}");
    }

    /// One worker is the exact case, not a stand-in: there is nothing to
    /// average, so it stays supported.
    #[test]
    fn single_worker_aggregation_is_the_identity() {
        let only = HashMap::from([("w".to_string(), Value::tensor(vec![2.0], vec![1]))]);
        let out = GradientAggregation::AllReduce
            .aggregate(std::slice::from_ref(&only))
            .unwrap();
        assert_eq!(out, only);
    }

    #[test]
    fn serde_roundtrip_pbt() {
        let strategy = TrainingStrategy::PopulationBased {
            population_size: 20,
            generations: 50,
            exploit: ExploitStrategy::Truncation { fraction: 0.2 },
            explore: ExploreStrategy::Perturbation { factor: 0.2 },
        };
        let json = serde_json::to_string(&strategy).unwrap();
        let parsed: TrainingStrategy = serde_json::from_str(&json).unwrap();
        assert!(matches!(
            parsed,
            TrainingStrategy::PopulationBased {
                population_size: 20,
                ..
            }
        ));
    }
}
