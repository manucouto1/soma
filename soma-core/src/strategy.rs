//! Training strategies for distributed execution.
//!
//! A [`TrainingStrategy`] is a graph-level attribute that controls HOW the
//! Scheduler distributes work across workers and HOW workers coordinate
//! during training (gradient aggregation, state sync, communication).
//!
//! Only the description lives here. *Running* one — sharding inputs,
//! calling workers in a round loop, aggregating gradients — is execution,
//! and is in `somatize_runtime::strategy` along with the traits that
//! describe it.
//!
//! Subgraphs inherit the parent's strategy unless overridden.

use crate::filter::RemoteTarget;
use crate::graph::NodeId;
use serde::{Deserialize, Serialize};

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
        /// Number of graph replicas — one worker and one data shard each.
        num_replicas: usize,
        /// How the replicas' gradients are combined after each step.
        aggregation: GradientAggregation,
    },

    /// Arbitrary model partitioning: each Partition maps a set of
    /// node IDs to a worker target. Any topology is supported.
    ModelParallel {
        /// Which nodes run where. Nodes not covered by any
        /// [`Partition`] stay on the default (local) target.
        partitions: Vec<Partition>,
        /// How activations and gradients move between partitions.
        communication: CommunicationProtocol,
    },

    /// Federated learning: data stays on workers, only model updates
    /// are shared. The coordinator aggregates after each round.
    Federated {
        /// Total number of participating clients (the pool
        /// [`ClientSelection`] draws from each round).
        num_clients: usize,
        /// Number of train→aggregate rounds to run.
        rounds: usize,
        /// How client updates are combined into the global model.
        aggregation: FederatedAggregation,
        /// Which clients participate in each round.
        client_selection: ClientSelection,
    },

    /// Population-Based Training: evolutionary hyperparameter optimization.
    /// Each generation trains a population, evaluates, then evolves.
    PopulationBased {
        /// Number of concurrently trained population members.
        population_size: usize,
        /// Number of train→evaluate→exploit/explore cycles.
        generations: usize,
        /// How underperformers copy from top performers.
        exploit: ExploitStrategy,
        /// How copied hyperparameters are mutated afterwards.
        explore: ExploreStrategy,
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
    Decentralized {
        /// Name of the gossip topology (e.g. `"ring"`), interpreted by
        /// the executing backend.
        topology: String,
    },
}

/// A partition maps a set of node IDs to a worker target.
///
/// Used in `ModelParallel` to define which nodes run on which worker.
/// The user has full control over the partitioning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Partition {
    /// The nodes this partition claims. Need not be contiguous in the
    /// graph — any subset can be pinned to a target.
    pub node_ids: Vec<NodeId>,
    /// Worker the nodes run on, by id or by capability tag.
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
    Pipeline {
        /// Rows per micro-batch: smaller batches overlap partitions
        /// more at the cost of per-message overhead.
        micro_batch_size: usize,
    },
}

/// Aggregation method for federated learning rounds.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "method")]
#[non_exhaustive]
pub enum FederatedAggregation {
    /// Federated Averaging: weighted mean of client updates.
    FedAvg,
    /// FedProx: adds proximal term to prevent client drift.
    FedProx {
        /// Proximal term weight: larger values pull clients harder
        /// toward the global model.
        mu: f64,
    },
    /// FedYogi: adaptive federated optimization.
    FedYogi {
        /// First-moment (momentum) decay rate.
        beta1: f64,
        /// Second-moment decay rate.
        beta2: f64,
        /// Adaptivity floor: bounds how large the effective per-round
        /// step can grow.
        tau: f64,
    },
}

/// How clients are selected per federated round.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "method")]
#[non_exhaustive]
pub enum ClientSelection {
    /// All available clients participate.
    All,
    /// Random subset of clients.
    Random {
        /// Fraction of `num_clients` sampled each round, in `(0, 1]`.
        fraction: f64,
    },
    /// Only clients matching specific tags.
    ByCapability {
        /// Capability tags a client must carry to be eligible.
        required_tags: Vec<String>,
    },
}

/// PBT exploit strategy: how underperformers learn from top performers.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "method")]
#[non_exhaustive]
pub enum ExploitStrategy {
    /// Bottom fraction copies weights+hyperparams from top fraction.
    Truncation {
        /// Fraction of the population treated as top/bottom (the
        /// `PbtRunner` clamps it to at most half the population).
        fraction: f64,
    },
    /// Each member is compared to a random other; loser copies winner
    /// on any strict fitness loss.
    Binary,
}

/// PBT explore strategy: how hyperparameters are mutated after exploit.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "method")]
#[non_exhaustive]
pub enum ExploreStrategy {
    /// Multiply each hyperparameter by a random factor in [1-factor, 1+factor].
    Perturbation {
        /// Half-width of the multiplicative jitter (0.2 → factors in
        /// [0.8, 1.2]).
        factor: f64,
    },
    /// Resample hyperparameters from the original search space.
    Resample,
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
