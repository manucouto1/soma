//! Running a [`TrainingStrategy`].
//!
//! The strategy *types* are contracts — a strategy is a graph-level
//! attribute, part of what a graph is — so they live in `soma-core`
//! beside the graph. Running one is not: it shards inputs, calls workers
//! in a round loop, aggregates gradients and redistributes states. That
//! is execution, and execution lives here.
//!
//! See the "soma-core holds contracts, not execution" entry in the design
//! decisions.

use somatize_core::error::{Result, SomaError};
use somatize_core::strategy::{FederatedAggregation, GradientAggregation, TrainingStrategy};
use somatize_core::value::Value;
use std::collections::HashMap;

// ── The execution contracts ──
//
// These describe how a strategy is *run*, so they belong beside the
// running of it. `soma-core` keeps the strategy types themselves, which
// are graph attributes and therefore contracts.

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
                Err(SomaError::Other(
                    "ModelParallel strategy execution not yet implemented".into(),
                ))
            }

            TrainingStrategy::PopulationBased { .. } => {
                // TODO: PBT cycle
                Err(SomaError::Other(
                    "PopulationBased strategy execution not yet implemented".into(),
                ))
            }

            TrainingStrategy::Custom { .. } => Err(SomaError::Other(
                "Custom strategy requires a user-provided coordinator".into(),
            )),

            // `TrainingStrategy` is `#[non_exhaustive]` and now lives in
            // another crate, so this arm cannot be deleted. It refuses
            // rather than falling back to something plausible: running a
            // strategy this build does not understand as if it were
            // `Local` would train on one worker and report success.
            other => Err(SomaError::Other(format!(
                "this runtime does not know how to run {other:?}. It was \
                 probably described by a newer version"
            ))),
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
        Err(SomaError::Other(format!(
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
        Err(SomaError::Other(format!(
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
}
