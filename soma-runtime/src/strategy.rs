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

use crate::executor::RunMode;
use crate::node_catalog::NodeCatalog;
use crate::runner::Transport;
use somatize_compiler::ExecutionPlan;
use somatize_core::error::{Result, SomaError};
use somatize_core::strategy::{FederatedAggregation, GradientAggregation, TrainingStrategy};
use somatize_core::value::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

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
    /// Combine per-worker gradients (keyed by node id) into the one set
    /// every worker then applies.
    fn aggregate(&self, gradients: &[HashMap<String, Value>]) -> Result<HashMap<String, Value>>;
}

/// Contract for federated state aggregation.
pub trait StateAggregator {
    /// Combine per-worker trained states (keyed by node id) into the one
    /// set redistributed to every worker.
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

/// Element-wise mean of one node's state across contributors.
///
/// Refuses rather than guesses, on purpose. A key one contributor lacks,
/// a shape that disagrees, or a non-tensor state that is not identical
/// everywhere all produce an error naming what differed — because the
/// alternative is an average over a subset, silently, in the middle of
/// training.
fn mean_of(label: &str, contributions: &[(usize, &Value)]) -> Result<Value> {
    let (first_idx, first) = contributions[0];
    match first {
        Value::Tensor { values, shape } => {
            let mut acc = vec![0.0f64; values.len()];
            for (idx, value) in contributions {
                let (v, s) = match value {
                    Value::Tensor { values, shape } => (values, shape),
                    other => {
                        return Err(SomaError::Other(format!(
                            "aggregating `{label}`: contributor {idx} has a \
                             {other:?} where contributor {first_idx} has a tensor"
                        )));
                    }
                };
                if s != shape {
                    return Err(SomaError::Other(format!(
                        "aggregating `{label}`: contributor {idx} has shape {s:?}, \
                         contributor {first_idx} has {shape:?}"
                    )));
                }
                for (slot, x) in acc.iter_mut().zip(v.iter()) {
                    *slot += *x;
                }
            }
            let n = contributions.len() as f64;
            for slot in &mut acc {
                *slot /= n;
            }
            Ok(Value::tensor(acc, shape.clone()))
        }
        // A Python filter's state is a dict — `{"mu": 1.5}` — which
        // arrives as Json. Averaging its numeric leaves is exactly what
        // FedAvg means for it, and refusing would have made this useless
        // for the filters people actually write.
        Value::Json(_) => {
            let mut jsons = Vec::with_capacity(contributions.len());
            for (idx, value) in contributions {
                match value {
                    Value::Json(j) => jsons.push((*idx, j.as_ref())),
                    other => {
                        return Err(SomaError::Other(format!(
                            "aggregating `{label}`: contributor {idx} has a \
                             {other:?} where contributor {first_idx} has a dict"
                        )));
                    }
                }
            }
            Ok(Value::json(mean_json(label, &jsons)?))
        }
        other => {
            // Nothing to average. Identical everywhere is a legitimate
            // constant; anything else has no mean and inventing one would
            // be worse than stopping.
            for (idx, value) in &contributions[1..] {
                if *value != other {
                    return Err(SomaError::Other(format!(
                        "aggregating `{label}`: it is not a tensor or a dict, \
                         and contributor {idx} disagrees with contributor \
                         {first_idx}. A non-numeric state has no mean"
                    )));
                }
            }
            Ok(other.clone())
        }
    }
}

/// Element-wise mean of JSON states, leaf by leaf.
///
/// Numbers average. Objects recurse, and must carry the same keys.
/// Arrays average position-wise, and must be the same length. Anything
/// else — a string, a bool, a null — passes through only when every
/// contributor agrees, because there is no such thing as the mean of two
/// different strings.
fn mean_json(label: &str, values: &[(usize, &serde_json::Value)]) -> Result<serde_json::Value> {
    use serde_json::Value as J;
    let (first_idx, first) = values[0];
    match first {
        J::Number(_) => {
            let mut sum = 0.0;
            for (idx, v) in values {
                sum += v.as_f64().ok_or_else(|| {
                    SomaError::Other(format!(
                        "aggregating `{label}`: contributor {idx} has {v} where \
                         contributor {first_idx} has a number"
                    ))
                })?;
            }
            Ok(serde_json::json!(sum / values.len() as f64))
        }
        J::Object(first_map) => {
            let mut out = serde_json::Map::new();
            for key in first_map.keys() {
                let mut inner = Vec::with_capacity(values.len());
                for (idx, v) in values {
                    let child = v.get(key).ok_or_else(|| {
                        SomaError::Other(format!(
                            "aggregating `{label}`: contributor {idx} is missing \
                             `{key}`"
                        ))
                    })?;
                    inner.push((*idx, child));
                }
                out.insert(key.clone(), mean_json(&format!("{label}.{key}"), &inner)?);
            }
            Ok(J::Object(out))
        }
        J::Array(first_arr) => {
            let mut out = Vec::with_capacity(first_arr.len());
            for i in 0..first_arr.len() {
                let mut inner = Vec::with_capacity(values.len());
                for (idx, v) in values {
                    let arr = v.as_array().ok_or_else(|| {
                        SomaError::Other(format!(
                            "aggregating `{label}`: contributor {idx} is not an array"
                        ))
                    })?;
                    if arr.len() != first_arr.len() {
                        return Err(SomaError::Other(format!(
                            "aggregating `{label}`: contributor {idx} has {} elements, \
                             contributor {first_idx} has {}",
                            arr.len(),
                            first_arr.len()
                        )));
                    }
                    inner.push((*idx, &arr[i]));
                }
                out.push(mean_json(&format!("{label}[{i}]"), &inner)?);
            }
            Ok(J::Array(out))
        }
        other => {
            for (idx, v) in &values[1..] {
                if *v != other {
                    return Err(SomaError::Other(format!(
                        "aggregating `{label}`: contributor {idx} has {v}, contributor \
                         {first_idx} has {other}. Neither is numeric, so there is no mean"
                    )));
                }
            }
            Ok(other.clone())
        }
    }
}

/// Average every node's entry across contributors, key by key.
fn mean_by_key(what: &str, entries: &[HashMap<String, Value>]) -> Result<HashMap<String, Value>> {
    let mut out = HashMap::new();
    for key in entries[0].keys() {
        let mut contributions = Vec::with_capacity(entries.len());
        for (idx, entry) in entries.iter().enumerate() {
            match entry.get(key) {
                Some(value) => contributions.push((idx, value)),
                None => {
                    return Err(SomaError::Other(format!(
                        "aggregating {what}: `{key}` is missing from contributor \
                         {idx}. Averaging over whoever happens to have it would \
                         quietly weight the others"
                    )));
                }
            }
        }
        out.insert(key.clone(), mean_of(key, &contributions)?);
    }
    Ok(out)
}

impl GradientAggregator for GradientAggregation {
    fn aggregate(&self, gradients: &[HashMap<String, Value>]) -> Result<HashMap<String, Value>> {
        // A single worker needs no aggregation: its gradients *are* the
        // result, so this case is exact rather than a stand-in.
        if gradients.len() == 1 {
            return Ok(gradients[0].clone());
        }
        // The arithmetic below is the same mean AllReduce would compute,
        // but nothing can reach it yet: gradients have to come off the
        // worker first, and `soma-worker/src/server.rs` answers
        // `GetGradients`/`ApplyGradients` with "not implemented for
        // SubprocessFilter". Naming that is more useful than naming the
        // averaging, which is right here.
        match self {
            GradientAggregation::AllReduce => mean_by_key("gradients", gradients),
            other => Err(SomaError::Other(format!(
                "{other:?} is not implemented; only AllReduce (an element-wise \
                 mean) is. Note that no gradient can reach this function yet: \
                 soma-worker/src/server.rs refuses GetGradients and \
                 ApplyGradients for SubprocessFilter"
            ))),
        }
    }
}

impl StateAggregator for FederatedAggregation {
    fn aggregate(&self, states: &[HashMap<String, Value>]) -> Result<HashMap<String, Value>> {
        if states.is_empty() {
            return Err(SomaError::Other(
                "federated aggregation over zero clients".into(),
            ));
        }
        if states.len() == 1 {
            return Ok(states[0].clone());
        }
        match self {
            FederatedAggregation::FedAvg => mean_by_key("client states", states),
            // Both need something this function is not given. FedProx
            // needs the global model to measure drift against; FedYogi
            // needs the optimizer moments it carries between rounds. A
            // plain mean would be FedAvg wearing their name.
            FederatedAggregation::FedProx { .. } => Err(SomaError::Other(
                "FedProx needs the previous global model to compute its proximal \
                 term, and this aggregator only receives the clients' states. \
                 FedAvg works today"
                    .into(),
            )),
            FederatedAggregation::FedYogi { .. } => Err(SomaError::Other(
                "FedYogi needs the optimizer moments carried between rounds, and \
                 this aggregator is stateless. FedAvg works today"
                    .into(),
            )),
            other => Err(SomaError::Other(format!(
                "this runtime does not know how to aggregate with {other:?}"
            ))),
        }
    }
}

/// A [`StrategyContext`] over one [`Transport`] per worker.
///
/// This is the piece that was missing: `StrategyExecutor` was written and
/// had nowhere to run, because nothing implemented the context it takes.
///
/// It deliberately does **not** send `GetState`/`SetState` over the wire,
/// even though the worker now answers them: a Fit already returns its
/// trained states in the plan result, so asking again would be a second
/// round trip for something already in hand. Gradients are different —
/// nothing else carries them — so those two do go to the worker.
pub struct TransportContext<'a> {
    transports: Vec<Arc<dyn Transport>>,
    plan: &'a ExecutionPlan,
    catalog: &'a NodeCatalog,
    seed: Option<i64>,
    /// The states each worker returned from its last fit, by worker index.
    states: Mutex<Vec<HashMap<String, Value>>>,
}

impl<'a> TransportContext<'a> {
    /// One transport per worker, in the order the strategy will index them.
    pub fn new(
        transports: Vec<Arc<dyn Transport>>,
        plan: &'a ExecutionPlan,
        catalog: &'a NodeCatalog,
        seed: Option<i64>,
    ) -> Self {
        let n = transports.len();
        Self {
            transports,
            plan,
            catalog,
            seed,
            states: Mutex::new(vec![HashMap::new(); n]),
        }
    }

    fn transport(&self, idx: usize) -> Result<&Arc<dyn Transport>> {
        self.transports.get(idx).ok_or_else(|| {
            SomaError::Other(format!(
                "worker {idx} was asked for, but only {} are registered",
                self.transports.len()
            ))
        })
    }
}

impl StrategyContext for TransportContext<'_> {
    fn num_workers(&self) -> usize {
        self.transports.len()
    }

    fn execute_on_worker(
        &self,
        worker_idx: usize,
        _plan: &serde_json::Value,
        input: &Value,
        y: Option<&Value>,
    ) -> Result<HashMap<String, Value>> {
        // The trait's `plan` argument is a JSON placeholder every strategy
        // passes as `{}`; the executable plan is the one this context was
        // built with.
        let (_, states) = self.transport(worker_idx)?.execute(
            self.plan,
            self.catalog,
            input,
            &RunMode::Fit { y: y.cloned() },
            self.seed,
        )?;
        if let Ok(mut cache) = self.states.lock() {
            cache[worker_idx] = states.clone();
        }
        Ok(states)
    }

    fn get_state(&self, worker_idx: usize, node_ids: &[String]) -> Result<HashMap<String, Value>> {
        let cache = self
            .states
            .lock()
            .map_err(|e| SomaError::Other(format!("state cache poisoned: {e}")))?;
        let states = cache.get(worker_idx).ok_or_else(|| {
            SomaError::Other(format!("worker {worker_idx} has no recorded state"))
        })?;
        if node_ids.is_empty() {
            return Ok(states.clone());
        }
        Ok(node_ids
            .iter()
            .filter_map(|id| states.get(id).map(|v| (id.clone(), v.clone())))
            .collect())
    }

    fn set_state(&self, worker_idx: usize, states: &HashMap<String, Value>) -> Result<()> {
        // Into the catalog, which is what the next plan serializes its
        // filter states from.
        for (node_id, state) in states {
            self.catalog.try_set_state(node_id.clone(), state.clone())?;
        }
        // And into this worker's record, because that is what the call
        // means: worker `worker_idx` now holds these. Without it the
        // federated loop's closing `get_state(0)` returns worker 0's own
        // last fit instead of the aggregate just distributed to it — one
        // client's answer presented as the average of all of them.
        if let Ok(mut cache) = self.states.lock()
            && let Some(slot) = cache.get_mut(worker_idx)
        {
            for (node_id, state) in states {
                slot.insert(node_id.clone(), state.clone());
            }
        }
        Ok(())
    }

    fn get_gradients(
        &self,
        worker_idx: usize,
        node_ids: &[String],
    ) -> Result<HashMap<String, Value>> {
        self.transport(worker_idx)?.get_gradients(node_ids)
    }

    fn apply_gradients(&self, worker_idx: usize, gradients: &HashMap<String, Value>) -> Result<()> {
        self.transport(worker_idx)?.apply_gradients(gradients)
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
    use somatize_core::strategy::ClientSelection;

    fn one(node: &str, values: Vec<f64>) -> HashMap<String, Value> {
        let n = values.len();
        HashMap::from([(node.to_string(), Value::tensor(values, vec![n]))])
    }

    /// FedAvg is an element-wise mean, and this is the whole reason the
    /// federated loop could not run: both aggregators used to refuse for
    /// more than one contributor, so a second client was an error.
    #[test]
    fn fedavg_averages_element_wise() {
        let out = FederatedAggregation::FedAvg
            .aggregate(&[one("w", vec![1.0, 10.0]), one("w", vec![3.0, 20.0])])
            .unwrap();
        assert_eq!(out["w"], Value::tensor(vec![2.0, 15.0], vec![2]));

        let out = FederatedAggregation::FedAvg
            .aggregate(&[
                one("w", vec![0.0]),
                one("w", vec![3.0]),
                one("w", vec![6.0]),
            ])
            .unwrap();
        assert_eq!(out["w"], Value::tensor(vec![3.0], vec![1]));
    }

    /// The same arithmetic for gradients. Nothing can reach it yet — the
    /// worker refuses to hand gradients over — but the error must be about
    /// that, not about the averaging.
    #[test]
    fn allreduce_averages_and_the_others_say_what_they_are_not() {
        let out = GradientAggregation::AllReduce
            .aggregate(&[one("w", vec![2.0]), one("w", vec![4.0])])
            .unwrap();
        assert_eq!(out["w"], Value::tensor(vec![3.0], vec![1]));

        let err = GradientAggregation::ParameterServer
            .aggregate(&[one("w", vec![1.0]), one("w", vec![2.0])])
            .expect_err("only AllReduce is implemented");
        assert!(err.to_string().contains("server.rs"), "{err}");
    }

    /// A subset average is a wrong number that looks like a right one.
    #[test]
    fn a_contributor_missing_a_key_is_an_error_naming_it() {
        let err = FederatedAggregation::FedAvg
            .aggregate(&[one("w", vec![1.0]), one("other", vec![2.0])])
            .expect_err("averaging over whoever has the key would misweight");
        let msg = err.to_string();
        assert!(
            msg.contains("`w`") && msg.contains("contributor 1"),
            "{msg}"
        );
    }

    #[test]
    fn mismatched_shapes_name_both() {
        let err = FederatedAggregation::FedAvg
            .aggregate(&[one("w", vec![1.0, 2.0]), one("w", vec![3.0])])
            .expect_err("shapes that disagree have no mean");
        let msg = err.to_string();
        assert!(msg.contains("[1]") && msg.contains("[2]"), "{msg}");
    }

    /// FedProx and FedYogi are not FedAvg wearing a different name; each
    /// needs something this aggregator is never given.
    #[test]
    fn the_adaptive_variants_say_what_they_would_need() {
        let two = [one("w", vec![1.0]), one("w", vec![3.0])];
        let err = FederatedAggregation::FedProx { mu: 0.1 }
            .aggregate(&two)
            .unwrap_err()
            .to_string();
        assert!(err.contains("global model"), "{err}");
        let err = FederatedAggregation::FedYogi {
            beta1: 0.9,
            beta2: 0.99,
            tau: 1e-3,
        }
        .aggregate(&two)
        .unwrap_err()
        .to_string();
        assert!(err.contains("moments"), "{err}");
    }

    /// One worker is the exact case, not a stand-in: there is nothing to
    /// average, so it stays supported.
    #[test]
    fn single_worker_aggregation_is_the_identity() {
        let only = one("w", vec![2.0]);
        let out = GradientAggregation::AllReduce
            .aggregate(std::slice::from_ref(&only))
            .unwrap();
        assert_eq!(out, only);
    }

    /// The federated loop, driven end to end over fake transports.
    ///
    /// Each "worker" reports a state derived from the shard it was given,
    /// so a run that quietly used one client cannot produce the mean of
    /// two — which is what this asserts.
    #[test]
    fn the_federated_loop_converges_to_the_mean_of_its_clients() {
        use somatize_compiler::ExecutionPlan;
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct ShardMean {
            calls: AtomicUsize,
        }
        impl Transport for ShardMean {
            fn execute(
                &self,
                _plan: &ExecutionPlan,
                _filters: &NodeCatalog,
                input: &Value,
                _mode: &RunMode,
                _seed: Option<i64>,
            ) -> Result<(Value, HashMap<String, Value>)> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                let mean = match input {
                    Value::Tensor { values, .. } if !values.is_empty() => {
                        values.iter().sum::<f64>() / values.len() as f64
                    }
                    _ => 0.0,
                };
                Ok((Value::Empty, one("m", vec![mean])))
            }
            fn get_state(&self, _: &[String]) -> Result<HashMap<String, Value>> {
                Ok(HashMap::new())
            }
            fn set_state(&self, _: &HashMap<String, Value>) -> Result<()> {
                Ok(())
            }
            fn get_gradients(&self, _: &[String]) -> Result<HashMap<String, Value>> {
                Ok(HashMap::new())
            }
            fn apply_gradients(&self, _: &HashMap<String, Value>) -> Result<()> {
                Ok(())
            }
        }

        let transports: Vec<Arc<dyn Transport>> = vec![
            Arc::new(ShardMean {
                calls: AtomicUsize::new(0),
            }),
            Arc::new(ShardMean {
                calls: AtomicUsize::new(0),
            }),
        ];
        let plan = ExecutionPlan::Execute {
            node_id: "m".into(),
        };
        let catalog = NodeCatalog::new();
        let ctx = TransportContext::new(transports, &plan, &catalog, None);

        // 0..8 split in two: means 1.5 and 5.5, whose mean is 3.5.
        let input = Value::tensor((0..8).map(|i| i as f64).collect(), vec![8]);
        let strategy = TrainingStrategy::Federated {
            num_clients: 2,
            rounds: 2,
            aggregation: FederatedAggregation::FedAvg,
            client_selection: ClientSelection::All,
        };
        let out = strategy
            .fit(&ctx, &input, None, &["m".to_string()])
            .expect("the federated loop must run");

        let Value::Tensor { values, .. } = &out["m"] else {
            panic!("expected a tensor, got {:?}", out["m"]);
        };
        assert!((values[0] - 3.5).abs() < 1e-9, "got {}", values[0]);
        // Not either client alone — a single-client path cannot pass this.
        assert!((values[0] - 1.5).abs() > 1e-6 && (values[0] - 5.5).abs() > 1e-6);
    }

    /// DataParallel drives its workers through the context now.
    ///
    /// It used to be impossible: `soma-worker/src/server.rs` refused
    /// `GetGradients`/`ApplyGradients`, so the loop could not get past its
    /// first collection. The server dispatches them today, and this
    /// asserts the loop completes rather than erroring — the gradients a
    /// parameterless filter contributes are empty, and an empty average is
    /// the right answer for it.
    #[test]
    fn data_parallel_runs_its_loop() {
        use somatize_compiler::ExecutionPlan;

        struct Noop;
        impl Transport for Noop {
            fn execute(
                &self,
                _: &ExecutionPlan,
                _: &NodeCatalog,
                _: &Value,
                _: &RunMode,
                _: Option<i64>,
            ) -> Result<(Value, HashMap<String, Value>)> {
                Ok((Value::Empty, HashMap::new()))
            }
            fn get_state(&self, _: &[String]) -> Result<HashMap<String, Value>> {
                Ok(HashMap::new())
            }
            fn set_state(&self, _: &HashMap<String, Value>) -> Result<()> {
                Ok(())
            }
            fn get_gradients(&self, _: &[String]) -> Result<HashMap<String, Value>> {
                Ok(HashMap::new())
            }
            fn apply_gradients(&self, _: &HashMap<String, Value>) -> Result<()> {
                Ok(())
            }
        }

        let transports: Vec<Arc<dyn Transport>> = vec![Arc::new(Noop), Arc::new(Noop)];
        let plan = ExecutionPlan::Execute {
            node_id: "m".into(),
        };
        let catalog = NodeCatalog::new();
        let ctx = TransportContext::new(transports, &plan, &catalog, None);

        let out = TrainingStrategy::DataParallel {
            num_replicas: 2,
            aggregation: GradientAggregation::AllReduce,
        }
        .fit(
            &ctx,
            &Value::tensor(vec![1.0, 2.0], vec![2]),
            None,
            &["m".to_string()],
        )
        .expect("DataParallel drives the workers through the context");
        assert!(
            out.is_empty(),
            "a filter with no parameters contributes no gradients: {out:?}"
        );
    }
}
