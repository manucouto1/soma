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
use somatize_core::filter::RemoteTarget;
use somatize_core::strategy::{
    FederatedAggregation, GradientAggregation, Partition, TrainingStrategy,
};
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

    /// Read a worker's state *now*, over the wire, rather than recalling
    /// what its last fit returned.
    ///
    /// The two differ exactly when something changed the model after the
    /// fit — which is what [`apply_gradients`](Self::apply_gradients) does.
    /// A data-parallel round that finished with `get_state` handed back the
    /// weights each replica had *before* the averaged gradient was applied,
    /// so the training it had just done was discarded on the way out.
    ///
    /// Defaults to `get_state`, for a context whose two answers cannot
    /// differ.
    fn read_back_state(
        &self,
        worker_idx: usize,
        node_ids: &[String],
    ) -> Result<HashMap<String, Value>> {
        self.get_state(worker_idx, node_ids)
    }

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

    /// Run *part* of the graph on a worker, returning the activation and
    /// the states it learned.
    ///
    /// This is what model parallelism needs and data parallelism does
    /// not: every other strategy runs the whole plan on each worker and
    /// only ever wants the states back. Here each worker holds a slice of
    /// the model, so its output is the next worker's input.
    ///
    /// Defaults to refusing, so a context that cannot address part of a
    /// plan says so instead of silently running all of it.
    fn execute_partition(
        &self,
        _worker_idx: usize,
        _node_ids: &[String],
        _input: &Value,
        _y: Option<&Value>,
    ) -> Result<(Value, HashMap<String, Value>)> {
        Err(SomaError::Other(
            "this context cannot run part of a plan, so a model-parallel \
             partition has nowhere to go"
                .into(),
        ))
    }

    /// Which worker answers to `target`.
    ///
    /// Every other strategy indexes workers by position, because every
    /// worker is interchangeable to it. A partition is pinned to one, so
    /// it has to be found by id or tag.
    fn worker_for(&self, target: &RemoteTarget) -> Result<usize> {
        Err(SomaError::Other(format!(
            "this context does not know which worker is which, so {target:?} \
             cannot be resolved"
        )))
    }
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
                let (shards, y_shards) = shard_pair(input, y, n)?;

                // Fit on each worker with its shard — inputs and targets
                // split together, so example i still meets target i.
                for (i, shard) in shards.iter().enumerate() {
                    ctx.execute_on_worker(i, &serde_json::json!({}), shard, y_shards[i].as_ref())?;
                }

                // Collect and aggregate gradients
                let mut all_grads = Vec::new();
                for i in 0..n {
                    all_grads.push(ctx.get_gradients(i, node_ids)?);
                }
                let averaged = aggregation.aggregate(&all_grads)?;

                // Apply to all workers. This is where the step happens: the
                // replicas move together, on the mean of what they each saw.
                for i in 0..n {
                    ctx.apply_gradients(i, &averaged)?;
                }

                // Read worker 0 back over the wire. `get_state` would return
                // what its fit returned — the weights from *before* the
                // averaged gradient was applied — so the round would train
                // and then hand back the untrained model.
                ctx.read_back_state(0, node_ids)
            }

            TrainingStrategy::Federated {
                num_clients,
                rounds,
                aggregation,
                ..
            } => {
                let n = (*num_clients).min(ctx.num_workers());
                let (shards, y_shards) = shard_pair(input, y, n)?;

                for _round in 0..*rounds {
                    // Each client trains on its shard
                    for (i, shard) in shards.iter().enumerate().take(n) {
                        ctx.execute_on_worker(
                            i,
                            &serde_json::json!({}),
                            shard,
                            y_shards[i].as_ref(),
                        )?;
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

            TrainingStrategy::ModelParallel { partitions, .. } => {
                let stages = order_partitions(partitions, node_ids)?;

                // Each stage runs where it was pinned, and hands its
                // activation to the next one. That is the whole of model
                // parallelism on the forward path: the model is split, the
                // data is not.
                let mut activation = input.clone();
                let mut states: HashMap<String, Value> = HashMap::new();
                for (partition, ids) in &stages {
                    let worker = ctx.worker_for(&partition.target)?;
                    let (output, learned) = ctx.execute_partition(worker, ids, &activation, y)?;
                    states.extend(learned);
                    activation = output;
                }
                Ok(states)
            }

            TrainingStrategy::PopulationBased { .. } => {
                // Not a missing implementation — a wrong home. PBT gives
                // each member DIFFERENT hyperparameters, and applying them
                // means rebuilding the graph's filters with new configs.
                // A strategy only gets to send a plan; the configs live in
                // the caller's language. Which is exactly the shape of
                // `Study`, and why `PbtRunner` takes a callback.
                Err(SomaError::Other(
                    "population-based training is not a distribution strategy: \
                     each member needs its own hyperparameters applied to the \
                     graph, and a worker is sent a plan, not a way to rebuild \
                     the filters. It runs as an executor instead, driven from \
                     Python:\n    pbt = soma.Pbt(search_space=[...], \
                     population_size=8, generations=5)\n    \
                     best = pbt.run(train, evaluate)"
                        .into(),
                ))
            }

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
    // Guarded here as well as at both call sites: `entries[0]` below is a
    // panic, and a panic is the one failure mode a caller cannot report.
    if entries.is_empty() {
        return Err(SomaError::Other(format!(
            "averaging {what} over zero contributors: there is nothing to \
             take a mean of"
        )));
    }
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
        // Zero contributors reached `mean_by_key`, which indexes
        // `entries[0]` — a panic, from a `num_replicas` of 0 that nothing
        // validated. The federated aggregator below has always guarded
        // this; this one did not.
        if gradients.is_empty() {
            return Err(SomaError::Other(
                "aggregating gradients from zero replicas: a data-parallel \
                 round with no workers to average over"
                    .into(),
            ));
        }
        match self {
            GradientAggregation::AllReduce => mean_by_key("gradients", gradients),
            other => Err(SomaError::Other(format!(
                "{other:?} is not implemented; only AllReduce (an element-wise \
                 mean) is"
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
    /// `(id, tags)` per worker, in transport order. Empty unless
    /// [`with_targets`](Self::with_targets) was used — only model
    /// parallelism needs to tell workers apart, and only it pays for
    /// knowing.
    identities: Vec<WorkerIdentity>,
}

/// How a worker can be named by a `RemoteTarget`.
#[derive(Debug, Clone)]
pub struct WorkerIdentity {
    /// The worker's id — its address, as registered.
    pub id: String,
    /// Capability tags it was registered with.
    pub tags: Vec<String>,
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
            identities: Vec::new(),
        }
    }

    /// Name the workers, so a partition pinned to an id or a tag can find
    /// one. Without this, `worker_for` refuses rather than guessing.
    pub fn with_targets(mut self, identities: Vec<WorkerIdentity>) -> Self {
        self.identities = identities;
        self
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

    fn worker_for(&self, target: &RemoteTarget) -> Result<usize> {
        if self.identities.is_empty() {
            return Err(SomaError::Other(format!(
                "this context was built without worker identities, so {target:?} \
                 cannot be resolved. Build it with `with_targets`"
            )));
        }
        let found = match target {
            RemoteTarget::WorkerId(id) => self.identities.iter().position(|w| &w.id == id),
            RemoteTarget::Tag(tag) => self
                .identities
                .iter()
                .position(|w| w.tags.iter().any(|t| t == tag)),
        };
        found.ok_or_else(|| {
            SomaError::Other(format!(
                "no registered worker answers to {target:?}. Registered: {}",
                self.identities
                    .iter()
                    .map(|w| format!("{} {:?}", w.id, w.tags))
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })
    }

    fn execute_partition(
        &self,
        worker_idx: usize,
        node_ids: &[String],
        input: &Value,
        y: Option<&Value>,
    ) -> Result<(Value, HashMap<String, Value>)> {
        // A stage's plan is its own nodes, in order — not the whole plan
        // this context holds, which is what every other strategy sends.
        let stage = ExecutionPlan::Sequence(
            node_ids
                .iter()
                .map(|node_id| ExecutionPlan::Execute {
                    node_id: node_id.clone(),
                })
                .collect(),
        );
        let (output, states) = self.transport(worker_idx)?.execute(
            &stage,
            self.catalog,
            input,
            &RunMode::Fit { y: y.cloned() },
            self.seed,
        )?;
        if let Ok(mut cache) = self.states.lock()
            && let Some(slot) = cache.get_mut(worker_idx)
        {
            slot.extend(states.clone());
        }
        Ok((output, states))
    }

    fn read_back_state(
        &self,
        worker_idx: usize,
        node_ids: &[String],
    ) -> Result<HashMap<String, Value>> {
        let states = self.transport(worker_idx)?.get_state(node_ids)?;
        // Record it, so a later `get_state` agrees with the wire.
        if let Ok(mut cache) = self.states.lock()
            && let Some(slot) = cache.get_mut(worker_idx)
        {
            for (id, value) in &states {
                slot.insert(id.clone(), value.clone());
            }
        }
        Ok(states)
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

/// Put the partitions in execution order, checking they can be a chain.
///
/// A partition is a *stage*: it runs somewhere, and its output is the next
/// stage's input. That only means something if the partitions tile the
/// plan — so a node claimed twice, a node claimed by nobody, and a
/// partition whose nodes are interleaved with another's are all errors
/// here rather than a pipeline that quietly drops or repeats a node.
fn order_partitions<'a>(
    partitions: &'a [Partition],
    node_ids: &[String],
) -> Result<Vec<(&'a Partition, Vec<String>)>> {
    if partitions.is_empty() {
        return Err(SomaError::Other(
            "model-parallel training with no partitions: there is nothing to \
             say where any node runs"
                .into(),
        ));
    }
    let position: HashMap<&str, usize> = node_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (id.as_str(), i))
        .collect();

    let mut claimed: HashMap<&str, usize> = HashMap::new();
    let mut stages: Vec<(&Partition, Vec<usize>)> = Vec::new();
    for (p_idx, partition) in partitions.iter().enumerate() {
        let mut positions = Vec::with_capacity(partition.node_ids.len());
        for node in &partition.node_ids {
            let Some(&pos) = position.get(node.as_str()) else {
                return Err(SomaError::Other(format!(
                    "partition {p_idx} claims `{node}`, which is not in this \
                     graph. Its nodes are: {}",
                    node_ids.join(", ")
                )));
            };
            if let Some(&first) = claimed.get(node.as_str()) {
                return Err(SomaError::Other(format!(
                    "`{node}` is claimed by partitions {first} and {p_idx}. A \
                     node runs in one place"
                )));
            }
            claimed.insert(node.as_str(), p_idx);
            positions.push(pos);
        }
        positions.sort_unstable();
        stages.push((partition, positions));
    }

    let unclaimed: Vec<&str> = node_ids
        .iter()
        .map(String::as_str)
        .filter(|id| !claimed.contains_key(id))
        .collect();
    if !unclaimed.is_empty() {
        return Err(SomaError::Other(format!(
            "no partition claims {}. Every node needs a worker; model \
             parallelism has no default target",
            unclaimed.join(", ")
        )));
    }

    stages.sort_by_key(|(_, positions)| positions.first().copied().unwrap_or(0));
    // Contiguous, once ordered: stage k must own a solid run of the plan.
    let mut next = 0usize;
    for (p_idx, (_, positions)) in stages.iter().enumerate() {
        for &pos in positions {
            if pos != next {
                return Err(SomaError::Other(format!(
                    "partition {p_idx} is interleaved with another: it owns \
                     `{}` but not `{}`, which runs before it. A stage has to \
                     own a contiguous run of the graph",
                    node_ids[pos], node_ids[next]
                )));
            }
            next += 1;
        }
    }

    Ok(stages
        .into_iter()
        .map(|(partition, positions)| {
            let ids = positions.iter().map(|&i| node_ids[i].clone()).collect();
            (partition, ids)
        })
        .collect())
}

/// Split inputs and targets into `n` shards **together**.
///
/// Sharding `x` and sending every worker the whole `y` is the bug this
/// exists to make impossible. It is not caught by anything downstream: a
/// 4-row output against an 8-row target does not fail, it *broadcasts*, so
/// each replica computed a loss between things that were never paired,
/// backpropagated it, and reported a successful round. Only the diverging
/// weights showed it.
///
/// Row counts that disagree are an error naming both, since pairing
/// example `i` with target `i` is the one assumption every shard rests on.
fn shard_pair(x: &Value, y: Option<&Value>, n: usize) -> Result<(Vec<Value>, Vec<Option<Value>>)> {
    let x_shards = shard_value(x, n);
    let Some(y) = y else {
        return Ok((x_shards, vec![None; n]));
    };
    if let (Some(xr), Some(yr)) = (rows_of(x), rows_of(y))
        && xr != yr
    {
        return Err(SomaError::Other(format!(
            "sharding across {n} workers: the input has {xr} rows and the \
             targets have {yr}. Each shard pairs example i with target i, \
             so the two must agree"
        )));
    }
    let y_shards = shard_value(y, n);
    if y_shards.len() != x_shards.len() {
        return Err(SomaError::Other(format!(
            "sharding across {n} workers: the input split into {} shards and \
             the targets into {}",
            x_shards.len(),
            y_shards.len()
        )));
    }
    Ok((x_shards, y_shards.into_iter().map(Some).collect()))
}

/// Leading dimension of a tensor, when it has one.
fn rows_of(value: &Value) -> Option<usize> {
    match value {
        Value::Tensor { shape, .. } if !shape.is_empty() => Some(shape[0]),
        _ => None,
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

    fn part(nodes: &[&str], tag: &str) -> Partition {
        Partition {
            node_ids: nodes.iter().map(|s| s.to_string()).collect(),
            target: RemoteTarget::Tag(tag.into()),
        }
    }

    fn ids(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    /// The ordinary case: two stages, in plan order whatever order they
    /// were declared in.
    #[test]
    fn partitions_are_ordered_by_the_plan_not_by_declaration() {
        let declared = [part(&["c", "d"], "gpu1"), part(&["a", "b"], "gpu0")];
        let stages = order_partitions(&declared, &ids(&["a", "b", "c", "d"])).unwrap();
        assert_eq!(stages.len(), 2);
        assert_eq!(stages[0].1, ids(&["a", "b"]));
        assert_eq!(stages[1].1, ids(&["c", "d"]));
    }

    /// A node claimed twice would run twice, on two machines, and the
    /// second activation would silently overwrite the first.
    #[test]
    fn a_node_in_two_partitions_is_refused() {
        let declared = [part(&["a", "b"], "gpu0"), part(&["b"], "gpu1")];
        let err = order_partitions(&declared, &ids(&["a", "b"]))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("`b` is claimed by partitions 0 and 1"),
            "{err}"
        );
    }

    /// A node claimed by nobody has no worker, and model parallelism has
    /// no default target to fall back on.
    #[test]
    fn an_unclaimed_node_is_refused_by_name() {
        let declared = [part(&["a"], "gpu0")];
        let err = order_partitions(&declared, &ids(&["a", "b"]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("no partition claims b"), "{err}");
    }

    /// Interleaved stages are not a pipeline: `a`,`c` on one worker and
    /// `b` on another would need the activation to cross back.
    #[test]
    fn interleaved_partitions_are_refused() {
        let declared = [part(&["a", "c"], "gpu0"), part(&["b"], "gpu1")];
        let err = order_partitions(&declared, &ids(&["a", "b", "c"]))
            .unwrap_err()
            .to_string();
        assert!(err.contains("interleaved"), "{err}");
    }

    #[test]
    fn no_partitions_at_all_is_refused() {
        let err = order_partitions(&[], &ids(&["a"])).unwrap_err().to_string();
        assert!(err.contains("nothing to say where any node runs"), "{err}");
    }

    /// The activation is threaded: stage 2 receives what stage 1 produced,
    /// not the graph's input. A context that ignored the chaining would
    /// hand both stages the same thing and still "succeed".
    #[test]
    fn model_parallel_threads_the_activation_between_stages() {
        use std::sync::Mutex as StdMutex;

        #[derive(Default)]
        struct Chain {
            seen: StdMutex<Vec<(usize, Vec<String>, Value)>>,
        }
        impl StrategyContext for Chain {
            fn num_workers(&self) -> usize {
                2
            }
            fn execute_on_worker(
                &self,
                _: usize,
                _: &serde_json::Value,
                _: &Value,
                _: Option<&Value>,
            ) -> Result<HashMap<String, Value>> {
                unreachable!("model parallelism runs partitions, not whole plans")
            }
            fn execute_partition(
                &self,
                worker_idx: usize,
                node_ids: &[String],
                input: &Value,
                _: Option<&Value>,
            ) -> Result<(Value, HashMap<String, Value>)> {
                self.seen
                    .lock()
                    .unwrap()
                    .push((worker_idx, node_ids.to_vec(), input.clone()));
                // Each stage adds one, so the output identifies its stage.
                let next = match input {
                    Value::Tensor { values, shape } => {
                        Value::tensor(values.iter().map(|v| v + 1.0).collect(), shape.clone())
                    }
                    other => other.clone(),
                };
                let states = node_ids
                    .iter()
                    .map(|id| (id.clone(), Value::tensor(vec![1.0], vec![1])))
                    .collect();
                Ok((next, states))
            }
            fn worker_for(&self, target: &RemoteTarget) -> Result<usize> {
                match target {
                    RemoteTarget::Tag(t) if t == "gpu0" => Ok(0),
                    RemoteTarget::Tag(t) if t == "gpu1" => Ok(1),
                    other => Err(SomaError::Other(format!("no worker for {other:?}"))),
                }
            }
            fn get_state(&self, _: usize, _: &[String]) -> Result<HashMap<String, Value>> {
                Ok(HashMap::new())
            }
            fn set_state(&self, _: usize, _: &HashMap<String, Value>) -> Result<()> {
                Ok(())
            }
            fn get_gradients(&self, _: usize, _: &[String]) -> Result<HashMap<String, Value>> {
                Ok(HashMap::new())
            }
            fn apply_gradients(&self, _: usize, _: &HashMap<String, Value>) -> Result<()> {
                Ok(())
            }
        }

        let ctx = Chain::default();
        let states = TrainingStrategy::ModelParallel {
            partitions: vec![part(&["a"], "gpu0"), part(&["b"], "gpu1")],
            communication: somatize_core::strategy::CommunicationProtocol::DataStore,
        }
        .fit(
            &ctx,
            &Value::tensor(vec![10.0], vec![1]),
            None,
            &ids(&["a", "b"]),
        )
        .unwrap();

        let seen = ctx.seen.lock().unwrap();
        assert_eq!(seen.len(), 2, "one call per stage");
        assert_eq!(seen[0].0, 0, "stage 1 on gpu0");
        assert_eq!(seen[0].2, Value::tensor(vec![10.0], vec![1]));
        assert_eq!(seen[1].0, 1, "stage 2 on gpu1");
        assert_eq!(
            seen[1].2,
            Value::tensor(vec![11.0], vec![1]),
            "stage 2 must receive stage 1's output, not the graph input"
        );
        // Both stages' states come back, not just the last one's.
        assert_eq!(states.len(), 2);
        assert!(states.contains_key("a") && states.contains_key("b"));
    }

    /// A context with no idea which worker is which refuses rather than
    /// sending the partition to whoever is first.
    #[test]
    fn an_unnamed_worker_pool_refuses_a_pinned_partition() {
        let plan = ExecutionPlan::Empty;
        let catalog = NodeCatalog::new();
        let ctx = TransportContext::new(Vec::new(), &plan, &catalog, None);
        let err = ctx
            .worker_for(&RemoteTarget::Tag("gpu".into()))
            .unwrap_err()
            .to_string();
        assert!(err.contains("with_targets"), "{err}");

        let ctx = TransportContext::new(Vec::new(), &plan, &catalog, None).with_targets(vec![
            WorkerIdentity {
                id: "ws://a".into(),
                tags: vec!["cpu".into()],
            },
        ]);
        assert!(ctx.worker_for(&RemoteTarget::Tag("cpu".into())).unwrap() == 0);
        assert!(
            ctx.worker_for(&RemoteTarget::WorkerId("ws://a".into()))
                .unwrap()
                == 0
        );
        let err = ctx
            .worker_for(&RemoteTarget::Tag("gpu".into()))
            .unwrap_err()
            .to_string();
        assert!(err.contains("no registered worker"), "{err}");
    }

    /// Zero contributors used to reach `mean_by_key`, which indexes
    /// `entries[0]`. A panic is the one failure a caller cannot report,
    /// and it was reachable from Python: `num_replicas=0` passed straight
    /// through, both loops ran zero times, and the aggregator got `&[]`.
    #[test]
    fn aggregating_over_zero_contributors_errors_rather_than_panicking() {
        let err = GradientAggregation::AllReduce
            .aggregate(&[])
            .unwrap_err()
            .to_string();
        assert!(err.contains("zero replicas"), "{err}");

        let err = FederatedAggregation::FedAvg
            .aggregate(&[])
            .unwrap_err()
            .to_string();
        assert!(err.contains("zero clients"), "{err}");

        // And the shared helper guards itself, so a third caller added
        // later cannot reintroduce the panic.
        let err = mean_by_key("things", &[]).unwrap_err().to_string();
        assert!(err.contains("zero contributors"), "{err}");
    }

    /// Inputs and targets split together. Sharding only `x` sent every
    /// replica the whole `y`: shapes that broadcast rather than fail, so
    /// each one trained on pairs that were never pairs.
    #[test]
    fn shard_pair_splits_targets_alongside_inputs() {
        let x = Value::tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]);
        let y = Value::tensor(vec![10.0, 20.0, 30.0, 40.0], vec![4, 1]);
        let (xs, ys) = shard_pair(&x, Some(&y), 2).unwrap();
        assert_eq!(xs[0], Value::tensor(vec![1.0, 2.0], vec![2, 1]));
        assert_eq!(ys[0], Some(Value::tensor(vec![10.0, 20.0], vec![2, 1])));
        assert_eq!(xs[1], Value::tensor(vec![3.0, 4.0], vec![2, 1]));
        assert_eq!(ys[1], Some(Value::tensor(vec![30.0, 40.0], vec![2, 1])));
    }

    #[test]
    fn shard_pair_refuses_row_counts_that_disagree() {
        let x = Value::tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]);
        let y = Value::tensor(vec![10.0, 20.0], vec![2, 1]);
        let err = shard_pair(&x, Some(&y), 2).unwrap_err().to_string();
        assert!(
            err.contains("4 rows") && err.contains("2"),
            "the error should name both counts: {err}"
        );
    }

    #[test]
    fn shard_pair_without_targets_yields_none_per_shard() {
        let x = Value::tensor(vec![1.0, 2.0], vec![2, 1]);
        let (xs, ys) = shard_pair(&x, None, 2).unwrap();
        assert_eq!(xs.len(), 2);
        assert_eq!(ys, vec![None, None]);
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

    /// The same arithmetic for gradients, and it is reached now: a
    /// data-parallel round averages real gradients off real workers. The
    /// doc comment here used to say nothing could reach it, which was
    /// true until the worker learned to hand gradients over.
    #[test]
    fn allreduce_averages_and_the_others_say_what_they_are_not() {
        let out = GradientAggregation::AllReduce
            .aggregate(&[one("w", vec![2.0]), one("w", vec![4.0])])
            .unwrap();
        assert_eq!(out["w"], Value::tensor(vec![3.0], vec![1]));

        let err = GradientAggregation::ParameterServer
            .aggregate(&[one("w", vec![1.0]), one("w", vec![2.0])])
            .expect_err("only AllReduce is implemented");
        let err = err.to_string();
        assert!(err.contains("ParameterServer"), "name the variant: {err}");
        assert!(err.contains("AllReduce"), "name what does work: {err}");
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
