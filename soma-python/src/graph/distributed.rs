//! The same graph, on another machine.
//!
//! Everything `PyGraph` holds about *where* work runs: the registered
//! workers, the coordinator that finds more of them, the data store large
//! payloads travel through, and the training strategy that decides how a
//! fit is split across them.

use super::{PyGraph, registry};
use crate::prelude::*;

/// A worker this graph may dispatch to.
///
/// Was a bare `(String, Option<String>, Vec<String>)`, destructured
/// positionally at eight call sites — one of which spelled it `(a, t, _)`.
pub(super) struct RemoteWorker {
    pub(super) address: String,
    pub(super) token: Option<String>,
    pub(super) tags: Vec<String>,
}

// ── Picking a worker ──

/// The worker a whole-plan dispatch should go to.
///
/// Policy, in one place: the first node that names a target picks by tag,
/// and anything else takes the first registered worker. Both dispatches
/// wrote this out, so "which worker" had two answers that only happened
/// to agree.
fn pick_worker(g: &PyGraph) -> PyResult<(String, Option<String>)> {
    let first_target = g
        .graph
        .nodes
        .iter()
        .find_map(|n| n.target.as_deref())
        .unwrap_or("default");

    g.workers
        .iter()
        .find(|w| first_target == "default" || w.tags.contains(&first_target.to_string()))
        .or_else(|| g.workers.first())
        .map(|w| (w.address.clone(), w.token.clone()))
        .ok_or_else(|| PyRuntimeError::new_err("no workers available"))
}

/// Build a transport from the first registered worker (if any).
pub(super) fn make_transport(
    g: &PyGraph,
) -> Option<Arc<dyn somatize_runtime::execution::runner::Transport>> {
    let first = g.workers.first()?;
    Some(Arc::new(somatize_worker::WsTransport::new(
        &first.address,
        first.token.clone(),
    )))
}

/// A session wired to one transport per registered worker.
///
/// The strategy indexes its workers — `execute_on_worker(i, …)` — so a
/// single transport cannot serve it, which is why this exists beside
/// [`make_transport`].
pub(super) fn session_with_transports(
    g: &PyGraph,
    transports: Vec<Arc<dyn somatize_runtime::execution::runner::Transport>>,
) -> somatize_core::error::Result<somatize_runtime::GraphSession> {
    Ok(
        somatize_runtime::GraphSession::new(g.graph.clone(), g.library.clone())
            .with_cache(g.cache.clone())
            .with_event_bus(g.event_bus.clone())
            .with_transports(transports)
            // In the same order the transports were built, which is the
            // order `g.workers` is in — a strategy that pins a partition
            // to a tag resolves it against these.
            .with_worker_identities(
                g.workers
                    .iter()
                    .map(|w| somatize_runtime::distributed::WorkerIdentity {
                        id: w.address.clone(),
                        tags: w.tags.clone(),
                    })
                    .collect(),
            ),
    )
}

/// One `WsTransport` per registered worker, in registration order.
pub(super) fn transports(
    g: &PyGraph,
) -> Vec<Arc<dyn somatize_runtime::execution::runner::Transport>> {
    g.workers
        .iter()
        .map(|w| {
            Arc::new(somatize_worker::WsTransport::new(
                &w.address,
                w.token.clone(),
            )) as Arc<dyn somatize_runtime::execution::runner::Transport>
        })
        .collect()
}

// ── Dispatch ──

/// Put this graph's filters on every worker before a strategy runs.
///
/// A strategy drives workers through `Transport::execute`, which cannot
/// carry them: the worker rebuilds a Python filter by unpickling
/// `SerializedFilter::pickled_filter`, and those bytes live only here — a
/// `NodeCatalog` holds live filters, never the pickle. So the strategy's
/// first call would arrive at a worker that has never heard of the node,
/// and fail with `node not found in graph`.
///
/// An `ExecutionPlan::Empty` carrying the filters registers them and
/// executes nothing. The worker keeps its catalog between messages, so
/// every later call in the round loop finds them.
pub(super) fn register_filters_on_all(g: &PyGraph) -> PyResult<()> {
    use somatize_worker::protocol::{CoordinatorToWorker, SerializedPlan};

    for worker in &g.workers {
        let mut plan = SerializedPlan::new(
            somatize_core::util::timestamp_id("register"),
            somatize_compiler::ExecutionPlan::Empty,
        );
        plan.filters = registry::serialized_filters(g);
        somatize_worker::WsTransport::new(&worker.address, worker.token.clone())
            .send_msg(&CoordinatorToWorker::AssignPlan { plan })
            .map_err(|e| soma_err_to_py(e.into()))?;
    }
    Ok(())
}

/// How the input data reaches the worker.
///
/// - DataStore configured → upload to the store, send a reference
/// - Payload ≥ [`INLINE_THRESHOLD_BYTES`] → HTTP bulk upload, send a reference
/// - Otherwise → inline, in the WS message
///
/// [`INLINE_THRESHOLD_BYTES`]: somatize_core::data::store::INLINE_THRESHOLD_BYTES
fn resolve_transport(
    g: &PyGraph,
    x: &Value,
    transport: &somatize_worker::WsTransport,
) -> Result<somatize_worker::protocol::InputSource, PyErr> {
    use somatize_worker::protocol::InputSource;

    // A store the caller configured is always used: they opted in.
    if let Some(store) = &g.data_store {
        let data_bytes = serde_json::to_vec(x).unwrap_or_default();
        let key = CacheKey::hash_data(&data_bytes);
        let data_ref = store.put(&key, x).map_err(soma_err_to_py)?;
        return Ok(InputSource::Reference { data_ref });
    }

    let size_bytes = serde_json::to_vec(x).map(|v| v.len()).unwrap_or(0);
    if size_bytes >= somatize_core::data::store::INLINE_THRESHOLD_BYTES {
        let data_ref = transport.upload(x).map_err(|e| soma_err_to_py(e.into()))?;
        return Ok(InputSource::Reference { data_ref });
    }

    Ok(InputSource::Inline { value: x.clone() })
}

/// Send a plan to a remote worker via WebSocket.
///
/// Returns `(output, trained_states)` — the states are non-empty after a
/// `Fit`, keyed by bare node id (the worker strips the `__state_` prefix
/// before it answers).
///
/// Compiles against `library` rather than the rebuilt catalog because
/// every caller runs this inside `py.allow_threads` and rebuilding reads
/// Python objects. A study that re-samples an agent's prompt therefore
/// dispatches the prompt the graph was built with — see `rebuild_catalog`.
pub(super) fn dispatch_to_worker(
    g: &PyGraph,
    x: &Value,
    mode: somatize_worker::protocol::ExecutionMode,
    seed: Option<i64>,
) -> Result<(Value, HashMap<String, Value>), PyErr> {
    use somatize_worker::protocol::*;

    let compile_mode = match &mode {
        ExecutionMode::Fit { .. } => CompileMode::NoCache,
        _ => CompileMode::Inference,
    };
    let compile_result =
        somatize_compiler::compile(&g.graph, &g.library, compile_mode, Some(g.cache.as_ref()))
            .map_err(soma_err_to_py)?;

    let (addr, token) = pick_worker(g)?;
    let transport = somatize_worker::WsTransport::new(&addr, token);
    let plan = SerializedPlan {
        protocol_version: PROTOCOL_VERSION,
        plan_id: somatize_core::util::timestamp_id("remote_plan"),
        plan: compile_result.plan,
        input: Some(resolve_transport(g, x, &transport)?),
        // Cloudpickle bytes travel with the plan: they are the only way
        // the worker can reconstruct a Python filter.
        filters: registry::serialized_filters(g),
        mode,
        seed,
        metadata: serde_json::json!({}),
    };

    // The socket, the framing and the size limits belong to the transport.
    // This function decides *which* worker gets *which* filters — policy —
    // and hands the result over.
    let reply = transport
        .send_msg(&CoordinatorToWorker::AssignPlan { plan })
        .map_err(|e| soma_err_to_py(e.into()))?;

    match reply {
        WorkerToCoordinator::PlanResult { result, .. } => match result {
            PlanResult::Success { output, states, .. } => {
                let value = transport
                    .resolve_output(&output)
                    .map_err(|e| soma_err_to_py(e.into()))?;
                Ok((value, states))
            }
            PlanResult::Failed { error, .. } => {
                Err(PyRuntimeError::new_err(format!("remote: {error}")))
            }
        },
        other => Err(PyRuntimeError::new_err(format!(
            "worker answered with {other:?} instead of a plan result"
        ))),
    }
}

/// Stream data to a worker in chunks via WebSocket Binary frames.
/// The worker drives a `StreamRun` — no full materialization.
pub(super) fn dispatch_streamed(
    g: &PyGraph,
    x: &Value,
    chunk_size: usize,
    seed: Option<i64>,
) -> Result<Value, PyErr> {
    use somatize_worker::protocol::*;

    // compile_stream, not compile: the remote chunk loop honours the same
    // contract as the local one, so a diamond or a step is refused here,
    // by name, before anything crosses the wire.
    let compile_result = somatize_compiler::compile_stream(&g.graph, &g.library, chunk_size)
        .map_err(soma_err_to_py)?;

    let (addr, token) = pick_worker(g)?;
    let plan = SerializedPlan {
        protocol_version: PROTOCOL_VERSION,
        plan_id: somatize_core::util::timestamp_id("stream"),
        plan: compile_result.plan,
        input: None, // the input arrives as chunks
        filters: registry::serialized_filters(g),
        mode: ExecutionMode::Forward,
        seed,
        metadata: serde_json::json!({}),
    };

    somatize_worker::WsTransport::new(&addr, token)
        .stream_plan(plan, chunk_value(x, chunk_size))
        .map_err(|e| soma_err_to_py(e.into()))
}

/// Split a `Value` into chunks for streaming, along the first dimension.
fn chunk_value(x: &Value, chunk_size: usize) -> Vec<Value> {
    match x {
        Value::Tensor { values, shape } if !values.is_empty() => {
            let row_size = if shape.len() > 1 {
                shape[1..].iter().product()
            } else {
                1
            };
            let n_rows = shape[0];
            let mut chunks = Vec::new();
            for start in (0..n_rows).step_by(chunk_size) {
                let end = (start + chunk_size).min(n_rows);
                let chunk_vals = values[start * row_size..end * row_size].to_vec();
                let mut chunk_shape = shape.clone();
                chunk_shape[0] = end - start;
                chunks.push(Value::tensor(chunk_vals, chunk_shape));
            }
            chunks
        }
        // Non-tensor or small data: one chunk.
        _ => vec![x.clone()],
    }
}

// ── The `#[pymethods]` bodies ──

/// Register a remote worker for direct connection (mode B).
pub(super) fn add_worker(
    g: &mut PyGraph,
    address: String,
    token: Option<String>,
    tags: Option<Vec<String>>,
) {
    g.workers.push(RemoteWorker {
        address,
        token,
        tags: tags.unwrap_or_default(),
    });
}

/// Known workers, from `add_worker` and from the coordinator.
pub(super) fn workers(g: &PyGraph, py: Python<'_>) -> PyResult<PyObject> {
    let list = PyList::empty(py);

    for worker in &g.workers {
        let dict = PyDict::new(py);
        dict.set_item("address", &worker.address)?;
        dict.set_item("tags", &worker.tags)?;
        dict.set_item("source", "direct")?;
        list.append(dict)?;
    }

    if let Some((url, token)) = &g.coordinator {
        let client = reqwest::blocking::Client::new();
        let mut request = client.get(format!("{url}/workers"));
        if let Some(t) = token {
            request = request.query(&[("token", t.as_str())]);
        }
        if let Ok(resp) = request.send()
            && let Ok(text) = resp.text()
        {
            let json_mod = py.import("json")?;
            if let Ok(parsed) = json_mod.call_method1("loads", (text,))
                && let Ok(items) = parsed.downcast::<PyList>()
            {
                for item in items.iter() {
                    list.append(item)?;
                }
            }
        }
    }

    Ok(list.into_any().unbind())
}

/// Send a `Shutdown` message to one worker.
fn send_shutdown(address: &str, token: Option<&str>, reason: &str) -> PyResult<()> {
    somatize_worker::WsTransport::new(address, token.map(str::to_string))
        .notify(&somatize_worker::protocol::CoordinatorToWorker::Shutdown {
            reason: reason.to_string(),
        })
        .map_err(|e| soma_err_to_py(e.into()))
}

/// Shut down one registered worker, by address.
pub(super) fn shutdown_worker(g: &PyGraph, address: &str, reason: Option<String>) -> PyResult<()> {
    let token = g
        .workers
        .iter()
        .find(|w| w.address == address)
        .and_then(|w| w.token.clone());
    send_shutdown(address, token.as_deref(), &reason.unwrap_or_default())
}

/// Shut down every registered worker. One that will not answer is
/// reported and skipped: the others still have to be told.
pub(super) fn shutdown_workers(g: &PyGraph, reason: Option<String>) -> PyResult<()> {
    let reason = reason.unwrap_or_default();
    for worker in &g.workers {
        if let Err(e) = send_shutdown(&worker.address, worker.token.as_deref(), &reason) {
            eprintln!("Warning: failed to shutdown {}: {e}", worker.address);
        }
    }
    Ok(())
}

/// Set the graph's training strategy.
#[allow(clippy::too_many_arguments)]
pub(super) fn set_strategy(
    g: &mut PyGraph,
    kind: &str,
    num_replicas: Option<usize>,
    num_clients: Option<usize>,
    rounds: Option<usize>,
    aggregation: Option<&str>,
    generations: Option<usize>,
    population_size: Option<usize>,
    partitions: Option<&Bound<'_, pyo3::types::PyAny>>,
) -> PyResult<()> {
    use somatize_core::distributed::{
        ClientSelection, CommunicationProtocol, ExploitStrategy, ExploreStrategy,
        FederatedAggregation, GradientAggregation, Partition, TrainingStrategy,
    };
    use somatize_core::graph::filter::RemoteTarget;

    // A count of zero is never what anyone meant, and it used to travel
    // all the way in: `num_replicas: 0` made the fit and gradient loops
    // run zero times and then handed an empty slice to an aggregator that
    // indexed `[0]`. The panic named none of this.
    for (name, value) in [
        ("num_replicas", num_replicas),
        ("num_clients", num_clients),
        ("population_size", population_size),
        ("rounds", rounds),
        ("generations", generations),
    ] {
        if value == Some(0) {
            return Err(PyValueError::new_err(format!(
                "{name}=0: a strategy needs at least one. Leave it unset \
                 for the default"
            )));
        }
    }

    let strategy = match kind {
        "local" => TrainingStrategy::Local,
        "data_parallel" => TrainingStrategy::DataParallel {
            num_replicas: num_replicas.unwrap_or(1),
            aggregation: match aggregation.unwrap_or("all_reduce") {
                "all_reduce" => GradientAggregation::AllReduce,
                "parameter_server" => GradientAggregation::ParameterServer,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "unknown gradient aggregation '{other}'. \
                         Available: all_reduce, parameter_server"
                    )));
                }
            },
        },
        "federated" => TrainingStrategy::Federated {
            num_clients: num_clients.unwrap_or(2),
            rounds: rounds.unwrap_or(1),
            aggregation: match aggregation.unwrap_or("fed_avg") {
                "fed_avg" => FederatedAggregation::FedAvg,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "unknown federated aggregation '{other}'. Only fed_avg \
                         runs today: fed_prox needs the previous global model \
                         and fed_yogi needs optimizer moments, neither of which \
                         the aggregator is given"
                    )));
                }
            },
            client_selection: ClientSelection::All,
        },
        "model_parallel" => {
            let Some(spec) = partitions else {
                return Err(PyValueError::new_err(
                    "model_parallel needs partitions=[{\"nodes\": [...], \
                     \"tag\": \"gpu0\"}, ...]: it splits the MODEL across \
                     workers, so something has to say which nodes go where",
                ));
            };
            let mut parsed = Vec::new();
            for (i, item) in spec.try_iter()?.enumerate() {
                let item = item?;
                let nodes: Vec<String> = item
                    .get_item("nodes")
                    .map_err(|_| PyValueError::new_err(format!("partition {i} has no \"nodes\"")))?
                    .extract()?;
                // A tag matches any worker carrying it; a worker names
                // exactly one. Both are how `add_worker` registers them.
                let target = if let Ok(tag) = item.get_item("tag") {
                    RemoteTarget::Tag(tag.extract()?)
                } else if let Ok(worker) = item.get_item("worker") {
                    RemoteTarget::WorkerId(worker.extract()?)
                } else {
                    return Err(PyValueError::new_err(format!(
                        "partition {i} names neither a \"tag\" nor a \"worker\", \
                         so its nodes have nowhere to run"
                    )));
                };
                parsed.push(Partition {
                    node_ids: nodes,
                    target,
                });
            }
            TrainingStrategy::ModelParallel {
                partitions: parsed,
                communication: CommunicationProtocol::DataStore,
            }
        }
        "population_based" => TrainingStrategy::PopulationBased {
            population_size: population_size.unwrap_or(4),
            generations: generations.unwrap_or(1),
            exploit: ExploitStrategy::Truncation { fraction: 0.2 },
            explore: ExploreStrategy::Perturbation { factor: 0.2 },
        },
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown strategy '{other}'. Available: local, data_parallel, \
                 federated, model_parallel, population_based"
            )));
        }
    };
    g.graph.set_strategy(strategy);
    Ok(())
}

/// The graph's training strategy, as the string `set_strategy` takes.
pub(super) fn strategy(g: &PyGraph) -> String {
    use somatize_core::distributed::TrainingStrategy as T;
    match g.graph.effective_strategy() {
        T::Local => "local",
        T::DataParallel { .. } => "data_parallel",
        T::Federated { .. } => "federated",
        T::ModelParallel { .. } => "model_parallel",
        T::PopulationBased { .. } => "population_based",
        _ => "unknown",
    }
    .to_string()
}
