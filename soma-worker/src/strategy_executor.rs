//! Strategy executor — orchestrates distributed training across workers.
//!
//! Implements the coordination logic for each TrainingStrategy:
//! - DataParallel: replicate plan, fit with shards, AllReduce gradients
//! - ModelParallel: forward/backward across partitions
//! - Federated: fit rounds, aggregate states
//!
//! Uses WebSocket messages (CoordinatorToWorker) to communicate with workers.

use somatize_compiler::ExecutionPlan;
use somatize_core::error::{Result, SomaError};
use somatize_core::strategy::*;
use somatize_core::value::Value;
use std::collections::HashMap;

use crate::protocol::*;

/// A connection to a worker for strategy orchestration.
pub struct WorkerConnection {
    pub address: String,
    pub token: Option<String>,
    pub tags: Vec<String>,
}

impl WorkerConnection {
    /// Send a CoordinatorToWorker message and wait for a response.
    fn send(&self, msg: &CoordinatorToWorker) -> Result<WorkerToCoordinator> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| SomaError::Other(format!("tokio: {e}")))?;

        rt.block_on(async {
            let url = if let Some(t) = &self.token {
                format!("{}/ws?token={t}", self.address)
            } else {
                format!("{}/ws", self.address)
            };

            let ws_config = {
                let mut c = tokio_tungstenite::tungstenite::protocol::WebSocketConfig::default();
                c.max_message_size = None;
                c.max_frame_size = None;
                c
            };

            let (mut ws, _) =
                tokio_tungstenite::connect_async_with_config(&url, Some(ws_config), false)
                    .await
                    .map_err(|e| SomaError::Other(format!("WS connect: {e}")))?;

            use futures_util::{SinkExt, StreamExt};
            use tokio_tungstenite::tungstenite::Message;

            let json = serde_json::to_string(msg)
                .map_err(|e| SomaError::Other(format!("serialize: {e}")))?;

            ws.send(Message::Text(json.into()))
                .await
                .map_err(|e| SomaError::Other(format!("WS send: {e}")))?;

            // Wait for response
            while let Some(Ok(Message::Text(response))) = ws.next().await {
                if let Ok(result) = serde_json::from_str::<WorkerToCoordinator>(&response) {
                    let _ = ws.close(None).await;
                    return Ok(result);
                }
            }

            Err(SomaError::Other("worker closed without response".into()))
        })
    }

    /// Send a plan for execution and wait for PlanResult.
    fn execute_plan(&self, plan: SerializedPlan) -> Result<PlanResult> {
        let msg = CoordinatorToWorker::AssignPlan { plan };
        match self.send(&msg)? {
            WorkerToCoordinator::PlanResult { result, .. } => Ok(result),
            other => Err(SomaError::Other(format!(
                "expected PlanResult, got: {other:?}"
            ))),
        }
    }

    /// Request trained states from the worker.
    fn get_state(&self, plan_id: &str, node_ids: &[String]) -> Result<HashMap<String, Value>> {
        let msg = CoordinatorToWorker::GetState {
            plan_id: plan_id.to_string(),
            node_ids: node_ids.to_vec(),
        };
        match self.send(&msg)? {
            WorkerToCoordinator::StateResult { states, .. } => Ok(states),
            other => Err(SomaError::Other(format!(
                "expected StateResult, got: {other:?}"
            ))),
        }
    }

    /// Load aggregated states into the worker's filters.
    fn set_state(&self, plan_id: &str, states: &HashMap<String, Value>) -> Result<()> {
        let msg = CoordinatorToWorker::SetState {
            plan_id: plan_id.to_string(),
            states: states.clone(),
        };
        self.send(&msg)?;
        Ok(())
    }

    /// Request gradients from the worker.
    fn get_gradients(&self, plan_id: &str, node_ids: &[String]) -> Result<HashMap<String, Value>> {
        let msg = CoordinatorToWorker::GetGradients {
            plan_id: plan_id.to_string(),
            node_ids: node_ids.to_vec(),
        };
        match self.send(&msg)? {
            WorkerToCoordinator::GradientsResult { gradients, .. } => Ok(gradients),
            other => Err(SomaError::Other(format!(
                "expected GradientsResult, got: {other:?}"
            ))),
        }
    }

    /// Apply aggregated gradients on the worker.
    fn apply_gradients(&self, plan_id: &str, gradients: &HashMap<String, Value>) -> Result<()> {
        let msg = CoordinatorToWorker::ApplyGradients {
            plan_id: plan_id.to_string(),
            gradients: gradients.clone(),
        };
        self.send(&msg)?;
        Ok(())
    }
}

/// Execute a training strategy across workers.
pub fn execute_strategy(
    strategy: &TrainingStrategy,
    plan: &ExecutionPlan,
    filters: &[SerializedFilter],
    workers: &[WorkerConnection],
    input: &Value,
    y: Option<&Value>,
) -> Result<HashMap<String, Value>> {
    match strategy {
        TrainingStrategy::Local => Err(SomaError::Other(
            "Local strategy should not use strategy_executor".into(),
        )),
        TrainingStrategy::DataParallel {
            num_replicas,
            aggregation,
        } => execute_data_parallel(plan, filters, workers, input, y, *num_replicas, aggregation),
        TrainingStrategy::Federated {
            num_clients,
            rounds,
            aggregation,
            client_selection,
        } => execute_federated(
            plan,
            filters,
            workers,
            input,
            y,
            *num_clients,
            *rounds,
            aggregation,
            client_selection,
        ),
        _ => Err(SomaError::Other(format!(
            "Strategy {strategy:?} not yet implemented in strategy_executor"
        ))),
    }
}

/// DataParallel: replicate plan on N workers, each sees a data shard,
/// then AllReduce gradients.
fn execute_data_parallel(
    plan: &ExecutionPlan,
    filters: &[SerializedFilter],
    workers: &[WorkerConnection],
    input: &Value,
    y: Option<&Value>,
    num_replicas: usize,
    aggregation: &GradientAggregation,
) -> Result<HashMap<String, Value>> {
    let n = num_replicas.min(workers.len());
    if n == 0 {
        return Err(SomaError::Other("no workers available".into()));
    }

    let plan_id = somatize_core::util::timestamp_id("dp");
    let node_ids = plan
        .node_ids()
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();

    // Split input into N shards along first dimension
    let shards = shard_value(input, n);

    // Send plan + shard to each worker
    let mut results = Vec::new();
    for (i, worker) in workers.iter().take(n).enumerate() {
        let shard_input = shards.get(i).cloned().unwrap_or(Value::Empty);
        let serialized_plan = SerializedPlan {
            plan_id: format!("{plan_id}_shard_{i}"),
            plan: plan.clone(),
            input: Some(InputSource::Inline { value: shard_input }),
            filters: filters.to_vec(),
            mode: ExecutionMode::Fit { y: y.cloned() },
            metadata: serde_json::json!({}),
        };
        let result = worker.execute_plan(serialized_plan)?;
        results.push((i, result));
    }

    // Aggregate gradients
    match aggregation {
        GradientAggregation::AllReduce => {
            // Collect gradients from all workers
            let mut all_gradients: Vec<HashMap<String, Value>> = Vec::new();
            for (i, worker) in workers.iter().take(n).enumerate() {
                let grads = worker.get_gradients(&format!("{plan_id}_shard_{i}"), &node_ids)?;
                all_gradients.push(grads);
            }

            // Average gradients (simple AllReduce)
            let averaged = average_gradients(&all_gradients);

            // Apply averaged gradients to all workers
            for (i, worker) in workers.iter().take(n).enumerate() {
                worker.apply_gradients(&format!("{plan_id}_shard_{i}"), &averaged)?;
            }
        }
        GradientAggregation::ParameterServer => {
            // Same as AllReduce for now — coordinator is the parameter server
            let mut all_gradients: Vec<HashMap<String, Value>> = Vec::new();
            for (i, worker) in workers.iter().take(n).enumerate() {
                let grads = worker.get_gradients(&format!("{plan_id}_shard_{i}"), &node_ids)?;
                all_gradients.push(grads);
            }
            let averaged = average_gradients(&all_gradients);
            for (i, worker) in workers.iter().take(n).enumerate() {
                worker.apply_gradients(&format!("{plan_id}_shard_{i}"), &averaged)?;
            }
        }
        _ => {
            return Err(SomaError::Other(format!(
                "Aggregation {aggregation:?} not yet implemented"
            )));
        }
    }

    // Collect final states from first worker (all should be identical after sync)
    workers[0].get_state(&format!("{plan_id}_shard_0"), &node_ids)
}

/// Federated: each client trains on its local data, coordinator aggregates states.
#[allow(clippy::too_many_arguments)]
fn execute_federated(
    plan: &ExecutionPlan,
    filters: &[SerializedFilter],
    workers: &[WorkerConnection],
    input: &Value,
    y: Option<&Value>,
    num_clients: usize,
    rounds: usize,
    aggregation: &FederatedAggregation,
    _client_selection: &ClientSelection,
) -> Result<HashMap<String, Value>> {
    let n = num_clients.min(workers.len());
    if n == 0 {
        return Err(SomaError::Other("no workers available".into()));
    }

    let plan_id = somatize_core::util::timestamp_id("fed");
    let node_ids = plan
        .node_ids()
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
    let shards = shard_value(input, n);

    for round in 0..rounds {
        tracing::info!("Federated round {}/{rounds}", round + 1);

        // Each client trains on its local shard
        for (i, worker) in workers.iter().take(n).enumerate() {
            let shard = shards.get(i).cloned().unwrap_or(Value::Empty);
            let serialized_plan = SerializedPlan {
                plan_id: format!("{plan_id}_round_{round}_client_{i}"),
                plan: plan.clone(),
                input: Some(InputSource::Inline { value: shard }),
                filters: filters.to_vec(),
                mode: ExecutionMode::Fit { y: y.cloned() },
                metadata: serde_json::json!({}),
            };
            worker.execute_plan(serialized_plan)?;
        }

        // Collect states from all clients
        let mut all_states: Vec<HashMap<String, Value>> = Vec::new();
        for (i, worker) in workers.iter().take(n).enumerate() {
            let states =
                worker.get_state(&format!("{plan_id}_round_{round}_client_{i}"), &node_ids)?;
            all_states.push(states);
        }

        // Aggregate states
        let aggregated = match aggregation {
            FederatedAggregation::FedAvg => average_states(&all_states),
            FederatedAggregation::FedProx { .. } => average_states(&all_states), // simplified
            _ => average_states(&all_states),
        };

        // Distribute aggregated states back to all clients
        for (i, worker) in workers.iter().take(n).enumerate() {
            worker.set_state(&format!("{plan_id}_round_{round}_client_{i}"), &aggregated)?;
        }
    }

    // Return final aggregated states
    workers[0].get_state(
        &format!("{plan_id}_round_{}_client_0", rounds - 1),
        &node_ids,
    )
}

// ── Helpers ──

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
        _ => {
            // Can't shard — give the same data to everyone
            (0..n).map(|_| value.clone()).collect()
        }
    }
}

/// Average gradients from N workers (simple AllReduce).
/// Each gradient set is HashMap<node_id, Value::Bytes(torch.save bytes)>.
/// For now, returns the first worker's gradients (proper averaging requires
/// deserializing torch tensors in Rust — deferred to when we have a tensor lib).
fn average_gradients(all_gradients: &[HashMap<String, Value>]) -> HashMap<String, Value> {
    // TODO: Proper tensor averaging. For now, use first worker's gradients.
    // Real implementation would: deserialize each gradient tensor, sum, divide by N.
    all_gradients.first().cloned().unwrap_or_default()
}

/// Average states from N workers (FedAvg).
/// Same limitation as average_gradients — proper averaging deferred.
fn average_states(all_states: &[HashMap<String, Value>]) -> HashMap<String, Value> {
    // TODO: Proper tensor averaging for FedAvg.
    all_states.first().cloned().unwrap_or_default()
}
