//! LocalRunner — executes plans locally using the Executor.
//!
//! This is the default runner. The worker's RemoteRunner delegates here
//! after preparing the environment (deserializing filters, resolving input).

use super::Runner;
use crate::EventBus;
use crate::executor::{Context, Executable, GraphInfo};
use crate::filter_library::FilterLibrary;

use somatize_compiler::ExecutionPlan;
use somatize_core::cache::{CacheKey, CacheStore};
use somatize_core::error::{Result, SomaError};
use somatize_core::event::Event;
use somatize_core::filter::FilterKind;
use somatize_core::util::timestamp_id;
use somatize_core::value::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// Executes plans locally — same logic for local and remote execution.
pub struct LocalRunner;

impl Runner for LocalRunner {
    fn fit(
        &self,
        plan: &ExecutionPlan,
        filters: &FilterLibrary,
        cache: &dyn CacheStore,
        event_bus: &Arc<EventBus>,
        input: &Value,
        y: Option<&Value>,
    ) -> Result<(Value, HashMap<String, Value>)> {
        let node_id_refs = plan.node_ids();
        let node_ids: Vec<String> = node_id_refs.iter().map(|s| s.to_string()).collect();
        let graph_info = GraphInfo::for_linear(&node_id_refs);
        let run_id = timestamp_id("fit");
        let mut outputs: HashMap<String, Value> = HashMap::new();
        let mut trained_states: HashMap<String, Value> = HashMap::new();

        // Set initial input for first node
        if let Some(first) = node_ids.first() {
            outputs.insert(format!("__input_{first}"), input.clone());
        }

        for node_id in &node_ids {
            let filter = filters
                .get(node_id)
                .ok_or_else(|| SomaError::NodeNotFound(node_id.to_string()))?;

            let meta = filter.meta();

            event_bus.emit(Event::NodeStarted {
                run_id: run_id.clone(),
                node_id: node_id.to_string(),
                kind: meta.kind,
            });

            // Resolve input from predecessors
            let preds = graph_info.predecessors(node_id);
            let node_input = match preds.len() {
                0 => input.clone(),
                1 => outputs
                    .get(&preds[0])
                    .cloned()
                    .unwrap_or_else(|| input.clone()),
                _ => {
                    let mut merged = serde_json::Map::new();
                    for pred_id in preds {
                        if let Some(val) = outputs.get(pred_id.as_str()) {
                            let json_val =
                                serde_json::to_value(val).unwrap_or(serde_json::Value::Null);
                            merged.insert(pred_id.clone(), json_val);
                        }
                    }
                    Value::Json(serde_json::Value::Object(merged))
                }
            };

            let start = std::time::Instant::now();

            // Fit trainable filters
            let state = if meta.kind == FilterKind::Trainable {
                let data_hash =
                    CacheKey::hash_data(&serde_json::to_vec(&node_input).unwrap_or_default());
                let state_key = CacheKey::for_state(&filter.config_hash(), &data_hash);

                let s = if let Some(cached) = cache.get(&state_key)? {
                    cached
                } else {
                    let fitted = filter.fit(&node_input, y)?;
                    let _ = cache.put(&state_key, &fitted);
                    fitted
                };
                trained_states.insert(node_id.clone(), s.clone());
                s
            } else {
                filters.get_state(node_id).cloned().unwrap_or(Value::Empty)
            };

            // Forward with state
            let output = filter.forward(&node_input, &state)?;

            event_bus.emit(Event::NodeCompleted {
                run_id: run_id.clone(),
                node_id: node_id.to_string(),
                duration: start.elapsed(),
                output_summary: format!("{output}"),
            });

            outputs.insert(node_id.clone(), output);
        }

        let last_output = outputs.values().last().cloned().unwrap_or(Value::Empty);

        Ok((last_output, trained_states))
    }

    fn forward(
        &self,
        plan: &ExecutionPlan,
        filters: &FilterLibrary,
        cache: &dyn CacheStore,
        event_bus: &Arc<EventBus>,
        input: &Value,
    ) -> Result<Value> {
        let node_ids = plan.node_ids();
        let graph_info = GraphInfo::for_linear(&node_ids);

        let mut ctx =
            Context::new(event_bus.clone(), timestamp_id("forward")).with_graph_info(graph_info);

        // Set input for root nodes
        if let Some(first) = node_ids.first() {
            ctx.set(format!("__input_{first}"), input.clone());
        }
        ctx.set("__input__", input.clone());

        plan.execute(&mut ctx, filters, cache)?;

        // Return last executed node's output
        ctx.execution_order
            .last()
            .and_then(|id| ctx.store.remove(id))
            .and_then(|vv| vv.as_value().cloned())
            .ok_or_else(|| SomaError::Other("no output produced".into()))
    }
}
