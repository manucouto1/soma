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
use somatize_core::filter::{Filter, FilterKind};
use somatize_core::util::timestamp_id;
use somatize_core::value::Value;
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

/// Resolve every node_id in ``node_ids`` to its filter instance.
/// Returns ``None`` if any id is missing from the library — in that case
/// the runner falls back to sequential execution with a NodeNotFound error.
fn collect_peers(
    filters: &FilterLibrary,
    node_ids: &[String],
) -> Option<Vec<(String, Arc<dyn Filter>)>> {
    node_ids
        .iter()
        .map(|id| filters.get(id).map(|f| (id.clone(), f)))
        .collect()
}

/// Executes plans locally — same logic for local and remote execution.
pub struct LocalRunner;

impl LocalRunner {
    /// Fit a Sequence plan, handling Composite steps as blocks.
    #[allow(clippy::too_many_arguments)]
    fn fit_sequence(
        &self,
        steps: &[ExecutionPlan],
        filters: &FilterLibrary,
        cache: &dyn CacheStore,
        event_bus: &Arc<EventBus>,
        run_id: &str,
        input: &Value,
        y: Option<&Value>,
    ) -> Result<(Value, HashMap<String, Value>)> {
        let mut current_input = input.clone();
        let mut all_outputs = HashMap::new();

        for step in steps {
            let handled = if let ExecutionPlan::Composite { node_ids } = step
                && let Some(peers) = collect_peers(filters, node_ids)
                && let Some(filter) = filters.get(&node_ids[0])
                && let Some(result) = filter.composite_fit(&peers, &current_input, y)
            {
                let (output, states) = result?;
                for (id, state) in &states {
                    all_outputs.insert(format!("__state_{id}"), state.clone());
                }
                if let Some(last_id) = node_ids.last() {
                    all_outputs.insert(last_id.clone(), output.clone());
                }
                current_input = output;
                true
            } else {
                false
            };

            if !handled {
                let sub_result = <Self as Runner>::fit(
                    self,
                    step,
                    filters,
                    cache,
                    event_bus,
                    run_id,
                    &current_input,
                    y,
                )?;
                current_input = sub_result.0;
                all_outputs.extend(sub_result.1);
            }
        }

        Ok((current_input, all_outputs))
    }
}

impl Runner for LocalRunner {
    fn fit(
        &self,
        plan: &ExecutionPlan,
        filters: &FilterLibrary,
        cache: &dyn CacheStore,
        event_bus: &Arc<EventBus>,
        run_id: &str,
        input: &Value,
        y: Option<&Value>,
    ) -> Result<(Value, HashMap<String, Value>)> {
        // Handle Composite plan: delegate to composite_fit on the first filter
        if let ExecutionPlan::Composite { node_ids } = plan
            && let Some(peers) = collect_peers(filters, node_ids)
            && let Some(filter) = filters.get(&node_ids[0])
            && let Some(result) = filter.composite_fit(&peers, input, y)
        {
            return result;
            // Fallback: treat as sequential if composite_fit not supported
        }

        // Handle Sequence that may contain Composite steps
        if let ExecutionPlan::Sequence(steps) = plan {
            return self.fit_sequence(steps, filters, cache, event_bus, run_id, input, y);
        }

        // Single node or other plan types: sequential fit
        let node_id_refs = plan.node_ids();
        let node_ids: Vec<String> = node_id_refs.iter().map(|s| s.to_string()).collect();
        let graph_info = GraphInfo::for_linear(&node_id_refs);
        let mut outputs: HashMap<String, Value> = HashMap::new();
        let mut trained_states: HashMap<String, Value> = HashMap::new();

        if let Some(first) = node_ids.first() {
            outputs.insert(format!("__input_{first}"), input.clone());
        }

        for node_id in &node_ids {
            let filter = filters
                .get(node_id)
                .ok_or_else(|| SomaError::NodeNotFound(node_id.to_string()))?;

            let meta = filter.meta();

            event_bus.emit(Event::NodeStarted {
                run_id: run_id.to_string(),
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
                            let json_val = val.to_plain_json();
                            merged.insert(pred_id.clone(), json_val);
                        }
                    }
                    Value::json(serde_json::Value::Object(merged))
                }
            };

            let start = std::time::Instant::now();

            // Fit trainable filters. Use Cow so the non-trainable branch
            // can borrow the existing state (via the StateStore Arc)
            // without cloning potentially huge tensors on every forward
            // call.
            let library_state: Option<Arc<Value>> = if meta.kind == FilterKind::Trainable {
                None
            } else {
                filters.get_state(node_id)
            };
            let empty_state = Value::Empty;
            let state: Cow<Value> = if meta.kind == FilterKind::Trainable {
                // Labels are part of the key: the same features trained
                // against different labels must not collide.
                let state_key = Some(CacheKey::for_state(
                    &filter.config_hash(),
                    &CacheKey::for_value(&node_input),
                    y.map(CacheKey::for_value).as_ref(),
                ));

                let cached_state = match &state_key {
                    Some(key) => cache.get(key)?,
                    None => None,
                };
                let s = if let Some(cached) = cached_state {
                    cached
                } else {
                    let fitted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        filter.fit(&node_input, y)
                    }))
                    .map_err(|panic| {
                        let msg = panic
                            .downcast_ref::<String>()
                            .map(|s| s.as_str())
                            .or_else(|| panic.downcast_ref::<&str>().copied())
                            .unwrap_or("unknown panic");
                        SomaError::Execution {
                            node_id: node_id.clone(),
                            message: format!("fit panicked: {msg}"),
                        }
                    })??;
                    if let Some(key) = &state_key {
                        let origin = somatize_core::cache::Origin::Computed {
                            node_id: node_id.clone(),
                            run_id: run_id.to_string(),
                        };
                        let _ = cache.put_computed(key, &fitted, &origin, start.elapsed(), true);
                    }
                    fitted
                };
                trained_states.insert(node_id.clone(), s.clone());
                Cow::Owned(s)
            } else {
                match library_state.as_deref() {
                    Some(v) => Cow::Borrowed(v),
                    None => Cow::Borrowed(&empty_state),
                }
            };

            // Forward with state (catch panics)
            let output = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                filter.forward(&node_input, &state)
            }))
            .map_err(|panic| {
                let msg = panic
                    .downcast_ref::<String>()
                    .map(|s| s.as_str())
                    .or_else(|| panic.downcast_ref::<&str>().copied())
                    .unwrap_or("unknown panic");
                SomaError::Execution {
                    node_id: node_id.clone(),
                    message: format!("forward panicked: {msg}"),
                }
            })??;

            event_bus.emit(Event::NodeCompleted {
                run_id: run_id.to_string(),
                node_id: node_id.to_string(),
                duration: start.elapsed(),
                output_summary: format!("{output}"),
            });

            outputs.insert(node_id.clone(), output);
        }

        let last_output = outputs.values().last().cloned().unwrap_or(Value::Empty);

        // Forward outputs keyed by node_id (for GraphSession inspection).
        // Trained states added with __state_ prefix (for Worker to extract).
        for (id, state) in &trained_states {
            outputs.insert(format!("__state_{id}"), state.clone());
        }
        Ok((last_output, outputs))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::MemoryCache;
    use somatize_core::filter::{FilterMeta, StreamMode};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Counts fit() invocations — probe for state-cache key tests.
    struct CountingFitFilter {
        fits: Arc<AtomicUsize>,
    }

    impl Filter for CountingFitFilter {
        fn config_hash(&self) -> CacheKey {
            CacheKey::from_parts(&[b"CountingFit"])
        }
        fn fit(&self, _x: &Value, y: Option<&Value>) -> Result<Value> {
            self.fits.fetch_add(1, Ordering::SeqCst);
            Ok(y.cloned().unwrap_or(Value::Empty))
        }
        fn forward(&self, x: &Value, _state: &Value) -> Result<Value> {
            Ok(x.clone())
        }
        fn meta(&self) -> FilterMeta {
            FilterMeta {
                name: "CountingFit".into(),
                kind: FilterKind::Trainable,
                cacheable: true,
                differentiable: false,
                deterministic: true,
                stream_mode: StreamMode::FixedState,
                distribution: somatize_core::filter::Distribution::Local,
                input_schema: None,
                output_schema: None,
            }
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[test]
    fn state_cache_key_is_sensitive_to_labels() {
        let fits = Arc::new(AtomicUsize::new(0));
        let mut filters = FilterLibrary::new();
        filters.register("clf", Box::new(CountingFitFilter { fits: fits.clone() }));

        let cache = MemoryCache::default();
        let bus = Arc::new(EventBus::new(64));
        let plan = ExecutionPlan::Execute {
            node_id: "clf".into(),
        };
        let runner = LocalRunner;
        let x = Value::tensor(vec![1.0, 2.0], vec![2]);
        let y_a = Value::tensor(vec![0.0, 1.0], vec![2]);
        let y_b = Value::tensor(vec![1.0, 0.0], vec![2]);

        runner
            .fit(&plan, &filters, &cache, &bus, "test_run", &x, Some(&y_a))
            .unwrap();
        assert_eq!(fits.load(Ordering::SeqCst), 1);

        // Same features, same labels → cached state, no refit.
        runner
            .fit(&plan, &filters, &cache, &bus, "test_run", &x, Some(&y_a))
            .unwrap();
        assert_eq!(fits.load(Ordering::SeqCst), 1);

        // Same features, DIFFERENT labels → must refit.
        runner
            .fit(&plan, &filters, &cache, &bus, "test_run", &x, Some(&y_b))
            .unwrap();
        assert_eq!(
            fits.load(Ordering::SeqCst),
            2,
            "different labels must not reuse the cached state"
        );

        // Unsupervised (no labels) → distinct from both supervised keys.
        runner
            .fit(&plan, &filters, &cache, &bus, "test_run", &x, None)
            .unwrap();
        assert_eq!(fits.load(Ordering::SeqCst), 3);
    }
}
