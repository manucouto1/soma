use crate::cache::MemoryCache;
use crate::event_bus::EventBus;
use soma_core::cache::{CacheKey, CacheStore};
use soma_core::error::{Result, SomaError};
use soma_core::event::Event;
use soma_core::filter::Filter;
use soma_core::value::Value;
use std::sync::Arc;
use std::time::Instant;

/// A pipeline of filters that can be fitted and used for prediction.
///
/// This is the main user-facing API for sequential filter pipelines.
pub struct Pipeline {
    /// Named filters in order.
    filters: Vec<(String, Box<dyn Filter>)>,
    /// Trained states, keyed by filter name.
    states: Vec<(String, Option<Value>)>,
    /// Cache store.
    cache: Arc<dyn CacheStore>,
    /// Event bus.
    event_bus: Arc<EventBus>,
    /// Whether the pipeline has been fitted.
    fitted: bool,
}

impl Pipeline {
    /// Create a new pipeline from a list of named filters.
    pub fn new(filters: Vec<(String, Box<dyn Filter>)>) -> Self {
        let state_slots: Vec<(String, Option<Value>)> =
            filters.iter().map(|(name, _)| (name.clone(), None)).collect();
        Self {
            filters,
            states: state_slots,
            cache: Arc::new(MemoryCache::default()),
            event_bus: Arc::new(EventBus::default()),
            fitted: false,
        }
    }

    /// Set a custom cache store.
    pub fn with_cache(mut self, cache: Arc<dyn CacheStore>) -> Self {
        self.cache = cache;
        self
    }

    /// Set a custom event bus.
    pub fn with_event_bus(mut self, bus: Arc<EventBus>) -> Self {
        self.event_bus = bus;
        self
    }

    /// Subscribe to pipeline events.
    pub fn subscribe(&self) -> tokio::sync::broadcast::Receiver<Event> {
        self.event_bus.subscribe()
    }

    /// Fit the pipeline: train each filter sequentially.
    ///
    /// Each filter's state is cached. If the filter config and training data
    /// haven't changed, the cached state is reused.
    pub fn fit(&mut self, x: &Value, y: Option<&Value>) -> Result<()> {
        let run_id = format!("fit_{}", timestamp_id());

        self.event_bus.emit(Event::RunStarted {
            run_id: run_id.clone(),
            plan_summary: soma_core::event::PlanSummary {
                total_nodes: self.filters.len(),
                cached_nodes: 0,
                parallel_branches: 0,
            },
        });

        let start = Instant::now();
        let mut current_x = x.clone();

        for (i, (name, filter)) in self.filters.iter().enumerate() {
            let fit_start = Instant::now();

            self.event_bus.emit(Event::NodeStarted {
                run_id: run_id.clone(),
                node_id: name.clone(),
                kind: filter.meta().kind,
            });

            // Check state cache
            let data_hash = CacheKey::hash_data(
                &serde_json::to_vec(&current_x).unwrap_or_default(),
            );
            let state_key = CacheKey::for_state(&filter.config_hash(), &data_hash);

            let state = if let Some(cached_state) = self.cache.get(&state_key)? {
                self.event_bus.emit(Event::NodeCacheHit {
                    run_id: run_id.clone(),
                    node_id: name.clone(),
                    key: state_key,
                    tier: soma_core::cache::CacheTier::Memory,
                    load_time: fit_start.elapsed(),
                });
                cached_state
            } else {
                let state = filter.fit(&current_x, y)?;
                self.cache.put(&state_key, &state)?;
                state
            };

            // Forward to get input for next filter (detached - training mode)
            current_x = filter.forward(&current_x, &state)?;

            self.states[i].1 = Some(state);

            self.event_bus.emit(Event::NodeCompleted {
                run_id: run_id.clone(),
                node_id: name.clone(),
                duration: fit_start.elapsed(),
                output_summary: format!("{current_x}"),
            });
        }

        self.fitted = true;

        self.event_bus.emit(Event::RunCompleted {
            run_id,
            duration: start.elapsed(),
        });

        Ok(())
    }

    /// Predict: forward data through all fitted filters.
    /// Uses caching for outputs.
    pub fn predict(&self, x: &Value) -> Result<Value> {
        if !self.fitted {
            return Err(SomaError::Execution {
                node_id: "pipeline".into(),
                message: "pipeline must be fitted before predict".into(),
            });
        }

        let run_id = format!("predict_{}", timestamp_id());
        let start = Instant::now();

        self.event_bus.emit(Event::RunStarted {
            run_id: run_id.clone(),
            plan_summary: soma_core::event::PlanSummary {
                total_nodes: self.filters.len(),
                cached_nodes: 0,
                parallel_branches: 0,
            },
        });

        let mut current = x.clone();

        for (name, filter) in &self.filters {
            let state = self
                .states
                .iter()
                .find(|(n, _)| n == name)
                .and_then(|(_, s)| s.as_ref())
                .ok_or_else(|| SomaError::Execution {
                    node_id: name.clone(),
                    message: "missing state for filter".into(),
                })?;

            let node_start = Instant::now();
            self.event_bus.emit(Event::NodeStarted {
                run_id: run_id.clone(),
                node_id: name.clone(),
                kind: filter.meta().kind,
            });

            current = filter.forward(&current, state)?;

            self.event_bus.emit(Event::NodeCompleted {
                run_id: run_id.clone(),
                node_id: name.clone(),
                duration: node_start.elapsed(),
                output_summary: format!("{current}"),
            });
        }

        self.event_bus.emit(Event::RunCompleted {
            run_id,
            duration: start.elapsed(),
        });

        Ok(current)
    }

    /// Whether the pipeline has been fitted.
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// Get the search space aggregated from all filters.
    pub fn filter_names(&self) -> Vec<&str> {
        self.filters.iter().map(|(n, _)| n.as_str()).collect()
    }
}

fn timestamp_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{nanos:x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use soma_core::filter::{FilterKind, FilterMeta, StreamMode};

    // ── Test filters ──

    struct ScaleFilter {
        scale: f64,
    }

    impl Filter for ScaleFilter {
        fn config_hash(&self) -> CacheKey {
            CacheKey::from_parts(&[b"Scale", &self.scale.to_le_bytes()])
        }

        fn fit(&self, x: &Value, _y: Option<&Value>) -> Result<Value> {
            let (data, _) = x.as_tensor().ok_or(SomaError::Other("need tensor".into()))?;
            let mean = data.iter().sum::<f64>() / data.len() as f64;
            Ok(Value::json(serde_json::json!({ "mean": mean })))
        }

        fn forward(&self, x: &Value, state: &Value) -> Result<Value> {
            let (data, shape) = x.as_tensor().ok_or(SomaError::Other("need tensor".into()))?;
            let mean = state.as_json().and_then(|j| j["mean"].as_f64()).unwrap_or(0.0);
            let result: Vec<f64> = data.iter().map(|v| (v - mean) * self.scale).collect();
            Ok(Value::tensor(result, shape.to_vec()))
        }

        fn meta(&self) -> FilterMeta {
            FilterMeta {
                name: "Scale".into(),
                kind: FilterKind::Trainable,
                cacheable: true,
                differentiable: true,
                stream_mode: StreamMode::FixedState,
            }
        }
    }

    struct SquareFilter;

    impl Filter for SquareFilter {
        fn config_hash(&self) -> CacheKey {
            CacheKey::from_parts(&[b"Square"])
        }

        fn fit(&self, _x: &Value, _y: Option<&Value>) -> Result<Value> {
            Ok(Value::Empty)
        }

        fn forward(&self, x: &Value, _state: &Value) -> Result<Value> {
            let (data, shape) = x.as_tensor().ok_or(SomaError::Other("need tensor".into()))?;
            let result: Vec<f64> = data.iter().map(|v| v * v).collect();
            Ok(Value::tensor(result, shape.to_vec()))
        }

        fn meta(&self) -> FilterMeta {
            FilterMeta {
                name: "Square".into(),
                kind: FilterKind::Stateless,
                cacheable: true,
                differentiable: true,
                stream_mode: StreamMode::FixedState,
            }
        }
    }

    #[test]
    fn pipeline_fit_and_predict() {
        let mut pipeline = Pipeline::new(vec![
            ("scaler".into(), Box::new(ScaleFilter { scale: 2.0 })),
            ("square".into(), Box::new(SquareFilter)),
        ]);

        let train = Value::tensor(vec![2.0, 4.0, 6.0], vec![3]);
        pipeline.fit(&train, None).unwrap();
        assert!(pipeline.is_fitted());

        let test = Value::tensor(vec![3.0, 5.0], vec![2]);
        let result = pipeline.predict(&test).unwrap();

        // mean of [2,4,6] = 4. scale=2
        // scaler: (3-4)*2=-2, (5-4)*2=2
        // square: (-2)^2=4, (2)^2=4
        let (data, _) = result.as_tensor().unwrap();
        assert_eq!(data, &[4.0, 4.0]);
    }

    #[test]
    fn predict_before_fit_errors() {
        let pipeline = Pipeline::new(vec![
            ("scaler".into(), Box::new(ScaleFilter { scale: 1.0 })),
        ]);

        let result = pipeline.predict(&Value::tensor(vec![1.0], vec![1]));
        assert!(result.is_err());
    }

    #[test]
    fn pipeline_emits_events() {
        let bus = Arc::new(EventBus::new(64));
        let mut rx = bus.subscribe();

        let mut pipeline = Pipeline::new(vec![
            ("scaler".into(), Box::new(ScaleFilter { scale: 1.0 })),
        ])
        .with_event_bus(bus);

        let data = Value::tensor(vec![1.0, 2.0, 3.0], vec![3]);
        pipeline.fit(&data, None).unwrap();

        // Should have: RunStarted, NodeStarted, NodeCompleted, RunCompleted
        let mut events = Vec::new();
        while let Ok(e) = rx.try_recv() {
            events.push(e);
        }

        assert!(events.iter().any(|e| matches!(e, Event::RunStarted { .. })));
        assert!(events.iter().any(|e| matches!(e, Event::NodeStarted { .. })));
        assert!(events.iter().any(|e| matches!(e, Event::NodeCompleted { .. })));
        assert!(events.iter().any(|e| matches!(e, Event::RunCompleted { .. })));
    }

    #[test]
    fn pipeline_caches_state() {
        let cache = Arc::new(MemoryCache::new(1024 * 1024));

        let mut pipeline = Pipeline::new(vec![
            ("scaler".into(), Box::new(ScaleFilter { scale: 1.0 })),
        ])
        .with_cache(cache.clone());

        let data = Value::tensor(vec![1.0, 2.0, 3.0], vec![3]);
        pipeline.fit(&data, None).unwrap();

        // Cache should have at least the state entry
        assert!(!cache.is_empty());
    }

    #[test]
    fn filter_names() {
        let pipeline = Pipeline::new(vec![
            ("scaler".into(), Box::new(ScaleFilter { scale: 1.0 })),
            ("square".into(), Box::new(SquareFilter)),
        ]);

        assert_eq!(pipeline.filter_names(), vec!["scaler", "square"]);
    }
}
