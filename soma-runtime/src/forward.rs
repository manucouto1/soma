//! Forward execution strategies for [`crate::GraphSession`].
//!
//! Each strategy defines HOW input data flows through a compiled graph:
//! - [`Standard`] — full input at once, with inference caching
//! - [`Stream`] — chunked input through [`crate::StreamExecutor`], respecting StreamMode
//! - [`Batched`] — rows from a [`DataStore`], batch by batch (memory-bounded)

use crate::event_bus::EventBus;
use crate::node_catalog::NodeCatalog;
use crate::runner::Runner;
use somatize_compiler::{CompileMode, CompileResult, compile, compile_stream};
use somatize_core::cache::CacheStore;
use somatize_core::error::{Result, SomaError};
use somatize_core::graph::Graph;
use somatize_core::store::{DataRef, DataStore};
use somatize_core::value::Value;
use std::sync::Arc;

/// How a forward pass feeds data through the compiled graph.
pub trait ForwardStrategy {
    /// Execute a forward pass, returning the final output.
    fn forward(
        &self,
        graph: &Graph,
        library: &NodeCatalog,
        cache: &dyn CacheStore,
        event_bus: &Arc<EventBus>,
        data_store: Option<&Arc<dyn DataStore>>,
        x: &Value,
    ) -> Result<Value>;
}

/// Standard forward: full input at once with inference caching.
pub struct Standard;

impl ForwardStrategy for Standard {
    fn forward(
        &self,
        graph: &Graph,
        library: &NodeCatalog,
        cache: &dyn CacheStore,
        event_bus: &Arc<EventBus>,
        _data_store: Option<&Arc<dyn DataStore>>,
        x: &Value,
    ) -> Result<Value> {
        let CompileResult { plan, .. } =
            compile(graph, library, CompileMode::Inference, Some(cache))?;

        let runner = crate::runner::LocalRunner;
        runner.forward(&plan, library, cache, event_bus, x)
    }
}

/// Streaming forward: chunk input and process through StreamExecutor.
/// Each filter's StreamMode (FixedState/Evolving/Barrier) defines its per-chunk contract.
pub struct Stream {
    pub chunk_size: usize,
}

impl ForwardStrategy for Stream {
    fn forward(
        &self,
        graph: &Graph,
        library: &NodeCatalog,
        cache: &dyn CacheStore,
        event_bus: &Arc<EventBus>,
        _data_store: Option<&Arc<dyn DataStore>>,
        x: &Value,
    ) -> Result<Value> {
        let CompileResult { plan, .. } = compile_stream(graph, library, self.chunk_size)?;

        let runner = crate::runner::LocalRunner;
        runner.forward(&plan, library, cache, event_bus, x)
    }
}

/// Batched forward: read rows from a DataStore in fixed-size batches.
/// Keeps memory bounded — only one batch is materialized at a time.
pub struct Batched<'a> {
    pub data_ref: &'a DataRef,
    pub batch_size: usize,
}

impl ForwardStrategy for Batched<'_> {
    fn forward(
        &self,
        graph: &Graph,
        library: &NodeCatalog,
        cache: &dyn CacheStore,
        event_bus: &Arc<EventBus>,
        data_store: Option<&Arc<dyn DataStore>>,
        _x: &Value,
    ) -> Result<Value> {
        let store = data_store.ok_or_else(|| SomaError::Execution {
            node_id: "session".into(),
            message: "Batched strategy requires a data store (use with_data_store)".into(),
        })?;

        let meta = store.meta(self.data_ref)?;
        let total_rows = meta.total_rows;
        if total_rows == 0 {
            return Ok(Value::Empty);
        }

        // Compile once, reuse for each batch.
        let CompileResult { plan, .. } =
            compile(graph, library, CompileMode::Inference, Some(cache))?;
        let runner = crate::runner::LocalRunner;

        let mut all_values: Vec<f64> = Vec::new();
        let mut result_shape: Option<Vec<usize>> = None;
        let mut rows_processed = 0;

        while rows_processed < total_rows {
            let batch_len = self.batch_size.min(total_rows - rows_processed);
            let batch = store.get_rows(self.data_ref, rows_processed, batch_len)?;
            let output = runner.forward(&plan, library, cache, event_bus, &batch)?;

            if let Value::Tensor { values, shape } = &output {
                if result_shape.is_none() {
                    result_shape = Some(shape.clone());
                }
                all_values.extend_from_slice(values.as_slice());
            } else {
                return Ok(output);
            }

            rows_processed += batch_len;
        }

        match result_shape {
            Some(mut shape) => {
                shape[0] = total_rows;
                Ok(Value::tensor(all_values, shape))
            }
            None => Ok(Value::Empty),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::MemoryCache;
    use crate::node_catalog::NodeCatalog;
    use somatize_core::cache::CacheKey;
    use somatize_core::error::Result as SomaResult;
    use somatize_core::filter::{Distribution, Filter, FilterKind, FilterMeta, StreamMode};
    use somatize_core::graph::{Graph, Node};

    struct DoublerFilter;
    impl Filter for DoublerFilter {
        fn config_hash(&self) -> CacheKey {
            CacheKey::from_parts(&[b"Doubler"])
        }
        fn fit(&self, _x: &Value, _y: Option<&Value>) -> SomaResult<Value> {
            Ok(Value::Empty)
        }
        fn forward(&self, x: &Value, _state: &Value) -> SomaResult<Value> {
            match x {
                Value::Tensor { values, shape } => {
                    let doubled: Vec<f64> = values.iter().map(|v| v * 2.0).collect();
                    Ok(Value::tensor(doubled, shape.clone()))
                }
                _ => Ok(x.clone()),
            }
        }
        fn meta(&self) -> FilterMeta {
            FilterMeta {
                name: "Doubler".into(),
                kind: FilterKind::Stateless,
                cacheable: false,
                differentiable: false,
                deterministic: true,
                stream_mode: StreamMode::FixedState,
                distribution: Distribution::Local,
                input_schema: None,
                output_schema: None,
            }
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    fn make_session() -> (Graph, NodeCatalog, Arc<dyn CacheStore>, Arc<EventBus>) {
        let mut graph = Graph::new();
        graph.nodes.push(Node::new("double", "Double", "double"));

        let mut library = NodeCatalog::new();
        library.register("double", Box::new(DoublerFilter));

        let cache: Arc<dyn CacheStore> = Arc::new(MemoryCache::default());
        let bus = Arc::new(EventBus::new(64));
        (graph, library, cache, bus)
    }

    #[test]
    fn standard_forward() {
        let (graph, library, cache, bus) = make_session();
        let input = Value::tensor(vec![1.0, 2.0, 3.0], vec![3]);

        let result = Standard
            .forward(&graph, &library, cache.as_ref(), &bus, None, &input)
            .unwrap();
        let (data, _) = result.as_tensor().unwrap();
        assert_eq!(data, &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn stream_forward() {
        let (graph, library, cache, bus) = make_session();
        let input = Value::tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6]);

        let result = Stream { chunk_size: 2 }
            .forward(&graph, &library, cache.as_ref(), &bus, None, &input)
            .unwrap();
        let (data, shape) = result.as_tensor().unwrap();
        assert_eq!(data, &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
        assert_eq!(shape, &[6]);
    }

    #[test]
    fn stream_matches_standard() {
        let (graph, library, cache, bus) = make_session();
        let input = Value::tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4]);

        let standard = Standard
            .forward(&graph, &library, cache.as_ref(), &bus, None, &input)
            .unwrap();
        let streamed = Stream { chunk_size: 2 }
            .forward(&graph, &library, cache.as_ref(), &bus, None, &input)
            .unwrap();
        assert_eq!(standard, streamed);
    }
}
