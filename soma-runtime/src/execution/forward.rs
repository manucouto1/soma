//! Forward execution strategies for [`crate::GraphSession`].
//!
//! Each strategy defines HOW input data flows through a compiled graph:
//! - [`Standard`] — full input at once, with inference caching
//! - [`Stream`] — chunked input through [`crate::StreamRun`], respecting StreamMode
//! - [`Batched`] — rows from a [`DataStore`], batch by batch (memory-bounded)

use crate::execution::runner::{LocalRunner, RunContext, Runner};
use somatize_compiler::{CompileMode, CompileResult, compile, compile_stream};
use somatize_core::data::store::{DataRef, DataStore};
use somatize_core::data::value::Value;
use somatize_core::error::Result;
use somatize_core::graph::Graph;
use std::sync::Arc;

/// How a forward pass feeds data through the compiled graph.
///
/// The context is the run's, built once by the caller — not a second
/// struct describing the same four things. `ForwardEnv` used to sit here
/// with `catalog`/`cache`/`event_bus`/`driver`, which is [`RunContext`]
/// minus the run id and the topology, and a reader had to hold both.
pub trait ForwardStrategy {
    /// Execute a forward pass, returning the final output.
    fn forward(&self, graph: &Graph, ctx: &RunContext<'_>, x: &Value) -> Result<Value>;
}

/// Full input at once, with inference caching.
pub struct Standard;

impl ForwardStrategy for Standard {
    fn forward(&self, graph: &Graph, ctx: &RunContext<'_>, x: &Value) -> Result<Value> {
        let CompileResult { plan, .. } =
            compile(graph, ctx.catalog, CompileMode::Inference, Some(ctx.cache))?;
        LocalRunner.forward(&plan, ctx, x)
    }
}

/// Chunked input through [`crate::StreamRun`], respecting each
/// filter's `StreamMode`.
pub struct Stream {
    /// Rows per chunk fed through the stream plan.
    pub chunk_size: usize,
}

impl ForwardStrategy for Stream {
    fn forward(&self, graph: &Graph, ctx: &RunContext<'_>, x: &Value) -> Result<Value> {
        let CompileResult { plan, .. } = compile_stream(graph, ctx.catalog, self.chunk_size)?;
        LocalRunner.forward(&plan, ctx, x)
    }
}

/// Batched forward: read rows from a DataStore in fixed-size batches.
/// Keeps memory bounded — only one batch is materialized at a time.
pub struct Batched<'a> {
    /// Where the rows come from.
    ///
    /// A field, not something looked up in a shared environment: this is
    /// the only strategy that reads rows, and asking for one and finding
    /// none used to be a runtime error ("requires a data store") that
    /// nothing in the workspace could reach. Now it will not compile.
    pub store: &'a Arc<dyn DataStore>,
    /// Which dataset to read.
    pub data_ref: &'a DataRef,
    /// Rows materialized per batch — the memory bound.
    pub batch_size: usize,
}

impl ForwardStrategy for Batched<'_> {
    fn forward(&self, graph: &Graph, ctx: &RunContext<'_>, _x: &Value) -> Result<Value> {
        let store = self.store;
        let meta = store.meta(self.data_ref)?;
        let total_rows = meta.total_rows;
        if total_rows == 0 {
            return Ok(Value::Empty);
        }

        // Compile once, reuse for each batch.
        let CompileResult { plan, .. } =
            compile(graph, ctx.catalog, CompileMode::Inference, Some(ctx.cache))?;

        let mut all_values: Vec<f64> = Vec::new();
        let mut result_shape: Option<Vec<usize>> = None;
        let mut rows_processed = 0;

        while rows_processed < total_rows {
            let batch_len = self.batch_size.min(total_rows - rows_processed);
            let batch = store.get_rows(self.data_ref, rows_processed, batch_len)?;
            let output = LocalRunner.forward(&plan, ctx, &batch)?;

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
    use crate::execution::node_catalog::NodeCatalog;
    use crate::tracking::event_bus::EventBus;
    use somatize_core::cache::CacheKey;
    use somatize_core::cache::CacheStore;
    use somatize_core::error::Result as SomaResult;
    use somatize_core::graph::filter::{Distribution, Filter, FilterKind, FilterMeta, StreamMode};
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
    }

    fn make_session() -> (Graph, NodeCatalog, Arc<dyn CacheStore>, Arc<EventBus>) {
        let mut graph = Graph::new();
        graph.nodes.push(Node::new("double", "Double", "double"));

        let mut catalog = NodeCatalog::new();
        catalog.register("double", Box::new(DoublerFilter));

        let cache: Arc<dyn CacheStore> = Arc::new(MemoryCache::default());
        let bus = Arc::new(EventBus::new(64));
        (graph, catalog, cache, bus)
    }

    fn ctx<'a>(
        catalog: &'a NodeCatalog,
        cache: &'a dyn CacheStore,
        events: &'a Arc<EventBus>,
        graph: &Graph,
    ) -> RunContext<'a> {
        RunContext::new(
            catalog,
            cache,
            events,
            "test_forward",
            crate::execution::executor::GraphInfo::from_graph(graph),
        )
    }

    #[test]
    fn standard_forward() {
        let (graph, catalog, cache, bus) = make_session();
        let input = Value::tensor(vec![1.0, 2.0, 3.0], vec![3]);

        let result = Standard
            .forward(&graph, &ctx(&catalog, cache.as_ref(), &bus, &graph), &input)
            .unwrap();
        let (data, _) = result.as_tensor().unwrap();
        assert_eq!(data, &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn stream_forward() {
        let (graph, catalog, cache, bus) = make_session();
        let input = Value::tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6]);

        let result = Stream { chunk_size: 2 }
            .forward(&graph, &ctx(&catalog, cache.as_ref(), &bus, &graph), &input)
            .unwrap();
        let (data, shape) = result.as_tensor().unwrap();
        assert_eq!(data, &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
        assert_eq!(shape, &[6]);
    }

    #[test]
    fn stream_matches_standard() {
        let (graph, catalog, cache, bus) = make_session();
        let input = Value::tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4]);

        let standard = Standard
            .forward(&graph, &ctx(&catalog, cache.as_ref(), &bus, &graph), &input)
            .unwrap();
        let streamed = Stream { chunk_size: 2 }
            .forward(&graph, &ctx(&catalog, cache.as_ref(), &bus, &graph), &input)
            .unwrap();
        assert_eq!(standard, streamed);
    }
}
