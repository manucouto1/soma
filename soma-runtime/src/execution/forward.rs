//! Forward execution strategies for [`crate::GraphSession`].
//!
//! Each strategy defines HOW input data flows through a compiled graph:
//! - [`Standard`] — full input at once, with inference caching
//! - [`Stream`] — chunked input through [`crate::StreamRun`], respecting StreamMode
//! - [`Batched`] — rows from a [`DataStore`], batch by batch (memory-bounded)

use crate::execution::node_catalog::NodeCatalog;
use crate::execution::runner::{RunContext, Runner};
use crate::tracking::event_bus::EventBus;
use somatize_compiler::{CompileMode, CompileResult, compile, compile_stream};
use somatize_core::cache::CacheStore;
use somatize_core::data::store::{DataRef, DataStore};
use somatize_core::data::value::Value;
use somatize_core::error::{Result, SomaError};
use somatize_core::graph::Graph;
use std::sync::Arc;

/// What a forward pass runs against, besides the graph and the data.
///
/// A struct rather than six parameters, for the same reason as
/// [`RunContext`]: every strategy takes exactly this set, and a caller
/// forgetting one of six positional arguments is a bug the compiler cannot
/// name.
pub struct ForwardEnv<'a> {
    /// Implementations and trained states for every node in the graph.
    pub catalog: &'a NodeCatalog,
    /// Output cache consulted and filled during the pass.
    pub cache: &'a dyn CacheStore,
    /// Bus the pass emits its node events on.
    pub event_bus: &'a Arc<EventBus>,
    /// Row source [`Batched`] reads from; the other strategies ignore it.
    pub data_store: Option<&'a Arc<dyn DataStore>>,
    /// Performs and journals step effects; a graph without steps ignores
    /// it, which is why it is an `Option` and not a requirement.
    pub driver: Option<&'a crate::agentic::EffectDriver>,
}

/// How a forward pass feeds data through the compiled graph.
pub trait ForwardStrategy {
    /// Execute a forward pass, returning the final output.
    fn forward(&self, graph: &Graph, env: &ForwardEnv<'_>, x: &Value) -> Result<Value>;
}

/// Full input at once, with inference caching.
pub struct Standard;

impl ForwardStrategy for Standard {
    fn forward(&self, graph: &Graph, env: &ForwardEnv<'_>, x: &Value) -> Result<Value> {
        let CompileResult { plan, .. } =
            compile(graph, env.catalog, CompileMode::Inference, Some(env.cache))?;
        run_forward(graph, &plan, env, x)
    }
}

/// Chunked input through [`crate::StreamRun`], respecting each
/// filter's `StreamMode`.
pub struct Stream {
    /// Rows per chunk fed through the stream plan.
    pub chunk_size: usize,
}

impl ForwardStrategy for Stream {
    fn forward(&self, graph: &Graph, env: &ForwardEnv<'_>, x: &Value) -> Result<Value> {
        let CompileResult { plan, .. } = compile_stream(graph, env.catalog, self.chunk_size)?;
        run_forward(graph, &plan, env, x)
    }
}

/// Run a compiled plan against the graph's *real* topology.
///
/// The runner used to derive its own from the plan's node order, chaining
/// them as if every graph were a line. On a diamond it was simply wrong:
/// `a → {b, c} → d` answered `d(c(…))`, with `d` never seeing `b` and `a`
/// never seeing the input. Every strategy here has the graph, so every
/// strategy passes it.
fn run_forward(
    graph: &Graph,
    plan: &somatize_compiler::ExecutionPlan,
    env: &ForwardEnv<'_>,
    x: &Value,
) -> Result<Value> {
    let run_id = somatize_core::util::timestamp_id("forward");
    let mut ctx = RunContext::new(
        env.catalog,
        env.cache,
        env.event_bus,
        &run_id,
        crate::execution::executor::GraphInfo::from_graph(graph),
    );
    if let Some(driver) = env.driver {
        ctx = ctx.with_driver(driver.clone());
    }
    crate::execution::runner::LocalRunner.forward(plan, &ctx, x)
}

/// Batched forward: read rows from a DataStore in fixed-size batches.
/// Keeps memory bounded — only one batch is materialized at a time.
pub struct Batched<'a> {
    /// Which dataset to read from the [`ForwardEnv::data_store`].
    pub data_ref: &'a DataRef,
    /// Rows materialized per batch — the memory bound.
    pub batch_size: usize,
}

impl ForwardStrategy for Batched<'_> {
    fn forward(&self, graph: &Graph, env: &ForwardEnv<'_>, _x: &Value) -> Result<Value> {
        let store = env.data_store.ok_or_else(|| SomaError::Execution {
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
            compile(graph, env.catalog, CompileMode::Inference, Some(env.cache))?;

        let mut all_values: Vec<f64> = Vec::new();
        let mut result_shape: Option<Vec<usize>> = None;
        let mut rows_processed = 0;

        while rows_processed < total_rows {
            let batch_len = self.batch_size.min(total_rows - rows_processed);
            let batch = store.get_rows(self.data_ref, rows_processed, batch_len)?;
            let output = run_forward(graph, &plan, env, &batch)?;

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
    use somatize_core::cache::CacheKey;
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

    fn env<'a>(
        catalog: &'a NodeCatalog,
        cache: &'a dyn CacheStore,
        event_bus: &'a Arc<EventBus>,
    ) -> ForwardEnv<'a> {
        ForwardEnv {
            catalog,
            cache,
            event_bus,
            data_store: None,
            driver: None,
        }
    }

    #[test]
    fn standard_forward() {
        let (graph, catalog, cache, bus) = make_session();
        let input = Value::tensor(vec![1.0, 2.0, 3.0], vec![3]);

        let result = Standard
            .forward(&graph, &env(&catalog, cache.as_ref(), &bus), &input)
            .unwrap();
        let (data, _) = result.as_tensor().unwrap();
        assert_eq!(data, &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn stream_forward() {
        let (graph, catalog, cache, bus) = make_session();
        let input = Value::tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6]);

        let result = Stream { chunk_size: 2 }
            .forward(&graph, &env(&catalog, cache.as_ref(), &bus), &input)
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
            .forward(&graph, &env(&catalog, cache.as_ref(), &bus), &input)
            .unwrap();
        let streamed = Stream { chunk_size: 2 }
            .forward(&graph, &env(&catalog, cache.as_ref(), &bus), &input)
            .unwrap();
        assert_eq!(standard, streamed);
    }
}
