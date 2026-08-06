//! LocalRunner — executes plans locally using the Executor.
//!
//! This is the default runner. The worker delegates here
//! after preparing the environment (deserializing filters, resolving input).

use super::{RunContext, Runner};
use crate::execution::executor::{Context, RunMode};

use somatize_compiler::ExecutionPlan;
use somatize_core::data::keys::{GRAPH_INPUT, input_key};
use somatize_core::data::value::Value;
use somatize_core::error::{Result, SomaError};
use std::collections::HashMap;

/// Executes plans locally — same logic for local and remote execution.
pub struct LocalRunner;

impl LocalRunner {
    /// Build the run's context and walk the plan.
    ///
    /// Fit and forward differ only in [`RunMode`], so they share this. Fit
    /// used to have a walk of its own — flattening the plan into a list and
    /// re-deriving inputs, events, caching and panic handling — which is
    /// why it ignored `Parallel`, `Loop` and `Branch`, never emitted a cache
    /// event, and failed outright on a graph containing a step.
    fn walk(
        &self,
        plan: &ExecutionPlan,
        ctx: &RunContext<'_>,
        input: &Value,
        mode: RunMode,
    ) -> Result<Context> {
        let mut exec = Context::new(ctx.events.clone(), ctx.run_id)
            .with_graph_info(ctx.graph_info.clone())
            .with_seed(ctx.seed);
        exec.mode = mode;
        exec.driver = ctx.driver();

        if let Some(first) = plan.node_ids().first() {
            exec.set(input_key(first), input.clone());
        }
        exec.set(GRAPH_INPUT, input.clone());

        crate::execution::executor::execute(plan, &mut exec, ctx.catalog, ctx.cache)?;
        Ok(exec)
    }

    /// The output of the node that ran last, not of the last node listed.
    ///
    /// On a branch only one arm runs, so plan order is the wrong question.
    fn last_output(exec: &Context) -> Option<Value> {
        exec.execution_order()
            .iter()
            .rev()
            .find(|id| !somatize_core::data::keys::is_reserved(id))
            .and_then(|id| exec.get(id).cloned())
    }
}

impl Runner for LocalRunner {
    fn fit(
        &self,
        plan: &ExecutionPlan,
        ctx: &RunContext<'_>,
        input: &Value,
        y: Option<&Value>,
    ) -> Result<(Value, HashMap<String, Value>)> {
        let exec = self.walk(plan, ctx, input, RunMode::Fit { y: y.cloned() })?;

        // Node outputs plus the states that were learned. The `__state_*`
        // entries travel in the same map because that is what the session
        // and the worker already read out of it.
        let last = Self::last_output(&exec).unwrap_or(Value::Empty);
        let mut produced = exec.into_outputs();
        // The run's own inputs are not something a node produced.
        produced.retain(|id, _| !somatize_core::data::keys::is_input_key(id));

        Ok((last, produced))
    }

    fn forward(&self, plan: &ExecutionPlan, ctx: &RunContext<'_>, input: &Value) -> Result<Value> {
        let exec = self.walk(plan, ctx, input, RunMode::Forward)?;
        Self::last_output(&exec).ok_or_else(|| SomaError::Other("no output produced".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EventBus;
    use crate::cache::MemoryCache;
    use crate::execution::executor::GraphInfo;
    use crate::execution::node_catalog::NodeCatalog;
    use somatize_core::cache::{CacheKey, CacheStore};
    use somatize_core::graph::filter::{Filter, FilterKind, FilterMeta, StreamMode};
    use std::sync::Arc;
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
                distribution: somatize_core::graph::filter::Distribution::Local,
                input_schema: None,
                output_schema: None,
            }
        }
    }

    #[test]
    fn state_cache_key_is_sensitive_to_labels() {
        let fits = Arc::new(AtomicUsize::new(0));
        let mut filters = NodeCatalog::new();
        filters.register("clf", Box::new(CountingFitFilter { fits: fits.clone() }));

        let cache = MemoryCache::default();
        let bus = Arc::new(EventBus::new(64));
        let plan = ExecutionPlan::Execute {
            node_id: "clf".into(),
        };
        let runner = LocalRunner;
        // A single node: the fabricated linear topology is the real one.
        fn ctx<'a>(
            filters: &'a NodeCatalog,
            cache: &'a dyn CacheStore,
            bus: &'a Arc<EventBus>,
        ) -> RunContext<'a> {
            RunContext::new(filters, cache, bus, "test_run", GraphInfo::new())
        }
        let x = Value::tensor(vec![1.0, 2.0], vec![2]);
        let y_a = Value::tensor(vec![0.0, 1.0], vec![2]);
        let y_b = Value::tensor(vec![1.0, 0.0], vec![2]);

        runner
            .fit(&plan, &ctx(&filters, &cache, &bus), &x, Some(&y_a))
            .unwrap();
        assert_eq!(fits.load(Ordering::SeqCst), 1);

        // Same features, same labels → cached state, no refit.
        runner
            .fit(&plan, &ctx(&filters, &cache, &bus), &x, Some(&y_a))
            .unwrap();
        assert_eq!(fits.load(Ordering::SeqCst), 1);

        // Same features, DIFFERENT labels → must refit.
        runner
            .fit(&plan, &ctx(&filters, &cache, &bus), &x, Some(&y_b))
            .unwrap();
        assert_eq!(
            fits.load(Ordering::SeqCst),
            2,
            "different labels must not reuse the cached state"
        );

        // Unsupervised (no labels) → distinct from both supervised keys.
        runner
            .fit(&plan, &ctx(&filters, &cache, &bus), &x, None)
            .unwrap();
        assert_eq!(fits.load(Ordering::SeqCst), 3);
    }
}
