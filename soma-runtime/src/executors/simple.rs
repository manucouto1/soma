//! SimpleExecutor — one-shot execution: compile, fit, forward.
//!
//! The default executor for running a complete dataset through a graph.
//! No loops, no sampling, no chunking — just compile → fit → forward.

use crate::EventBus;
use crate::filter_library::FilterLibrary;
use crate::runner::Runner;

use somatize_compiler::{CompileMode, compile};
use somatize_core::cache::CacheStore;
use somatize_core::error::Result;
use somatize_core::graph::Graph;
use somatize_core::value::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// One-shot execution: compile the graph, fit, then forward.
pub struct SimpleExecutor;

impl SimpleExecutor {
    /// Fit + forward in one call.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        runner: &dyn Runner,
        graph: &Graph,
        filters: &FilterLibrary,
        cache: &dyn CacheStore,
        event_bus: &Arc<EventBus>,
        x: &Value,
        y: Option<&Value>,
    ) -> Result<(Value, HashMap<String, Value>)> {
        graph.validate()?;

        let plan = compile(graph, filters, CompileMode::NoCache, Some(cache))?.plan;
        runner.fit(&plan, filters, cache, event_bus, x, y)
    }

    /// Forward only (inference on fitted model).
    pub fn forward(
        &self,
        runner: &dyn Runner,
        graph: &Graph,
        filters: &FilterLibrary,
        cache: &dyn CacheStore,
        event_bus: &Arc<EventBus>,
        x: &Value,
    ) -> Result<Value> {
        let plan = compile(graph, filters, CompileMode::Inference, Some(cache))?.plan;
        runner.forward(&plan, filters, cache, event_bus, x)
    }
}
