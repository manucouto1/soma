//! What a graph needs before an effectful node can run.
//!
//! Tools, the provider that serves an unqualified model name, and the
//! [`EffectDriver`] that performs what a step asks for. A purely
//! computational pipeline never comes through here.

use super::{PyGraph, registry};
use crate::prelude::*;

/// Build the step library and effect driver an agentic plan needs.
///
/// Returns `None` for a graph with no steps, so a purely computational
/// pipeline never constructs a provider router, reads a catalog, or
/// touches an environment variable.
pub(super) fn step_runtime(
    g: &PyGraph,
    py: Python<'_>,
    catalog: &NodeCatalog,
) -> PyResult<Option<EffectDriver>> {
    if !catalog.has_steps() {
        return Ok(None);
    }
    // Captured before `somatize_llm::Catalog` shadows the name below.
    let node_catalog = Arc::new(catalog.clone());

    // Python tools and MCP tools land in one toolbox: to a model they are
    // the same thing, and a step names them the same way. Tools declared
    // on a live agent are collected here too, so an agent that gained one
    // since the graph was built can still call it.
    let mut toolbox = somatize_llm::Toolbox::new();
    for tool in g.tools.values() {
        toolbox.add(Arc::new(PyToolAdapter { tool: tool.clone() }));
    }
    for (_, obj) in g.nodes.steps() {
        for tool in to_step_spec(py, obj.bind(py))?.tools() {
            toolbox.add(Arc::new(PyToolAdapter { tool: tool.clone() }));
        }
    }
    for mcp in &g.mcp_toolboxes {
        toolbox.merge_from(mcp);
    }

    let catalog = somatize_llm::Catalog::load().map_err(soma_err_to_py)?;
    let mut router = somatize_llm::Router::from_catalog(catalog).map_err(soma_err_to_py)?;
    if let Some(default) = &g.default_provider {
        router = router.with_default(default);
    }

    // The journal shares the graph's cache directory, so an agentic run is
    // resumable by the same mechanism a computational one is.
    let cache_dir = default_cache_dir().ok_or_else(|| {
        PyRuntimeError::new_err(
            "an agentic graph needs somewhere to journal its effects; \
             set SOMA_CACHE_DIR or HOME",
        )
    })?;
    let store = Arc::new(FsActionStore::new(cache_dir).map_err(soma_err_to_py)?);
    let journal = EffectJournal::new(store.clone(), store);

    // The base handlers, shared with the graph handler below so a
    // sub-pipeline's own agents reach the same providers, tools and
    // journal — that is what makes agent → pipeline → agent one run.
    let base: Vec<Arc<dyn somatize_core::agentic::effect::EffectHandler>> = vec![
        Arc::new(somatize_llm::LlmHandler::new(router)),
        Arc::new(toolbox),
        Arc::new(somatize_runtime::agentic::SleepHandler),
    ];
    let graph_handler = somatize_runtime::agentic::GraphHandler::new((*node_catalog).clone())
        .with_cache(g.cache.clone())
        .with_step_runtime(base.clone(), journal.clone())
        .with_event_bus(g.event_bus.clone());

    let mut driver = EffectDriver::new(journal)
        .with_event_bus(g.event_bus.clone())
        .with_handler(Arc::new(graph_handler))
        // The driver carries its own catalog: this is where a `Spawn`
        // transition finds the nodes it names.
        .with_catalog(node_catalog);
    for handler in base {
        driver = driver.with_handler(handler);
    }

    Ok(Some(driver))
}

/// Start an MCP server and make everything it publishes callable.
///
/// Discovery happens now, so a misconfigured server fails here rather
/// than mid-run. The toolbox is kept for the graph's lifetime: dropping
/// the client kills the server's subprocess.
pub(super) fn add_mcp_server(
    g: &mut PyGraph,
    py: Python<'_>,
    command: String,
    args: Option<Vec<String>>,
) -> PyResult<Vec<String>> {
    let args = args.unwrap_or_default();
    let refs: Vec<&str> = args.iter().map(String::as_str).collect();

    // Spawning and handshaking is I/O; do not hold the GIL for it.
    let mut toolbox = somatize_llm::Toolbox::new();
    py.allow_threads(|| toolbox.add_mcp_server(&command, &refs))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let names: Vec<String> = toolbox.names().into_iter().map(String::from).collect();
    g.mcp_toolboxes.push(toolbox);
    Ok(names)
}

/// Answer what a suspended run was waiting for.
///
/// Every argument comes off the `SomaSuspended` exception that stopped the
/// run, `reason` included — it is part of the journal key, so the answer
/// has to be filed against the same pause the step described, not one
/// reconstructed from a guess.
pub(super) fn resume(
    g: &PyGraph,
    py: Python<'_>,
    run_id: &str,
    node_id: &str,
    turn: usize,
    reason: &Bound<'_, PyAny>,
    answer: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let reason: somatize_core::agentic::effect::SuspendReason =
        serde_json::from_value(py_any_to_json(reason)?).map_err(|e| {
            PyValueError::new_err(format!(
                "`reason` should be the one from the SomaSuspended exception: {e}"
            ))
        })?;

    let catalog = registry::rebuild_catalog(g, py)?;
    let driver = step_runtime(g, py, &catalog)?.ok_or_else(|| {
        PyRuntimeError::new_err("this graph has no effectful nodes, so nothing in it can suspend")
    })?;

    // Release the GIL, like every sibling entry point. Resuming drives the
    // effect journal forward, which can perform a model call — holding the
    // GIL across that blocks every other Python thread in the process for
    // the length of an HTTP request.
    let answer = py_to_value(py, answer)?;
    py.allow_threads(|| driver.resume_with(run_id, node_id, turn, &reason, answer))
        .map_err(soma_err_to_py)
}
