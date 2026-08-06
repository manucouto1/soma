//! Running the graph: fit, forward, compile.
//!
//! Each of the three rebuilds the catalog, hands the compiler the same
//! value, and lets the runtime do the walking. What differs between them
//! is the compile mode and where the plan runs — here, or on a worker.

use super::{PyGraph, agentic, distributed, registry};
use crate::prelude::*;

/// File what a fit learned, and mark the graph fitted.
///
/// The tail of every `fit` path — it was written out at each of the five
/// returns, and the five copies had drifted. All three producers now hand
/// over states keyed by node id: `Fitted::states` from a runner, the
/// worker's `PlanResult`, and a strategy's round. The `__state_` prefix is
/// a key inside the runner's value store and no longer reaches here, which
/// is what fixed the differentiable path reading each node's *output* as
/// its learned state.
fn absorb(g: &mut PyGraph, states: HashMap<String, Value>) -> PyResult<()> {
    for (node_id, state) in states {
        g.library
            .try_set_state(node_id, state)
            .map_err(soma_err_to_py)?;
    }
    g.fitted = true;
    Ok(())
}

// ── Fit ──

/// Fit every trainable filter, wherever the graph says it should run.
pub(super) fn fit(
    g: &mut PyGraph,
    py: Python<'_>,
    x: &Bound<'_, pyo3::types::PyAny>,
    y: Option<&Bound<'_, pyo3::types::PyAny>>,
    batch_size: Option<usize>,
    mode: &str,
    seed: Option<i64>,
) -> PyResult<()> {
    let x_val = py_to_value(py, x)?;
    let y_val = match y {
        Some(v) => Some(py_to_value(py, v)?),
        None => None,
    };

    if mode == "differentiable" {
        return fit_differentiable(g, py, &x_val, y_val.as_ref());
    }
    if mode != "inference" {
        return Err(PyRuntimeError::new_err(format!(
            "Unknown mode={mode:?}. Use 'inference' or 'differentiable'."
        )));
    }

    // A strategy over several workers goes through the runtime's
    // StrategyExecutor rather than a single dispatch: it shards the input,
    // runs a round per client and aggregates between rounds. One worker is
    // not a strategy — it is the ordinary path below.
    if g.graph.effective_strategy_is_distributed()
        && g.workers.len() > 1
        && g.graph.nodes.iter().all(|n| !n.is_local())
    {
        distributed::register_filters_on_all(g)?;
        let transports = distributed::transports(g);
        let states = py.allow_threads(|| {
            distributed::session_with_transports(g, transports)
                .and_then(|mut session| session.fit(&x_val, y_val.as_ref()))
        });
        let fitted = states.map_err(soma_err_to_py)?;
        return absorb(g, fitted.states);
    }

    // Dispatch fit to a worker if possible. Batching is the worker's
    // business either way — `batch_size` travels inside the mode — so the
    // batched and unbatched dispatches were the same call written twice.
    //
    // Release the GIL during WS dispatch so the worker thread can acquire
    // it for Python execution.
    if !g.workers.is_empty() && g.graph.nodes.iter().all(|n| !n.is_local()) {
        let mode = somatize_worker::protocol::ExecutionMode::Fit {
            y: y_val.clone(),
            batch_size,
        };
        let result = py.allow_threads(|| distributed::dispatch_to_worker(g, &x_val, mode, seed));
        let (_output, states) = result?;
        return absorb(g, states);
    }

    fit_local(g, py, &x_val, y_val.as_ref(), seed)
}

/// The local fit.
///
/// Through the compiler and the runner, like every other entry point.
/// This used to be a topological loop written here, walking
/// `graph.topological_sort()` and calling fit/forward node by node — so it
/// ignored parallelism, loops and branches, and it was the only fit
/// anywhere that salted its state keys with the seed. Now the runner
/// salts, and the loop is gone.
fn fit_local(
    g: &mut PyGraph,
    py: Python<'_>,
    x: &Value,
    y: Option<&Value>,
    seed: Option<i64>,
) -> PyResult<()> {
    g.graph.validate().map_err(soma_err_to_py)?;
    let catalog = registry::rebuild_catalog(g, py)?;
    let compile_result = compile(
        &g.graph,
        &catalog,
        CompileMode::NoCache,
        Some(g.cache.as_ref()),
    )
    .map_err(soma_err_to_py)?;

    let run_id = somatize_core::util::timestamp_id("graph_fit");
    let mut run_ctx = somatize_runtime::execution::runner::RunContext::new(
        &catalog,
        g.cache.as_ref(),
        &g.event_bus,
        &run_id,
        GraphInfo::from_graph(&g.graph),
    )
    .with_seed(seed);

    // A fit reaches steps now, so it has to be able to drive them —
    // `forward` has always attached this. Without it a graph mixing
    // filters and steps fitted the filters and then stopped at the first
    // step for want of a driver.
    if let Some(driver) = agentic::step_runtime(g, py, &catalog)? {
        run_ctx = run_ctx.with_driver(driver);
    }

    // Release the GIL inside the bracket: a Parallel plan runs branches on
    // scoped threads whose Python filters must acquire it.
    let fitted = g
        .event_bus
        .run_bracket(&run_id, compile_result.plan.summary(), || {
            py.allow_threads(|| LocalRunner.fit(&compile_result.plan, &run_ctx, x, y))
        })
        .map_err(soma_err_to_py)?;

    absorb(g, fitted.states)
}

/// The differentiable fit: compile with `CompileMode::Differentiable`,
/// which collapses consecutive differentiable filters into a `Composite`
/// block, and let the runner delegate the block to the first filter's
/// `composite_fit`. Gradients flow end to end inside that call.
fn fit_differentiable(
    g: &mut PyGraph,
    py: Python<'_>,
    x: &Value,
    y: Option<&Value>,
) -> PyResult<()> {
    // `mode="differentiable"` is the *local* loop: the caller drives
    // `context`/`backward`/`step` and owns when the parameters move. A
    // worker cannot be driven that way — it would need distributed
    // autograd — so this stays refused rather than running a fit that
    // computes gradients and never steps.
    //
    // Training a differentiable graph on workers is
    // `set_strategy("data_parallel")`, which is a complete round: each
    // replica fits its own shard, the gradients are averaged across
    // replicas and applied, and the stepped weights are read back. See
    // `guides/execution-modes.md`.
    if !g.workers.is_empty() {
        return Err(PyRuntimeError::new_err(
            "mode='differentiable' drives the training loop locally, so \
             it cannot run on workers. To train this graph on the \
             workers you registered, set a strategy instead:\n    \
             g.set_strategy(\"data_parallel\", num_replicas=N)\n    \
             g.fit(x, y)",
        ));
    }
    g.graph.validate().map_err(soma_err_to_py)?;
    let catalog = registry::rebuild_catalog(g, py)?;
    let compile_result = compile(
        &g.graph,
        &catalog,
        CompileMode::Differentiable,
        Some(g.cache.as_ref()),
    )
    .map_err(soma_err_to_py)?;

    let run_id = somatize_core::util::timestamp_id("fit");
    let run_ctx = somatize_runtime::execution::runner::RunContext::new(
        &catalog,
        g.cache.as_ref(),
        &g.event_bus,
        &run_id,
        GraphInfo::from_graph(&g.graph),
    );

    let fitted = g
        .event_bus
        .run_bracket(&run_id, compile_result.plan.summary(), || {
            LocalRunner.fit(&compile_result.plan, &run_ctx, x, y)
        })
        .map_err(soma_err_to_py)?;

    absorb(g, fitted.states)
}

// ── Forward ──

/// Forward, after deciding which engine the graph needs.
///
/// A graph whose filters carry torch modules is walked in Python, because
/// autograd does not survive the `Value` boundary: a tensor that becomes a
/// vector of f64 and back has lost the graph the optimiser needs.
///
/// That walk used to *replace* this method at import time, so which engine
/// ran depended on which modules had been imported, two implementations
/// answered to one name, and neither `help(Graph.forward)` nor any static
/// analysis could see it. The dispatch belongs here, where the graph knows
/// what it holds; the walk is a named function this calls.
pub(super) fn forward(
    slf: PyRef<'_, PyGraph>,
    py: Python<'_>,
    x: &Bound<'_, pyo3::types::PyAny>,
    stream: bool,
    chunk_size: Option<usize>,
    seed: Option<i64>,
    run_id: Option<String>,
) -> PyResult<PyObject> {
    if registry::has_differentiable_filters(&slf, py) {
        // The torch walk honours none of these: it does not chunk, it does
        // not salt a cache key (nothing it produces is cached), and it
        // emits no run bracket. They used to be accepted and discarded, so
        // `g.forward(x, seed=42)` on a torch graph reported success having
        // ignored the seed — and a seed that is silently ignored is worse
        // than one that is refused, because the run looks reproducible.
        let ignored: Vec<&str> = [
            ("stream", stream),
            ("chunk_size", chunk_size.is_some()),
            ("seed", seed.is_some()),
            ("run_id", run_id.is_some()),
        ]
        .iter()
        .filter(|(_, given)| *given)
        .map(|(name, _)| *name)
        .collect();
        if !ignored.is_empty() {
            return Err(PyValueError::new_err(format!(
                "this graph holds differentiable filters, so it is walked in \
                 Python for autograd — and that walk cannot honour {}. Run the \
                 graph in eval mode (`g.eval()`, after `g.freeze()`) to reach the \
                 Rust path, which can.",
                ignored.join(", ")
            )));
        }
        let graph = Py::from(slf);
        let walk = py.import("soma._orchestrator")?;
        return walk
            .call_method1("differentiable_forward", (graph, x))
            .map(|v| v.unbind());
    }
    forward_local(
        &slf,
        py,
        x,
        stream,
        chunk_size.unwrap_or(1024),
        seed,
        run_id,
    )
}

/// Rebuild the plan and run it here (or on workers). The non-autograd path.
#[allow(clippy::too_many_arguments)]
fn forward_local(
    g: &PyGraph,
    py: Python<'_>,
    x: &Bound<'_, pyo3::types::PyAny>,
    stream: bool,
    chunk_size: usize,
    seed: Option<i64>,
    run_id: Option<String>,
) -> PyResult<PyObject> {
    // A step has no fit phase — its behaviour comes from a model and a
    // prompt, not from learned state. A graph with nothing trainable in it
    // therefore has nothing to fit, and demanding a fit first would be
    // asking for a no-op.
    if !g.fitted && registry::has_trainable_filters(g) {
        return Err(PyRuntimeError::new_err(
            "graph must be fitted before forward",
        ));
    }
    let x_val = py_to_value(py, x)?;

    // Remote streaming: chunks over WS Binary to the worker. Release the
    // GIL during the dispatch so the worker thread can acquire it for
    // Python execution.
    if stream && !g.workers.is_empty() {
        let output =
            py.allow_threads(|| distributed::dispatch_streamed(g, &x_val, chunk_size, seed))?;
        return value_to_py(py, &output);
    }

    // Dispatch the entire plan remotely if workers are registered and no
    // node forces local.
    if !stream && !g.workers.is_empty() && g.graph.nodes.iter().all(|n| !n.is_local()) {
        let (output, _states) = py.allow_threads(|| {
            distributed::dispatch_to_worker(
                g,
                &x_val,
                somatize_worker::protocol::ExecutionMode::Forward,
                seed,
            )
        })?;
        return value_to_py(py, &output);
    }

    // Local execution — one path whether chunked or not. Streaming used to
    // be a hand-rolled sibling that attached no driver, no transport,
    // ignored a resumed run's id and picked its output differently; now the
    // ONLY difference is which compiler entry produced the plan.
    let catalog = registry::rebuild_catalog(g, py)?;
    let compile_result = if stream {
        somatize_compiler::compile_stream(&g.graph, &catalog, chunk_size)
    } else {
        somatize_compiler::compile(
            &g.graph,
            &catalog,
            CompileMode::Inference,
            Some(g.cache.as_ref()),
        )
    }
    .map_err(soma_err_to_py)?;

    // A caller resuming a suspended run passes its id back. The journal
    // keys an impure effect by `(run, node, turn, index)`, so a fresh id
    // would replay nothing and the answer already recorded would never be
    // found — which is why resuming did not work.
    let run_id = run_id.unwrap_or_else(|| somatize_core::util::timestamp_id("graph_forward"));
    let mut ctx = Context::new(g.event_bus.clone(), run_id)
        .with_graph_info(GraphInfo::from_graph(&g.graph))
        .with_seed(seed);

    if let Some(driver) = agentic::step_runtime(g, py, &catalog)? {
        ctx = ctx.with_driver(driver);
    }
    if let Some(transport) = distributed::make_transport(g) {
        ctx = ctx.with_transport(transport);
    }

    let roots = g.graph.roots();
    if roots.len() == 1 {
        ctx.set(
            somatize_core::data::keys::input_key(roots[0]),
            x_val.clone(),
        );
    }
    ctx.set(somatize_core::data::keys::GRAPH_INPUT, x_val);

    // Release the GIL: Parallel plans run branches on scoped threads whose
    // Python filters must acquire it — holding it here would deadlock the
    // join.
    py.allow_threads(|| {
        executor::execute(&compile_result.plan, &mut ctx, &catalog, g.cache.as_ref())
    })
    .map_err(soma_err_to_py)?;

    value_to_py(py, &pick_output(g, &ctx))
}

/// Which leaf is "the output" when there are several?
///
/// Prefer one that actually ran. A branch makes every arm a leaf, so
/// declaration order alone would return the arm that was *not* taken — an
/// empty value, from a node that never executed. Among leaves that did
/// produce something, declaration order still decides, so a parallel
/// fan-out answers the same as it always has.
fn pick_output(g: &PyGraph, ctx: &Context) -> Value {
    let leaves = g.graph.leaves();
    leaves
        .iter()
        .find_map(|id| ctx.get(id).cloned())
        .or_else(|| leaves.first().and_then(|id| ctx.get(id).cloned()))
        .or_else(|| {
            // The last node that actually ran — skipping the run's own
            // reserved entries, which `last()` alone would happily return
            // as though a node had produced them.
            ctx.execution_order()
                .iter()
                .rev()
                .find(|id| !somatize_core::data::keys::is_reserved(id))
                .and_then(|id| ctx.get(id).cloned())
        })
        .unwrap_or(Value::Empty)
}

// ── Compile ──

/// Compile the graph and return diagnostic information.
pub(super) fn compile_info(g: &PyGraph, py: Python<'_>, mode: &str) -> PyResult<PyObject> {
    let compile_mode = match mode {
        "inference" => CompileMode::Inference,
        "differentiable" => CompileMode::Differentiable,
        _ => {
            return Err(PyRuntimeError::new_err(format!(
                "Unknown mode: {mode}. Use 'inference' or 'differentiable' — \
                 the same two `fit` takes."
            )));
        }
    };

    // The rebuilt catalog, not `library`: passing the filter half alone is
    // how `.compile()` came to skip every step's schema while `.run()`
    // checked them.
    let catalog = registry::rebuild_catalog(g, py)?;
    let result =
        somatize_compiler::compile(&g.graph, &catalog, compile_mode, Some(g.cache.as_ref()))
            .map_err(soma_err_to_py)?;

    let dict = PyDict::new(py);
    let summary = result.plan.summary();
    dict.set_item("total_nodes", summary.total_nodes)?;
    dict.set_item("cached_nodes", summary.cached_nodes)?;
    dict.set_item("parallel_branches", summary.parallel_branches)?;

    // Structured diagnostics: {node, level, message} dicts, not Debug
    // strings — readable and machine-consumable.
    let diags = PyList::empty(py);
    for d in &result.diagnostics {
        let entry = PyDict::new(py);
        entry.set_item("node", &d.node_id)?;
        entry.set_item(
            "level",
            match d.level {
                somatize_compiler::DiagnosticLevel::Warning => "warning",
                somatize_compiler::DiagnosticLevel::Info => "info",
            },
        )?;
        entry.set_item("message", &d.message)?;
        diags.append(entry)?;
    }
    dict.set_item("diagnostics", diags)?;
    dict.set_item("plan_text", format!("{}", result.plan))?;
    dict.set_item("plan_mermaid", result.plan.to_mermaid())?;
    dict.set_item("plan_svg", result.plan.to_graph().to_svg())?;

    Ok(dict.into_any().unbind())
}
