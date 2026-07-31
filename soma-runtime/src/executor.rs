//! Plan executor — walks [`ExecutionPlan`] trees and runs filter nodes.
//!
//! Handles sequential, parallel (scoped threads), cached, remote, loop,
//! and branch execution. Uses [`GraphInfo`] for topology-aware input resolution.

use crate::event_bus::EventBus;
use crate::filter_library::FilterLibrary;
use somatize_compiler::ExecutionPlan;
use somatize_core::cache::CacheStore;
use somatize_core::control::{
    LoopCondition, LoopSignal, is_default_arm, read_arm_selector, read_loop_signal,
};
use somatize_core::error::{Result, SomaError};
use somatize_core::event::Event;
use somatize_core::store::DataStore;
use somatize_core::value::Value;
use somatize_core::virtual_value::VirtualValue;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// Graph topology information for input resolution.
///
/// Maps each node to its predecessor node IDs so the executor knows
/// where to read inputs from in the context store.
#[derive(Debug, Clone, Default)]
pub struct GraphInfo {
    /// node_id → list of predecessor node IDs
    predecessors: HashMap<String, Vec<String>>,
}

impl GraphInfo {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register predecessors for a node.
    pub fn set_predecessors(&mut self, node_id: impl Into<String>, preds: Vec<String>) {
        self.predecessors.insert(node_id.into(), preds);
    }

    /// Build GraphInfo from a somatize_core::graph::Graph.
    pub fn from_graph(graph: &somatize_core::graph::Graph) -> Self {
        let mut info = Self::new();
        for node in &graph.nodes {
            let preds: Vec<String> = graph
                .predecessors(&node.id)
                .into_iter()
                .map(|s| s.to_string())
                .collect();
            info.set_predecessors(node.id.clone(), preds);
        }
        info
    }

    /// Build GraphInfo for a linear pipeline (each node depends on the previous).
    pub fn for_linear(node_ids: &[&str]) -> Self {
        let mut info = Self::new();
        for (i, &id) in node_ids.iter().enumerate() {
            let preds = if i > 0 {
                vec![node_ids[i - 1].to_string()]
            } else {
                vec![]
            };
            info.set_predecessors(id, preds);
        }
        info
    }

    /// Get predecessors for a node.
    pub fn predecessors(&self, node_id: &str) -> &[String] {
        self.predecessors
            .get(node_id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }
}

/// Execution context passed to filters during runtime.
///
/// Node outputs are stored as [`VirtualValue`]s — they may be materialized
/// in memory, cached on disk, or deferred (not yet computed). The executor
/// resolves them on demand when a downstream node needs the data.
pub struct Context {
    /// Node outputs as virtual values (may be lazy).
    pub store: HashMap<String, VirtualValue>,
    /// Event bus for emitting runtime events.
    pub event_bus: Arc<EventBus>,
    /// Current run ID.
    pub run_id: String,
    /// Track execution order.
    pub execution_order: Vec<String>,
    /// Graph topology for input resolution.
    pub graph_info: GraphInfo,
    /// Optional transport for distributed plans.
    pub transport: Option<Arc<dyn crate::runner::Transport>>,
    /// Optional data store for persisting intermediate results.
    pub data_store: Option<Arc<dyn DataStore>>,
    /// Minimum value size (bytes) to spill to DataStore instead of keeping in memory.
    /// Default: 0 (disabled — all values stay in memory).
    pub spill_threshold: usize,
    /// Memoized content hashes of node outputs, keyed by node id.
    /// Invalidated whenever a node's output is (re)stored, so Loop
    /// iterations that overwrite an output never reuse a stale hash.
    output_hashes: HashMap<String, somatize_core::cache::CacheKey>,
    /// Experiment seed for this run. Hashed into every cache key so
    /// each seed owns an independent cache line (a 5-seed study is 5
    /// resumable computations, not one).
    pub seed: Option<i64>,
    /// Owned handle to the same cache passed to `execute()` — needed by
    /// the stream executor, which holds the cache across chunks.
    pub cache_arc: Option<Arc<dyn CacheStore>>,
    /// Effectful steps, by node id. Only needed when the plan contains
    /// `ExecutionPlan::Step`; a purely computational graph leaves it unset.
    pub steps: Option<Arc<crate::step_library::StepLibrary>>,
    /// Performs and journals step effects. Required alongside `steps`.
    pub driver: Option<crate::effects::EffectDriver>,
}

impl Context {
    pub fn new(event_bus: Arc<EventBus>, run_id: impl Into<String>) -> Self {
        Self {
            store: HashMap::new(),
            event_bus,
            run_id: run_id.into(),
            execution_order: Vec::new(),
            graph_info: GraphInfo::new(),
            transport: None,
            data_store: None,
            spill_threshold: 0,
            output_hashes: HashMap::new(),
            seed: None,
            cache_arc: None,
            steps: None,
            driver: None,
        }
    }

    /// Register the step library and effect driver an effectful plan needs.
    ///
    /// The driver is given the same library, so a step that fans out
    /// dynamically can reach the workers it names.
    pub fn with_steps(
        mut self,
        steps: Arc<crate::step_library::StepLibrary>,
        driver: crate::effects::EffectDriver,
    ) -> Self {
        self.driver = Some(driver.with_steps(steps.clone()));
        self.steps = Some(steps);
        self
    }

    pub fn with_graph_info(mut self, info: GraphInfo) -> Self {
        self.graph_info = info;
        self
    }

    /// Set the experiment seed (hashed into every cache key).
    pub fn with_seed(mut self, seed: Option<i64>) -> Self {
        self.seed = seed;
        self
    }

    /// Provide an owned cache handle (enables chunk-level caching in
    /// streaming plans; pass the same store given to `execute()`).
    pub fn with_cache_arc(mut self, cache: Arc<dyn CacheStore>) -> Self {
        self.cache_arc = Some(cache);
        self
    }

    pub fn with_transport(mut self, transport: Arc<dyn crate::runner::Transport>) -> Self {
        self.transport = Some(transport);
        self
    }

    pub fn with_data_store(mut self, store: Arc<dyn DataStore>) -> Self {
        self.data_store = Some(store);
        self
    }

    /// Set spill threshold: values larger than this (in bytes) are offloaded
    /// to the DataStore and replaced with a VirtualValue::Cached reference.
    /// Requires a DataStore to be set via `with_data_store()`.
    pub fn with_spill_threshold(mut self, bytes: usize) -> Self {
        self.spill_threshold = bytes;
        self
    }

    /// If a DataStore and spill threshold are configured, check if the value
    /// should be offloaded. Returns VirtualValue (materialized or cached ref).
    fn maybe_spill(&self, node_id: &str, value: Value) -> VirtualValue {
        if self.spill_threshold > 0
            && let Some(store) = &self.data_store
        {
            let size = value.size() * 8; // approximate bytes (f64 = 8 bytes)
            if size >= self.spill_threshold {
                let key = somatize_core::cache::CacheKey::from_parts(&[
                    self.run_id.as_bytes(),
                    node_id.as_bytes(),
                ]);
                let vv_for_schema = VirtualValue::materialized(value.clone());
                let schema = vv_for_schema.schema().clone();
                if let Ok(_data_ref) = store.put(&key, &value) {
                    tracing::debug!("spilled node `{node_id}` ({size} bytes) to DataStore");
                    return VirtualValue::cached(key, schema);
                }
            }
        }
        VirtualValue::materialized(value)
    }

    /// Get the materialized Value for a node, if present and materialized.
    pub fn get(&self, node_id: &str) -> Option<&Value> {
        self.store.get(node_id).and_then(|vv| vv.as_value())
    }

    /// Get the raw VirtualValue for a node.
    pub fn get_virtual(&self, node_id: &str) -> Option<&VirtualValue> {
        self.store.get(node_id)
    }

    /// Store a materialized value for a node.
    pub fn set(&mut self, node_id: impl Into<String>, value: Value) {
        let id = node_id.into();
        self.execution_order.push(id.clone());
        self.output_hashes.remove(&id);
        self.store.insert(id, VirtualValue::materialized(value));
    }

    /// Store a virtual value (which may be deferred or cached).
    pub fn set_virtual(&mut self, node_id: impl Into<String>, vv: VirtualValue) {
        let id = node_id.into();
        self.execution_order.push(id.clone());
        self.output_hashes.remove(&id);
        self.store.insert(id, vv);
    }

    /// Content hash of a node's resolved input, memoized through the
    /// single-predecessor fast path: the input of a 1-pred node IS that
    /// predecessor's output, so sibling consumers (diamonds) reuse the
    /// hash instead of re-serializing a potentially large value.
    fn input_hash(
        &mut self,
        node_id: &str,
        input: &Value,
    ) -> somatize_core::error::Result<somatize_core::cache::CacheKey> {
        let preds = self.graph_info.predecessors(node_id);
        let single_pred = match preds {
            [only] => Some(only.clone()),
            _ => None,
        };
        if let Some(pred) = single_pred {
            if let Some(h) = self.output_hashes.get(&pred) {
                return Ok(h.clone());
            }
            // Only trust the memo association when the pred's output is
            // actually what resolve_input handed us (it may have fallen
            // back to Value::Empty if the pred produced nothing).
            if self.store.contains_key(&pred) {
                let h = somatize_core::cache::CacheKey::for_value(input)?;
                self.output_hashes.insert(pred, h.clone());
                return Ok(h);
            }
        }
        somatize_core::cache::CacheKey::for_value(input)
    }

    fn snapshot(&self) -> Self {
        Self {
            store: self.store.clone(),
            event_bus: self.event_bus.clone(),
            run_id: self.run_id.clone(),
            execution_order: self.execution_order.clone(),
            graph_info: self.graph_info.clone(),
            transport: self.transport.clone(),
            data_store: self.data_store.clone(),
            spill_threshold: self.spill_threshold,
            output_hashes: self.output_hashes.clone(),
            seed: self.seed,
            cache_arc: self.cache_arc.clone(),
            steps: self.steps.clone(),
            driver: self.driver.clone(),
        }
    }
}

/// Execute a compiled plan synchronously.
/// For parallel branches, uses the async executor under the hood.
/// Contract for executing a plan.
pub trait Executable {
    fn execute(
        &self,
        ctx: &mut Context,
        filters: &FilterLibrary,
        cache: &dyn CacheStore,
    ) -> Result<()>;
}

impl Executable for ExecutionPlan {
    fn execute(
        &self,
        ctx: &mut Context,
        filters: &FilterLibrary,
        cache: &dyn CacheStore,
    ) -> Result<()> {
        match self {
            ExecutionPlan::Empty => Ok(()),

            ExecutionPlan::Execute { node_id } => execute_node(node_id, ctx, filters, cache),

            ExecutionPlan::Step { node_id, handoffs } => {
                execute_step(node_id, handoffs, ctx, filters, cache)
            }

            ExecutionPlan::Sequence(steps) => {
                for step in steps {
                    step.execute(ctx, filters, cache)?;
                }
                Ok(())
            }

            ExecutionPlan::Parallel(branches) => execute_parallel(branches, ctx, filters, cache),

            ExecutionPlan::Loop {
                node_id,
                body,
                max_iterations,
                until,
                carry_from,
            } => {
                let max = max_iterations.unwrap_or(100);
                let mut ran = 0usize;

                // The loop node's value is its *carry*: what the body reads
                // on each pass. A body entry's only predecessor is the loop
                // node itself (that control edge is what makes it the body),
                // so without seeding this the first iteration would run on
                // `Empty`. Seed it with the loop's own input; after each
                // iteration the condition node's output replaces it, which is
                // what makes a refine loop actually refine rather than redraft
                // the same thing N times.
                let seed = resolve_input(node_id, ctx);
                ctx.set(node_id.clone(), seed);

                for i in 0..max {
                    body.execute(ctx, filters, cache)?;
                    ran = i + 1;

                    // Advance the carry before testing the condition: even a
                    // loop with no stop signal has to move forward, or every
                    // pass repeats the first one.
                    if let Some(source) = carry_from
                        && let Some(value) = ctx.get(source).cloned()
                    {
                        ctx.set(node_id.clone(), value);
                    }

                    // Termination is read from the node the compiler resolved,
                    // never from whichever node happened to run last — with a
                    // parallel body "last" is a race.
                    let LoopCondition::WhenSignaled(cond_node) = until else {
                        continue; // Exhaust: always run the full count
                    };

                    let value = ctx.get(cond_node).ok_or_else(|| SomaError::Execution {
                        node_id: node_id.clone(),
                        message: format!(
                            "loop condition node `{cond_node}` produced no output on \
                             iteration {ran}"
                        ),
                    })?;

                    let signal = read_loop_signal(value).ok_or_else(|| SomaError::Execution {
                        node_id: node_id.clone(),
                        message: format!(
                            "loop condition node `{cond_node}` produced `{}`, which \
                                 carries no termination signal. Return a bool, \"done\"/\"stop\", \
                                 or {{\"done\": bool}}",
                            value.type_name()
                        ),
                    })?;

                    if signal == LoopSignal::Stop {
                        ctx.event_bus.emit(Event::NodeCompleted {
                            run_id: ctx.run_id.clone(),
                            node_id: node_id.clone(),
                            duration: std::time::Duration::ZERO,
                            output_summary: format!("Loop terminated at iteration {ran}"),
                        });
                        return Ok(());
                    }
                }

                ctx.event_bus.emit(Event::NodeCompleted {
                    run_id: ctx.run_id.clone(),
                    node_id: node_id.clone(),
                    duration: std::time::Duration::ZERO,
                    output_summary: format!("Loop exhausted {ran} iterations"),
                });
                Ok(())
            }

            ExecutionPlan::Branch { node_id, arms } => {
                // Execute the branch node first (it produces the condition
                // value). It may be an ordinary filter or an effectful step:
                // an LLM deciding where a request goes is the routing case
                // agentic graphs are built for.
                let request = resolve_input(node_id, ctx);
                if ctx.steps.as_ref().is_some_and(|s| s.contains(node_id)) {
                    execute_step(node_id, &[], ctx, filters, cache)?;
                } else {
                    execute_node(node_id, ctx, filters, cache)?;
                }

                let condition = ctx.get(node_id).cloned().unwrap_or(Value::Empty);

                let selector =
                    read_arm_selector(&condition).ok_or_else(|| SomaError::Execution {
                        node_id: node_id.clone(),
                        message: format!(
                            "branch condition produced `{}`, which names no arm. Return the \
                             arm's label as a string, a bool, or {{\"branch\": \"<label>\"}}",
                            condition.type_name()
                        ),
                    })?;

                let (label, plan) = arms
                    .iter()
                    .find(|(label, _)| label == &selector)
                    .or_else(|| arms.iter().find(|(label, _)| is_default_arm(label)))
                    .ok_or_else(|| SomaError::Execution {
                        node_id: node_id.clone(),
                        message: format!(
                            "branch selected `{selector}`, which matches no arm ({}) and \
                             there is no `default` arm",
                            arms.iter()
                                .map(|(l, _)| l.as_str())
                                .collect::<Vec<_>>()
                                .join(", ")
                        ),
                    })?;

                ctx.event_bus.emit(Event::NodeCompleted {
                    run_id: ctx.run_id.clone(),
                    node_id: node_id.clone(),
                    duration: std::time::Duration::ZERO,
                    output_summary: format!("Branch selected: {label}"),
                });

                // The selector is control, not data. An arm's only
                // predecessor is the branch node, so leaving the label there
                // would hand the chosen agent the string "billing" instead of
                // the customer's question — the handoff-context loss the
                // multi-agent failure literature keeps finding. The branch
                // passes its input through instead; put a filter *before* it
                // if the request needs transforming.
                ctx.set(node_id.clone(), request);
                plan.execute(ctx, filters, cache)
            }

            ExecutionPlan::Remote {
                node_id,
                target: _,
                plan,
            } => {
                if let Some(transport) = &ctx.transport {
                    // Gather input from predecessors
                    let input = ctx
                        .graph_info
                        .predecessors(node_id)
                        .first()
                        .and_then(|pred| ctx.get(pred));

                    let result = transport.execute_node(node_id, input)?;
                    ctx.set(node_id.clone(), result);
                    Ok(())
                } else {
                    // No transport — fall back to local execution
                    plan.execute(ctx, filters, cache)
                }
            }

            ExecutionPlan::Composite { node_ids } => {
                // Sequential fallback — execute each node in order.
                // A future Python-aware executor will pass tensors directly.
                for nid in node_ids {
                    execute_node(nid, ctx, filters, cache)?;
                }
                Ok(())
            }

            ExecutionPlan::Stream {
                node_ids,
                chunk_size,
            } => execute_stream(node_ids, *chunk_size, ctx, filters, cache),

            _ => {
                tracing::warn!("Unhandled ExecutionPlan variant");
                Ok(())
            }
        }
    }
}

/// Salt a cache key with the run's experiment seed, when set.
/// `None` leaves the key untouched (and distinct from any seeded key).
pub fn salt_with_seed(
    key: somatize_core::cache::CacheKey,
    seed: Option<i64>,
) -> somatize_core::cache::CacheKey {
    match seed {
        Some(s) => somatize_core::cache::CacheKey::from_parts(&[b"seed", &s.to_le_bytes(), &key.0]),
        None => key,
    }
}

/// Execute a plan (convenience function, delegates to `Executable` trait).
pub fn execute(
    plan: &ExecutionPlan,
    ctx: &mut Context,
    filters: &FilterLibrary,
    cache: &dyn CacheStore,
) -> Result<()> {
    plan.execute(ctx, filters, cache)
}

/// Execute a single filter node.
///
/// Cacheable nodes are memoized: the output key is
/// `hash(config + state + input)` — if a previous run (possibly in
/// another process, via a persistent cache) already computed this exact
/// forward, the stored output is used and the filter never runs.
/// Run an effectful step and store its output.
///
/// Unlike [`execute_node`] there is no output caching here. A step's result
/// is not a function of its input, so a content-addressed key would be a lie;
/// what makes a step re-runnable is the effect journal, which the driver
/// consults per effect.
fn execute_step(
    node_id: &str,
    handoffs: &[(String, ExecutionPlan)],
    ctx: &mut Context,
    filters: &FilterLibrary,
    cache: &dyn CacheStore,
) -> Result<()> {
    let start = Instant::now();
    let _span = tracing::info_span!("execute_step", %node_id).entered();

    let steps = ctx.steps.clone().ok_or_else(|| SomaError::Execution {
        node_id: node_id.to_string(),
        message: "the plan contains a step but no step library was registered; \
                  build the context with `with_steps(...)`"
            .into(),
    })?;
    let driver = ctx.driver.clone().ok_or_else(|| SomaError::Execution {
        node_id: node_id.to_string(),
        message: "the plan contains a step but no effect driver was registered; \
                  build the context with `with_steps(...)`"
            .into(),
    })?;
    let step = steps
        .get(node_id)
        .ok_or_else(|| SomaError::NodeNotFound(node_id.to_string()))?;

    ctx.event_bus.emit(Event::NodeStarted {
        run_id: ctx.run_id.clone(),
        node_id: node_id.to_string(),
        kind: somatize_core::filter::FilterKind::Opaque,
    });

    let input = resolve_input(node_id, ctx);
    let run_id = ctx.run_id.clone();

    let outcome = driver
        .run(step.as_ref(), &run_id, node_id, &input)
        .inspect_err(|e| {
            ctx.event_bus.emit(Event::NodeFailed {
                run_id: run_id.clone(),
                node_id: node_id.to_string(),
                error: e.to_string(),
            });
        })?;

    match outcome {
        crate::effects::StepOutcome::Done(output) => {
            let summary = format!("{output}");
            ctx.set(node_id, output);
            ctx.event_bus.emit(Event::NodeCompleted {
                run_id,
                node_id: node_id.to_string(),
                duration: start.elapsed(),
                output_summary: summary,
            });
            Ok(())
        }

        // A handoff: the step is finished, and names who continues. The
        // carried value is stored under *this* node, so the target resolves
        // it as an ordinary predecessor output — no special path.
        crate::effects::StepOutcome::Goto { target, carry } => {
            let plan = handoffs
                .iter()
                .find(|(t, _)| *t == target)
                .map(|(_, p)| p)
                .ok_or_else(|| SomaError::Execution {
                    node_id: node_id.to_string(),
                    message: if handoffs.is_empty() {
                        format!(
                            "step handed control to `{target}`, but it declares no \
                             handoffs. Add a control edge from `{node_id}` to `{target}`"
                        )
                    } else {
                        format!(
                            "step handed control to `{target}`, which is not among its \
                             declared handoffs ({})",
                            handoffs
                                .iter()
                                .map(|(t, _)| t.as_str())
                                .collect::<Vec<_>>()
                                .join(", ")
                        )
                    },
                })?;

            ctx.set(node_id, carry);
            ctx.event_bus.emit(Event::NodeCompleted {
                run_id,
                node_id: node_id.to_string(),
                duration: start.elapsed(),
                output_summary: format!("handed off to {target}"),
            });
            plan.execute(ctx, filters, cache)
        }

        // Not a failure: the run paused. It travels as an error so the rest
        // of the plan does not execute, and carries everything the caller
        // needs to answer and resume.
        crate::effects::StepOutcome::Suspended { turn, reason } => Err(SomaError::Suspended {
            run_id,
            node_id: node_id.to_string(),
            turn,
            reason: serde_json::to_string(&reason).unwrap_or_else(|_| "unknown".into()),
        }),
    }
}

fn execute_node(
    node_id: &str,
    ctx: &mut Context,
    filters: &FilterLibrary,
    cache: &dyn CacheStore,
) -> Result<()> {
    let start = Instant::now();

    let filter = filters
        .get(node_id)
        .ok_or_else(|| SomaError::NodeNotFound(node_id.to_string()))?;
    let meta = filter.meta();

    let _span = tracing::info_span!("execute_node", %node_id).entered();

    let input = resolve_input(node_id, ctx);
    // Borrow state via Arc — cloning the inner Value here would deep-copy
    // potentially huge tensors (encoder outputs, model weights) on every
    // forward call. Arc::clone is a cheap atomic increment.
    let state = filters.get_state(node_id);
    let state_ref: &Value = state.as_deref().unwrap_or(&Value::Empty);

    // An unhashable state/input means "uncacheable", not an error:
    // execution proceeds, only memoization is skipped. Nondeterministic
    // forwards (declared via `deterministic = false`) are also excluded:
    // serving a recorded result would silently freeze what the user
    // expects to vary. (Their fit STATES still cache — any recorded
    // training result is acceptable, constructive-trace semantics.)
    let out_key = if meta.cacheable && meta.deterministic {
        match (
            somatize_core::cache::CacheKey::for_value(state_ref),
            ctx.input_hash(node_id, &input),
        ) {
            (Ok(state_hash), Ok(input_hash)) => {
                let key = somatize_core::cache::CacheKey::for_output(
                    &filter.config_hash(),
                    &state_hash,
                    &input_hash,
                );
                Some(salt_with_seed(key, ctx.seed))
            }
            _ => None,
        }
    } else {
        None
    };

    if let Some(key) = &out_key
        && let Ok(Some(cached)) = cache.get(key)
    {
        ctx.set(node_id.to_string(), cached);
        ctx.event_bus.emit(Event::NodeCacheHit {
            run_id: ctx.run_id.clone(),
            node_id: node_id.to_string(),
            key: key.clone(),
            tier: somatize_core::cache::CacheTier::Memory,
            load_time: start.elapsed(),
        });
        return Ok(());
    }

    if let Some(key) = &out_key {
        ctx.event_bus.emit(Event::NodeCacheMiss {
            run_id: ctx.run_id.clone(),
            node_id: node_id.to_string(),
            key: key.clone(),
        });
    }

    ctx.event_bus.emit(Event::NodeStarted {
        run_id: ctx.run_id.clone(),
        node_id: node_id.to_string(),
        kind: meta.kind,
    });

    // catch_unwind: a panic in a user filter must not crash the process
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        filter.forward(&input, state_ref)
    }));

    let result = match result {
        Ok(inner) => inner,
        Err(panic) => {
            let msg = panic
                .downcast_ref::<String>()
                .map(|s| s.as_str())
                .or_else(|| panic.downcast_ref::<&str>().copied())
                .unwrap_or("unknown panic");
            tracing::error!(node_id, "filter panicked: {msg}");
            Err(SomaError::Execution {
                node_id: node_id.to_string(),
                message: format!("filter panicked: {msg}"),
            })
        }
    };

    match result {
        Ok(output) => {
            let duration = start.elapsed();
            let summary = format!("{output}");
            if let Some(key) = &out_key {
                // Best-effort: a full cache disk must never fail the run.
                let origin = somatize_core::cache::Origin::Computed {
                    node_id: node_id.to_string(),
                    run_id: ctx.run_id.clone(),
                };
                if let Err(e) =
                    cache.put_computed(key, &output, &origin, duration, meta.deterministic)
                {
                    tracing::warn!(node_id, error = %e, "failed to cache node output");
                }
            }
            let vv = ctx.maybe_spill(node_id, output);
            ctx.set_virtual(node_id, vv);
            ctx.event_bus.emit(Event::NodeCompleted {
                run_id: ctx.run_id.clone(),
                node_id: node_id.to_string(),
                duration,
                output_summary: summary,
            });
            Ok(())
        }
        Err(e) => {
            tracing::error!(node_id, error = %e, "node execution failed");
            ctx.event_bus.emit(Event::NodeFailed {
                run_id: ctx.run_id.clone(),
                node_id: node_id.to_string(),
                error: e.to_string(),
            });
            Err(e)
        }
    }
}

/// Execute parallel branches concurrently using std::thread::scope.
///
/// Each branch gets a snapshot of the context. After all branches complete,
/// their new outputs are merged back into the main context.
fn execute_parallel(
    branches: &[ExecutionPlan],
    ctx: &mut Context,
    filters: &FilterLibrary,
    cache: &dyn CacheStore,
) -> Result<()> {
    // What a branch contributes is what it *wrote*, not what happens to be
    // absent from the parent. Those differ the second time a parallel block
    // runs: inside a `Loop`, every body node already has a value from the
    // previous iteration, so filtering by "not already present" discarded
    // every fresh result and left downstream fan-in reading iteration one
    // forever — the nodes re-ran and their new outputs went nowhere.
    //
    // `execution_order` is appended to by every `set`/`set_virtual`, and a
    // snapshot copies it, so whatever a branch appended past this mark is
    // exactly its write set.
    let order_mark = ctx.execution_order.len();

    // Use scoped threads for true parallelism without Send requirements
    let results: Vec<Result<Vec<(String, VirtualValue)>>> = std::thread::scope(|s| {
        let handles: Vec<_> = branches
            .iter()
            .map(|branch| {
                let mut branch_ctx = ctx.snapshot();
                s.spawn(move || {
                    execute(branch, &mut branch_ctx, filters, cache)?;
                    let written: std::collections::HashSet<&String> =
                        branch_ctx.execution_order[order_mark..].iter().collect();
                    let new_entries: Vec<(String, VirtualValue)> = written
                        .into_iter()
                        .filter_map(|k| branch_ctx.store.get(k).map(|v| (k.clone(), v.clone())))
                        .collect();
                    Ok(new_entries)
                })
            })
            .collect();

        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // Merge results and propagate first error
    for result in results {
        let entries = result?;
        for (key, vv) in entries {
            ctx.set_virtual(key, vv);
        }
    }

    Ok(())
}

/// Resolve a VirtualValue to a concrete Value, loading from DataStore if needed.
fn resolve_value(vv: &VirtualValue, data_store: &Option<Arc<dyn DataStore>>) -> Option<Value> {
    match vv {
        VirtualValue::Materialized { value, .. } => Some(value.clone()),
        VirtualValue::Cached { key, .. } => {
            // Try to load from DataStore
            if let Some(store) = data_store {
                let data_ref = somatize_core::store::DataRef::Cached {
                    cache_key: key.clone(),
                };
                store.get(&data_ref).ok()
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Resolve the input for a node from the context store using graph topology.
/// If a predecessor was spilled to DataStore, loads it back.
pub fn resolve_input(node_id: &str, ctx: &Context) -> Value {
    let preds = ctx.graph_info.predecessors(node_id);

    let resolve_node = |id: &str| -> Option<Value> {
        ctx.store
            .get(id)
            .and_then(|vv| resolve_value(vv, &ctx.data_store))
    };

    match preds.len() {
        0 => ctx
            .execution_order
            .last()
            .and_then(|id| resolve_node(id))
            .unwrap_or(Value::Empty),
        1 => resolve_node(&preds[0]).unwrap_or(Value::Empty),
        _ => {
            let mut merged = serde_json::Map::new();
            for pred_id in preds {
                if let Some(val) = resolve_node(pred_id) {
                    let json_val = val.to_plain_json();
                    merged.insert(pred_id.clone(), json_val);
                }
            }
            Value::json(serde_json::Value::Object(merged))
        }
    }
}

/// Execute a stream plan: chunk input, process through StreamExecutor, concatenate.
fn execute_stream(
    node_ids: &[String],
    chunk_size: usize,
    ctx: &mut Context,
    filters: &FilterLibrary,
    _cache: &dyn CacheStore,
) -> Result<()> {
    use crate::executors::{FittedFilter, StreamExecutor};

    let start = Instant::now();

    // Build FittedFilter list from the library in plan order.
    let fitted: Vec<FittedFilter> = node_ids
        .iter()
        .map(|nid| {
            let filter = filters
                .get(nid)
                .ok_or_else(|| SomaError::NodeNotFound(nid.clone()))?;
            let state = filters
                .get_state(nid)
                .unwrap_or_else(|| Arc::new(Value::Empty));
            Ok(FittedFilter {
                name: nid.clone(),
                filter,
                state,
            })
        })
        .collect::<Result<_>>()?;

    // Resolve input from the first node's predecessors.
    let first_id = node_ids
        .first()
        .ok_or_else(|| SomaError::Other("stream plan has no nodes".into()))?;
    let input = resolve_input(first_id, ctx);

    // Chunk the input along the first tensor dimension.
    let chunks = chunk_value(&input, chunk_size);

    // Process chunks through the stream executor. Chunk-level caching
    // needs an owned handle (the executor keeps it across chunks) —
    // provided via `Context::with_cache_arc`.
    let mut executor = StreamExecutor::new(fitted);
    if let Some(c) = ctx.cache_arc.clone() {
        executor = executor.with_cache(c);
    }

    let last_id = node_ids.last().unwrap();

    // Incrementally concatenate tensor outputs to avoid holding all chunks
    // in memory simultaneously. Each chunk is dropped after its data is
    // extracted, keeping peak memory proportional to the final output size
    // rather than O(n_chunks × chunk_size).
    let mut all_data: Vec<f64> = Vec::new();
    let mut result_shape: Option<Vec<usize>> = None;
    let mut non_tensor_output: Option<Value> = None;

    let mut append_output = |output: Value| {
        match &output {
            Value::Tensor { values, shape } => {
                if result_shape.is_none() {
                    result_shape = Some(shape.clone());
                }
                all_data.extend_from_slice(values.as_slice());
                // `output` is dropped here, freeing the chunk's allocation
            }
            _ => {
                non_tensor_output = Some(output);
            }
        }
    };

    for (i, chunk) in chunks.into_iter().enumerate() {
        ctx.event_bus.emit(Event::NodeStarted {
            run_id: ctx.run_id.clone(),
            node_id: format!("{last_id}#chunk_{i}"),
            kind: somatize_core::filter::FilterKind::Stateless,
        });
        if let Some(output) = executor.process_chunk(chunk)? {
            append_output(output);
        }
    }

    // Flush barrier filters.
    if let Some(flushed) = executor.flush()? {
        append_output(flushed);
    }

    // Build the final result.
    let result = if let Some(mut shape) = result_shape {
        // Tensor output: fix the first dimension to reflect total rows.
        let row_size: usize = shape.iter().skip(1).product::<usize>().max(1);
        shape[0] = all_data.len() / row_size;
        Value::tensor(all_data, shape)
    } else {
        non_tensor_output.unwrap_or(Value::Empty)
    };

    let duration = start.elapsed();
    ctx.set(last_id.clone(), result);
    ctx.event_bus.emit(Event::NodeCompleted {
        run_id: ctx.run_id.clone(),
        node_id: last_id.clone(),
        duration,
        output_summary: format!(
            "stream: {} chunks through {} filters",
            executor.chunks_processed(),
            node_ids.len()
        ),
    });

    Ok(())
}

/// Split a Value::Tensor along the first dimension into chunks.
fn chunk_value(x: &Value, chunk_size: usize) -> Vec<Value> {
    match x {
        Value::Tensor { values, shape } if !values.is_empty() && chunk_size > 0 => {
            let row_size = if shape.len() > 1 {
                shape[1..].iter().product()
            } else {
                1
            };
            let n_rows = shape[0];
            let mut chunks = Vec::new();
            for start in (0..n_rows).step_by(chunk_size) {
                let end = (start + chunk_size).min(n_rows);
                let flat_start = start * row_size;
                let flat_end = end * row_size;
                let chunk_vals = values[flat_start..flat_end].to_vec();
                let mut chunk_shape = shape.clone();
                chunk_shape[0] = end - start;
                chunks.push(Value::tensor(chunk_vals, chunk_shape));
            }
            chunks
        }
        _ => vec![x.clone()],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::MemoryCache;
    use somatize_core::cache::CacheKey;
    use somatize_core::filter::{Filter, FilterKind, FilterMeta, StreamMode};

    struct DoublerFilter;

    impl Filter for DoublerFilter {
        fn config_hash(&self) -> CacheKey {
            CacheKey::from_parts(&[b"Doubler"])
        }
        fn fit(&self, _x: &Value, _y: Option<&Value>) -> Result<Value> {
            Ok(Value::Empty)
        }
        fn forward(&self, x: &Value, _state: &Value) -> Result<Value> {
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
                cacheable: true,
                differentiable: true,
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

    struct AdderFilter {
        amount: f64,
    }

    impl Filter for AdderFilter {
        fn config_hash(&self) -> CacheKey {
            CacheKey::from_parts(&[b"Adder", &self.amount.to_le_bytes()])
        }
        fn fit(&self, _x: &Value, _y: Option<&Value>) -> Result<Value> {
            Ok(Value::Empty)
        }
        fn forward(&self, x: &Value, _state: &Value) -> Result<Value> {
            match x {
                Value::Tensor { values, shape } => {
                    let added: Vec<f64> = values.iter().map(|v| v + self.amount).collect();
                    Ok(Value::tensor(added, shape.clone()))
                }
                _ => Ok(x.clone()),
            }
        }
        fn meta(&self) -> FilterMeta {
            FilterMeta {
                name: "Adder".into(),
                kind: FilterKind::Stateless,
                cacheable: true,
                differentiable: true,
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

    /// Slow filter that sleeps to verify parallelism.
    struct SlowFilter {
        id: String,
        delay_ms: u64,
    }

    impl Filter for SlowFilter {
        fn config_hash(&self) -> CacheKey {
            CacheKey::from_parts(&[b"Slow", self.id.as_bytes()])
        }
        fn fit(&self, _x: &Value, _y: Option<&Value>) -> Result<Value> {
            Ok(Value::Empty)
        }
        fn forward(&self, x: &Value, _state: &Value) -> Result<Value> {
            std::thread::sleep(std::time::Duration::from_millis(self.delay_ms));
            Ok(x.clone())
        }
        fn meta(&self) -> FilterMeta {
            FilterMeta {
                name: format!("Slow_{}", self.id),
                kind: FilterKind::Stateless,
                cacheable: false,
                differentiable: true,
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

    fn setup() -> (Arc<EventBus>, MemoryCache) {
        (Arc::new(EventBus::new(64)), MemoryCache::default())
    }

    #[test]
    fn execute_single_node() {
        let (bus, cache) = setup();
        let mut ctx = Context::new(bus, "run_1");
        ctx.set("input", Value::tensor(vec![1.0, 2.0, 3.0], vec![3]));
        ctx.graph_info
            .set_predecessors("doubler", vec!["input".into()]);

        let mut filters = FilterLibrary::new();
        filters.register("doubler", Box::new(DoublerFilter));

        let plan = ExecutionPlan::Execute {
            node_id: "doubler".into(),
        };

        execute(&plan, &mut ctx, &filters, &cache).unwrap();

        let result = ctx.get("doubler").unwrap();
        let (data, _) = result.as_tensor().unwrap();
        assert_eq!(data, &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn execute_sequence_with_graph_info() {
        let (bus, cache) = setup();
        let mut ctx = Context::new(bus, "run_1");
        ctx.set("input", Value::tensor(vec![1.0, 2.0], vec![2]));

        let graph_info = GraphInfo::for_linear(&["input", "add", "double"]);
        ctx.graph_info = graph_info;

        let mut filters = FilterLibrary::new();
        filters.register("add", Box::new(AdderFilter { amount: 10.0 }));
        filters.register("double", Box::new(DoublerFilter));

        let plan = ExecutionPlan::Sequence(vec![
            ExecutionPlan::Execute {
                node_id: "add".into(),
            },
            ExecutionPlan::Execute {
                node_id: "double".into(),
            },
        ]);

        execute(&plan, &mut ctx, &filters, &cache).unwrap();

        let result = ctx.get("double").unwrap();
        let (data, _) = result.as_tensor().unwrap();
        assert_eq!(data, &[22.0, 24.0]);
    }

    #[test]
    fn execute_emits_events() {
        let bus = Arc::new(EventBus::new(64));
        let cache = MemoryCache::default();
        let mut rx = bus.subscribe();

        let mut ctx = Context::new(bus, "run_1");
        ctx.set("input", Value::tensor(vec![1.0], vec![1]));
        ctx.graph_info
            .set_predecessors("double", vec!["input".into()]);

        let mut filters = FilterLibrary::new();
        filters.register("double", Box::new(DoublerFilter));

        execute(
            &ExecutionPlan::Execute {
                node_id: "double".into(),
            },
            &mut ctx,
            &filters,
            &cache,
        )
        .unwrap();

        // Cacheable node, cold cache: miss → started → completed.
        let e1 = rx.try_recv().unwrap();
        assert!(matches!(e1, Event::NodeCacheMiss { .. }), "got {e1:?}");
        let e2 = rx.try_recv().unwrap();
        assert!(matches!(e2, Event::NodeStarted { .. }), "got {e2:?}");
        let e3 = rx.try_recv().unwrap();
        assert!(matches!(e3, Event::NodeCompleted { .. }), "got {e3:?}");
    }

    #[test]
    fn execute_missing_filter_errors() {
        let (bus, cache) = setup();
        let mut ctx = Context::new(bus, "run_1");
        let filters = FilterLibrary::new();

        let result = execute(
            &ExecutionPlan::Execute {
                node_id: "nonexistent".into(),
            },
            &mut ctx,
            &filters,
            &cache,
        );
        assert!(matches!(result, Err(SomaError::NodeNotFound(_))));
    }

    #[test]
    fn execute_empty_plan() {
        let (bus, cache) = setup();
        let mut ctx = Context::new(bus, "run_1");
        let filters = FilterLibrary::new();
        execute(&ExecutionPlan::Empty, &mut ctx, &filters, &cache).unwrap();
    }

    #[test]
    fn parallel_merge_keeps_rerun_outputs() {
        // Running the same parallel block twice must leave the *second*
        // results in the context. Merging by "keys the parent lacks" passed
        // the first pass and silently dropped every later one, so anything
        // downstream of a parallel body inside a `Loop` read iteration one
        // forever while the nodes dutifully re-ran.
        let (bus, cache) = setup();
        let mut ctx = Context::new(bus, "run_1");
        ctx.graph_info
            .set_predecessors("double", vec!["input".into()]);
        ctx.graph_info.set_predecessors("add", vec!["input".into()]);

        let mut filters = FilterLibrary::new();
        filters.register("double", Box::new(DoublerFilter));
        filters.register("add", Box::new(AdderFilter { amount: 100.0 }));

        let plan = ExecutionPlan::Parallel(vec![
            ExecutionPlan::Execute {
                node_id: "double".into(),
            },
            ExecutionPlan::Execute {
                node_id: "add".into(),
            },
        ]);

        ctx.set("input", Value::tensor(vec![5.0], vec![1]));
        execute(&plan, &mut ctx, &filters, &cache).unwrap();
        assert_eq!(ctx.get("double").unwrap().as_tensor().unwrap().0, &[10.0]);

        // Same plan, new input — as a second loop iteration would.
        ctx.set("input", Value::tensor(vec![7.0], vec![1]));
        execute(&plan, &mut ctx, &filters, &cache).unwrap();

        assert_eq!(
            ctx.get("double").unwrap().as_tensor().unwrap().0,
            &[14.0],
            "second pass output was discarded by the merge"
        );
        assert_eq!(ctx.get("add").unwrap().as_tensor().unwrap().0, &[107.0]);
    }

    #[test]
    fn execute_parallel_branches_merge_outputs() {
        let (bus, cache) = setup();
        let mut ctx = Context::new(bus, "run_1");
        ctx.set("input", Value::tensor(vec![5.0], vec![1]));
        ctx.graph_info
            .set_predecessors("double", vec!["input".into()]);
        ctx.graph_info.set_predecessors("add", vec!["input".into()]);

        let mut filters = FilterLibrary::new();
        filters.register("double", Box::new(DoublerFilter));
        filters.register("add", Box::new(AdderFilter { amount: 100.0 }));

        let plan = ExecutionPlan::Parallel(vec![
            ExecutionPlan::Execute {
                node_id: "double".into(),
            },
            ExecutionPlan::Execute {
                node_id: "add".into(),
            },
        ]);

        execute(&plan, &mut ctx, &filters, &cache).unwrap();

        let double_out = ctx.get("double").unwrap().as_tensor().unwrap().0;
        assert_eq!(double_out, &[10.0]);
        let add_out = ctx.get("add").unwrap().as_tensor().unwrap().0;
        assert_eq!(add_out, &[105.0]);
    }

    #[test]
    fn parallel_branches_run_concurrently() {
        let (bus, cache) = setup();
        let mut ctx = Context::new(bus, "run_1");
        ctx.set("input", Value::tensor(vec![1.0], vec![1]));
        ctx.graph_info
            .set_predecessors("slow_a", vec!["input".into()]);
        ctx.graph_info
            .set_predecessors("slow_b", vec!["input".into()]);

        let mut filters = FilterLibrary::new();
        filters.register(
            "slow_a",
            Box::new(SlowFilter {
                id: "a".into(),
                delay_ms: 200,
            }),
        );
        filters.register(
            "slow_b",
            Box::new(SlowFilter {
                id: "b".into(),
                delay_ms: 200,
            }),
        );

        let plan = ExecutionPlan::Parallel(vec![
            ExecutionPlan::Execute {
                node_id: "slow_a".into(),
            },
            ExecutionPlan::Execute {
                node_id: "slow_b".into(),
            },
        ]);

        let start = Instant::now();
        execute(&plan, &mut ctx, &filters, &cache).unwrap();
        let elapsed = start.elapsed();

        // If truly parallel: ~200ms. If sequential: ~400ms. The wide
        // margin keeps the discrimination robust on loaded CI runners.
        assert!(
            elapsed.as_millis() < 350,
            "parallel branches took {}ms, expected <350ms (sequential would be ~400ms)",
            elapsed.as_millis()
        );

        assert!(ctx.get("slow_a").is_some());
        assert!(ctx.get("slow_b").is_some());
    }

    #[test]
    fn resolve_input_single_predecessor() {
        let bus = Arc::new(EventBus::new(8));
        let mut ctx = Context::new(bus, "r");
        ctx.set("A", Value::tensor(vec![42.0], vec![1]));
        ctx.graph_info.set_predecessors("B", vec!["A".into()]);

        let input = resolve_input("B", &ctx);
        let (data, _) = input.as_tensor().unwrap();
        assert_eq!(data, &[42.0]);
    }

    #[test]
    fn resolve_input_multiple_predecessors() {
        let bus = Arc::new(EventBus::new(8));
        let mut ctx = Context::new(bus, "r");
        ctx.set("A", Value::tensor(vec![1.0], vec![1]));
        ctx.set("B", Value::tensor(vec![2.0], vec![1]));
        ctx.graph_info
            .set_predecessors("C", vec!["A".into(), "B".into()]);

        let input = resolve_input("C", &ctx);
        let json = input.as_json().unwrap();
        assert!(json.get("A").is_some());
        assert!(json.get("B").is_some());
    }

    #[test]
    fn resolve_input_no_predecessors_fallback() {
        let bus = Arc::new(EventBus::new(8));
        let mut ctx = Context::new(bus, "r");
        ctx.set("prev", Value::tensor(vec![7.0], vec![1]));

        let input = resolve_input("root", &ctx);
        let (data, _) = input.as_tensor().unwrap();
        assert_eq!(data, &[7.0]);
    }

    #[test]
    fn graph_info_from_linear() {
        let info = GraphInfo::for_linear(&["a", "b", "c"]);
        assert!(info.predecessors("a").is_empty());
        assert_eq!(info.predecessors("b"), &["a"]);
        assert_eq!(info.predecessors("c"), &["b"]);
    }

    #[test]
    fn execute_stream_chunks_input() {
        let (bus, cache) = setup();
        let mut ctx = Context::new(bus, "run_stream");
        // 6-element input, chunk_size=2 → 3 chunks
        ctx.set(
            "__input__",
            Value::tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6]),
        );
        ctx.graph_info
            .set_predecessors("double", vec!["__input__".into()]);

        let mut filters = FilterLibrary::new();
        filters.register("double", Box::new(DoublerFilter));

        let plan = ExecutionPlan::Stream {
            node_ids: vec!["double".into()],
            chunk_size: 2,
        };

        execute(&plan, &mut ctx, &filters, &cache).unwrap();

        let result = ctx.get("double").unwrap();
        let (data, shape) = result.as_tensor().unwrap();
        assert_eq!(data, &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
        assert_eq!(shape, &[6]);
    }

    #[test]
    fn execute_stream_chain() {
        let (bus, cache) = setup();
        let mut ctx = Context::new(bus, "run_stream_chain");
        ctx.set(
            "__input__",
            Value::tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4]),
        );
        ctx.graph_info
            .set_predecessors("double", vec!["__input__".into()]);
        ctx.graph_info
            .set_predecessors("add", vec!["double".into()]);

        let mut filters = FilterLibrary::new();
        filters.register("double", Box::new(DoublerFilter));
        filters.register("add", Box::new(AdderFilter { amount: 10.0 }));

        let plan = ExecutionPlan::Stream {
            node_ids: vec!["double".into(), "add".into()],
            chunk_size: 2,
        };

        execute(&plan, &mut ctx, &filters, &cache).unwrap();

        // double → add: [1,2,3,4] → [2,4,6,8] → [12,14,16,18]
        let result = ctx.get("add").unwrap();
        let (data, shape) = result.as_tensor().unwrap();
        assert_eq!(data, &[12.0, 14.0, 16.0, 18.0]);
        assert_eq!(shape, &[4]);
    }

    /// Counts forward() invocations — the probe for cache-hit tests.
    struct CountingFilter {
        forwards: Arc<std::sync::atomic::AtomicUsize>,
        cacheable: bool,
        config: f64,
    }

    impl Filter for CountingFilter {
        fn config_hash(&self) -> CacheKey {
            CacheKey::from_parts(&[b"Counting", &self.config.to_le_bytes()])
        }
        fn fit(&self, _x: &Value, _y: Option<&Value>) -> Result<Value> {
            Ok(Value::Empty)
        }
        fn forward(&self, x: &Value, _state: &Value) -> Result<Value> {
            self.forwards
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            match x {
                Value::Tensor { values, shape } => {
                    let out: Vec<f64> = values.iter().map(|v| v + self.config).collect();
                    Ok(Value::tensor(out, shape.clone()))
                }
                _ => Ok(x.clone()),
            }
        }
        fn meta(&self) -> FilterMeta {
            FilterMeta {
                name: "Counting".into(),
                kind: FilterKind::Stateless,
                cacheable: self.cacheable,
                differentiable: true,
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

    fn counting_setup(
        cacheable: bool,
    ) -> (
        FilterLibrary,
        Arc<std::sync::atomic::AtomicUsize>,
        GraphInfo,
    ) {
        let forwards = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut filters = FilterLibrary::new();
        filters.register(
            "a",
            Box::new(CountingFilter {
                forwards: forwards.clone(),
                cacheable,
                config: 1.0,
            }),
        );
        filters.register(
            "b",
            Box::new(CountingFilter {
                forwards: forwards.clone(),
                cacheable,
                config: 2.0,
            }),
        );
        let info = GraphInfo::for_linear(&["input", "a", "b"]);
        (filters, forwards, info)
    }

    fn run_chain(cache: &dyn CacheStore, filters: &FilterLibrary, info: &GraphInfo) -> Value {
        let bus = Arc::new(EventBus::new(64));
        let mut ctx = Context::new(bus, "run").with_graph_info(info.clone());
        ctx.set("input", Value::tensor(vec![1.0, 2.0], vec![2]));
        let plan = ExecutionPlan::Sequence(vec![
            ExecutionPlan::Execute {
                node_id: "a".into(),
            },
            ExecutionPlan::Execute {
                node_id: "b".into(),
            },
        ]);
        execute(&plan, &mut ctx, filters, cache).unwrap();
        ctx.get("b").unwrap().clone()
    }

    #[test]
    fn second_run_hits_cache_and_skips_execution() {
        let (filters, forwards, info) = counting_setup(true);
        let cache = MemoryCache::default();

        let first = run_chain(&cache, &filters, &info);
        assert_eq!(forwards.load(std::sync::atomic::Ordering::SeqCst), 2);

        let second = run_chain(&cache, &filters, &info);
        assert_eq!(
            forwards.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "second run must not execute any filter"
        );
        assert_eq!(first, second);
    }

    #[test]
    fn uncacheable_filter_always_executes() {
        let (filters, forwards, info) = counting_setup(false);
        let cache = MemoryCache::default();

        run_chain(&cache, &filters, &info);
        run_chain(&cache, &filters, &info);
        assert_eq!(forwards.load(std::sync::atomic::Ordering::SeqCst), 4);
    }

    #[test]
    fn cache_survives_process_restart() {
        use crate::cache::LocalCache;
        let dir = std::env::temp_dir().join(format!(
            "soma_exec_restart_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let (filters, forwards, info) = counting_setup(true);

        {
            let cache = LocalCache::new(&dir).unwrap();
            run_chain(&cache, &filters, &info);
        }
        assert_eq!(forwards.load(std::sync::atomic::Ordering::SeqCst), 2);

        // "Restart": a fresh cache instance over the same directory.
        {
            let cache = LocalCache::new(&dir).unwrap();
            run_chain(&cache, &filters, &info);
        }
        assert_eq!(
            forwards.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "after restart the persisted cache must serve both nodes"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn different_input_misses_cache() {
        let (filters, forwards, info) = counting_setup(true);
        let cache = MemoryCache::default();

        run_chain(&cache, &filters, &info);

        let bus = Arc::new(EventBus::new(64));
        let mut ctx = Context::new(bus, "run2").with_graph_info(info.clone());
        ctx.set("input", Value::tensor(vec![9.0, 9.0], vec![2]));
        let plan = ExecutionPlan::Sequence(vec![
            ExecutionPlan::Execute {
                node_id: "a".into(),
            },
            ExecutionPlan::Execute {
                node_id: "b".into(),
            },
        ]);
        execute(&plan, &mut ctx, &filters, &cache).unwrap();
        assert_eq!(
            forwards.load(std::sync::atomic::Ordering::SeqCst),
            4,
            "different input data must not hit the cache"
        );
    }

    #[test]
    fn cache_hit_emits_cache_hit_event() {
        let (filters, _forwards, info) = counting_setup(true);
        let cache = MemoryCache::default();
        run_chain(&cache, &filters, &info);

        let bus = Arc::new(EventBus::new(64));
        let mut rx = bus.subscribe();
        let mut ctx = Context::new(bus, "run2").with_graph_info(info.clone());
        ctx.set("input", Value::tensor(vec![1.0, 2.0], vec![2]));
        execute(
            &ExecutionPlan::Execute {
                node_id: "a".into(),
            },
            &mut ctx,
            &filters,
            &cache,
        )
        .unwrap();

        let event = rx.try_recv().unwrap();
        assert!(
            matches!(event, Event::NodeCacheHit { ref node_id, .. } if node_id == "a"),
            "expected NodeCacheHit for `a`, got: {event:?}"
        );
    }

    #[test]
    fn spill_roundtrip_through_datastore() {
        use somatize_core::store::LocalDataStore;
        let dir = std::env::temp_dir().join(format!(
            "soma_spill_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let store: Arc<dyn DataStore> = Arc::new(LocalDataStore::new(&dir));

        let (filters, _forwards, info) = counting_setup(true);
        let bus = Arc::new(EventBus::new(64));
        let mut ctx = Context::new(bus, "run_spill")
            .with_graph_info(info.clone())
            .with_data_store(store)
            .with_spill_threshold(1); // spill everything
        ctx.set("input", Value::tensor(vec![1.0, 2.0], vec![2]));

        let plan = ExecutionPlan::Sequence(vec![
            ExecutionPlan::Execute {
                node_id: "a".into(),
            },
            ExecutionPlan::Execute {
                node_id: "b".into(),
            },
        ]);
        execute(&plan, &mut ctx, &filters, &cache_for_spill())
            .expect("spilled intermediate must be readable downstream");

        // `a`'s output was spilled; `b` must still have received it:
        // input + 1.0 + 2.0 = [4.0, 5.0]
        let out = resolve_value(ctx.get_virtual("b").unwrap(), &ctx.data_store).unwrap();
        let (data, _) = out.as_tensor().unwrap();
        assert_eq!(data, &[4.0, 5.0]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    fn cache_for_spill() -> MemoryCache {
        MemoryCache::default()
    }

    /// Config carries a salt that does NOT affect the output — models a
    /// cosmetic/irrelevant config change upstream.
    struct SaltedFilter {
        salt: f64,
        forwards: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl Filter for SaltedFilter {
        fn config_hash(&self) -> CacheKey {
            CacheKey::from_parts(&[b"Salted", &self.salt.to_le_bytes()])
        }
        fn fit(&self, _x: &Value, _y: Option<&Value>) -> Result<Value> {
            Ok(Value::Empty)
        }
        fn forward(&self, x: &Value, _state: &Value) -> Result<Value> {
            self.forwards
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            match x {
                Value::Tensor { values, shape } => Ok(Value::tensor(
                    values.iter().map(|v| v + 1.0).collect(),
                    shape.clone(),
                )),
                _ => Ok(x.clone()),
            }
        }
        fn meta(&self) -> FilterMeta {
            FilterMeta {
                name: "Salted".into(),
                kind: FilterKind::Stateless,
                cacheable: true,
                differentiable: true,
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
    fn early_cutoff_downstream_hits_when_upstream_output_unchanged() {
        // Downstream keys derive from input CONTENT hashes, not from
        // upstream provenance — so a config change in A that produces
        // identical bytes must not invalidate B (rustc/salsa-style
        // early cutoff; impossible under the old deep-provenance keys).
        let a_forwards = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let b_forwards = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let cache = MemoryCache::default();
        let info = GraphInfo::for_linear(&["input", "a", "b"]);
        let plan = ExecutionPlan::Sequence(vec![
            ExecutionPlan::Execute {
                node_id: "a".into(),
            },
            ExecutionPlan::Execute {
                node_id: "b".into(),
            },
        ]);

        let run = |salt: f64| {
            let mut filters = FilterLibrary::new();
            filters.register(
                "a",
                Box::new(SaltedFilter {
                    salt,
                    forwards: a_forwards.clone(),
                }),
            );
            filters.register(
                "b",
                Box::new(CountingFilter {
                    forwards: b_forwards.clone(),
                    cacheable: true,
                    config: 2.0,
                }),
            );
            let bus = Arc::new(EventBus::new(64));
            let mut ctx = Context::new(bus, "run").with_graph_info(info.clone());
            ctx.set("input", Value::tensor(vec![1.0, 2.0], vec![2]));
            execute(&plan, &mut ctx, &filters, &cache).unwrap();
        };

        run(1.0);
        assert_eq!(a_forwards.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(b_forwards.load(std::sync::atomic::Ordering::SeqCst), 1);

        // A's config changed (new salt) → A re-executes. Its output is
        // byte-identical, so B must hit.
        run(2.0);
        assert_eq!(
            a_forwards.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "A's config changed, it must re-execute"
        );
        assert_eq!(
            b_forwards.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "B's input content is unchanged — early cutoff must serve it from cache"
        );
    }

    #[test]
    fn nondeterministic_filter_is_never_cached() {
        struct RandomishFilter {
            forwards: Arc<std::sync::atomic::AtomicUsize>,
        }
        impl Filter for RandomishFilter {
            fn config_hash(&self) -> CacheKey {
                CacheKey::from_parts(&[b"Randomish"])
            }
            fn fit(&self, _x: &Value, _y: Option<&Value>) -> Result<Value> {
                Ok(Value::Empty)
            }
            fn forward(&self, x: &Value, _state: &Value) -> Result<Value> {
                self.forwards
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(x.clone())
            }
            fn meta(&self) -> FilterMeta {
                FilterMeta {
                    name: "Randomish".into(),
                    kind: FilterKind::Stateless,
                    cacheable: true,
                    differentiable: false,
                    deterministic: false, // declared nondeterministic
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

        let forwards = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut filters = FilterLibrary::new();
        filters.register(
            "rng",
            Box::new(RandomishFilter {
                forwards: forwards.clone(),
            }),
        );
        let cache = MemoryCache::default();
        let info = GraphInfo::for_linear(&["input", "rng"]);
        for _ in 0..2 {
            let bus = Arc::new(EventBus::new(64));
            let mut ctx = Context::new(bus, "run").with_graph_info(info.clone());
            ctx.set("input", Value::tensor(vec![1.0], vec![1]));
            execute(
                &ExecutionPlan::Execute {
                    node_id: "rng".into(),
                },
                &mut ctx,
                &filters,
                &cache,
            )
            .unwrap();
        }
        assert_eq!(
            forwards.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "a filter declared nondeterministic must run every time"
        );
    }

    #[test]
    fn execute_stream_single_chunk() {
        let (bus, cache) = setup();
        let mut ctx = Context::new(bus, "run_stream_single");
        ctx.set("__input__", Value::tensor(vec![5.0, 10.0], vec![2]));
        ctx.graph_info
            .set_predecessors("double", vec!["__input__".into()]);

        let mut filters = FilterLibrary::new();
        filters.register("double", Box::new(DoublerFilter));

        // chunk_size larger than input → single chunk
        let plan = ExecutionPlan::Stream {
            node_ids: vec!["double".into()],
            chunk_size: 1000,
        };

        execute(&plan, &mut ctx, &filters, &cache).unwrap();

        let result = ctx.get("double").unwrap();
        let (data, _) = result.as_tensor().unwrap();
        assert_eq!(data, &[10.0, 20.0]);
    }
}
