//! Runner module — trait-based execution contracts.
//!
//! A [`Runner`] defines the contract for executing plans (fit + forward).
//! [`LocalRunner`] executes locally using the Executor.
//! The worker prepares the environment and delegates to `LocalRunner`.

pub mod local;
pub mod remote;

use somatize_compiler::ExecutionPlan;
use somatize_core::cache::CacheStore;
use somatize_core::data::value::Value;
use somatize_core::error::Result;
use std::collections::HashMap;

use crate::EventBus;
use crate::execution::executor::GraphInfo;
use crate::execution::node_catalog::NodeCatalog;
use std::sync::Arc;

/// Everything a runner needs besides the plan and the data.
///
/// A struct rather than six more parameters, and one of them is the point:
/// `graph_info`. Both runner methods used to build
/// `GraphInfo::for_linear(plan.node_ids())` — chaining the plan's nodes in
/// flattened order as if every graph were a chain. On a diamond that is
/// simply wrong: `GraphSession::forward` on `a → {b, c} → d` answered with
/// `d(c(...))`, `d` never seeing `b` and `a` never seeing the input.
///
/// The caller supplies the real topology now. A caller that genuinely has
/// only a plan can still pass `GraphInfo::for_linear`, but it has to say so.
pub struct RunContext<'a> {
    /// Implementations and trained states for every node in the plan.
    pub catalog: &'a NodeCatalog,
    /// Output cache consulted and filled by `run_node`.
    pub cache: &'a dyn CacheStore,
    /// Bus the run emits its node events on.
    pub events: &'a Arc<EventBus>,
    /// Tags every node event of this run — callers that emit a
    /// `RunStarted`/`RunCompleted` bracket pass the same id so readers can
    /// group a run's events.
    pub run_id: &'a str,
    /// The real topology for input resolution — the reason this struct
    /// exists; see the type docs.
    pub graph_info: GraphInfo,
    /// The run's experiment seed, folded into every cache key.
    ///
    /// Without it two seeds share a state cache line, so the second one
    /// trains on the first one's recorded state and the sweep measures
    /// one seed N times. Only the Python fit path used to salt.
    pub seed: Option<i64>,
    /// Performs and journals step effects.
    ///
    /// Needed only when the plan contains a step. It lives here rather than
    /// being built inside the runner because a driver carries the journal —
    /// which is what makes a resumed run replay instead of re-calling a
    /// model — and only the caller knows where that journal lives.
    pub driver: Option<crate::agentic::EffectDriver>,
}

impl<'a> RunContext<'a> {
    /// A context over the real topology; use [`Self::linear`] only when a
    /// plan is genuinely all you have.
    pub fn new(
        catalog: &'a NodeCatalog,
        cache: &'a dyn CacheStore,
        events: &'a Arc<EventBus>,
        run_id: &'a str,
        graph_info: GraphInfo,
    ) -> Self {
        Self {
            catalog,
            cache,
            events,
            run_id,
            graph_info,
            seed: None,
            driver: None,
        }
    }

    /// Fold this run's seed into the cache keys.
    pub fn with_seed(mut self, seed: Option<i64>) -> Self {
        self.seed = seed;
        self
    }

    /// Register the effect driver a plan containing steps needs.
    ///
    /// The driver should already carry its catalog
    /// ([`crate::agentic::EffectDriver::with_catalog`]) if a step may fan
    /// out dynamically — the same rule as
    /// [`crate::execution::executor::Context::with_driver`], so the two entry points
    /// cannot drift apart on who attaches it.
    pub fn with_driver(mut self, driver: crate::agentic::EffectDriver) -> Self {
        self.driver = Some(driver);
        self
    }

    /// For a caller that has only a plan: treat it as a chain.
    ///
    /// Correct for a linear pipeline and a fabrication for anything else,
    /// which is why it is spelled out at the call site rather than being
    /// what you get by default.
    pub fn linear(
        catalog: &'a NodeCatalog,
        cache: &'a dyn CacheStore,
        events: &'a Arc<EventBus>,
        run_id: &'a str,
        plan: &ExecutionPlan,
    ) -> Self {
        let ids = plan.node_ids();
        Self::new(catalog, cache, events, run_id, GraphInfo::for_linear(&ids))
    }

    /// Clone the driver for a run's own context.
    pub(crate) fn driver(&self) -> Option<crate::agentic::EffectDriver> {
        self.driver.clone()
    }
}

/// What a fit produced: what the nodes computed, and what they learned.
///
/// A fit used to answer with one `HashMap<String, Value>` holding both,
/// telling them apart by a `__state_` prefix on the key — and every one of
/// the four consumers separated them again, in its own three lines. They
/// did not all agree: the differentiable path in the Python bindings read
/// the map as though every entry were a state, so each node's *output* was
/// filed as its learned state and which one survived depended on `HashMap`
/// order.
///
/// The prefix is a key inside the runner's value store. It stops here.
#[derive(Debug, Clone)]
pub struct Fitted {
    /// What the last node that ran produced — the fit's "result" for a
    /// caller that wants one value.
    pub last: Value,
    /// What each node computed, by node id.
    pub outputs: HashMap<String, Value>,
    /// What each trainable node learned, by node id.
    pub states: HashMap<String, Value>,
}

impl Default for Fitted {
    fn default() -> Self {
        Self {
            last: Value::Empty,
            outputs: HashMap::new(),
            states: HashMap::new(),
        }
    }
}

/// Contract for executing plans. Every execution mode (local, remote, stream)
/// implements this trait. One interface, polymorphic dispatch.
pub trait Runner: Send + Sync {
    /// Train: fit each filter, forward to propagate outputs.
    fn fit(
        &self,
        plan: &ExecutionPlan,
        ctx: &RunContext<'_>,
        input: &Value,
        y: Option<&Value>,
    ) -> Result<Fitted>;

    /// Inference: forward data through the compiled plan.
    fn forward(&self, plan: &ExecutionPlan, ctx: &RunContext<'_>, input: &Value) -> Result<Value>;
}

pub use local::LocalRunner;
pub use remote::Transport;
