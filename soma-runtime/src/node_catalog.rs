//! One registry for every node in a graph — implementations, metadata,
//! and trained states.
//!
//! [`NodeCatalog`] holds both kinds of node. Filters and steps used to
//! live in separate registries joined by a borrow-pair adapter, which
//! meant three ways to answer "what is this node", and callers that
//! reached for the filter half alone silently skipped half the schema
//! validation. There is one place to ask now.
//!
//! The compiler reads metadata through [`NodeRegistry`]; the executor
//! reads implementations and states directly. No intermediate conversion.
//!
//! States live in a pluggable [`StateStore`] — by default
//! [`MemoryStateStore`], but users can inject a disk- or S3-backed store
//! for pipelines whose trained states don't fit comfortably in RAM. Steps
//! hold no trained state: their history is the journal.

use somatize_compiler::NodeRegistry;
use somatize_core::cache::CacheKey;
use somatize_core::error::Result;
use somatize_core::filter::Filter;
#[cfg(test)]
use somatize_core::filter::FilterMeta;
use somatize_core::node::NodeMeta;
use somatize_core::state::{MemoryStateStore, StateStore};
use somatize_core::step::Step;
use somatize_core::value::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// What sits behind a node id.
///
/// The only place in the workspace that names the two kinds. Everything
/// downstream asks [`NodeCatalog::node_meta`] instead, which answers for
/// both.
#[derive(Clone)]
pub enum NodeImpl {
    Filter(Arc<dyn Filter>),
    Step(Arc<dyn Step>),
}

impl NodeImpl {
    pub fn meta(&self) -> NodeMeta {
        match self {
            Self::Filter(f) => f.meta().into(),
            Self::Step(s) => s.meta().into(),
        }
    }

    pub fn config_hash(&self) -> CacheKey {
        match self {
            Self::Filter(f) => f.config_hash(),
            Self::Step(s) => s.config_hash(),
        }
    }
}

/// Every node a graph can execute, plus the states its filters have learned.
///
/// ```ignore
/// let mut lib = NodeCatalog::new();
/// lib.register("scaler", Box::new(MyScaler { scale: 2.0 }));
/// lib.register_step("researcher", Box::new(ReactStep::new("claude-opus-5")));
///
/// // Use as compiler registry
/// let result = somatize_compiler::compile(&graph, &lib, mode, cache)?;
///
/// // Use directly with executor — no conversion needed
/// executor::execute(&plan, &mut ctx, &lib, &cache)?;
/// ```
/// Cloning shares both the nodes and the state store, so a clone sees
/// whatever the original has fitted. That is what lets one catalog serve
/// many graph runs — an agent running pipelines back to back, for instance.
#[derive(Clone)]
pub struct NodeCatalog {
    nodes: HashMap<String, NodeImpl>,
    states: Arc<dyn StateStore>,
}

impl NodeCatalog {
    /// Create a new catalog with an in-memory state store.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            states: Arc::new(MemoryStateStore::new()),
        }
    }

    /// Create a catalog with a custom [`StateStore`] backend.
    pub fn with_state_store(states: Arc<dyn StateStore>) -> Self {
        Self {
            nodes: HashMap::new(),
            states,
        }
    }

    /// Register a filter for a given node ID.
    pub fn register(&mut self, node_id: impl Into<String>, filter: Box<dyn Filter>) {
        self.nodes
            .insert(node_id.into(), NodeImpl::Filter(Arc::from(filter)));
    }

    /// Register a step for a given node ID.
    pub fn register_step(&mut self, node_id: impl Into<String>, step: Box<dyn Step>) {
        self.nodes
            .insert(node_id.into(), NodeImpl::Step(Arc::from(step)));
    }

    /// Register an already-shared step, for callers that built one
    /// elsewhere (the Python bindings do).
    pub fn register_step_arc(&mut self, node_id: impl Into<String>, step: Arc<dyn Step>) {
        self.nodes.insert(node_id.into(), NodeImpl::Step(step));
    }

    /// Number of registered nodes, of either kind.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the catalog is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// What sits behind a node id.
    pub fn node(&self, node_id: &str) -> Option<&NodeImpl> {
        self.nodes.get(node_id)
    }

    /// A node's contract, whichever kind it is.
    pub fn node_meta(&self, node_id: &str) -> Option<NodeMeta> {
        self.nodes.get(node_id).map(NodeImpl::meta)
    }

    /// Get a filter by node ID. `None` if the id is a step, or unknown.
    pub fn get(&self, node_id: &str) -> Option<Arc<dyn Filter>> {
        match self.nodes.get(node_id) {
            Some(NodeImpl::Filter(f)) => Some(f.clone()),
            _ => None,
        }
    }

    /// Get a step by node ID. `None` if the id is a filter, or unknown.
    pub fn step(&self, node_id: &str) -> Option<Arc<dyn Step>> {
        match self.nodes.get(node_id) {
            Some(NodeImpl::Step(s)) => Some(s.clone()),
            _ => None,
        }
    }

    /// Does the catalog hold any effectful node at all?
    ///
    /// The question a caller asks before building an effect driver, which
    /// costs a provider catalog read and a journal directory.
    pub fn has_steps(&self) -> bool {
        self.nodes.values().any(|n| matches!(n, NodeImpl::Step(_)))
    }

    /// Copy every node of `other` into this catalog.
    ///
    /// Registering the same id twice with the same configuration is a no-op;
    /// the same id behind a *different* configuration is an error, because
    /// whichever one lost would silently answer for the other's cache
    /// entries. States are not merged — they follow this catalog's store.
    pub fn merge_from(&mut self, other: &NodeCatalog) -> somatize_core::error::Result<()> {
        for (id, node) in &other.nodes {
            if let Some(existing) = self.nodes.get(id)
                && existing.config_hash() != node.config_hash()
            {
                return Err(somatize_core::error::SomaError::Other(format!(
                    "node {id:?} is already registered with a different \
                     configuration; rename one of the two"
                )));
            }
            self.nodes.insert(id.clone(), node.clone());
        }
        Ok(())
    }

    /// Registered node ids, sorted, so listings are stable.
    pub fn node_ids(&self) -> Vec<&str> {
        let mut ids: Vec<&str> = self.nodes.keys().map(String::as_str).collect();
        ids.sort_unstable();
        ids
    }

    /// Learn a node's state, if it has one to learn.
    ///
    /// `None` means "nothing to learn here" — a stateless filter, an
    /// opaque one, or a step — which is a different answer from an error.
    pub fn learn(&self, node_id: &str, x: &Value, y: Option<&Value>) -> Option<Result<Value>> {
        match self.nodes.get(node_id)? {
            NodeImpl::Filter(f)
                if f.meta().kind == somatize_core::filter::FilterKind::Trainable =>
            {
                Some(f.fit(x, y))
            }
            _ => None,
        }
    }

    /// Store a trained state for a node.
    ///
    /// Errors bubble up from the underlying [`StateStore`] (e.g. I/O on
    /// a disk-backed backend). The in-memory default never fails.
    pub fn try_set_state(&self, node_id: impl Into<String>, state: Value) -> Result<()> {
        let id = node_id.into();
        self.states.set(&id, state)
    }

    /// Retrieve the trained state for a node. The returned `Arc<Value>`
    /// can be dereferenced (`&*arc`) for the forward hot path without
    /// cloning the underlying value.
    pub fn get_state(&self, node_id: &str) -> Option<Arc<Value>> {
        self.states.get(node_id).ok().flatten()
    }

    /// Drop all stored states (but keep the nodes).
    pub fn clear_states(&self) {
        let _ = self.states.clear();
    }

    /// Access the underlying [`StateStore`] (e.g. to share it across
    /// sessions or inspect its contents).
    pub fn state_store(&self) -> &Arc<dyn StateStore> {
        &self.states
    }
}

impl Default for NodeCatalog {
    fn default() -> Self {
        Self::new()
    }
}

/// Implements [`NodeRegistry`] so the compiler can read metadata directly
/// from the registered implementations — filters *and* steps.
///
/// This is what closed the hole where a caller passing the filter half
/// alone got the graph compiled with every step edge unchecked.
impl NodeRegistry for NodeCatalog {
    fn node_meta(&self, node_id: &str) -> Option<NodeMeta> {
        NodeCatalog::node_meta(self, node_id)
    }

    fn config_hash(&self, node_id: &str) -> Option<CacheKey> {
        self.nodes.get(node_id).map(NodeImpl::config_hash)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use somatize_core::error::Result;
    use somatize_core::filter::{FilterKind, StreamMode};

    struct DummyFilter {
        name: String,
    }

    impl Filter for DummyFilter {
        fn config_hash(&self) -> CacheKey {
            CacheKey::from_parts(&[self.name.as_bytes()])
        }
        fn fit(&self, _x: &Value, _y: Option<&Value>) -> Result<Value> {
            Ok(Value::Empty)
        }
        fn forward(&self, x: &Value, _state: &Value) -> Result<Value> {
            Ok(x.clone())
        }
        fn meta(&self) -> FilterMeta {
            FilterMeta {
                name: self.name.clone(),
                kind: FilterKind::Stateless,
                cacheable: true,
                differentiable: false,
                deterministic: true,
                stream_mode: StreamMode::FixedState,
                distribution: somatize_core::filter::Distribution::Local,
                input_schema: None,
                output_schema: None,
            }
        }
    }

    #[test]
    fn register_and_query() {
        let mut lib = NodeCatalog::new();
        lib.register("a", Box::new(DummyFilter { name: "A".into() }));
        lib.register("b", Box::new(DummyFilter { name: "B".into() }));

        assert_eq!(lib.len(), 2);
        assert!(lib.get("a").is_some());
        assert!(lib.get("missing").is_none());
    }

    #[test]
    fn implements_filter_registry() {
        let mut lib = NodeCatalog::new();
        lib.register(
            "node_1",
            Box::new(DummyFilter {
                name: "Scaler".into(),
            }),
        );

        let meta = lib.meta("node_1").unwrap();
        assert_eq!(meta.name, "Scaler");
        assert!(meta.cacheable);

        let hash = lib.config_hash("node_1").unwrap();
        assert_eq!(hash, CacheKey::from_parts(&[b"Scaler"]));

        assert!(lib.meta("nonexistent").is_none());
    }

    /// A store that refuses every write, the way a full disk or a revoked
    /// S3 credential would.
    struct FailingStateStore;

    impl StateStore for FailingStateStore {
        fn set(&self, _node_id: &str, _state: Value) -> Result<()> {
            Err(somatize_core::error::SomaError::Other("disk full".into()))
        }
        fn get(&self, _node_id: &str) -> Result<Option<Arc<Value>>> {
            Ok(None)
        }
        fn remove(&self, _node_id: &str) -> Result<()> {
            Ok(())
        }
        fn clear(&self) -> Result<()> {
            Ok(())
        }
        fn keys(&self) -> Result<Vec<String>> {
            Ok(Vec::new())
        }
    }

    /// A failing state store used to `panic!`, which aborts the host
    /// process — a library taking the whole application down because a
    /// disk filled up. It reports the failure now.
    #[test]
    fn a_failing_state_store_is_reported_not_fatal() {
        let lib = NodeCatalog::with_state_store(Arc::new(FailingStateStore));

        let err = lib.try_set_state("a", Value::Empty).unwrap_err();
        assert!(err.to_string().contains("disk full"), "got: {err}");
    }

    struct DummyStep {
        name: String,
    }

    impl Step for DummyStep {
        fn config_hash(&self) -> CacheKey {
            CacheKey::from_parts(&[self.name.as_bytes()])
        }
        fn meta(&self) -> somatize_core::step::StepMeta {
            somatize_core::step::StepMeta::new(&self.name)
        }
        fn poll(
            &self,
            _ctx: &somatize_core::step::StepCtx<'_>,
        ) -> Result<somatize_core::step::Transition> {
            Ok(somatize_core::step::Transition::Done(Value::Empty))
        }
    }

    /// `get()` and `step()` are typed views of one map: each answers only
    /// for its own kind, so a caller can never run a step as a filter (or
    /// the reverse) by holding the wrong accessor.
    #[test]
    fn a_step_registers_beside_filters_not_as_one() {
        let mut lib = NodeCatalog::new();
        lib.register("f", Box::new(DummyFilter { name: "F".into() }));
        lib.register_step("s", Box::new(DummyStep { name: "S".into() }));

        assert_eq!(lib.len(), 2);
        assert!(lib.step("s").is_some());
        assert!(lib.get("s").is_none(), "a step must not answer as a filter");
        assert!(
            lib.step("f").is_none(),
            "a filter must not answer as a step"
        );
        assert!(lib.step("missing").is_none());
    }

    /// The executor's output-cache guard reads `cacheable && deterministic`
    /// straight off `NodeMeta` — there is no `if is_step` anywhere. A step
    /// whose meta said anything but `false/false` would be silently
    /// memoized by content, freezing its first model answer forever.
    #[test]
    fn a_steps_node_meta_declares_it_effectful_and_uncacheable() {
        let mut lib = NodeCatalog::new();
        lib.register("f", Box::new(DummyFilter { name: "F".into() }));
        lib.register_step("s", Box::new(DummyStep { name: "S".into() }));

        let step_meta = lib.node_meta("s").unwrap();
        assert!(step_meta.effectful);
        assert!(!step_meta.cacheable);
        assert!(!step_meta.deterministic);

        let filter_meta = lib.node_meta("f").unwrap();
        assert!(!filter_meta.effectful);
        assert!(filter_meta.cacheable);
    }

    /// `has_steps` decides whether a session pays for an effect driver — a
    /// provider catalog read and a journal directory. Answering `true` for
    /// a filter-only catalog would charge every plain pipeline that cost.
    #[test]
    fn has_steps_flips_when_the_first_step_arrives() {
        let mut lib = NodeCatalog::new();
        assert!(!lib.has_steps());

        lib.register("f", Box::new(DummyFilter { name: "F".into() }));
        assert!(!lib.has_steps(), "a filter is not a step");

        lib.register_step("s", Box::new(DummyStep { name: "S".into() }));
        assert!(lib.has_steps());
    }

    /// `merge_from` copies both kinds; the same id under the same config is
    /// a no-op, and the same id under a *different* config is refused —
    /// whichever implementation lost would silently answer for the other's
    /// cache entries.
    #[test]
    fn merge_from_merges_and_rejects_a_config_collision() {
        let mut lib = NodeCatalog::new();
        lib.register("x", Box::new(DummyFilter { name: "X".into() }));

        let mut other = NodeCatalog::new();
        // Same id, same config: allowed. Plus one of each kind to copy in.
        other.register("x", Box::new(DummyFilter { name: "X".into() }));
        other.register("y", Box::new(DummyFilter { name: "Y".into() }));
        other.register_step("s", Box::new(DummyStep { name: "S".into() }));

        lib.merge_from(&other).unwrap();
        assert_eq!(lib.len(), 3);
        assert!(lib.get("y").is_some());
        assert!(lib.step("s").is_some(), "steps must merge too");

        let mut clashing = NodeCatalog::new();
        clashing.register(
            "x",
            Box::new(DummyFilter {
                name: "DIFFERENT".into(),
            }),
        );
        let err = lib.merge_from(&clashing).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("x"), "should name the colliding id: {msg}");
        assert!(msg.contains("different"), "should say why: {msg}");
    }

    #[test]
    fn state_management() {
        let mut lib = NodeCatalog::new();
        lib.register("a", Box::new(DummyFilter { name: "A".into() }));

        assert!(lib.get_state("a").is_none());

        lib.try_set_state("a", Value::json(serde_json::json!({"mean": 5.0})))
            .unwrap();
        let state = lib.get_state("a").unwrap();
        assert_eq!(state.as_json().unwrap()["mean"], 5.0);
    }

    #[test]
    fn clear_states_keeps_filters() {
        let mut lib = NodeCatalog::new();
        lib.register("a", Box::new(DummyFilter { name: "A".into() }));
        lib.try_set_state("a", Value::Empty).unwrap();

        assert!(lib.get_state("a").is_some());
        lib.clear_states();
        assert!(lib.get_state("a").is_none());
        assert!(lib.get("a").is_some());
    }
}
