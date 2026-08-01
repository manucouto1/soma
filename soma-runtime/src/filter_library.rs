//! Unified filter registry — holds implementations, metadata, and trained states.
//!
//! [`FilterLibrary`] is the single registry for the entire pipeline:
//! the compiler reads metadata via [`FilterRegistry`], and the executor
//! reads filters and states directly. No intermediate conversion needed.
//!
//! States live in a pluggable [`StateStore`] — by default
//! [`MemoryStateStore`], but users can inject a disk- or S3-backed store
//! for pipelines whose trained states don't fit comfortably in RAM.

use somatize_compiler::FilterRegistry;
use somatize_core::cache::CacheKey;
use somatize_core::error::Result;
use somatize_core::filter::{Filter, FilterMeta};
use somatize_core::state::{MemoryStateStore, StateStore};
use somatize_core::value::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// Unified registry of filter implementations, metadata, and trained states.
///
/// ```ignore
/// let mut lib = FilterLibrary::new();
/// lib.register("scaler", Box::new(MyScaler { scale: 2.0 }));
/// lib.register("model", Box::new(MyModel::default()));
///
/// // Use as compiler registry
/// let result = somatize_compiler::compile(&graph, &lib, mode, cache)?;
///
/// // Use directly with executor — no conversion needed
/// executor::execute(&plan, &mut ctx, &lib, &cache)?;
/// ```
/// Cloning shares both the filters and the state store, so a clone sees
/// whatever the original has fitted. That is what lets one library serve
/// many graph runs — an agent running pipelines back to back, for instance.
#[derive(Clone)]
pub struct FilterLibrary {
    filters: HashMap<String, Arc<dyn Filter>>,
    states: Arc<dyn StateStore>,
}

impl FilterLibrary {
    /// Create a new library with an in-memory state store.
    pub fn new() -> Self {
        Self {
            filters: HashMap::new(),
            states: Arc::new(MemoryStateStore::new()),
        }
    }

    /// Create a library with a custom [`StateStore`] backend.
    pub fn with_state_store(states: Arc<dyn StateStore>) -> Self {
        Self {
            filters: HashMap::new(),
            states,
        }
    }

    /// Register a filter for a given node ID.
    pub fn register(&mut self, node_id: impl Into<String>, filter: Box<dyn Filter>) {
        self.filters.insert(node_id.into(), Arc::from(filter));
    }

    /// Number of registered filters.
    pub fn len(&self) -> usize {
        self.filters.len()
    }

    /// Whether the library is empty.
    pub fn is_empty(&self) -> bool {
        self.filters.is_empty()
    }

    /// Get a filter by node ID.
    pub fn get(&self, node_id: &str) -> Option<Arc<dyn Filter>> {
        self.filters.get(node_id).cloned()
    }

    /// Store a trained state for a node.
    ///
    /// Errors bubble up from the underlying [`StateStore`] (e.g. I/O on
    /// a disk-backed backend). The in-memory default never fails.
    pub fn try_set_state(&self, node_id: impl Into<String>, state: Value) -> Result<()> {
        let id = node_id.into();
        self.states.set(&id, state)
    }

    /// Store a trained state, reporting a store failure through the log.
    ///
    /// Prefer [`FilterLibrary::try_set_state`]: a state that failed to
    /// store is a state the next `forward` will not find, and only the
    /// caller knows whether that is worth stopping for.
    #[deprecated(
        since = "0.3.2",
        note = "use `try_set_state`, which reports a failing StateStore instead of swallowing it"
    )]
    pub fn set_state(&self, node_id: impl Into<String>, state: Value) {
        let id = node_id.into();
        if let Err(e) = self.states.set(&id, state) {
            tracing::error!(node_id = %id, "StateStore::set failed: {e}");
        }
    }

    /// Retrieve the trained state for a node. The returned `Arc<Value>`
    /// can be dereferenced (`&*arc`) for the forward hot path without
    /// cloning the underlying value.
    pub fn get_state(&self, node_id: &str) -> Option<Arc<Value>> {
        self.states.get(node_id).ok().flatten()
    }

    /// Drop all stored states (but keep filters).
    pub fn clear_states(&self) {
        let _ = self.states.clear();
    }

    /// Access the underlying [`StateStore`] (e.g. to share it across
    /// sessions or inspect its contents).
    pub fn state_store(&self) -> &Arc<dyn StateStore> {
        &self.states
    }
}

impl Default for FilterLibrary {
    fn default() -> Self {
        Self::new()
    }
}

/// Implements [`FilterRegistry`] so the compiler can read metadata
/// directly from the registered filter implementations.
impl FilterRegistry for FilterLibrary {
    fn meta(&self, node_id: &str) -> Option<FilterMeta> {
        self.filters.get(node_id).map(|f| f.meta())
    }

    fn config_hash(&self, node_id: &str) -> Option<CacheKey> {
        self.filters.get(node_id).map(|f| f.config_hash())
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

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[test]
    fn register_and_query() {
        let mut lib = FilterLibrary::new();
        lib.register("a", Box::new(DummyFilter { name: "A".into() }));
        lib.register("b", Box::new(DummyFilter { name: "B".into() }));

        assert_eq!(lib.len(), 2);
        assert!(lib.get("a").is_some());
        assert!(lib.get("missing").is_none());
    }

    #[test]
    fn implements_filter_registry() {
        let mut lib = FilterLibrary::new();
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
        let lib = FilterLibrary::with_state_store(Arc::new(FailingStateStore));

        let err = lib.try_set_state("a", Value::Empty).unwrap_err();
        assert!(err.to_string().contains("disk full"), "got: {err}");
    }

    #[test]
    fn state_management() {
        let mut lib = FilterLibrary::new();
        lib.register("a", Box::new(DummyFilter { name: "A".into() }));

        assert!(lib.get_state("a").is_none());

        lib.try_set_state("a", Value::json(serde_json::json!({"mean": 5.0})))
            .unwrap();
        let state = lib.get_state("a").unwrap();
        assert_eq!(state.as_json().unwrap()["mean"], 5.0);
    }

    #[test]
    fn clear_states_keeps_filters() {
        let mut lib = FilterLibrary::new();
        lib.register("a", Box::new(DummyFilter { name: "A".into() }));
        lib.try_set_state("a", Value::Empty).unwrap();

        assert!(lib.get_state("a").is_some());
        lib.clear_states();
        assert!(lib.get_state("a").is_none());
        assert!(lib.get("a").is_some());
    }
}
