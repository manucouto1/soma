//! Unified filter registry — holds both implementations and metadata.
//!
//! [`FilterLibrary`] bridges the compiler (which needs [`FilterRegistry`])
//! and the executor (which needs [`FilterStore`]). Register filters once,
//! use everywhere.

use crate::executor::FilterStore;
use soma_compiler::FilterRegistry;
use soma_core::cache::CacheKey;
use soma_core::filter::{Filter, FilterMeta};
use std::collections::HashMap;
use std::sync::Arc;

/// A unified registry of filter implementations and their metadata.
///
/// ```ignore
/// let mut lib = FilterLibrary::new();
/// lib.register("scaler", Box::new(MyScaler { scale: 2.0 }));
/// lib.register("model", Box::new(MyModel::default()));
///
/// // Use as compiler registry
/// let result = soma_compiler::compile(&graph, &lib, mode, cache)?;
///
/// // Use as executor filter store
/// let store = lib.to_filter_store();
/// ```
pub struct FilterLibrary {
    filters: HashMap<String, Arc<dyn Filter>>,
}

impl FilterLibrary {
    pub fn new() -> Self {
        Self {
            filters: HashMap::new(),
        }
    }

    /// Register a filter for a given node ID.
    pub fn register(&mut self, node_id: impl Into<String>, filter: Box<dyn Filter>) {
        self.filters.insert(node_id.into(), Arc::from(filter));
    }

    /// Build a [`FilterStore`] for the executor, transferring all filters and states.
    pub fn to_filter_store(&self) -> FilterStore {
        let mut store = FilterStore::new();
        for (id, filter) in &self.filters {
            store.register(id.clone(), Box::new(ArcFilter(filter.clone())));
        }
        store
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

/// Thin wrapper to re-box an `Arc<dyn Filter>` into `Box<dyn Filter>`.
/// Needed because FilterStore takes `Box<dyn Filter>` but we store `Arc`.
struct ArcFilter(Arc<dyn Filter>);

impl Filter for ArcFilter {
    fn config_hash(&self) -> CacheKey {
        self.0.config_hash()
    }
    fn fit(&self, x: &soma_core::value::Value, y: Option<&soma_core::value::Value>) -> soma_core::error::Result<soma_core::value::Value> {
        self.0.fit(x, y)
    }
    fn forward(&self, x: &soma_core::value::Value, state: &soma_core::value::Value) -> soma_core::error::Result<soma_core::value::Value> {
        self.0.forward(x, state)
    }
    fn meta(&self) -> FilterMeta {
        self.0.meta()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use soma_core::error::Result;
    use soma_core::filter::{FilterKind, StreamMode};
    use soma_core::value::Value;

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
                stream_mode: StreamMode::FixedState,
                distribution: soma_core::filter::Distribution::Local,
                input_schema: None,
                output_schema: None,
            }
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
        lib.register("node_1", Box::new(DummyFilter { name: "Scaler".into() }));

        // FilterRegistry methods
        let meta = lib.meta("node_1").unwrap();
        assert_eq!(meta.name, "Scaler");
        assert!(meta.cacheable);

        let hash = lib.config_hash("node_1").unwrap();
        assert_eq!(hash, CacheKey::from_parts(&[b"Scaler"]));

        assert!(lib.meta("nonexistent").is_none());
    }

    #[test]
    fn to_filter_store_transfers_filters() {
        let mut lib = FilterLibrary::new();
        lib.register("a", Box::new(DummyFilter { name: "A".into() }));
        lib.register("b", Box::new(DummyFilter { name: "B".into() }));

        let store = lib.to_filter_store();
        assert!(store.get("a").is_some());
        assert!(store.get("b").is_some());
        assert!(store.get("c").is_none());
    }

    #[test]
    fn filter_store_filters_work() {
        let mut lib = FilterLibrary::new();
        lib.register("a", Box::new(DummyFilter { name: "A".into() }));

        let store = lib.to_filter_store();
        let filter = store.get("a").unwrap();
        let result = filter.forward(&Value::tensor(vec![1.0], vec![1]), &Value::Empty).unwrap();
        assert_eq!(result, Value::tensor(vec![1.0], vec![1]));
    }
}
