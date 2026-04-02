//! Unified filter registry — holds implementations, metadata, and trained states.
//!
//! [`FilterLibrary`] is the single registry for the entire pipeline:
//! the compiler reads metadata via [`FilterRegistry`], and the executor
//! reads filters and states directly. No intermediate conversion needed.

use somatize_compiler::FilterRegistry;
use somatize_core::cache::CacheKey;
use somatize_core::filter::{Filter, FilterMeta};
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
pub struct FilterLibrary {
    filters: HashMap<String, Arc<dyn Filter>>,
    states: HashMap<String, Value>,
}

impl FilterLibrary {
    pub fn new() -> Self {
        Self {
            filters: HashMap::new(),
            states: HashMap::new(),
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
    pub fn set_state(&mut self, node_id: impl Into<String>, state: Value) {
        self.states.insert(node_id.into(), state);
    }

    /// Retrieve the trained state for a node.
    pub fn get_state(&self, node_id: &str) -> Option<&Value> {
        self.states.get(node_id)
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
                stream_mode: StreamMode::FixedState,
                distribution: somatize_core::filter::Distribution::Local,
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

    #[test]
    fn state_management() {
        let mut lib = FilterLibrary::new();
        lib.register("a", Box::new(DummyFilter { name: "A".into() }));

        assert!(lib.get_state("a").is_none());

        lib.set_state("a", Value::json(serde_json::json!({"mean": 5.0})));
        let state = lib.get_state("a").unwrap();
        assert_eq!(state.as_json().unwrap()["mean"], 5.0);
    }
}
