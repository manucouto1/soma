//! Registry of effectful steps, keyed by node id.
//!
//! Deliberately the mirror image of [`crate::filter_library::FilterLibrary`]:
//! the same registration shape, so a graph can mix computational and
//! effectful nodes without the caller learning a second idiom.
//!
//! Steps hold no trained state — that is the point of the split. A filter's
//! state is learned and cached; a step's state is the conversation it carries
//! in its values, and its history is the journal.

use crate::filter_library::FilterLibrary;
use somatize_compiler::FilterRegistry;
use somatize_core::cache::CacheKey;
use somatize_core::filter::FilterMeta;
use somatize_core::step::{Step, StepMeta};
use std::collections::HashMap;
use std::sync::Arc;

/// Registry of step implementations.
///
/// ```ignore
/// let mut steps = StepLibrary::new();
/// steps.register("researcher", Box::new(ReactStep::new("claude-opus-5", tools)));
/// ```
#[derive(Default, Clone)]
pub struct StepLibrary {
    steps: HashMap<String, Arc<dyn Step>>,
}

impl StepLibrary {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a step for a node id.
    pub fn register(&mut self, node_id: impl Into<String>, step: Box<dyn Step>) {
        self.steps.insert(node_id.into(), Arc::from(step));
    }

    /// Register an already-shared step, for callers that built one
    /// elsewhere (the Python bindings do).
    pub fn register_arc(&mut self, node_id: impl Into<String>, step: Arc<dyn Step>) {
        self.steps.insert(node_id.into(), step);
    }

    pub fn get(&self, node_id: &str) -> Option<Arc<dyn Step>> {
        self.steps.get(node_id).cloned()
    }

    pub fn contains(&self, node_id: &str) -> bool {
        self.steps.contains_key(node_id)
    }

    pub fn meta(&self, node_id: &str) -> Option<StepMeta> {
        self.steps.get(node_id).map(|s| s.meta())
    }

    pub fn len(&self) -> usize {
        self.steps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Registered node ids, sorted, so listings are stable.
    pub fn node_ids(&self) -> Vec<&str> {
        let mut ids: Vec<&str> = self.steps.keys().map(String::as_str).collect();
        ids.sort_unstable();
        ids
    }
}

/// A compiler registry spanning both libraries.
///
/// The compiler needs one place to ask "what does this node accept, and what
/// does it produce", whether the node is a filter or a step. Without it, a
/// graph mixing the two can only be schema-checked on half its edges — and
/// the half it would skip is exactly the agent handoffs.
pub struct GraphRegistry<'a> {
    pub filters: &'a FilterLibrary,
    pub steps: &'a StepLibrary,
}

impl<'a> GraphRegistry<'a> {
    pub fn new(filters: &'a FilterLibrary, steps: &'a StepLibrary) -> Self {
        Self { filters, steps }
    }
}

impl FilterRegistry for GraphRegistry<'_> {
    fn meta(&self, node_id: &str) -> Option<FilterMeta> {
        self.filters.meta(node_id)
    }

    fn config_hash(&self, node_id: &str) -> Option<CacheKey> {
        self.filters
            .config_hash(node_id)
            .or_else(|| self.steps.get(node_id).map(|s| s.config_hash()))
    }

    fn step_meta(&self, node_id: &str) -> Option<StepMeta> {
        self.steps.meta(node_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use somatize_core::error::Result;
    use somatize_core::step::{StepCtx, Transition};
    use somatize_core::value::Value;

    struct Noop(&'static str);

    impl Step for Noop {
        fn config_hash(&self) -> CacheKey {
            CacheKey::from_parts(&[self.0.as_bytes()])
        }
        fn meta(&self) -> StepMeta {
            StepMeta::new(self.0)
        }
        fn poll(&self, _ctx: &StepCtx<'_>) -> Result<Transition> {
            Ok(Transition::Done(Value::Empty))
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[test]
    fn registers_and_looks_up() {
        let mut lib = StepLibrary::new();
        assert!(lib.is_empty());

        lib.register("a", Box::new(Noop("A")));
        lib.register("b", Box::new(Noop("B")));

        assert_eq!(lib.len(), 2);
        assert!(lib.contains("a"));
        assert!(!lib.contains("missing"));
        assert_eq!(lib.meta("b").unwrap().name, "B");
        assert!(lib.get("missing").is_none());
        assert_eq!(lib.node_ids(), vec!["a", "b"]);
    }

    #[test]
    fn re_registering_replaces() {
        let mut lib = StepLibrary::new();
        lib.register("a", Box::new(Noop("first")));
        lib.register("a", Box::new(Noop("second")));
        assert_eq!(lib.len(), 1);
        assert_eq!(lib.meta("a").unwrap().name, "second");
    }
}
