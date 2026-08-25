//! The store: which implementation belongs to each node.
//!
//! Apart from the [`Graph`](crate::Graph) on purpose: a graph is data — it
//! serializes, compares, gets sent elsewhere — and an implementation is not.
//! What joins them is the node id, and nothing else.

use crate::{Node, NodeId};
use std::collections::HashMap;
use std::sync::Arc;

/// A graph's implementations, by node id.
#[derive(Default, Clone)]
pub struct Catalog {
    nodes: HashMap<NodeId, Arc<dyn Node>>,
}

impl Catalog {
    /// An empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a node's implementation, returning whatever was there before.
    pub fn insert(&mut self, id: impl Into<NodeId>, node: Arc<dyn Node>) -> Option<Arc<dyn Node>> {
        self.nodes.insert(id.into(), node)
    }

    /// The implementation registered for a node.
    pub fn get(&self, id: &NodeId) -> Option<&Arc<dyn Node>> {
        self.nodes.get(id)
    }

    /// How many implementations there are.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether there are none.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

impl std::fmt::Debug for Catalog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Catalog")
            .field("nodes", &self.nodes.keys().collect::<Vec<_>>())
            .finish()
    }
}
