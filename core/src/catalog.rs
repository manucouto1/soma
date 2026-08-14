//! El almacén: qué implementación corresponde a cada nodo.
//!
//! Va aparte del [`Graph`](crate::Graph) a propósito. El grafo es dato —se
//! serializa, se compara, se manda a otro sitio—; una implementación no lo es.
//! Lo que los une es el id del nodo, y nada más.

use crate::{Node, NodeId};
use std::collections::HashMap;
use std::sync::Arc;

/// Las implementaciones de un grafo, por id de nodo.
#[derive(Default, Clone)]
pub struct Catalog {
    nodes: HashMap<NodeId, Arc<dyn Node>>,
}

impl Catalog {
    /// Un almacén vacío.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registra la implementación de un nodo, devolviendo la que hubiera antes.
    pub fn insert(&mut self, id: impl Into<NodeId>, node: Arc<dyn Node>) -> Option<Arc<dyn Node>> {
        self.nodes.insert(id.into(), node)
    }

    /// La implementación registrada para un nodo.
    pub fn get(&self, id: &NodeId) -> Option<&Arc<dyn Node>> {
        self.nodes.get(id)
    }

    /// Cuántas implementaciones hay.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Si no hay ninguna.
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
