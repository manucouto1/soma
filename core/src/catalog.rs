//! El almacén: qué implementación corresponde a cada nodo.
//!
//! Va aparte del [`Graph`](crate::Graph) a propósito. El grafo es dato —se
//! serializa, se compara, se manda a otro sitio—; una implementación no lo es.
//! Lo que los une es el id del nodo, y nada más.

use crate::{Filter, NodeId, Step};
use std::collections::HashMap;
use std::sync::Arc;

/// Qué es un nodo, de las dos cosas que puede ser.
///
/// Sin `#[non_exhaustive]`: quien ejecuta tiene que decidir por cada variante.
#[derive(Clone)]
pub enum NodeImpl {
    /// Una función: termina siempre en una llamada.
    Filter(Arc<dyn Filter>),
    /// Una máquina de estados: puede pedir cosas antes de terminar.
    Step(Arc<dyn Step>),
}

impl std::fmt::Debug for NodeImpl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Filter(_) => "Filter",
            Self::Step(_) => "Step",
        })
    }
}

/// Las implementaciones de un grafo, por id de nodo.
#[derive(Default, Clone, Debug)]
pub struct Catalog {
    nodes: HashMap<NodeId, NodeImpl>,
}

impl Catalog {
    /// Un almacén vacío.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registra un filtro, devolviendo lo que hubiera antes bajo ese id.
    pub fn insert_filter(
        &mut self,
        id: impl Into<NodeId>,
        filter: Arc<dyn Filter>,
    ) -> Option<NodeImpl> {
        self.nodes.insert(id.into(), NodeImpl::Filter(filter))
    }

    /// Registra un step, devolviendo lo que hubiera antes bajo ese id.
    pub fn insert_step(&mut self, id: impl Into<NodeId>, step: Arc<dyn Step>) -> Option<NodeImpl> {
        self.nodes.insert(id.into(), NodeImpl::Step(step))
    }

    /// Qué es el nodo, si está registrado.
    pub fn get(&self, id: &NodeId) -> Option<&NodeImpl> {
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
