//! Lo que puede salir mal al construir un grafo.

use crate::NodeId;
use std::fmt;

/// Un intento de construir un grafo que no puede existir.
///
/// Las cuatro variantes son las cuatro formas de romper el invariante, y se
/// devuelven en el momento de la inserción: no hay un `validate()` que llamar
/// después, porque no hay un instante en el que el grafo esté mal formado.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
    /// Ya hay un nodo con ese id.
    DuplicateNode(NodeId),
    /// El id no nombra ningún nodo de este grafo.
    UnknownNode(NodeId),
    /// Esa arista ya está puesta.
    DuplicateEdge {
        /// Origen.
        from: NodeId,
        /// Destino.
        to: NodeId,
    },
    /// Añadir esa arista cerraría un ciclo.
    WouldCycle {
        /// Origen.
        from: NodeId,
        /// Destino, que ya alcanza al origen.
        to: NodeId,
    },
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateNode(id) => write!(f, "ya hay un nodo llamado `{id}`"),
            Self::UnknownNode(id) => write!(f, "`{id}` no nombra ningún nodo de este grafo"),
            Self::DuplicateEdge { from, to } => write!(f, "la arista `{from}` → `{to}` ya existe"),
            Self::WouldCycle { from, to } => write!(
                f,
                "la arista `{from}` → `{to}` cerraría un ciclo: `{to}` ya alcanza a `{from}`"
            ),
        }
    }
}

impl std::error::Error for GraphError {}
