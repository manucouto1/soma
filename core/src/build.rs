//! Declarar un grafo como una expresión, en vez de a base de llamadas.
//!
//! ```ignore
//! let (graph, catalog) = (node("fuente", Sumar(1.0))
//!     >> (node("izq", Sumar(10.0)) | node("der", Sumar(100.0)))
//!     >> node("juntar", Media))
//! .somatize()?;
//! ```
//!
//! `>>` encadena y `|` abre en ramas, que es la misma sintaxis que el DSL de
//! Python. En Rust sale de implementar [`std::ops::Shr`] y [`std::ops::BitOr`]
//! sobre un tipo propio: no hace falta un macro, aunque también valdría.
//!
//! Un [`Wire`] es un grafo a medio declarar. Guarda por dónde se entra
//! (`heads`) y por dónde se sale (`terminals`), que es lo único que hace falta
//! para pegarle otro delante o detrás. Los nodos y las aristas no se
//! materializan hasta [`Wire::somatize`], así que juntar dos trozos es
//! concatenar dos listas y no fusionar dos grafos.

use crate::{Catalog, Graph, GraphError, Node, NodeId};
use std::ops::{BitOr, Shr};
use std::sync::Arc;

/// Un grafo a medio declarar.
pub struct Wire {
    parts: Result<Parts, GraphError>,
}

struct Parts {
    nodes: Vec<(NodeId, Arc<dyn Node>)>,
    edges: Vec<(NodeId, NodeId)>,
    heads: Vec<NodeId>,
    terminals: Vec<NodeId>,
}

/// Un nodo suelto.
pub fn node(id: impl Into<NodeId>, implementation: impl Node + 'static) -> Wire {
    single(id.into(), Arc::new(implementation))
}

fn single(id: NodeId, implementation: Arc<dyn Node>) -> Wire {
    Wire {
        parts: Ok(Parts {
            nodes: vec![(id.clone(), implementation)],
            edges: Vec::new(),
            heads: vec![id.clone()],
            terminals: vec![id],
        }),
    }
}

impl Shr for Wire {
    type Output = Wire;

    /// `a >> b`: todo lo que sale de `a` entra en todo lo que empieza `b`.
    fn shr(self, next: Wire) -> Wire {
        combine(self, next, |left, right| Parts {
            edges: left
                .terminals
                .iter()
                .flat_map(|from| right.heads.iter().map(|to| (from.clone(), to.clone())))
                .chain(left.edges)
                .chain(right.edges)
                .collect(),
            nodes: left.nodes.into_iter().chain(right.nodes).collect(),
            heads: left.heads,
            terminals: right.terminals,
        })
    }
}

impl BitOr for Wire {
    type Output = Wire;

    /// `a | b`: dos ramas que no se tocan. Lo que entre les llega a las dos, y
    /// lo que salga de las dos sale de aquí.
    fn bitor(self, other: Wire) -> Wire {
        combine(self, other, |left, right| Parts {
            nodes: left.nodes.into_iter().chain(right.nodes).collect(),
            edges: left.edges.into_iter().chain(right.edges).collect(),
            heads: left.heads.into_iter().chain(right.heads).collect(),
            terminals: left.terminals.into_iter().chain(right.terminals).collect(),
        })
    }
}

/// Junta dos trozos, quedándose con el primer fallo si alguno lo trae.
fn combine(left: Wire, right: Wire, join: impl FnOnce(Parts, Parts) -> Parts) -> Wire {
    Wire {
        parts: match (left.parts, right.parts) {
            (Ok(left), Ok(right)) => Ok(join(left, right)),
            (Err(e), _) | (_, Err(e)) => Err(e),
        },
    }
}

impl Wire {
    /// Materializa lo declarado: la estructura y el almacén.
    ///
    /// Son dos cosas y ninguna contiene a la otra, así que salen las dos —
    /// es la misma separación de siempre: el grafo es dato, una
    /// implementación no.
    ///
    /// # Errores
    /// El primer [`GraphError`] que dé montarlo: un id repetido, sobre todo.
    pub fn somatize(self) -> Result<(Graph, Catalog), GraphError> {
        let parts = self.parts?;
        let mut graph = Graph::new();
        let mut catalog = Catalog::new();

        for (id, implementation) in parts.nodes {
            graph.add_node(id.clone())?;
            catalog.insert(id, implementation);
        }
        for (from, to) in parts.edges {
            graph.add_edge(from, to)?;
        }
        Ok((graph, catalog))
    }
}
