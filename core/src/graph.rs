//! Qué está conectado con qué.
//!
//! Un `Graph` del núcleo es **solo topología**: identidades y aristas. Qué
//! hace un nodo —un filtro, un agente, un subgrafo— no es asunto suyo, porque
//! crear un grafo no necesita saberlo. Ese mapa (id → implementación) vive en
//! quien tenga implementaciones que guardar; hoy, el crate de bindings.
//!
//! Es la razón de que el núcleo no dependa de nada.

use std::collections::HashSet;
use std::fmt;

/// El nombre de un nodo dentro de un grafo.
///
/// Es un tipo propio y no un `String` para que el compilador no deje pasar un
/// id de otra cosa donde va el de un nodo. Cuesta poco hoy y hay más ids por
/// venir.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(String);

impl NodeId {
    /// El id como texto.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for NodeId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for NodeId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Una conexión dirigida entre dos nodos.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Edge {
    /// De dónde sale.
    pub source: NodeId,
    /// A dónde llega.
    pub target: NodeId,
}

/// Un grafo dirigido acíclico de nodos con nombre.
///
/// El invariante —ids únicos, aristas entre nodos que existen, sin ciclos— lo
/// sostienen los constructores: no hay forma de tener un `Graph` que lo
/// incumpla, así que `topological_sort` no puede fallar y no devuelve
/// `Result`.
///
/// Guarda nodos y aristas en orden de inserción y calcula la adyacencia cuando
/// hace falta. Es O(n) donde podría ser O(1), y es a propósito: el código se
/// lee de un vistazo y ningún caso de uso ha pedido todavía otra cosa.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Graph {
    nodes: Vec<NodeId>,
    edges: Vec<Edge>,
}

impl Graph {
    /// Un grafo vacío.
    pub fn new() -> Self {
        Self::default()
    }

    // ── Construcción ──

    /// Añade un nodo.
    ///
    /// # Errores
    /// [`GraphError::DuplicateNode`] si el id ya está cogido.
    pub fn add_node(&mut self, id: impl Into<NodeId>) -> Result<&NodeId, GraphError> {
        let id = id.into();
        if self.contains(&id) {
            return Err(GraphError::DuplicateNode(id));
        }
        self.nodes.push(id);
        Ok(self.nodes.last().expect("acabamos de insertarlo"))
    }

    /// Conecta dos nodos.
    ///
    /// # Errores
    /// [`GraphError::UnknownNode`] si alguno de los extremos no existe —los dos
    /// nodos se declaran antes que la arista que los une—,
    /// [`GraphError::DuplicateEdge`] si ya estaban conectados así, y
    /// [`GraphError::WouldCycle`] si la arista cerraría un ciclo.
    pub fn add_edge(
        &mut self,
        source: impl Into<NodeId>,
        target: impl Into<NodeId>,
    ) -> Result<&Edge, GraphError> {
        let (source, target) = (source.into(), target.into());
        for end in [&source, &target] {
            if !self.contains(end) {
                return Err(GraphError::UnknownNode(end.clone()));
            }
        }
        if self
            .edges
            .iter()
            .any(|e| e.source == source && e.target == target)
        {
            return Err(GraphError::DuplicateEdge {
                from: source,
                to: target,
            });
        }
        if source == target || self.reaches(&target, &source) {
            return Err(GraphError::WouldCycle {
                from: source,
                to: target,
            });
        }
        self.edges.push(Edge { source, target });
        Ok(self.edges.last().expect("acabamos de insertarla"))
    }

    /// Un id libre a partir del que quieras, sufijando `_2`, `_3`, … si hace falta.
    pub fn free_id(&self, wanted: &str) -> NodeId {
        let mut candidate = NodeId::from(wanted);
        let mut n = 1;
        while self.contains(&candidate) {
            n += 1;
            candidate = NodeId::from(format!("{wanted}_{n}"));
        }
        candidate
    }

    // ── Consultas ──

    /// Los nodos, en orden de inserción.
    pub fn nodes(&self) -> &[NodeId] {
        &self.nodes
    }

    /// Las aristas, en orden de inserción.
    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    /// Cuántos nodos hay.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// `true` mientras el grafo no tenga nodos.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Si el id nombra un nodo de este grafo.
    pub fn contains(&self, id: &NodeId) -> bool {
        self.nodes.contains(id)
    }

    /// Los nodos que entran en `id`, en orden de inserción de sus aristas.
    pub fn predecessors(&self, id: &NodeId) -> Vec<&NodeId> {
        self.edges
            .iter()
            .filter(|e| &e.target == id)
            .map(|e| &e.source)
            .collect()
    }

    /// Los nodos a los que sale `id`, en orden de inserción de sus aristas.
    pub fn successors(&self, id: &NodeId) -> Vec<&NodeId> {
        self.edges
            .iter()
            .filter(|e| &e.source == id)
            .map(|e| &e.target)
            .collect()
    }

    /// Los nodos sin predecesores: por donde entra la ejecución.
    pub fn roots(&self) -> Vec<&NodeId> {
        self.nodes
            .iter()
            .filter(|id| !self.edges.iter().any(|e| e.target == **id))
            .collect()
    }

    /// Los nodos sin sucesores: por donde sale.
    pub fn leaves(&self) -> Vec<&NodeId> {
        self.nodes
            .iter()
            .filter(|id| !self.edges.iter().any(|e| e.source == **id))
            .collect()
    }

    /// Los nodos en un orden en que cada uno va después de sus predecesores.
    ///
    /// No devuelve `Result` porque no puede fallar: el ciclo se rechazó al
    /// poner la arista. Los empates se rompen por orden de inserción, así que
    /// el resultado es determinista.
    pub fn topological_sort(&self) -> Vec<&NodeId> {
        let mut pending: Vec<usize> = self
            .nodes
            .iter()
            .map(|id| self.predecessors(id).len())
            .collect();
        let mut order = Vec::with_capacity(self.nodes.len());
        let mut placed = HashSet::new();

        while order.len() < self.nodes.len() {
            let next = pending
                .iter()
                .enumerate()
                .find(|(i, n)| **n == 0 && !placed.contains(i))
                .map(|(i, _)| i)
                .expect("un DAG no vacío siempre tiene un nodo sin pendientes");

            placed.insert(next);
            let id = &self.nodes[next];
            for succ in self.successors(id) {
                let i = self
                    .nodes
                    .iter()
                    .position(|n| n == succ)
                    .expect("una arista solo apunta a nodos del grafo");
                pending[i] -= 1;
            }
            order.push(id);
        }
        order
    }

    /// Si `from` alcanza a `to` siguiendo aristas. Es la comprobación de ciclo.
    fn reaches(&self, from: &NodeId, to: &NodeId) -> bool {
        let mut frontier = vec![from];
        let mut seen = HashSet::new();
        while let Some(current) = frontier.pop() {
            if current == to {
                return true;
            }
            if seen.insert(current) {
                frontier.extend(self.successors(current));
            }
        }
        false
    }
}

// ── Lo que puede salir mal al construir uno ──

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
