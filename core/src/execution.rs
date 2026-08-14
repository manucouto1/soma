//! El motor: recorrer el grafo y ejecutar cada nodo.
//!
//! Vive en el núcleo, no en los bindings, porque recorrer el grafo *es* lógica
//! de dominio: el orden, de dónde sale la entrada de cada nodo y qué pasa
//! cuando uno falla. Python solo aporta las implementaciones.
//!
//! Dos cosas que el motor **no** sabe hacer todavía, y las dos fallan con un
//! error que dice cuál es la decisión pendiente en lugar de inventarse una:
//! juntar dos ramas en un nodo, y un grafo con más de una hoja. Ninguna tiene
//! respuesta obvia —¿los dos valores se combinan cómo?— y ningún caso de uso
//! la ha pedido.

use crate::{Catalog, FilterError, Graph, NodeId, Value};
use std::collections::HashMap;
use std::fmt;

/// Ejecuta el grafo entero y devuelve lo que produjo su hoja.
///
/// Es una función y no un método de [`Graph`] a propósito: ejecutar no es algo
/// que un grafo sepa hacer solo. Necesita además el almacén y una entrada, y
/// mañana necesitará una caché y un bus de eventos — cuando eso llegue, esto
/// pasa a ser el tipo que los guarda (en el original se llama `GraphSession`).
///
/// Cada nodo recibe la salida de su predecesor; los que no tienen ninguno
/// reciben `input`. Un grafo vacío devuelve `input` sin tocarlo.
///
/// # Errores
/// Ver [`RunError`]. El primer nodo que falle para el run: no hay recuperación
/// parcial, porque nadie ha dicho todavía qué debería significar.
pub fn run(graph: &Graph, catalog: &Catalog, input: Value) -> Result<Value, RunError> {
    let leaves = graph.leaves();
    let output_node = match leaves.as_slice() {
        [] => return Ok(input),
        [single] => (*single).clone(),
        many => {
            return Err(RunError::ManyLeaves(
                many.iter().map(|id| (*id).clone()).collect(),
            ));
        }
    };

    let mut outputs: HashMap<NodeId, Value> = HashMap::with_capacity(graph.len());
    for node in graph.topological_sort() {
        let node_input = resolve_input(graph, node, &outputs, &input)?;
        let filter = catalog
            .get(node)
            .ok_or_else(|| RunError::NoImplementation(node.clone()))?;
        let output = filter
            .forward(&node_input)
            .map_err(|source| RunError::Filter {
                node: node.clone(),
                source,
            })?;
        outputs.insert(node.clone(), output);
    }

    Ok(outputs
        .remove(&output_node)
        .expect("la hoja está en el grafo, así que el recorrido la ejecutó"))
}

/// Qué recibe un nodo: la salida de su predecesor, o la entrada del grafo.
fn resolve_input(
    graph: &Graph,
    node: &NodeId,
    outputs: &HashMap<NodeId, Value>,
    graph_input: &Value,
) -> Result<Value, RunError> {
    match graph.predecessors(node).as_slice() {
        [] => Ok(graph_input.clone()),
        [single] => Ok(outputs
            .get(*single)
            .expect("el orden topológico ya ejecutó a los predecesores")
            .clone()),
        many => Err(RunError::Fanin {
            node: node.clone(),
            sources: many.iter().map(|id| (*id).clone()).collect(),
        }),
    }
}

// ── Lo que puede salir mal al ejecutar ──

/// Por qué no se pudo ejecutar el grafo.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunError {
    /// El nodo está en el grafo pero nadie registró qué hace.
    NoImplementation(NodeId),
    /// El filtro del nodo falló.
    Filter {
        /// Dónde pasó.
        node: NodeId,
        /// Lo que dijo el filtro.
        source: FilterError,
    },
    /// A ese nodo llegan varias aristas y no está decidido cómo se combinan.
    Fanin {
        /// El nodo que recibe.
        node: NodeId,
        /// De dónde le llega.
        sources: Vec<NodeId>,
    },
    /// El grafo termina en varios sitios y no está decidido cuál es la salida.
    ManyLeaves(Vec<NodeId>),
}

impl fmt::Display for RunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoImplementation(id) => {
                write!(f, "el nodo `{id}` no tiene implementación registrada")
            }
            Self::Filter { node, source } => write!(f, "el nodo `{node}` falló: {source}"),
            Self::Fanin { node, sources } => write!(
                f,
                "a `{node}` llegan {} aristas ({}) y todavía no está decidido cómo se combinan",
                sources.len(),
                join(sources)
            ),
            Self::ManyLeaves(leaves) => write!(
                f,
                "el grafo termina en {} nodos ({}) y todavía no está decidido cuál es la salida",
                leaves.len(),
                join(leaves)
            ),
        }
    }
}

impl std::error::Error for RunError {}

fn join(ids: &[NodeId]) -> String {
    ids.iter()
        .map(|id| format!("`{id}`"))
        .collect::<Vec<_>>()
        .join(", ")
}
