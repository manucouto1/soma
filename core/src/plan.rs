//! La forma decidida de una ejecución.
//!
//! Un [`Graph`] dice qué nodos hay y cómo se conectan. Un `Plan` dice **cómo
//! se recorren**, y es una decisión aparte: la misma estructura puede
//! ejecutarse en secuencia, en paralelo o repartida entre máquinas.
//!
//! Es un enum y no un trait de ejecutores a propósito. Las formas de ejecutar
//! son un conjunto **cerrado** que decidimos nosotros, así que el compilador
//! puede llevar la cuenta: el día que entre `Remote { target, inner }`, el
//! `match` del motor deja de compilar y hay que decidir qué hacer, en vez de
//! caer en un brazo comodín. Un trait con N implementadores no da eso, y el
//! original —diez variantes y un solo `match`— llegó a la misma conclusión.
//!
//! Cada paso lleva escrito **de dónde sale su entrada**. Es lo que hace que un
//! plan sea autónomo: al ejecutarlo no hace falta volver a mirar el grafo, y
//! los abanicos —hacia fuera y hacia dentro— salen sin ninguna variante
//! especial.

use crate::{Catalog, Graph, NodeId};
use std::fmt;

/// Cómo se recorre un grafo.
///
/// Sin `#[non_exhaustive]`: quien ejecuta tiene que decidir por cada variante,
/// y un brazo comodín es una respuesta equivocada en silencio.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Plan {
    /// No hay nada que hacer.
    Empty,
    /// Avanzar un nodo hasta que termine.
    Execute {
        /// Cuál.
        node: NodeId,
        /// De dónde sale su entrada. Vacío = la entrada del grafo.
        from: Vec<NodeId>,
    },
    /// Uno detrás de otro. Cada uno lee lo que necesita de lo ya producido,
    /// así que el orden importa y es el topológico.
    Sequence(Vec<Plan>),
}

/// Decide cómo se recorre este grafo.
///
/// El catálogo solo se mira para comprobar que cada nodo tiene implementación.
/// La forma ya no depende de **qué** sea cada uno: todos se avanzan igual, y si
/// uno pide algo por el camino eso lo dice su `Transition`, no su tipo.
///
/// # Errores
/// Ver [`CompileError`].
pub fn compile(graph: &Graph, catalog: &Catalog) -> Result<Plan, CompileError> {
    if graph.is_empty() {
        return Ok(Plan::Empty);
    }

    let mut steps = Vec::with_capacity(graph.len());
    for node in graph.topological_sort() {
        let from: Vec<NodeId> = graph.predecessors(node).into_iter().cloned().collect();
        if catalog.get(node).is_none() {
            return Err(CompileError::NoImplementation(node.clone()));
        }
        steps.push(Plan::Execute {
            node: node.clone(),
            from,
        });
    }

    Ok(match steps.len() {
        1 => steps.pop().expect("acabamos de comprobar que hay uno"),
        _ => Plan::Sequence(steps),
    })
}

// ── Lo que puede salir mal al compilar ──

/// Por qué no se pudo decidir cómo recorrer el grafo.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompileError {
    /// El nodo está en el grafo pero nadie registró qué hace.
    NoImplementation(NodeId),
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoImplementation(id) => {
                write!(f, "el nodo `{id}` no tiene implementación registrada")
            }
        }
    }
}

impl std::error::Error for CompileError {}
