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
//! Aquí se detecta además lo estructural, antes de ejecutar nada. Hoy solo hay
//! una cosa sin decidir: un nodo al que **entran** dos aristas, porque nadie
//! ha dicho cómo se combinan los dos valores. Que de un nodo **salgan** dos no
//! tiene ningún problema: las dos ramas reciben lo mismo, y lo que producen es
//! una [`Value::List`].

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
    /// Llamar al filtro de un nodo.
    Execute(NodeId),
    /// Conducir un step por turnos hasta que termine.
    Step(NodeId),
    /// Uno detrás de otro, pasando la salida de cada uno al siguiente.
    Sequence(Vec<Plan>),
    /// Varias ramas independientes, todas con la misma entrada. Producen una
    /// [`Value::List`](crate::Value::List) con sus salidas, en orden.
    ///
    /// Se llama `Parallel` por lo que significa —las ramas no dependen entre
    /// sí— no por cómo se ejecuta hoy, que es una detrás de otra. Repartirlas
    /// entre hilos es una decisión que no cambia el resultado, y no la ha
    /// pedido nadie.
    Parallel(Vec<Plan>),
}

/// Decide cómo se recorre este grafo.
///
/// Necesita el catálogo porque la forma depende de **qué es** cada nodo: un
/// filtro se llama una vez, un step se conduce por turnos.
///
/// # Errores
/// Ver [`CompileError`]. Todo lo estructural se detecta aquí; lo que quede
/// para [`crate::Executor::run`] son fallos de las implementaciones.
pub fn compile(graph: &Graph, catalog: &Catalog) -> Result<Plan, CompileError> {
    if graph.is_empty() {
        return Ok(Plan::Empty);
    }
    branches(graph, catalog, &graph.roots())
}

/// Varias ramas que arrancan a la vez, o una sola sin envolver.
fn branches(graph: &Graph, catalog: &Catalog, heads: &[&NodeId]) -> Result<Plan, CompileError> {
    let mut plans = heads
        .iter()
        .map(|head| chain(graph, catalog, head))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(match plans.len() {
        1 => plans.pop().expect("acabamos de comprobar que hay uno"),
        _ => Plan::Parallel(plans),
    })
}

/// La cadena que arranca en `head` y sigue mientras no se bifurque.
fn chain(graph: &Graph, catalog: &Catalog, head: &NodeId) -> Result<Plan, CompileError> {
    let mut steps = Vec::new();
    let mut current = head.clone();
    loop {
        let sources = graph.predecessors(&current);
        if sources.len() > 1 {
            return Err(CompileError::Fanin {
                node: current.clone(),
                sources: sources.iter().map(|id| (*id).clone()).collect(),
            });
        }
        steps.push(match catalog.get(&current) {
            Some(crate::NodeImpl::Filter(_)) => Plan::Execute(current.clone()),
            Some(crate::NodeImpl::Step(_)) => Plan::Step(current.clone()),
            None => return Err(CompileError::NoImplementation(current.clone())),
        });

        let next = graph.successors(&current);
        match next.as_slice() {
            [] => break,
            [one] => current = (*one).clone(),
            many => {
                steps.push(branches(graph, catalog, many)?);
                break;
            }
        }
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
    /// A ese nodo llegan varias aristas y no está decidido cómo se combinan.
    Fanin {
        /// El nodo que recibe.
        node: NodeId,
        /// De dónde le llega.
        sources: Vec<NodeId>,
    },
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoImplementation(id) => {
                write!(f, "el nodo `{id}` no tiene implementación registrada")
            }
            Self::Fanin { node, sources } => write!(
                f,
                "a `{node}` llegan {} aristas ({}) y todavía no está decidido cómo se combinan",
                sources.len(),
                join(sources)
            ),
        }
    }
}

impl std::error::Error for CompileError {}

fn join(ids: &[NodeId]) -> String {
    ids.iter()
        .map(|id| format!("`{id}`"))
        .collect::<Vec<_>>()
        .join(", ")
}
