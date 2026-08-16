//! La forma decidida de una ejecución.
//!
//! Un [`Graph`] dice qué nodos hay y cómo se conectan. Un `Plan` dice **cómo
//! se recorren**, y es una decisión aparte: la misma estructura puede
//! ejecutarse en secuencia, a la vez o repartida entre máquinas.
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
//!
//! # Cómo se recupera la forma
//!
//! [`compile`] no aplana el grafo: lo **descompone**. `>>` y `|` del DSL son
//! composición en serie y en paralelo, así que el árbol que escribiste está
//! ahí y se puede recuperar del grafo — que es obligatorio, porque el mismo
//! grafo construido con `node()`/`edge()` en un bucle tiene que dar el mismo
//! plan (decisión 6 de CU5). El árbol de la expresión es el **oráculo**, no el
//! origen.
//!
//! La descomposición son cuatro casos, y el orden importa:
//!
//! | caso | sale |
//! |---|---|
//! | ningún nodo | [`Plan::Empty`] |
//! | un nodo | [`Plan::Execute`] |
//! | el subgrafo se parte en componentes | [`Plan::Wave`], una rama por componente |
//! | hay un **corte serie** | [`Plan::Sequence`] de los dos lados |
//! | no hay corte | secuencia plana: no es serie-paralelo |
//!
//! Un **corte serie** `(A, B)` es lo que haría un `>>`: las aristas que cruzan
//! van de **todos** los sinks de `A` a **todos** los sources de `B`, y de
//! ningún otro sitio. Es la definición de composición en serie, y comprobarla
//! entera —no solo que exista un nodo de paso— es lo que evita que una rama de
//! varios nodos se parta por la mitad.
//!
//! Basta probar los **prefijos de un orden topológico**, y eso es demostrable:
//! en una composición serie todo nodo de `A` alcanza un sink de `A`, todo sink
//! de `A` tiene arista a todo source de `B`, y todo nodo de `B` es alcanzable
//! desde un source de `B`. Luego todo nodo de `A` precede a todo nodo de `B` en
//! *cualquier* orden topológico, o sea que `A` es prefijo de todos ellos. No
//! hay que enumerar subconjuntos.
//!
//! # Lo que no es serie-paralelo
//!
//! Hay DAGs sin árbol —es un teorema, no un hueco de esto—. El patrón mínimo
//! prohibido es la «N»: `a→c`, `a→d`, `b→d`. Ver Valdes, Tarjan y Lawler, *The
//! recognition of series parallel digraphs*, SIAM J. Comput. 11(2), 1982.
//!
//! Y hay una frontera afortunada: **la imagen del DSL son exactamente los
//! grafos serie-paralelos**, porque `>>` conecta todos los terminales con
//! todas las cabezas y `|` es unión disjunta, y no hay una tercera operación.
//! La N solo se puede construir con `node()`/`edge()`. Para ésos, el último
//! caso los deja como estaban antes de que existieran las waves: una secuencia
//! plana. Sin paralelismo, sin regresión, y visible en `plan()`.

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
    /// Ramas que se lanzan **a la vez**.
    ///
    /// Significa «se ejecutan a la vez», no «son independientes». Esa es la
    /// diferencia con el `Plan::Parallel` que se quitó en CU4: aquel solo
    /// describía la estructura, y encima se rompía en el diamante porque sus
    /// ramas se solapaban —las dos reclamaban el nodo de unión y se ejecutaba
    /// dos veces—. Las ramas de una wave son **componentes conexas** del
    /// subgrafo, así que son disjuntas por construcción y ningún nodo puede
    /// aparecer en dos.
    ///
    /// Cada rama es un plan entero, no un paso suelto: en
    /// `a >> (b >> b2 | c) >> d` la rama larga corre de principio a fin en el
    /// mismo hilo. Agruparlas por nivel topológico también sería correcto,
    /// pero dejaría a `b2` esperando a `c` sin necesitarla, y el dispositivo
    /// de torch es *thread-local*, así que una rama que salta de hilo no puede
    /// fijarlo una sola vez.
    Wave(Vec<Plan>),
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

    let order = graph.topological_sort();
    for node in &order {
        if catalog.get(node).is_none() {
            return Err(CompileError::NoImplementation((*node).clone()));
        }
    }

    Ok(decompose(graph, &order))
}

// ── La descomposición ──

/// La forma de un subconjunto de nodos, en orden topológico.
///
/// El subconjunto está siempre **cerrado bajo caminos**: si dos de sus nodos
/// están unidos por un camino, todo el camino está dentro. Los tres que se
/// construyen aquí lo cumplen —un prefijo del orden topológico, su
/// complemento, y una componente conexa—, y por eso la alcanzabilidad dentro
/// del subgrafo coincide con la del grafo entero y no hay que inducirlo.
fn decompose<'g>(graph: &'g Graph, nodes: &[&'g NodeId]) -> Plan {
    match nodes {
        [] => Plan::Empty,
        [only] => step(graph, only),
        _ => {
            let parts = components(graph, nodes);
            if parts.len() > 1 {
                return Plan::Wave(parts.iter().map(|part| decompose(graph, part)).collect());
            }

            let Some(cut) = series_cut(graph, nodes) else {
                // Sin corte no hay árbol que recuperar: no es serie-paralelo.
                // Se recorre en secuencia, que es lo que se hacía antes de que
                // existieran las waves.
                return Plan::Sequence(nodes.iter().map(|node| step(graph, node)).collect());
            };

            // El corte más pequeño, y el resto se sigue cortando. Aplanar la
            // recursión por la derecha deja `Sequence` con sus pasos en fila
            // en vez de anidada, que es como se lee.
            let mut steps = vec![decompose(graph, &nodes[..cut])];
            match decompose(graph, &nodes[cut..]) {
                Plan::Sequence(rest) => steps.extend(rest),
                other => steps.push(other),
            }
            Plan::Sequence(steps)
        }
    }
}

/// Un paso suelto, con de dónde sale su entrada.
///
/// Los predecesores son los del grafo entero y no los del subconjunto: el
/// motor los busca en lo ya producido, que incluye lo de antes de esta rama.
fn step(graph: &Graph, node: &NodeId) -> Plan {
    Plan::Execute {
        node: node.clone(),
        from: graph.predecessors(node).into_iter().cloned().collect(),
    }
}

/// Las componentes conexas —sin mirar la dirección— del subgrafo.
///
/// Cada una conserva el orden topológico de la entrada, y salen ordenadas por
/// su primer nodo, así que dos ejecuciones dan lo mismo.
fn components<'g>(graph: &'g Graph, nodes: &[&'g NodeId]) -> Vec<Vec<&'g NodeId>> {
    let mut unassigned: Vec<bool> = vec![true; nodes.len()];
    let mut out = Vec::new();

    for start in 0..nodes.len() {
        if !unassigned[start] {
            continue;
        }
        unassigned[start] = false;
        let mut group = vec![start];
        let mut frontier = vec![start];

        while let Some(i) = frontier.pop() {
            for j in 0..nodes.len() {
                if unassigned[j] && adjacent(graph, nodes[i], nodes[j]) {
                    unassigned[j] = false;
                    group.push(j);
                    frontier.push(j);
                }
            }
        }

        group.sort_unstable();
        out.push(group.into_iter().map(|i| nodes[i]).collect());
    }
    out
}

/// Si hay una arista entre los dos, en cualquier dirección.
fn adjacent(graph: &Graph, a: &NodeId, b: &NodeId) -> bool {
    graph.successors(a).contains(&b) || graph.successors(b).contains(&a)
}

/// Por dónde parte la secuencia: el corte serie más pequeño, si lo hay.
fn series_cut(graph: &Graph, nodes: &[&NodeId]) -> Option<usize> {
    (1..nodes.len()).find(|cut| is_series_cut(graph, &nodes[..*cut], &nodes[*cut..]))
}

/// Si `before >> after` es exactamente lo que hay entre los dos.
///
/// O sea: las aristas que cruzan van de **todos** los sinks de `before` a
/// **todos** los sources de `after`, y de ningún otro sitio. Las dos mitades
/// de la comprobación hacen falta: sin la primera, una arista que sale de un
/// nodo interior pasaría por buena; sin la segunda, dos ramas que no se juntan
/// del todo también.
fn is_series_cut(graph: &Graph, before: &[&NodeId], after: &[&NodeId]) -> bool {
    let sinks: Vec<&NodeId> = before
        .iter()
        .copied()
        .filter(|node| !graph.successors(node).iter().any(|s| before.contains(s)))
        .collect();
    let sources: Vec<&NodeId> = after
        .iter()
        .copied()
        .filter(|node| !graph.predecessors(node).iter().any(|p| after.contains(p)))
        .collect();

    let crosses_outside_the_ends = before.iter().any(|node| {
        graph
            .successors(node)
            .iter()
            .any(|succ| after.contains(succ) && !(sinks.contains(node) && sources.contains(succ)))
    });
    if crosses_outside_the_ends {
        return false;
    }

    sinks.iter().all(|sink| {
        let onward = graph.successors(sink);
        sources.iter().all(|source| onward.contains(source))
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
