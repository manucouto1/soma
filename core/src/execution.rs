//! El motor: recorrer un [`Plan`] y ejecutar lo que dice.
//!
//! Vive en el núcleo, no en los bindings, porque recorrer *es* lógica de
//! dominio. Python solo aporta las implementaciones.
//!
//! El motor no mira el grafo: cada paso del plan lleva escrito de dónde sale
//! su entrada, y aquí solo se busca en lo ya producido. Un plan es una
//! estructura decidida, no un grafo que haya que interpretar.
//!
//! Cuando a un nodo le llega **una** cosa, recibe esa cosa. Cuando le llegan
//! varias, recibe un [`Value::Map`] con la clave del nodo que produjo cada
//! una: fan-in no es una variante del plan ni un tipo de nodo, es la forma que
//! toma una entrada con varios orígenes. Agregarlas —promediar, votar,
//! concatenar— es trabajo del nodo que las recibe, o sea biblioteca.

use crate::{Catalog, Driver, DriverError, FilterError, NodeImpl, Plan, StepCtx, StepError};
use crate::{NodeId, Transition, Value};
use std::collections::HashMap;
use std::fmt;

/// Cuántas veces se le pregunta a un step antes de darlo por colgado.
///
/// Un step que no termina es un bug del step, no una espera legítima: quien
/// espera de verdad pide algo y el driver tarda. El tope existe para que ese
/// bug se note como un error con nombre en vez de como un proceso parado.
const MAX_TURNS: usize = 64;

/// Ejecuta planes.
///
/// Es un tipo y no una función suelta porque ejecutar necesita contexto —hoy
/// el almacén y el driver— y mañana necesitará más: una caché, un bus de
/// eventos. Ese "mañana" es lo que en el original se llama `GraphSession`.
pub struct Executor<'a> {
    catalog: &'a Catalog,
    driver: Option<&'a dyn Driver>,
}

impl<'a> Executor<'a> {
    /// Un ejecutor que solo sabe de filtros.
    ///
    /// Sin driver, un plan con steps falla con [`RunError::NoDriver`] en vez de
    /// inventarse qué hacer con lo que pidan.
    pub fn new(catalog: &'a Catalog) -> Self {
        Self {
            catalog,
            driver: None,
        }
    }

    /// El mismo ejecutor, con quien atenderá lo que pidan los steps.
    pub fn with_driver(mut self, driver: &'a dyn Driver) -> Self {
        self.driver = Some(driver);
        self
    }

    /// Ejecuta el plan y devuelve lo que produjo.
    ///
    /// # Errores
    /// Ver [`RunError`]. El primer fallo para la ejecución: no hay recuperación
    /// parcial, porque nadie ha dicho todavía qué debería significar.
    pub fn run(&self, plan: &Plan, input: Value) -> Result<Value, RunError> {
        let mut produced: HashMap<NodeId, Value> = HashMap::new();
        let last = self.walk(plan, &input, &mut produced)?;

        // La salida del grafo es la de sus hojas: los nodos cuya salida no lee
        // nadie. Una hoja → ese valor. Varias → un mapa con la clave de cada
        // una, igual que una entrada con varios orígenes. Las dos direcciones
        // del abanico tienen la misma forma, así que un diamante da la vuelta.
        let leaves = terminals(plan);
        Ok(match leaves.as_slice() {
            [] | [_] => last,
            many => Value::map(
                many.iter()
                    .map(|id| {
                        let value = produced
                            .get(id)
                            .cloned()
                            .expect("el recorrido ejecutó todos los pasos del plan");
                        (id.to_string(), value)
                    })
                    .collect::<Vec<_>>(),
            ),
        })
    }

    /// Ejecuta un plan, apuntando lo que produce cada nodo, y devuelve la
    /// salida de su último paso.
    fn walk(
        &self,
        plan: &Plan,
        graph_input: &Value,
        produced: &mut HashMap<NodeId, Value>,
    ) -> Result<Value, RunError> {
        match plan {
            Plan::Empty => Ok(graph_input.clone()),
            Plan::Execute { node, from } => {
                let input = gather(from, graph_input, produced);
                let output = self.run_filter(node, input)?;
                produced.insert(node.clone(), output.clone());
                Ok(output)
            }
            Plan::Step { node, from } => {
                let input = gather(from, graph_input, produced);
                let output = self.drive_step(node, input)?;
                produced.insert(node.clone(), output.clone());
                Ok(output)
            }
            Plan::Sequence(plans) => {
                let mut last = graph_input.clone();
                for plan in plans {
                    last = self.walk(plan, graph_input, produced)?;
                }
                Ok(last)
            }
        }
    }

    /// Una llamada, y ya está.
    fn run_filter(&self, node: &NodeId, input: Value) -> Result<Value, RunError> {
        let NodeImpl::Filter(filter) = self.implementation(node)? else {
            return Err(RunError::WrongKind {
                node: node.clone(),
                expected: "filtro",
            });
        };
        filter.forward(&input).map_err(|source| RunError::Filter {
            node: node.clone(),
            source,
        })
    }

    /// Preguntar, atender lo que pida, volver a preguntar. Hasta que termine.
    fn drive_step(&self, node: &NodeId, input: Value) -> Result<Value, RunError> {
        let NodeImpl::Step(step) = self.implementation(node)? else {
            return Err(RunError::WrongKind {
                node: node.clone(),
                expected: "step",
            });
        };

        let mut results: Vec<Value> = Vec::new();
        for turn in 0..MAX_TURNS {
            let ctx = StepCtx {
                input: &input,
                turn,
                results: &results,
            };
            match step.poll(&ctx).map_err(|source| RunError::Step {
                node: node.clone(),
                source,
            })? {
                Transition::Done(output) => return Ok(output),
                Transition::Await(requests) => {
                    let driver = self
                        .driver
                        .ok_or_else(|| RunError::NoDriver(node.clone()))?;
                    results = driver
                        .perform(&requests)
                        .map_err(|source| RunError::Driver {
                            node: node.clone(),
                            source,
                        })?;
                }
            }
        }
        Err(RunError::TurnLimit {
            node: node.clone(),
            turns: MAX_TURNS,
        })
    }

    /// Lo que el catálogo dice que es este nodo.
    fn implementation(&self, node: &NodeId) -> Result<&NodeImpl, RunError> {
        self.catalog
            .get(node)
            .ok_or_else(|| RunError::NoImplementation(node.clone()))
    }
}

// ── Lo que puede salir mal al ejecutar ──

/// Por qué no se pudo terminar la ejecución.
///
/// Lo estructural —fan-in, varias hojas, un nodo sin implementación en el
/// grafo— ya se descartó en [`compile`](crate::compile). Lo de aquí son
/// fallos de las implementaciones, o de un plan que no cuadra con el catálogo
/// con el que se ejecuta.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunError {
    /// El plan nombra un nodo que este catálogo no conoce.
    NoImplementation(NodeId),
    /// El plan dice una cosa y el catálogo tiene otra: se compiló con un
    /// catálogo y se ejecutó con otro.
    WrongKind {
        /// El nodo en discordia.
        node: NodeId,
        /// Lo que el plan esperaba encontrar.
        expected: &'static str,
    },
    /// El filtro del nodo falló.
    Filter {
        /// Dónde pasó.
        node: NodeId,
        /// Lo que dijo el filtro.
        source: FilterError,
    },
    /// El step del nodo falló.
    Step {
        /// Dónde pasó.
        node: NodeId,
        /// Lo que dijo el step.
        source: StepError,
    },
    /// El step pidió algo y no hay quien lo atienda.
    NoDriver(NodeId),
    /// El driver no pudo atender lo que el step pidió.
    Driver {
        /// De qué step venía la petición.
        node: NodeId,
        /// Lo que dijo el driver.
        source: DriverError,
    },
    /// El step siguió pidiendo turnos sin terminar nunca.
    TurnLimit {
        /// Quién.
        node: NodeId,
        /// Cuántos turnos se le dieron.
        turns: usize,
    },
}

impl fmt::Display for RunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoImplementation(id) => {
                write!(f, "el nodo `{id}` no tiene implementación registrada")
            }
            Self::WrongKind { node, expected } => write!(
                f,
                "el plan esperaba que `{node}` fuera un {expected}, y el catálogo dice otra cosa"
            ),
            Self::Filter { node, source } => write!(f, "el nodo `{node}` falló: {source}"),
            Self::Step { node, source } => write!(f, "el step `{node}` falló: {source}"),
            Self::NoDriver(node) => write!(
                f,
                "`{node}` pidió algo y este ejecutor no tiene driver que lo atienda"
            ),
            Self::Driver { node, source } => {
                write!(f, "atendiendo lo que pidió `{node}`: {source}")
            }
            Self::TurnLimit { node, turns } => write!(
                f,
                "`{node}` gastó los {turns} turnos sin terminar; probablemente no sabe parar"
            ),
        }
    }
}

impl std::error::Error for RunError {}

/// Qué recibe un nodo, según de dónde le llegue.
///
/// Nada → la entrada del grafo. Una cosa → esa cosa. Varias → un mapa con la
/// clave de quien produjo cada una, en el orden en que se declararon las
/// aristas.
fn gather(from: &[NodeId], graph_input: &Value, produced: &HashMap<NodeId, Value>) -> Value {
    let recall = |id: &NodeId| {
        produced
            .get(id)
            .cloned()
            .expect("el orden topológico ya ejecutó a los predecesores")
    };
    match from {
        [] => graph_input.clone(),
        [single] => recall(single),
        many => Value::map(
            many.iter()
                .map(|id| (id.to_string(), recall(id)))
                .collect::<Vec<_>>(),
        ),
    }
}

/// Los nodos del plan cuya salida no lee ningún otro: las hojas.
fn terminals(plan: &Plan) -> Vec<NodeId> {
    let mut produced = Vec::new();
    let mut consumed = Vec::new();
    collect(plan, &mut produced, &mut consumed);
    produced
        .into_iter()
        .filter(|id| !consumed.contains(id))
        .collect()
}

/// Qué produce y qué lee cada paso, aplanando las secuencias.
fn collect(plan: &Plan, produced: &mut Vec<NodeId>, consumed: &mut Vec<NodeId>) {
    match plan {
        Plan::Empty => {}
        Plan::Execute { node, from } | Plan::Step { node, from } => {
            produced.push(node.clone());
            consumed.extend(from.iter().cloned());
        }
        Plan::Sequence(plans) => {
            for plan in plans {
                collect(plan, produced, consumed);
            }
        }
    }
}
