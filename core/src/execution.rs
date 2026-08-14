//! El motor: recorrer un [`Plan`] y ejecutar lo que dice.
//!
//! Vive en el núcleo, no en los bindings, porque recorrer *es* lógica de
//! dominio. Python solo aporta las implementaciones.
//!
//! Fíjate en lo que **no** hay aquí: resolver de dónde sale la entrada de cada
//! nodo. Eso lo decidió [`compile`](crate::compile), y lo que quedó fue una
//! [`Plan::Sequence`] donde cada uno recibe la salida del anterior. Un plan es
//! una estructura ya decidida, no un grafo que haya que interpretar.

use crate::{Catalog, Driver, DriverError, FilterError, NodeImpl, Plan, StepCtx, StepError};
use crate::{NodeId, Transition, Value};
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
        match plan {
            Plan::Empty => Ok(input),
            Plan::Execute(node) => self.run_filter(node, input),
            Plan::Step(node) => self.drive_step(node, input),
            Plan::Sequence(plans) => plans
                .iter()
                .try_fold(input, |carried, plan| self.run(plan, carried)),
            Plan::Parallel(branches) => {
                // Todas reciben la misma entrada; lo que sale es una lista con
                // sus salidas, en el orden en que se declararon las aristas.
                let outputs = branches
                    .iter()
                    .map(|branch| self.run(branch, input.clone()))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Value::list(outputs))
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
