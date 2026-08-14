//! El núcleo de soma-next: la estructura del grafo, los contratos de lo que se
//! ejecuta, la forma de la ejecución y el motor que la recorre.
//!
//! Aquí no entra `#[pyclass]`. En cuanto un tipo del núcleo lo lleva, deja de
//! poder usarse sin un intérprete de Python cargado, y eso no se deshace.
//!
//! Las piezas y sus papeles, que es fácil confundir:
//!
//! | pieza | papel |
//! |---|---|
//! | [`Graph`] | la **estructura**: qué nodos hay y cómo se conectan. Dato puro |
//! | [`Catalog`] | el **almacén**: qué implementación corresponde a cada nodo |
//! | [`Filter`] / [`Step`] | los dos **contratos** de unidad ejecutable |
//! | [`Driver`] | quien **atiende** lo que un step pide |
//! | [`Plan`] | la **forma decidida** de una ejecución |
//! | [`compile`] | de la estructura a la forma |
//! | [`Executor`] | el **motor** |
//!
//! Un fichero por tipo, con sus `impl` inherentes y los errores que producen
//! sus operaciones. Ver la regla completa en `CLAUDE.md`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod catalog;
mod driver;
mod execution;
mod filter;
mod graph;
mod plan;
mod step;
mod value;

pub use catalog::{Catalog, NodeImpl};
pub use driver::{Driver, DriverError};
pub use execution::{Executor, RunError};
pub use filter::{Filter, FilterError};
pub use graph::{Edge, Graph, GraphError, NodeId};
pub use plan::{CompileError, Plan, compile};
pub use step::{Step, StepCtx, StepError, Transition};
pub use value::Value;
