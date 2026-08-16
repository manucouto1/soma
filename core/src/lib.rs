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
//! | [`Placement`] | **dónde** corre cada nodo. Dato puro, y aparte del plan |
//! | [`Device`] | el sitio: `cpu`, `cuda:0`, `meta` |
//! | [`Node`] | el **contrato** de lo que ejecuta un nodo |
//! | [`Driver`] | quien **atiende** lo que un step pide |
//! | [`Plan`] | la **forma decidida** de una ejecución |
//! | [`compile`] | de la estructura a la forma |
//! | [`Executor`] | el **motor** |
//! | [`Wire`] | declarar un grafo como expresión: `a >> (b \| c) >> d` |
//!
//! Un fichero por tipo, con sus `impl` inherentes y los errores que producen
//! sus operaciones. Ver la regla completa en `CLAUDE.md`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod build;
mod catalog;
mod device;
mod driver;
mod execution;
mod graph;
mod node;
mod placement;
mod plan;
mod value;

pub use build::{Wire, node};
pub use catalog::Catalog;
pub use device::{Device, DeviceError};
pub use driver::{Driver, DriverError};
pub use execution::{Executor, RunError};
pub use graph::{Edge, Graph, GraphError, NodeId};
pub use node::{Ctx, Node, NodeError, Transition};
pub use placement::Placement;
pub use plan::{CompileError, Plan, compile};
pub use value::Value;
