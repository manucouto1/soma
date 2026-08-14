//! El núcleo de soma-next: la estructura del grafo, el contrato de lo que se
//! ejecuta y el motor que lo recorre.
//!
//! Aquí no entra `#[pyclass]`. En cuanto un tipo del núcleo lo lleva, deja de
//! poder usarse sin un intérprete de Python cargado, y eso no se deshace.
//!
//! Tres piezas y sus papeles, que es fácil confundir:
//!
//! - [`Graph`] es la **estructura**: qué nodos hay y cómo se conectan. Dato puro.
//! - [`Catalog`] es el **almacén**: qué implementación corresponde a cada nodo.
//! - [`Filter`] es el **contrato** de una unidad ejecutable.
//! - [`run`] es el **motor**.
//!
//! Un fichero por tipo, con sus `impl` inherentes y los errores que producen
//! sus operaciones. Ver la regla completa en `CLAUDE.md`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod execution;
mod filter;
mod graph;
mod value;

pub use execution::{RunError, run};
pub use filter::{Catalog, Filter, FilterError};
pub use graph::{Edge, Graph, GraphError, NodeId};
pub use value::Value;
