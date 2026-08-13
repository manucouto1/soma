//! El núcleo de soma-next: tipos puros, sin runtime, sin red, sin Python.
//!
//! Aquí no entra `#[pyclass]`. En cuanto un tipo del núcleo lo lleva, deja de
//! poder usarse sin un intérprete de Python cargado, y eso no se deshace.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod error;
mod graph;

pub use error::GraphError;
pub use graph::{Edge, Graph, NodeId};
