//! El núcleo de soma-next: tipos puros, sin runtime, sin red, sin Python.
//!
//! Aquí no entra `#[pyclass]`. En cuanto un tipo del núcleo lo lleva, deja de
//! poder usarse sin un intérprete de Python cargado, y eso no se deshace.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

/// Un grafo de cómputo.
///
/// ESQUELETO. Solo existe para probar que la costura Rust→PyO3→Python→pytest
/// está viva de punta a punta. `node`, `edge` y —antes que ninguna de las
/// dos— la respuesta a *qué es un nodo* son decisiones de diseño del caso de
/// uso 1, y se toman a mano, no se heredan de aquí.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Graph {
    _priv: (),
}

impl Graph {
    /// Un grafo vacío.
    pub fn new() -> Self {
        Self::default()
    }

    /// Cuántos nodos tiene. Hoy, ninguno: todavía no hay forma de añadirlos.
    pub fn len(&self) -> usize {
        0
    }

    /// `true` mientras el grafo no tenga nodos.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
