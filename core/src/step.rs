//! El contrato de una unidad que puede **no terminar**.
//!
//! Un [`Filter`](crate::Filter) es una función: entra un valor, sale otro. Un
//! `Step` es una máquina de estados: se le pregunta con `poll`, y puede
//! contestar que ha terminado o que necesita algo del mundo antes de seguir.
//! Esa es la única diferencia esencial entre los dos, y de ella salen todas
//! las demás.
//!
//! Lo que el step pide es **opaco para el núcleo**: un `Value` que el driver
//! sabe interpretar. Por eso aquí no hay ni LLMs, ni herramientas, ni diario
//! de efectos — eso es biblioteca y persistencia, no el contrato.

use crate::Value;

/// Algo que avanza por turnos y puede pedir cosas antes de terminar.
///
/// `Send + Sync` por la misma razón que [`Filter`](crate::Filter): acaba dentro
/// de un `#[pyclass]`.
pub trait Step: Send + Sync {
    /// Avanza un turno.
    ///
    /// Se llama con `turn == 0` y sin resultados; después, con lo que el driver
    /// devolvió de lo que se pidió en el turno anterior, en el mismo orden.
    ///
    /// # Errores
    /// Lo que el step quiera decir; el motor lo envuelve con el nodo.
    fn poll(&self, ctx: &StepCtx<'_>) -> Result<Transition, StepError>;
}

/// Lo que un step sabe cuando le preguntan.
#[derive(Debug, Clone, Copy)]
pub struct StepCtx<'a> {
    /// Lo que le llegó por su arista de entrada. El mismo en todos los turnos.
    pub input: &'a Value,
    /// Cuántas veces se le ha preguntado ya; empieza en 0.
    pub turn: usize,
    /// Lo que devolvió el driver de lo pedido en el turno anterior, en orden.
    /// Vacío en el turno 0.
    pub results: &'a [Value],
}

/// Cómo sigue la cosa después de un turno.
///
/// Deliberadamente **sin** `#[non_exhaustive]`: quien ejecuta un step tiene que
/// decidir qué hacer con cada variante, y un brazo comodín ahí es una respuesta
/// equivocada en silencio. Añadir una variante *debe* romper a todo el mundo.
#[derive(Debug, Clone, PartialEq)]
pub enum Transition {
    /// Terminado, con esta salida.
    Done(Value),
    /// Necesita que alguien haga esto antes de seguir. Se le volverá a
    /// preguntar con los resultados.
    Await(Vec<Value>),
}

/// Lo que un step puede contestar cuando no puede avanzar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepError(String);

impl StepError {
    /// Un fallo descrito con un mensaje.
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }

    /// El mensaje.
    pub fn message(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for StepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for StepError {}
