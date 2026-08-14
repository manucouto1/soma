//! El contrato de una unidad que **termina siempre**.
//!
//! Un filtro es una función: entra un valor, sale otro, se acabó. La otra
//! mitad —lo que puede no terminar— es un [`Step`](crate::Step).
//!
//! El contrato tiene un método. El del original tiene cinco, y los otros
//! cuatro no sirven para ejecutar: `config_hash` es para la clave de caché,
//! `meta` para el compilador, `fit` para entrenar y `composite_fit` para que
//! el autograd cruce entre filtros. Entrarán con su caso de uso.

use crate::Value;

/// Algo que transforma un valor en otro.
///
/// `Send + Sync` no es decoración: un `Graph` de Python es un `#[pyclass]`, y
/// PyO3 exige que un pyclass sea `Send`. Como el grafo lleva dentro el
/// catálogo, y el catálogo `Arc<dyn Filter>`, la cota sube hasta aquí. El
/// original la tiene por la misma razón.
pub trait Filter: Send + Sync {
    /// Transforma la entrada.
    ///
    /// # Errores
    /// Lo que el filtro quiera decir; el motor lo envuelve con el nodo en el
    /// que pasó.
    fn forward(&self, input: &Value) -> Result<Value, FilterError>;
}

/// Lo que un filtro puede contestar cuando no puede transformar la entrada.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilterError(String);

impl FilterError {
    /// Un fallo descrito con un mensaje.
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }

    /// El mensaje.
    pub fn message(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for FilterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for FilterError {}
