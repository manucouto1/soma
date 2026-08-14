//! Quien hace lo que un step pide.
//!
//! El núcleo no sabe qué es una petición: para él es un `Value`. Quien la
//! interpreta es el driver — llamar a un modelo, ejecutar una herramienta,
//! consultar un índice. Esa ignorancia es lo que mantiene la capa agéntica
//! fuera del núcleo.

use crate::Value;

/// Realiza lo que un [`Step`](crate::Step) pidió.
pub trait Driver: Send + Sync {
    /// Atiende las peticiones y devuelve un resultado por cada una, en orden.
    ///
    /// # Errores
    /// Lo que el driver quiera decir; el motor lo envuelve con el nodo.
    fn perform(&self, requests: &[Value]) -> Result<Vec<Value>, DriverError>;
}

/// Lo que un driver puede contestar cuando no puede atender una petición.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DriverError(String);

impl DriverError {
    /// Un fallo descrito con un mensaje.
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }

    /// El mensaje.
    pub fn message(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for DriverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for DriverError {}
