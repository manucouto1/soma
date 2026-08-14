//! Lo que viaja por una arista.
//!
//! Es el precio de que el núcleo ejecute: si el motor está en Rust, los datos
//! tienen que tener una forma que Rust entienda.
//!
//! Cinco variantes, y ninguna de adorno. No hay `Json` porque pediría
//! `serde_json` y el núcleo no depende de nada; no hay `Object` opaco porque
//! solo sirve para mandar algo por un cable y no hay cable; y no hay `Tensor`
//! con forma porque nadie produce uno todavía — cuando haya un puente a numpy
//! o a torch traerá consigo su propia decisión sobre copiar o no copiar.

use std::sync::Arc;

/// Un dato que cruza de un nodo al siguiente.
///
/// `Arc` donde los datos pesan, porque un valor se clona en cada arista y
/// clonar no debe copiar.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// Nada. Lo que recibe un nodo raíz cuando no le pasas entrada.
    Null,
    /// Un número.
    Number(f64),
    /// Texto UTF-8.
    Text(Arc<str>),
    /// Bytes sin interpretar.
    Bytes(Arc<Vec<u8>>),
    /// Varios valores en orden. Es lo que produce un abanico.
    List(Arc<Vec<Value>>),
}

impl Value {
    /// Un número.
    pub fn number(x: f64) -> Self {
        Self::Number(x)
    }

    /// Texto.
    pub fn text(s: impl AsRef<str>) -> Self {
        Self::Text(Arc::from(s.as_ref()))
    }

    /// Una lista.
    pub fn list(values: impl Into<Vec<Value>>) -> Self {
        Self::List(Arc::new(values.into()))
    }

    /// Cómo llamar a esta variante en un mensaje de error.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Null => "null",
            Self::Number(_) => "number",
            Self::Text(_) => "text",
            Self::Bytes(_) => "bytes",
            Self::List(_) => "list",
        }
    }
}

impl From<&str> for Value {
    fn from(s: &str) -> Self {
        Self::text(s)
    }
}

impl From<f64> for Value {
    fn from(x: f64) -> Self {
        Self::Number(x)
    }
}
