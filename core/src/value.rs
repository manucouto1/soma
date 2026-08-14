//! Lo que viaja por una arista.
//!
//! Es el precio de que el núcleo ejecute: si el motor está en Rust, los datos
//! tienen que tener una forma que Rust entienda, y eso obliga a decidir *ahora*
//! qué puede cruzar una arista.
//!
//! Cuatro variantes, no seis. `Json` pediría `serde_json` y el núcleo no
//! depende de nada; un `Object` opaco (un pickle) solo sirve para mandarlo por
//! un cable, y no hay cable. Las dos entrarán cuando un caso de uso las pida,
//! y el error de conversión dice exactamente qué faltó.

use std::sync::Arc;

/// Un dato que cruza de un nodo al siguiente.
///
/// `Arc` en todas partes porque un valor se clona en cada arista y clonar no
/// debe copiar los datos.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// Nada. Lo que recibe un nodo raíz cuando no le pasas entrada.
    Null,
    /// Texto UTF-8.
    Text(Arc<str>),
    /// Bytes sin interpretar.
    Bytes(Arc<Vec<u8>>),
    /// Datos numéricos: los valores en orden row-major, más su forma.
    Tensor {
        /// Los números, aplanados.
        values: Arc<Vec<f64>>,
        /// El tamaño de cada dimensión; su producto es `values.len()`.
        shape: Vec<usize>,
    },
}

impl Value {
    /// Un tensor de una dimensión.
    pub fn vector(values: impl Into<Vec<f64>>) -> Self {
        let values = values.into();
        Self::Tensor {
            shape: vec![values.len()],
            values: Arc::new(values),
        }
    }

    /// Un solo número.
    pub fn scalar(x: f64) -> Self {
        Self::Tensor {
            values: Arc::new(vec![x]),
            shape: vec![],
        }
    }

    /// Texto.
    pub fn text(s: impl AsRef<str>) -> Self {
        Self::Text(Arc::from(s.as_ref()))
    }

    /// Cómo llamar a esta variante en un mensaje de error.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Null => "null",
            Self::Text(_) => "text",
            Self::Bytes(_) => "bytes",
            Self::Tensor { .. } => "tensor",
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
        Self::scalar(x)
    }
}
