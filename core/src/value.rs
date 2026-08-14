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
    /// Varios valores en orden.
    List(Arc<Vec<Value>>),
    /// Varios valores con nombre. Es lo que recibe un nodo al que llegan
    /// varias aristas, con la clave del nodo que produjo cada uno.
    ///
    /// **Ordenado**, y no por capricho: un `HashMap` itera distinto en cada
    /// proceso, así que pasarlo a lista daría un orden distinto cada vez y el
    /// hash por contenido —cuando llegue la caché— sería inservible. Los pares
    /// van en el orden en que se declararon las aristas, que es además lo
    /// simétrico con un `dict` de Python.
    Map(Arc<Vec<(String, Value)>>),
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

    /// Un mapa, en el orden en que le pases los pares.
    pub fn map(pairs: impl Into<Vec<(String, Value)>>) -> Self {
        Self::Map(Arc::new(pairs.into()))
    }

    /// El valor guardado bajo esa clave, si es un mapa y la tiene.
    pub fn get(&self, key: &str) -> Option<&Value> {
        let Self::Map(pairs) = self else {
            return None;
        };
        pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// Los valores de un mapa, en orden — pasar un mapa a lista es esto.
    pub fn values(&self) -> Option<Vec<&Value>> {
        let Self::Map(pairs) = self else {
            return None;
        };
        Some(pairs.iter().map(|(_, v)| v).collect())
    }

    /// Cómo llamar a esta variante en un mensaje de error.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Null => "null",
            Self::Number(_) => "number",
            Self::Text(_) => "text",
            Self::Bytes(_) => "bytes",
            Self::List(_) => "list",
            Self::Map(_) => "map",
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
