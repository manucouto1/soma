//! Lo que viaja por una arista.
//!
//! Es el precio de que el núcleo ejecute: si el motor está en Rust, los datos
//! tienen que tener una forma que Rust entienda.
//!
//! Cinco de sus variantes son datos que el núcleo entiende y puede comparar.
//! No hay `Json` porque pediría `serde_json` y el núcleo no depende de nada, ni
//! `Tensor` con forma porque nadie produce uno.
//!
//! La sexta, [`Value::Opaque`], es de otra naturaleza: transporta algo que el
//! núcleo **no mira**. Existe porque hay valores que no pueden convertirse sin
//! destruirse — un tensor de torch a mitad de una gráfica de autograd es el
//! caso que la motivó: pasarlo a números y de vuelta lo deja sin `grad_fn`.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

/// Un dato que cruza de un nodo al siguiente.
///
/// `Arc` donde los datos pesan, porque un valor se clona en cada arista y
/// clonar no debe copiar.
#[derive(Clone)]
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
    /// Algo que el núcleo transporta sin mirarlo. Quien lo metió sabe qué es.
    ///
    /// Es la vía por la que un valor cruza el grafo **sin convertirse**, y el
    /// tipo es `dyn Any` y no algo de Python porque el núcleo no depende de
    /// PyO3 ni va a empezar: quien lo mete guarda dentro lo que quiera y lo
    /// recupera con `downcast_ref`.
    ///
    /// Lo que significa la variante, y de donde sale todo lo demás: **este
    /// valor solo existe en este proceso y en este run**. De ahí que no se
    /// pueda hashear por contenido (luego el nodo no se memoiza), ni
    /// serializar (luego no viaja a otra máquina), ni comparar salvo por
    /// identidad. Las tres consecuencias son las correctas: memoizar un tensor
    /// a mitad de autograd sería un error, no una optimización.
    Opaque(Arc<dyn Any + Send + Sync>),
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

    /// Envuelve algo para que cruce el grafo sin que nadie lo toque.
    pub fn opaque(x: impl Any + Send + Sync) -> Self {
        Self::Opaque(Arc::new(x))
    }

    /// Lo que hay dentro de un opaco, si es de este tipo.
    pub fn downcast<T: Any + Send + Sync>(&self) -> Option<&T> {
        let Self::Opaque(inner) = self else {
            return None;
        };
        inner.downcast_ref::<T>()
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
            Self::Opaque(_) => "opaque",
        }
    }
}

impl PartialEq for Value {
    /// Dos opacos son iguales solo si son **el mismo**: el núcleo no sabe qué
    /// llevan dentro, así que no puede comparar su contenido. Un valor clonado
    /// al recorrer una arista conserva el mismo `Arc`, que es el caso que
    /// importa; envolver dos veces el mismo objeto da dos valores distintos.
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Null, Self::Null) => true,
            (Self::Number(a), Self::Number(b)) => a == b,
            (Self::Text(a), Self::Text(b)) => a == b,
            (Self::Bytes(a), Self::Bytes(b)) => a == b,
            (Self::List(a), Self::List(b)) => a == b,
            (Self::Map(a), Self::Map(b)) => a == b,
            (Self::Opaque(a), Self::Opaque(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Null => f.write_str("Null"),
            Self::Number(x) => write!(f, "Number({x})"),
            Self::Text(s) => write!(f, "Text({s:?})"),
            Self::Bytes(b) => write!(f, "Bytes({} bytes)", b.len()),
            Self::List(items) => f.debug_tuple("List").field(items).finish(),
            Self::Map(pairs) => f.debug_tuple("Map").field(pairs).finish(),
            // No se imprime lo que lleva dentro porque no se sabe qué es.
            Self::Opaque(_) => f.write_str("Opaque(..)"),
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
