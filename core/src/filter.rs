//! El contrato de una unidad ejecutable, y dónde se guardan.
//!
//! Un filtro es una **función**: entra un valor, sale otro, termina siempre.
//! Es la mitad fácil; la otra —una unidad que puede *no* terminar y pedir algo
//! al mundo antes de seguir— es un `Step`, y todavía no existe aquí.
//!
//! El contrato tiene un método. El del original tiene cinco, y los otros
//! cuatro no sirven para ejecutar: `config_hash` es para la clave de caché,
//! `meta` para el compilador, `fit` para entrenar y `composite_fit` para que
//! el autograd cruce entre filtros. Entrarán con su caso de uso.

use crate::{NodeId, Value};
use std::collections::HashMap;
use std::sync::Arc;

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

/// El almacén: qué implementación corresponde a cada nodo.
///
/// Va aparte del [`Graph`](crate::Graph) a propósito. El grafo es dato —se
/// serializa, se compara, se manda a otro sitio—; una implementación no lo es.
/// Lo que los une es el id del nodo, y nada más.
#[derive(Default, Clone)]
pub struct Catalog {
    filters: HashMap<NodeId, Arc<dyn Filter>>,
}

impl Catalog {
    /// Un almacén vacío.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registra la implementación de un nodo, devolviendo la que hubiera antes.
    pub fn insert(
        &mut self,
        id: impl Into<NodeId>,
        filter: Arc<dyn Filter>,
    ) -> Option<Arc<dyn Filter>> {
        self.filters.insert(id.into(), filter)
    }

    /// La implementación registrada para un nodo.
    pub fn get(&self, id: &NodeId) -> Option<&Arc<dyn Filter>> {
        self.filters.get(id)
    }

    /// Cuántas implementaciones hay.
    pub fn len(&self) -> usize {
        self.filters.len()
    }

    /// Si no hay ninguna.
    pub fn is_empty(&self) -> bool {
        self.filters.is_empty()
    }
}

impl std::fmt::Debug for Catalog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Catalog")
            .field("nodes", &self.filters.keys().collect::<Vec<_>>())
            .finish()
    }
}
