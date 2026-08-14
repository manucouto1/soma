//! Un objeto Python visto como un `Filter` de Rust.
//!
//! Es el adaptador entero: 30 líneas. El del original tiene 505, y 263 de
//! ellas son `new()` — identidad determinista, cloudpickle, dependencias
//! transitivas, requirements, código fuente, esquemas. Todo eso sirve para
//! cachear y para mandar el filtro a un worker; nada de ello para ejecutarlo.

use crate::value::{from_py, to_py};
use pyo3::prelude::*;
use soma_next_core::{Filter, FilterError, Value};

/// Un objeto Python con `forward`, por el lado de Rust.
pub struct PyFilter {
    obj: PyObject,
}

impl PyFilter {
    /// Envuelve el objeto, comprobando que puede hacer de filtro.
    ///
    /// La comprobación es aquí y no a mitad de un run: un objeto sin `forward`
    /// es un error al registrarlo, cuando el usuario todavía sabe por qué.
    pub fn new(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        if !obj.hasattr("forward")? {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "un `{}` no puede ser un nodo: le falta forward()",
                obj.get_type().name()?
            )));
        }
        Ok(Self {
            obj: obj.clone().unbind(),
        })
    }
}

impl Filter for PyFilter {
    fn forward(&self, input: &Value) -> Result<Value, FilterError> {
        Python::with_gil(|py| {
            let py_input = to_py(py, input).map_err(as_filter_error)?;
            let result = self
                .obj
                .call_method1(py, "forward", (py_input,))
                .map_err(as_filter_error)?;
            from_py(result.bind(py)).map_err(as_filter_error)
        })
    }
}

/// Una excepción de Python, contada como fallo del filtro.
fn as_filter_error(e: PyErr) -> FilterError {
    FilterError::new(e.to_string())
}
