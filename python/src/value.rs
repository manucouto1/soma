//! Traducir entre `Value` y los objetos de Python. Solo eso.
//!
//! Es la frontera, y es donde se nota qué NO tiene `Value` todavía: un dict
//! de Python no cruza porque no hay variante `Json`, y el error lo dice con
//! todas las letras en vez de inventarse una representación.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyFloat, PyInt, PyList, PyString};
use soma_next_core::Value;

/// De un objeto Python al valor que cruza una arista.
pub fn from_py(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() {
        return Ok(Value::Null);
    }
    if let Ok(s) = obj.downcast::<PyString>() {
        return Ok(Value::text(s.to_cow()?));
    }
    if let Ok(b) = obj.downcast::<PyBytes>() {
        return Ok(Value::Bytes(std::sync::Arc::new(b.as_bytes().to_vec())));
    }
    // bool es subclase de int en Python; se deja fuera a propósito, porque
    // `True` como el tensor `1.0` es la clase de conversión silenciosa que
    // después nadie entiende.
    if obj.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(
            "un bool no cruza una arista todavía: no hay variante para él y \
             convertirlo a 1.0 sería mentir",
        ));
    }
    if obj.is_instance_of::<PyFloat>() || obj.is_instance_of::<PyInt>() {
        return Ok(Value::scalar(obj.extract::<f64>()?));
    }
    if let Ok(list) = obj.downcast::<PyList>() {
        let numbers: Vec<f64> = list
            .iter()
            .map(|item| item.extract::<f64>())
            .collect::<PyResult<_>>()
            .map_err(|_| {
                PyTypeError::new_err(
                    "una lista solo cruza si todos sus elementos son números; \
                     las listas anidadas y las de objetos llegarán con su caso de uso",
                )
            })?;
        return Ok(Value::vector(numbers));
    }
    Err(PyTypeError::new_err(format!(
        "un `{}` no cruza una arista: hoy solo pasan None, str, bytes, números \
         y listas de números",
        obj.get_type().name()?
    )))
}

/// Del valor que cruza una arista al objeto que ve el usuario.
pub fn to_py(py: Python<'_>, value: &Value) -> PyResult<PyObject> {
    Ok(match value {
        Value::Null => py.None(),
        Value::Text(s) => PyString::new(py, s).into(),
        Value::Bytes(b) => PyBytes::new(py, b).into(),
        Value::Tensor { values, shape } => match shape.len() {
            0 => values[0].into_pyobject(py)?.into_any().unbind(),
            1 => PyList::new(py, values.iter())?.into_any().unbind(),
            n => {
                return Err(PyTypeError::new_err(format!(
                    "un tensor de {n} dimensiones no sabe volver a Python todavía"
                )));
            }
        },
    })
}
