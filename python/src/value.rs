//! Traducir entre `Value` y los objetos de Python. Solo eso.
//!
//! La correspondencia es simétrica a propósito: lo que entra como lista sale
//! como lista. Una conversión que no da la vuelta —una lista de números que se
//! convierte en otra cosa— es la clase de sorpresa que después nadie entiende.
//!
//! [`PyOpaque`] es la excepción, y es explícita: lo que envuelvas con
//! `Opaque(x)` cruza **sin convertirse**, y sale por el otro lado siendo el
//! mismo objeto. Es la única forma de que un valor atraviese el grafo intacto,
//! y se pide a mano justamente para que no ocurra por accidente: un objeto
//! desconocido sigue dando error en vez de colarse opaco.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PyString};
use soma_next_core::Value;
use std::sync::Arc;

/// Marca un valor para que cruce el grafo sin que nadie lo toque.
///
/// Lo que envuelve puede ser cualquier cosa: un tensor de torch a mitad de una
/// gráfica de autograd, un DataFrame, una conexión. El nodo que lo recibe lo ve
/// **desenvuelto** —el objeto original, no este envoltorio—, así que solo se
/// escribe al devolverlo.
#[pyclass(name = "Opaque", module = "soma_next._soma_next", frozen)]
pub struct PyOpaque {
    /// El objeto tal cual.
    #[pyo3(get)]
    pub(crate) value: PyObject,
}

#[pymethods]
impl PyOpaque {
    #[new]
    fn new(value: PyObject) -> Self {
        Self { value }
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let nombre = self
            .value
            .bind(py)
            .get_type()
            .name()
            .map(|n| n.to_string())
            .unwrap_or_else(|_| "?".into());
        format!("Opaque({nombre})")
    }
}

/// De un objeto Python al valor que cruza una arista.
pub fn from_py(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if let Ok(envuelto) = obj.downcast::<PyOpaque>() {
        // Se guarda el PyObject dentro del opaco del núcleo. El núcleo no sabe
        // qué es ni tiene forma de averiguarlo.
        return Ok(Value::opaque(envuelto.get().value.clone_ref(obj.py())));
    }
    if obj.is_none() {
        return Ok(Value::Null);
    }
    if let Ok(s) = obj.downcast::<PyString>() {
        return Ok(Value::text(s.to_cow()?));
    }
    if let Ok(b) = obj.downcast::<PyBytes>() {
        return Ok(Value::Bytes(Arc::new(b.as_bytes().to_vec())));
    }
    // bool es subclase de int en Python; se deja fuera a propósito, porque
    // `True` como el número 1.0 es la clase de conversión silenciosa que
    // después nadie entiende.
    if obj.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(
            "un bool no cruza una arista todavía: no hay variante para él y \
             convertirlo a 1.0 sería mentir",
        ));
    }
    if obj.is_instance_of::<PyFloat>() || obj.is_instance_of::<PyInt>() {
        return Ok(Value::number(obj.extract::<f64>()?));
    }
    if let Ok(dict) = obj.downcast::<PyDict>() {
        // Un dict de Python conserva el orden de inserción, y `Value::Map`
        // también: la ida y la vuelta dan el mismo dict.
        let pairs = dict
            .iter()
            .map(|(k, v)| {
                let key: String = k.extract().map_err(|_| {
                    PyTypeError::new_err(
                        "las claves de un dict que cruza una arista tienen que ser texto",
                    )
                })?;
                Ok((key, from_py(&v)?))
            })
            .collect::<PyResult<Vec<_>>>()?;
        return Ok(Value::map(pairs));
    }
    if let Ok(list) = obj.downcast::<PyList>() {
        let items: Vec<Value> = list
            .iter()
            .map(|item| from_py(&item))
            .collect::<PyResult<_>>()?;
        return Ok(Value::list(items));
    }
    Err(PyTypeError::new_err(format!(
        "un `{}` no cruza una arista: hoy pasan None, str, bytes, números, \
         listas y dicts con claves de texto. Para que algo cruce sin \
         convertirse, envuélvelo: Opaque(x)",
        obj.get_type().name()?
    )))
}

/// Del valor que cruza una arista al objeto que ve el usuario.
pub fn to_py(py: Python<'_>, value: &Value) -> PyResult<PyObject> {
    Ok(match value {
        Value::Null => py.None(),
        Value::Number(x) => x.into_pyobject(py)?.into_any().unbind(),
        Value::Text(s) => PyString::new(py, s).into(),
        Value::Bytes(b) => PyBytes::new(py, b).into(),
        Value::Map(pairs) => {
            let dict = PyDict::new(py);
            for (key, value) in pairs.iter() {
                dict.set_item(key, to_py(py, value)?)?;
            }
            dict.into_any().unbind()
        }
        Value::Opaque(_) => match value.downcast::<PyObject>() {
            Some(obj) => obj.clone_ref(py),
            None => {
                return Err(PyTypeError::new_err(
                    "este valor opaco no lo puso Python, así que no hay nada que \
                     devolver aquí",
                ));
            }
        },
        Value::List(items) => {
            let converted = items
                .iter()
                .map(|v| to_py(py, v))
                .collect::<PyResult<Vec<_>>>()?;
            PyList::new(py, converted)?.into_any().unbind()
        }
    })
}
