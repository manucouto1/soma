//! Translating between `Value` and Python objects. Only that.
//!
//! The correspondence is deliberately symmetric: what goes in as a list comes
//! out as a list. A conversion that does not round-trip is the kind of surprise
//! nobody understands later.
//!
//! [`PyOpaque`] is the explicit exception: what you wrap with `Opaque(x)`
//! crosses **without being converted** and comes out as the same object. It is
//! asked for by hand precisely so it does not happen by accident — an unknown
//! object still raises rather than slipping through opaque.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PyString};
use soma_next_core::Value;
use std::sync::Arc;

/// Marks a value so it crosses the graph untouched. The node that receives it
/// sees it **unwrapped**, so it is only written on returning.
#[pyclass(name = "Opaque", module = "soma_next._soma_next", frozen)]
pub struct PyOpaque {
    /// The object as it is.
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
        let name = self
            .value
            .bind(py)
            .get_type()
            .name()
            .map(|n| n.to_string())
            .unwrap_or_else(|_| "?".into());
        format!("Opaque({name})")
    }
}

/// What to do with something there is no variant for.
#[derive(Clone, Copy)]
enum Unknown {
    /// Refuse it, which is what an **edge** does: there a bare tensor is a
    /// mistake with two right answers — convert it, or say `Opaque` and mean it
    /// — and the refusal is what makes the cost of the first one visible.
    Refused,
    /// Wrap it, which is what a **store** does: there it is bytes either way, so
    /// the refusal defends nothing and only makes whoever is keeping a map of
    /// tensors write `Opaque` a hundred times.
    Wrapped,
}

/// From a Python object to the value that crosses an edge.
pub fn from_py(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    convert(obj, Unknown::Refused)
}

/// The same, for something on its way into a store: whatever has no variant is
/// handed to the codecs instead of refused.
pub fn to_be_kept(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    convert(obj, Unknown::Wrapped)
}

fn convert(obj: &Bound<'_, PyAny>, unknown: Unknown) -> PyResult<Value> {
    if let Ok(wrapped) = obj.downcast::<PyOpaque>() {
        return Ok(Value::opaque(wrapped.get().value.clone_ref(obj.py())));
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
    // A bool is a subclass of int in Python, and left out on purpose.
    if obj.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(
            "a bool does not cross an edge yet: there is no variant for it and \
             converting it to 1.0 would be lying",
        ));
    }
    if obj.is_instance_of::<PyFloat>() || obj.is_instance_of::<PyInt>() {
        return Ok(Value::number(obj.extract::<f64>()?));
    }
    if let Ok(dict) = obj.downcast::<PyDict>() {
        // A dict keeps insertion order, and so does `Value::Map`.
        let pairs = dict
            .iter()
            .map(|(k, v)| {
                let key: String = k.extract().map_err(|_| {
                    PyTypeError::new_err("the keys of a dict that crosses an edge have to be text")
                })?;
                Ok((key, convert(&v, unknown)?))
            })
            .collect::<PyResult<Vec<_>>>()?;
        return Ok(Value::map(pairs));
    }
    if let Ok(list) = obj.downcast::<PyList>() {
        let items: Vec<Value> = list
            .iter()
            .map(|item| convert(&item, unknown))
            .collect::<PyResult<_>>()?;
        return Ok(Value::list(items));
    }
    match unknown {
        Unknown::Wrapped => Ok(Value::opaque(obj.clone().unbind())),
        Unknown::Refused => Err(PyTypeError::new_err(format!(
            "a `{}` does not cross an edge: today None, str, bytes, numbers, lists \
             and dicts with text keys do. For something to cross without being \
             converted, wrap it: Opaque(x)",
            obj.get_type().name()?
        ))),
    }
}

/// From the value that crosses an edge to the object the user sees.
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
                    "this opaque value was not put there by Python, so there is \
                     nothing to return here",
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
