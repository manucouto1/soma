//! Un objeto Python visto como un `Node` de Rust.
//!
//! Hay **dos adaptadores** y **un solo contrato**. La diferencia entre ellos no
//! es de tipo: es la convención de llamada que espera el objeto de Python.
//!
//! - [`PyFilterNode`] llama a `forward(x)` y espera un valor. Envuelve en
//!   `Done` y ya está.
//! - [`PyStepNode`] llama a `forward(x, ctx)` y espera `{"done": …}` o
//!   `{"await": [...]}`.
//!
//! Cuál de los dos se usa lo decide la herencia, en `soma_next._dsl`. Aquí solo
//! se traduce.

use crate::value::{from_py, to_py};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use soma_next_core::{Ctx, Driver, DriverError, Node, NodeError, Transition, Value};

/// Un objeto Python que devuelve un valor: `forward(x)`.
pub struct PyFilterNode {
    obj: PyObject,
}

impl PyFilterNode {
    /// Envuelve el objeto, comprobando que sabe lo que hay que saber.
    pub fn new(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        require_method(obj, "forward", "un nodo")?;
        Ok(Self {
            obj: obj.clone().unbind(),
        })
    }
}

impl Node for PyFilterNode {
    fn forward(&self, input: &Value, _ctx: &Ctx<'_>) -> Result<Transition, NodeError> {
        Python::with_gil(|py| {
            let py_input = to_py(py, input).map_err(as_node_error)?;
            let result = self
                .obj
                .call_method1(py, "forward", (py_input,))
                .map_err(as_node_error)?;
            from_py(result.bind(py))
                .map(Transition::Done)
                .map_err(as_node_error)
        })
    }
}

/// Un objeto Python que devuelve una transición: `forward(x, ctx)`.
pub struct PyStepNode {
    obj: PyObject,
}

impl PyStepNode {
    /// Envuelve el objeto, comprobando que sabe lo que hay que saber.
    pub fn new(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        require_method(obj, "forward", "un nodo")?;
        Ok(Self {
            obj: obj.clone().unbind(),
        })
    }
}

impl Node for PyStepNode {
    fn forward(&self, input: &Value, ctx: &Ctx<'_>) -> Result<Transition, NodeError> {
        Python::with_gil(|py| {
            let py_ctx = PyDict::new(py);
            let fill = || -> PyResult<PyObject> {
                py_ctx.set_item("turn", ctx.turn)?;
                py_ctx.set_item(
                    "results",
                    ctx.results
                        .iter()
                        .map(|v| to_py(py, v))
                        .collect::<PyResult<Vec<_>>>()?,
                )?;
                to_py(py, input)
            };
            let py_input = fill().map_err(as_node_error)?;

            let answer = self
                .obj
                .call_method1(py, "forward", (py_input, py_ctx))
                .map_err(as_node_error)?;
            transition_from_py(answer.bind(py))
        })
    }
}

/// Lee lo que devolvió un `forward(x, ctx)`.
fn transition_from_py(answer: &Bound<'_, PyAny>) -> Result<Transition, NodeError> {
    let dict = answer.downcast::<PyDict>().map_err(|_| {
        NodeError::new(format!(
            "forward(x, ctx) debe devolver {{\"done\": …}} o {{\"await\": [...]}}, devolvió un `{}`",
            type_name_of(answer)
        ))
    })?;

    if let Ok(Some(done)) = dict.get_item("done") {
        return from_py(&done).map(Transition::Done).map_err(as_node_error);
    }
    if let Ok(Some(requests)) = dict.get_item("await") {
        let values = requests
            .try_iter()
            .map_err(|_| NodeError::new("\"await\" tiene que ser una lista de peticiones"))?
            .map(|item| from_py(&item?))
            .collect::<PyResult<Vec<Value>>>()
            .map_err(as_node_error)?;
        return Ok(Transition::Await(values));
    }
    Err(NodeError::new(
        "forward(x, ctx) devolvió un dict sin \"done\" ni \"await\"",
    ))
}

/// Un objeto Python con `perform`, por el lado de Rust.
pub struct PyDriver {
    obj: PyObject,
}

impl PyDriver {
    /// Envuelve el objeto, comprobando que puede hacer de driver.
    pub fn new(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        require_method(obj, "perform", "un driver")?;
        Ok(Self {
            obj: obj.clone().unbind(),
        })
    }
}

impl Driver for PyDriver {
    fn perform(&self, requests: &[Value]) -> Result<Vec<Value>, DriverError> {
        Python::with_gil(|py| {
            let py_requests = requests
                .iter()
                .map(|v| to_py(py, v))
                .collect::<PyResult<Vec<_>>>()
                .map_err(as_driver_error)?;
            let answer = self
                .obj
                .call_method1(py, "perform", (py_requests,))
                .map_err(as_driver_error)?;
            let results = answer
                .bind(py)
                .try_iter()
                .map_err(|_| DriverError::new("perform() tiene que devolver una lista"))?
                .map(|item| from_py(&item?))
                .collect::<PyResult<Vec<Value>>>()
                .map_err(as_driver_error)?;

            if results.len() != requests.len() {
                return Err(DriverError::new(format!(
                    "se pidieron {} cosas y perform() devolvió {}; el nodo las lee por posición",
                    requests.len(),
                    results.len()
                )));
            }
            Ok(results)
        })
    }
}

/// Un objeto al que le falta el método que lo haría utilizable.
fn require_method(obj: &Bound<'_, PyAny>, name: &str, role: &str) -> PyResult<()> {
    if obj.hasattr(name)? {
        return Ok(());
    }
    Err(PyTypeError::new_err(format!(
        "un `{}` no puede ser {role}: le falta {name}()",
        obj.get_type().name()?
    )))
}

fn type_name_of(obj: &Bound<'_, PyAny>) -> String {
    obj.get_type()
        .name()
        .map(|n| n.to_string())
        .unwrap_or_else(|_| "?".into())
}

fn as_node_error(e: PyErr) -> NodeError {
    NodeError::new(e.to_string())
}

fn as_driver_error(e: PyErr) -> DriverError {
    DriverError::new(e.to_string())
}
