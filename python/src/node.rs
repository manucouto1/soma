//! Un objeto Python visto como un `Node` de Rust, y los tipos que cruzan con él.
//!
//! **Un adaptador, una convención de llamada**: `forward(input, ctx)` devuelve
//! una transición. La misma que en Rust, con los mismos nombres.
//!
//! `Ctx`, `Done` y `Await` son `#[pyclass]` y no diccionarios sueltos porque
//! son los mismos conceptos del núcleo cruzando la costura: así el adaptador
//! los reconoce por su tipo en vez de adivinar por las claves de un dict, y un
//! error de forma se cuenta con el nombre del tipo que llegó.

use crate::value::{from_py, to_py};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use soma_next_core::{Ctx, Driver, DriverError, Node, NodeError, Transition, Value};

/// Lo que un nodo sabe además de su entrada.
#[pyclass(name = "Ctx", module = "soma_next._soma_next", frozen)]
pub struct PyCtx {
    /// Cuántas veces se le ha preguntado ya; empieza en 0.
    #[pyo3(get)]
    turn: usize,
    /// Lo que devolvió el driver de lo pedido en el turno anterior, en orden.
    #[pyo3(get)]
    results: Vec<PyObject>,
}

#[pymethods]
impl PyCtx {
    fn __repr__(&self) -> String {
        format!("Ctx(turn={}, results={})", self.turn, self.results.len())
    }
}

/// Terminado, con esta salida.
#[pyclass(name = "Done", module = "soma_next._soma_next", frozen)]
pub struct PyDone {
    /// Lo que produjo el nodo.
    #[pyo3(get)]
    value: PyObject,
}

#[pymethods]
impl PyDone {
    #[new]
    fn new(value: PyObject) -> Self {
        Self { value }
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        format!(
            "Done({})",
            self.value
                .bind(py)
                .repr()
                .map(|r| r.to_string())
                .unwrap_or_default()
        )
    }
}

/// Necesita que alguien haga esto antes de seguir.
#[pyclass(name = "Await", module = "soma_next._soma_next", frozen)]
pub struct PyAwait {
    /// Las peticiones, en orden. El driver contesta una por cada una.
    #[pyo3(get)]
    requests: Vec<PyObject>,
}

#[pymethods]
impl PyAwait {
    #[new]
    fn new(requests: Vec<PyObject>) -> Self {
        Self { requests }
    }

    fn __repr__(&self) -> String {
        format!("Await({} peticiones)", self.requests.len())
    }
}

/// Un objeto Python con `forward(input, ctx)`, por el lado de Rust.
pub struct PyNode {
    obj: PyObject,
}

impl PyNode {
    /// Envuelve el objeto, comprobando que sabe lo que hay que saber.
    pub fn new(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        if !obj.hasattr("forward")? {
            return Err(PyTypeError::new_err(format!(
                "un `{}` no puede ser un nodo: le falta forward()",
                obj.get_type().name()?
            )));
        }
        Ok(Self {
            obj: obj.clone().unbind(),
        })
    }
}

impl Node for PyNode {
    fn forward(&self, input: &Value, ctx: &Ctx<'_>) -> Result<Transition, NodeError> {
        Python::with_gil(|py| {
            let prepare = || -> PyResult<(PyObject, Py<PyCtx>)> {
                let results = ctx
                    .results
                    .iter()
                    .map(|v| to_py(py, v))
                    .collect::<PyResult<Vec<_>>>()?;
                Ok((
                    to_py(py, input)?,
                    Py::new(
                        py,
                        PyCtx {
                            turn: ctx.turn,
                            results,
                        },
                    )?,
                ))
            };
            let (py_input, py_ctx) = prepare().map_err(as_node_error)?;

            let answer = self
                .obj
                .call_method1(py, "forward", (py_input, py_ctx))
                .map_err(as_node_error)?;
            transition_from_py(answer.bind(py))
        })
    }
}

/// Lee lo que devolvió un `forward`.
fn transition_from_py(answer: &Bound<'_, PyAny>) -> Result<Transition, NodeError> {
    if let Ok(done) = answer.downcast::<PyDone>() {
        let value = done.get().value.bind(answer.py());
        return from_py(value).map(Transition::Done).map_err(as_node_error);
    }
    if let Ok(waiting) = answer.downcast::<PyAwait>() {
        let py = answer.py();
        let requests = waiting
            .get()
            .requests
            .iter()
            .map(|r| from_py(r.bind(py)))
            .collect::<PyResult<Vec<Value>>>()
            .map_err(as_node_error)?;
        return Ok(Transition::Await(requests));
    }
    Err(NodeError::new(format!(
        "forward() debe devolver Done(valor) o Await([peticiones]), devolvió un `{}`",
        answer
            .get_type()
            .name()
            .map(|n| n.to_string())
            .unwrap_or_else(|_| "?".into())
    )))
}

/// Un objeto Python con `perform`, por el lado de Rust.
pub struct PyDriver {
    obj: PyObject,
}

impl PyDriver {
    /// Envuelve el objeto, comprobando que puede hacer de driver.
    pub fn new(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        if !obj.hasattr("perform")? {
            return Err(PyTypeError::new_err(format!(
                "un `{}` no puede ser un driver: le falta perform()",
                obj.get_type().name()?
            )));
        }
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

fn as_node_error(e: PyErr) -> NodeError {
    NodeError::new(e.to_string())
}

fn as_driver_error(e: PyErr) -> DriverError {
    DriverError::new(e.to_string())
}
