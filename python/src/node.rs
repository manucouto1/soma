//! A Python object seen as a Rust `Node`, and the types that cross with it.
//!
//! **One adapter, one calling convention**: `forward(input, ctx)` returns a
//! transition, the same one as in Rust and with the same names.
//!
//! `Ctx`, `Done` and `Await` are `#[pyclass]`es and not bare dictionaries: they
//! are the core's own concepts crossing the seam, so the adapter recognizes them
//! by type instead of guessing from a dict's keys.

use crate::value::{PyOpaque, from_py, to_py};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use soma_next_core::{Ctx, Device, Driver, DriverError, Node, NodeError, Transition, Value};

/// What a node knows beyond its input.
#[pyclass(name = "Ctx", module = "soma_next._soma_next", frozen)]
pub struct PyCtx {
    /// How many times it has already been asked; starts at 0.
    #[pyo3(get)]
    turn: usize,
    /// What the driver returned for the previous turn's requests, in order.
    #[pyo3(get)]
    results: Vec<PyObject>,
    /// Where this node was said to run — `"cuda:0"` — or `None`. Written the
    /// way torch writes it, so it can be handed straight to `.to()`.
    #[pyo3(get)]
    device: Option<String>,
}

#[pymethods]
impl PyCtx {
    fn __repr__(&self) -> String {
        match &self.device {
            Some(device) => format!(
                "Ctx(turn={}, results={}, device={device})",
                self.turn,
                self.results.len()
            ),
            None => format!("Ctx(turn={}, results={})", self.turn, self.results.len()),
        }
    }
}

/// Finished, with this output.
#[pyclass(name = "Done", module = "soma_next._soma_next", frozen)]
pub struct PyDone {
    /// What the node produced.
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

/// Needs someone to do this before continuing.
#[pyclass(name = "Await", module = "soma_next._soma_next", frozen)]
pub struct PyAwait {
    /// The requests, in order. The driver answers one per request.
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
        format!("Await({} requests)", self.requests.len())
    }
}

/// A Python object with `forward(input, ctx)`, from the Rust side.
pub struct PyNode {
    obj: PyObject,
}

impl PyNode {
    /// Wraps the object, checking that it knows what it has to know.
    pub fn new(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        if !obj.hasattr("forward")? {
            return Err(PyTypeError::new_err(format!(
                "a `{}` cannot be a node: it is missing forward()",
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
                            device: ctx.device.map(Device::to_string),
                        },
                    )?,
                ))
            };
            let (py_input, py_ctx) = prepare().map_err(as_node_error)?;

            let answer = self
                .obj
                .call_method1(py, "forward", (py_input, py_ctx))
                .map_err(as_node_error)?;
            if let Some(device) = ctx.device {
                obeyed(answer.bind(py), device)?;
            }
            transition_from_py(answer.bind(py))
        })
    }
}

/// Checks that what a placed node returned is where it was said to be.
///
/// The only defence against a node ignoring its `ctx.device` in silence: from
/// outside there is one thing to look at, where what it returned ended up. Only
/// what has a `.device` is checked — a tensor, loose or inside an `Opaque`.
fn obeyed(answer: &Bound<'_, PyAny>, device: &Device) -> Result<(), NodeError> {
    let Ok(done) = answer.downcast::<PyDone>() else {
        // Still asking for things: nothing produced to look at.
        return Ok(());
    };
    let py = answer.py();
    let value = done.get().value.bind(py);
    // An `Opaque` is not the value: it carries it inside.
    let value = match value.downcast::<PyOpaque>() {
        Ok(opaque) => opaque.get().value.bind(py).clone(),
        Err(_) => value.clone(),
    };

    let Ok(landed) = value.getattr("device") else {
        return Ok(());
    };
    let landed = landed.str().map_err(as_node_error)?.to_string();
    let declared = device.to_string();
    if landed != declared {
        return Err(NodeError::new(format!(
            "it declared `{declared}` but returned a value on `{landed}`; a device \
             is obeyed in the node: move the parameters and the input with \
             `.to(ctx.device)`"
        )));
    }
    Ok(())
}

/// Reads what a `forward` returned.
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
        "forward() must return Done(value) or Await([requests]), it returned a `{}`",
        answer
            .get_type()
            .name()
            .map(|n| n.to_string())
            .unwrap_or_else(|_| "?".into())
    )))
}

/// A Python object with `perform`, from the Rust side.
pub struct PyDriver {
    obj: PyObject,
}

impl PyDriver {
    /// Wraps the object, checking that it can act as a driver.
    pub fn new(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        if !obj.hasattr("perform")? {
            return Err(PyTypeError::new_err(format!(
                "a `{}` cannot be a driver: it is missing perform()",
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
                .map_err(|_| DriverError::new("perform() has to return a list"))?
                .map(|item| from_py(&item?))
                .collect::<PyResult<Vec<Value>>>()
                .map_err(as_driver_error)?;

            if results.len() != requests.len() {
                return Err(DriverError::new(format!(
                    "{} things were asked for and perform() returned {}; the node reads them by position",
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
