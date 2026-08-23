//! A Python object seen as a Rust `Node`, and the types that cross with it.
//!
//! **One adapter, one calling convention**: `forward(input, ctx)` returns the
//! output, the same as in Rust. There is no wrapper around it and nothing to
//! unwrap on this side.
//!
//! `Ctx` is a `#[pyclass]` and not a bare dictionary: it is the core's own
//! concept crossing the seam, so the adapter hands it by type instead of asking
//! Python to agree on a dict's keys.

use crate::value::{PyOpaque, from_py, to_py};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use soma_next_core::{Ctx, Device, Node, NodeError, Value};

/// What a node knows beyond its input.
#[pyclass(name = "Ctx", module = "soma_next._soma_next", frozen)]
pub struct PyCtx {
    /// Where this node was said to run — `"cuda:0"` — or `None`. Written the
    /// way torch writes it, so it can be handed straight to `.to()`.
    #[pyo3(get)]
    device: Option<String>,
}

#[pymethods]
impl PyCtx {
    /// One by hand, for whoever calls a node's `forward` themselves.
    ///
    /// The engine builds these; a test, a trace of an architecture, or somebody
    /// checking what their node does needs one too, and inventing an object
    /// with a `device` attribute is what people do when a library will not let
    /// them make the real thing.
    #[new]
    #[pyo3(signature = (device = None))]
    fn new(device: Option<String>) -> Self {
        Self { device }
    }

    fn __repr__(&self) -> String {
        match &self.device {
            Some(device) => format!("Ctx(device={device})"),
            None => "Ctx()".to_string(),
        }
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
    fn forward(&self, input: &Value, ctx: &Ctx<'_>) -> Result<Value, NodeError> {
        Python::with_gil(|py| {
            let prepare = || -> PyResult<(PyObject, Py<PyCtx>)> {
                Ok((
                    to_py(py, input)?,
                    Py::new(
                        py,
                        PyCtx {
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
            from_py(answer.bind(py)).map_err(as_node_error)
        })
    }
}

/// Checks that what a placed node returned is where it was said to be.
///
/// The only defence against a node ignoring its `ctx.device` in silence: from
/// outside there is one thing to look at, where what it returned ended up. Only
/// what has a `.device` is checked — a tensor, loose or inside an `Opaque`.
fn obeyed(answer: &Bound<'_, PyAny>, device: &Device) -> Result<(), NodeError> {
    let py = answer.py();
    // An `Opaque` is not the value: it carries it inside.
    let value = match answer.downcast::<PyOpaque>() {
        Ok(opaque) => opaque.get().value.bind(py).clone(),
        Err(_) => answer.clone(),
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

/// Whatever Python raised, said as the node's failure.
fn as_node_error(e: impl std::fmt::Display) -> NodeError {
    NodeError::new(e.to_string())
}
