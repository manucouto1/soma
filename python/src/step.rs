//! Un objeto Python visto como un `Step`, y otro como un `Driver`.
//!
//! Un step de Python es cualquier objeto con `poll(ctx)` que devuelva
//! `{"done": valor}` o `{"await": [peticiones]}`. Un diccionario y no dos
//! clases porque el contrato tiene hoy dos formas; el día que tenga cinco,
//! merecerá tipos.

use crate::value::{from_py, to_py};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use soma_next_core::{Driver, DriverError, Step, StepCtx, StepError, Transition, Value};

/// Un objeto Python con `poll`, por el lado de Rust.
pub struct PyStep {
    obj: PyObject,
}

impl PyStep {
    /// Envuelve el objeto, comprobando que puede hacer de step.
    pub fn new(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        require_method(obj, "poll", "un step")?;
        Ok(Self {
            obj: obj.clone().unbind(),
        })
    }
}

impl Step for PyStep {
    fn poll(&self, ctx: &StepCtx<'_>) -> Result<Transition, StepError> {
        Python::with_gil(|py| {
            let py_ctx = PyDict::new(py);
            let fill = || -> PyResult<()> {
                py_ctx.set_item("input", to_py(py, ctx.input)?)?;
                py_ctx.set_item("turn", ctx.turn)?;
                py_ctx.set_item(
                    "results",
                    ctx.results
                        .iter()
                        .map(|v| to_py(py, v))
                        .collect::<PyResult<Vec<_>>>()?,
                )?;
                Ok(())
            };
            fill().map_err(as_step_error)?;

            let answer = self
                .obj
                .call_method1(py, "poll", (py_ctx,))
                .map_err(as_step_error)?;
            transition_from_py(answer.bind(py))
        })
    }
}

/// Lee lo que devolvió `poll`.
fn transition_from_py(answer: &Bound<'_, PyAny>) -> Result<Transition, StepError> {
    let dict = answer.downcast::<PyDict>().map_err(|_| {
        StepError::new(format!(
            "poll() debe devolver {{\"done\": ...}} o {{\"await\": [...]}}, devolvió un `{}`",
            answer
                .get_type()
                .name()
                .map(|n| n.to_string())
                .unwrap_or_else(|_| "?".into())
        ))
    })?;

    if let Ok(Some(done)) = dict.get_item("done") {
        return from_py(&done).map(Transition::Done).map_err(as_step_error);
    }
    if let Ok(Some(requests)) = dict.get_item("await") {
        let list = requests
            .try_iter()
            .map_err(|_| StepError::new("\"await\" tiene que ser una lista de peticiones"))?;
        let values = list
            .map(|item| from_py(&item?))
            .collect::<PyResult<Vec<Value>>>()
            .map_err(as_step_error)?;
        return Ok(Transition::Await(values));
    }
    Err(StepError::new(
        "poll() devolvió un dict sin \"done\" ni \"await\"",
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
            let bound = answer.bind(py);
            let results = bound
                .try_iter()
                .map_err(|_| DriverError::new("perform() tiene que devolver una lista"))?
                .map(|item| from_py(&item?))
                .collect::<PyResult<Vec<Value>>>()
                .map_err(as_driver_error)?;

            if results.len() != requests.len() {
                return Err(DriverError::new(format!(
                    "se pidieron {} cosas y perform() devolvió {}; el step las lee por posición",
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

fn as_step_error(e: PyErr) -> StepError {
    StepError::new(e.to_string())
}

fn as_driver_error(e: PyErr) -> DriverError {
    DriverError::new(e.to_string())
}
