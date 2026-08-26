//! A dataset, from Python.

use crate::frame::PyFrame;
use crate::store::PyStore;
use crate::value;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use somatize_core::{Ctx, Node};
use somatize_data::Parquet;

/// A parquet file in a store, answering spans of rows. A node like any other and
/// deliberately not special: the DSL, the device, `.at()`, `.cached()` and the
/// record all reach it because `somatize.data.Parquet` puts a `forward` on it.
///
/// What it has that a node does not is [`version`](PySource::version) — what this
/// dataset **is** — and it costs one lookup and no bytes.
#[pyclass(name = "Source", module = "somatize._somatize", frozen)]
pub struct PySource {
    inner: Parquet,
}

#[pymethods]
impl PySource {
    /// The parquet file bound under that name, in that store.
    #[new]
    fn new(store: &PyStore, name: &str) -> PyResult<Self> {
        Parquet::at(store.shared(), name)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.message().to_string()))
    }

    /// What this dataset is: the digest of its content.
    #[getter]
    fn version(&self) -> &str {
        self.inner.version()
    }

    /// The name it was declared under.
    #[getter]
    fn name(&self) -> &str {
        self.inner.name()
    }

    /// The rows a span names. The node's half, and the only one the engine
    /// calls.
    fn forward(&self, py: Python<'_>, input: &Bound<'_, PyAny>) -> PyResult<PyFrame> {
        let span = value::from_py(input)?;
        // The GIL goes back while the file is read: a source is IO and decode,
        // and holding it would stop every other node in the wave.
        let out = py
            .allow_threads(|| self.inner.forward(&span, &Ctx { device: None }))
            .map_err(|e| PyValueError::new_err(e.message().to_string()))?;
        let frame = somatize_data::Frame::of(&out)
            .ok_or_else(|| PyValueError::new_err("a source answers with a frame"))?;
        Ok(PyFrame::new(frame.clone()))
    }

    fn __repr__(&self) -> String {
        format!(
            "Source({} · {})",
            self.inner.name(),
            &self.inner.version()[..12]
        )
    }
}
