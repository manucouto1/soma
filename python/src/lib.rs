//! La costura con Python. Traduce, no decide.
//!
//! Toda la lógica vive en `soma_next_core`. Lo que hay aquí es conversión de
//! tipos y nada más: si una regla del dominio acaba escrita en este crate,
//! está en el sitio equivocado.

use pyo3::prelude::*;

/// `soma_next.Graph` — envoltorio del `Graph` del núcleo.
#[pyclass(name = "Graph", module = "soma_next._soma_next")]
struct PyGraph {
    inner: soma_next_core::Graph,
}

#[pymethods]
impl PyGraph {
    #[new]
    fn new() -> Self {
        Self {
            inner: soma_next_core::Graph::new(),
        }
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("Graph({} nodos)", self.inner.len())
    }
}

#[pymodule]
fn _soma_next(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGraph>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
