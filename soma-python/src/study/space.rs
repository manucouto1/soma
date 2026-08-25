//! `soma_next.study.Space` — the knobs, and `Point` — one setting of them all.

use super::{hashed, to_py_err};
use pyo3::prelude::*;
use pyo3::types::PyString;
use somatize_study::{Dimension, Point, Setting, Space};

/// `soma_next.study.Space` — what is being searched over.
///
/// Built up, and every call gives back a **new** space: the one you had is still
/// the one you had, which is what makes handing the same base to two studies
/// safe.
#[pyclass(name = "Space", module = "soma_next._soma_next", frozen)]
#[derive(Clone)]
pub struct PySpace {
    pub(super) space: Space,
}

#[pymethods]
impl PySpace {
    /// Nothing to search yet.
    #[new]
    fn new() -> Self {
        Self {
            space: Space::new(),
        }
    }

    /// A knob that is anything between the two.
    ///
    /// `log=True` draws it evenly in the **logarithm**, which is the only sane
    /// way to search a learning rate: drawn linearly, four fifths of
    /// `1e-5..1e-1` sits above `0.02`.
    #[pyo3(signature = (name, low, high, *, log = false))]
    fn real(&self, name: String, low: f64, high: f64, log: bool) -> PyResult<Self> {
        self.plus(name, Dimension::Real { low, high, log })
    }

    /// A knob that is a whole number between the two, both included.
    fn int(&self, name: String, low: i64, high: i64) -> PyResult<Self> {
        self.plus(name, Dimension::Int { low, high })
    }

    /// A knob that is one of these, by name.
    fn choice(&self, name: String, options: Vec<String>) -> PyResult<Self> {
        self.plus(name, Dimension::Choice(options))
    }

    /// The point that text names, read against these knobs.
    ///
    /// The other half of `str(point)`, and it needs the space in front of it:
    /// `batch=64` on its own does not say whether 64 is a whole number or an
    /// option spelt `"64"`.
    ///
    /// It is what makes a study's history come back in **one scan** of the
    /// shared folder: a trial keeps its configuration as text next to its score,
    /// so nothing has to be fetched to know where it looked.
    fn read(&self, said: &str) -> PyResult<PyPoint> {
        self.space.read(said).map(PyPoint::from).map_err(to_py_err)
    }

    /// The knobs, in declaration order.
    fn names(&self) -> Vec<String> {
        self.space
            .dimensions()
            .iter()
            .map(|(name, _)| name.clone())
            .collect()
    }

    fn __len__(&self) -> usize {
        self.space.len()
    }

    fn __str__(&self) -> String {
        self.space.to_string()
    }

    fn __repr__(&self) -> String {
        format!("Space({})", self.space)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.space == other.space
    }
}

impl PySpace {
    /// One more knob, or the reason it is not one.
    fn plus(&self, name: String, dimension: Dimension) -> PyResult<Self> {
        Ok(Self {
            space: self
                .space
                .clone()
                .with(name, dimension)
                .map_err(to_py_err)?,
        })
    }
}

/// `soma_next.study.Point` — one configuration.
///
/// It behaves as a mapping, so `build(**point)` and `point["lr"]` both work, and
/// `str(point)` is the trial's **name** — derived from the values in the space's
/// order, so two machines that never spoke file it identically.
#[pyclass(name = "Point", module = "soma_next._soma_next", frozen)]
pub struct PyPoint {
    pub(super) point: Point,
}

#[pymethods]
impl PyPoint {
    /// The knobs, in the space's order. What makes `**point` work.
    fn keys(&self) -> Vec<String> {
        self.point
            .settings()
            .iter()
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// The values, in the same order.
    fn values<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyAny>> {
        self.point
            .settings()
            .iter()
            .map(|(_, setting)| said(py, setting))
            .collect()
    }

    /// Both, paired.
    fn items<'py>(&self, py: Python<'py>) -> Vec<(String, Bound<'py, PyAny>)> {
        self.point
            .settings()
            .iter()
            .map(|(name, setting)| (name.clone(), said(py, setting)))
            .collect()
    }

    fn __getitem__<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
        self.point
            .get(name)
            .map(|setting| said(py, setting))
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(name.to_string()))
    }

    fn __contains__(&self, name: &str) -> bool {
        self.point.get(name).is_some()
    }

    fn __iter__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let names: Vec<Bound<'_, PyString>> = self
            .keys()
            .iter()
            .map(|name| PyString::new(py, name))
            .collect();
        Ok(pyo3::types::PyList::new(py, names)?.try_iter()?.into())
    }

    fn __len__(&self) -> usize {
        self.point.len()
    }

    fn __str__(&self) -> String {
        self.point.to_string()
    }

    fn __repr__(&self) -> String {
        format!("Point({})", self.point)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.point == other.point
    }

    fn __hash__(&self) -> u64 {
        hashed(&self.point)
    }
}

impl From<Point> for PyPoint {
    fn from(point: Point) -> Self {
        Self { point }
    }
}

/// A setting as the Python value it is: a float, an int or a string. There is no
/// wrapper, because what a configuration is handed to is `build(**point)`.
fn said<'py>(py: Python<'py>, setting: &Setting) -> Bound<'py, PyAny> {
    match setting {
        Setting::Real(value) => value.into_pyobject(py).unwrap().into_any(),
        Setting::Int(value) => value.into_pyobject(py).unwrap().into_any(),
        Setting::Choice(option) => PyString::new(py, option).into_any(),
    }
}
