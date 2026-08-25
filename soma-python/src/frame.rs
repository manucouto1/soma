//! A frame, on the Python side of the wall.

use arrow_array::{
    Array, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, LargeStringArray,
    StringArray,
};
use arrow_schema::DataType;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};
use somatize_data::Frame;

/// The batch of columns a source answered with, as Python sees it.
///
/// # Why it arrives as this and not as a dataframe
///
/// Because which dataframe is not ours to decide. What comes out of a source is
/// Arrow, and `polars`, `pandas` and `pyarrow` all read Arrow — so this hands
/// over [`ipc`](PyFrame::ipc) and `somatize.data` turns it into whichever of
/// them is installed. A crate that imported one of the three would make it a
/// dependency of a worker that only counts rows.
///
/// # What it costs
///
/// A frame reaching a node is one `Arc` clone: the buffers stay where they are
/// and nothing is converted. Asking for [`ipc`](PyFrame::ipc) is where bytes get
/// written, and that is the caller's decision to make once, not the frontier's
/// to make on every edge.
#[pyclass(name = "Frame", module = "somatize._somatize", frozen)]
pub struct PyFrame {
    pub(crate) inner: Frame,
}

impl PyFrame {
    /// This frame, as Python holds it.
    pub fn new(inner: Frame) -> Self {
        Self { inner }
    }

    /// And the frame itself, for the way back across the wall.
    pub fn frame(&self) -> &Frame {
        &self.inner
    }
}

#[pymethods]
impl PyFrame {
    /// How many rows it brought. Short means the dataset ended.
    #[getter]
    fn rows(&self) -> usize {
        self.inner.rows()
    }

    /// What the columns are called, in order.
    #[getter]
    fn columns(&self) -> Vec<String> {
        self.inner
            .schema()
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect()
    }

    /// One column, as a list of Python values.
    ///
    /// The reason this exists next to [`ipc`](PyFrame::ipc): a worker whose
    /// whole image is 193 MB and has no tensors in it should not have to
    /// install a dataframe library to read a column of text. `to_polars` is for
    /// when there is work to do on the rows; this is for when a node wants the
    /// values and nothing else.
    ///
    /// The types it hands over are the ones a `Value` already has — text,
    /// numbers, booleans — and anything else says so rather than guessing. A
    /// missing value is `None`.
    fn column(&self, py: Python<'_>, name: &str) -> PyResult<PyObject> {
        let batch = self.inner.batch();
        let at = batch.schema().index_of(name).map_err(|_| {
            PyValueError::new_err(format!(
                "there is no column `{name}` here: it has {}",
                self.columns().join(", ")
            ))
        })?;
        let column = batch.column(at);
        let out = PyList::empty(py);
        match column.data_type() {
            DataType::Utf8 => each::<StringArray, _>(column, &out, |a, i| a.value(i).to_string())?,
            DataType::LargeUtf8 => {
                each::<LargeStringArray, _>(column, &out, |a, i| a.value(i).to_string())?
            }
            DataType::Int64 => each::<Int64Array, _>(column, &out, |a, i| a.value(i))?,
            DataType::Int32 => each::<Int32Array, _>(column, &out, |a, i| a.value(i))?,
            DataType::Float64 => each::<Float64Array, _>(column, &out, |a, i| a.value(i))?,
            DataType::Float32 => each::<Float32Array, _>(column, &out, |a, i| a.value(i))?,
            DataType::Boolean => each::<BooleanArray, _>(column, &out, |a, i| a.value(i))?,
            other => {
                return Err(PyValueError::new_err(format!(
                    "`{name}` is a {other}, and there is no way to hand one of those \
                     to Python one value at a time yet. `somatize.data.to_polars` or \
                     `to_arrow` read the whole frame, whatever is in it"
                )));
            }
        }
        Ok(out.into_any().unbind())
    }

    /// The Arrow IPC bytes of it, which is what every dataframe library reads.
    fn ipc<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let written = self
            .inner
            .written()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.message().to_string()))?;
        Ok(PyBytes::new(py, &written))
    }

    fn __len__(&self) -> usize {
        self.inner.rows()
    }

    fn __repr__(&self) -> String {
        format!(
            "Frame({} rows · {})",
            self.rows(),
            self.columns().join(", ")
        )
    }
}

/// Every value of a column, appended in order, with nulls as `None`.
///
/// One function and not seven: what changes between an `Int64Array` and a
/// `StringArray` is the type and the getter, and writing the loop once is what
/// keeps them from drifting into seven slightly different loops.
fn each<'py, A, T>(
    column: &dyn Array,
    out: &Bound<'py, PyList>,
    value: impl Fn(&A, usize) -> T,
) -> PyResult<()>
where
    A: Array + 'static,
    T: IntoPyObject<'py>,
{
    let column = column
        .as_any()
        .downcast_ref::<A>()
        .expect("the data type was just matched on");
    for at in 0..column.len() {
        match column.is_null(at) {
            true => out.append(out.py().None())?,
            false => out.append(value(column, at))?,
        }
    }
    Ok(())
}
