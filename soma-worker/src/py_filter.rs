//! Embedded Python filter execution via PyO3.
//!
//! Deserializes cloudpickle bytes into a live Python object and calls
//! fit/forward directly — no subprocess, no serialization overhead.
//! The model stays in GPU memory between calls.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use somatize_core::cache::CacheKey;
use somatize_core::error::{Result as SomaResult, SomaError};
use somatize_core::filter::{Filter, FilterKind, FilterMeta, StreamMode};
use somatize_core::value::Value;
use std::sync::Mutex;

// ── Value conversion (adapted from soma-python) ──

fn py_to_value(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if let Ok(lists) = obj.extract::<Vec<Vec<f64>>>() {
        let rows = lists.len();
        let cols = if rows > 0 { lists[0].len() } else { 0 };
        let flat: Vec<f64> = lists.into_iter().flatten().collect();
        return Ok(Value::tensor(flat, vec![rows, cols]));
    }

    if let Ok(arr) = obj.extract::<Vec<f64>>() {
        let len = arr.len();
        return Ok(Value::tensor(arr, vec![len]));
    }

    if obj.is_instance_of::<PyDict>() {
        let json_mod = py.import("json")?;
        let json_str: String = json_mod.call_method1("dumps", (obj,))?.extract()?;
        let val: serde_json::Value =
            serde_json::from_str(&json_str).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        return Ok(Value::Json(val));
    }

    if let Ok(s) = obj.extract::<String>()
        && let Ok(val) = serde_json::from_str(&s)
    {
        return Ok(Value::Json(val));
    }

    // Fallback: try to convert via json.dumps
    let json_mod = py.import("json")?;
    if let Ok(json_str) = json_mod.call_method1("dumps", (obj,)) {
        let s: String = json_str.extract()?;
        if let Ok(val) = serde_json::from_str(&s) {
            return Ok(Value::Json(val));
        }
    }

    Ok(Value::Empty)
}

fn value_to_py(py: Python<'_>, val: &Value) -> PyResult<PyObject> {
    match val {
        Value::Tensor { values, shape } => {
            if shape.len() == 2 {
                let rows = shape[0];
                let cols = shape[1];
                let result = PyList::empty(py);
                for r in 0..rows {
                    let row: Vec<f64> = values[r * cols..(r + 1) * cols].to_vec();
                    result.append(row)?;
                }
                Ok(result.into_any().unbind())
            } else {
                Ok(values.into_pyobject(py)?.into_any().unbind())
            }
        }
        Value::Json(v) => {
            let json_str = v.to_string();
            let json_mod = py.import("json")?;
            let obj = json_mod.call_method1("loads", (json_str,))?;
            Ok(obj.unbind())
        }
        Value::Bytes(b) => Ok(b.into_pyobject(py)?.into_any().unbind()),
        Value::Empty => Ok(py.None()),
        _ => Ok(py.None()),
    }
}

fn py_err_to_soma(e: PyErr) -> SomaError {
    SomaError::Other(e.to_string())
}

// ── EmbeddedPyFilter ──

/// A Python filter deserialized from cloudpickle bytes and kept alive in-process.
/// Uses PyO3 `auto-initialize` to embed the Python interpreter directly.
pub struct EmbeddedPyFilter {
    /// The live Python filter object (persists between calls).
    py_obj: PyObject,
    node_id: String,
    trainable: bool,
    /// Mutex for interior mutability — Filter trait requires &self.
    _lock: Mutex<()>,
}

// Safety: PyObject is Send when we use GIL properly.
// The Mutex ensures single-threaded access to the Python object.
unsafe impl Send for EmbeddedPyFilter {}
unsafe impl Sync for EmbeddedPyFilter {}

impl EmbeddedPyFilter {
    /// Deserialize a filter from cloudpickle bytes.
    /// Optionally prepend a venv's site-packages to sys.path.
    pub fn new(
        pickled_bytes: &[u8],
        node_id: String,
        trainable: bool,
        venv_site_packages: Option<&str>,
    ) -> SomaResult<Self> {
        Python::with_gil(|py| {
            // Add venv site-packages to sys.path if provided
            if let Some(site_pkgs) = venv_site_packages {
                let sys = py.import("sys").map_err(py_err_to_soma)?;
                let path = sys.getattr("path").map_err(py_err_to_soma)?;
                path.call_method1("insert", (0, site_pkgs))
                    .map_err(py_err_to_soma)?;
            }

            let cloudpickle = py.import("cloudpickle").map_err(py_err_to_soma)?;
            let pickled_py = pyo3::types::PyBytes::new(py, pickled_bytes);
            let obj = cloudpickle
                .call_method1("loads", (pickled_py,))
                .map_err(|e| SomaError::Other(format!("cloudpickle.loads failed: {e}")))?;

            Ok(Self {
                py_obj: obj.unbind(),
                node_id,
                trainable,
                _lock: Mutex::new(()),
            })
        })
    }

    // ── Strategy methods ──

    /// Serialize trained state: torch.save(state_dict) or cloudpickle.dumps(obj).
    pub fn get_state(&self) -> SomaResult<Value> {
        Python::with_gil(|py| {
            let locals = PyDict::new(py);
            locals
                .set_item("_obj", self.py_obj.bind(py))
                .map_err(py_err_to_soma)?;
            py.run(
                c"
import io as _io
_buf = _io.BytesIO()
if hasattr(_obj, 'state_dict'):
    import torch as _torch
    _torch.save(_obj.state_dict(), _buf)
else:
    import cloudpickle as _cp
    _buf.write(_cp.dumps(_obj))
_result = _buf.getvalue()
",
                None,
                Some(&locals),
            )
            .map_err(py_err_to_soma)?;
            let bytes: Vec<u8> = locals
                .get_item("_result")
                .map_err(py_err_to_soma)?
                .ok_or_else(|| SomaError::Other("get_state: no result".into()))?
                .extract()
                .map_err(py_err_to_soma)?;
            Ok(Value::Bytes(bytes))
        })
    }

    /// Load state: torch.load → load_state_dict or cloudpickle.loads.
    pub fn set_state(&self, state: &Value) -> SomaResult<()> {
        let bytes = match state {
            Value::Bytes(b) => b.clone(),
            _ => return Err(SomaError::Other("set_state expects Value::Bytes".into())),
        };
        Python::with_gil(|py| {
            let locals = PyDict::new(py);
            locals
                .set_item("_obj", self.py_obj.bind(py))
                .map_err(py_err_to_soma)?;
            locals
                .set_item("_state_bytes", pyo3::types::PyBytes::new(py, &bytes))
                .map_err(py_err_to_soma)?;
            py.run(
                c"
import io as _io
_buf = _io.BytesIO(_state_bytes)
if hasattr(_obj, 'load_state_dict'):
    import torch as _torch
    _obj.load_state_dict(_torch.load(_buf, weights_only=True))
else:
    import cloudpickle as _cp
    _obj = _cp.loads(_buf.read())
",
                None,
                Some(&locals),
            )
            .map_err(py_err_to_soma)?;
            Ok(())
        })
    }

    /// Get gradients for DataParallel AllReduce.
    pub fn get_gradients(&self) -> SomaResult<Value> {
        Python::with_gil(|py| {
            let locals = PyDict::new(py);
            locals
                .set_item("_obj", self.py_obj.bind(py))
                .map_err(py_err_to_soma)?;
            py.run(
                c"
import io as _io
_buf = _io.BytesIO()
if hasattr(_obj, 'parameters'):
    import torch as _torch
    _grads = {name: p.grad.clone() for name, p in _obj.named_parameters() if p.grad is not None}
    _torch.save(_grads, _buf)
_result = _buf.getvalue()
",
                None,
                Some(&locals),
            )
            .map_err(py_err_to_soma)?;
            let bytes: Vec<u8> = locals
                .get_item("_result")
                .map_err(py_err_to_soma)?
                .ok_or_else(|| SomaError::Other("get_gradients: no result".into()))?
                .extract()
                .map_err(py_err_to_soma)?;
            Ok(Value::Bytes(bytes))
        })
    }

    /// Apply aggregated gradients (from AllReduce).
    pub fn apply_gradients(&self, gradients: &Value) -> SomaResult<()> {
        let bytes = match gradients {
            Value::Bytes(b) => b.clone(),
            _ => {
                return Err(SomaError::Other(
                    "apply_gradients expects Value::Bytes".into(),
                ));
            }
        };
        Python::with_gil(|py| {
            let locals = PyDict::new(py);
            locals
                .set_item("_obj", self.py_obj.bind(py))
                .map_err(py_err_to_soma)?;
            locals
                .set_item("_grad_bytes", pyo3::types::PyBytes::new(py, &bytes))
                .map_err(py_err_to_soma)?;
            py.run(
                c"
import io as _io
if hasattr(_obj, 'named_parameters'):
    import torch as _torch
    _buf = _io.BytesIO(_grad_bytes)
    _grads = _torch.load(_buf, weights_only=True)
    for name, p in _obj.named_parameters():
        if name in _grads:
            p.grad = _grads[name]
    if hasattr(_obj, 'optimizer'):
        _obj.optimizer.step()
",
                None,
                Some(&locals),
            )
            .map_err(py_err_to_soma)?;
            Ok(())
        })
    }
}

impl Filter for EmbeddedPyFilter {
    fn config_hash(&self) -> CacheKey {
        CacheKey::from_parts(&[self.node_id.as_bytes()])
    }

    fn fit(&self, x: &Value, y: Option<&Value>) -> SomaResult<Value> {
        let _guard = self._lock.lock().unwrap();
        Python::with_gil(|py| {
            let py_x = value_to_py(py, x).map_err(py_err_to_soma)?;
            let py_y = match y {
                Some(v) => value_to_py(py, v).map_err(py_err_to_soma)?,
                None => py.None(),
            };
            let result = self
                .py_obj
                .call_method1(py, "fit", (py_x, py_y))
                .map_err(|e| SomaError::Other(format!("Python fit() error: {e}")))?;
            py_to_value(py, result.bind(py)).map_err(py_err_to_soma)
        })
    }

    fn forward(&self, x: &Value, state: &Value) -> SomaResult<Value> {
        let _guard = self._lock.lock().unwrap();
        Python::with_gil(|py| {
            let py_x = value_to_py(py, x).map_err(py_err_to_soma)?;
            let py_state = value_to_py(py, state).map_err(py_err_to_soma)?;
            let result = self
                .py_obj
                .call_method1(py, "forward", (py_x, py_state))
                .map_err(|e| SomaError::Other(format!("Python forward() error: {e}")))?;
            py_to_value(py, result.bind(py)).map_err(py_err_to_soma)
        })
    }

    fn meta(&self) -> FilterMeta {
        FilterMeta {
            name: self.node_id.clone(),
            kind: if self.trainable {
                FilterKind::Trainable
            } else {
                FilterKind::Stateless
            },
            cacheable: true,
            differentiable: false,
            stream_mode: StreamMode::FixedState,
            distribution: somatize_core::filter::Distribution::Local,
            input_schema: None,
            output_schema: None,
        }
    }
}
