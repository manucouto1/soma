//! Values across the boundary: Python objects to `Value` and back.

use crate::prelude::*;

pub(crate) fn py_any_to_json(obj: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    let py = obj.py();
    let json_mod = py.import("json")?;
    let text: String = json_mod.call_method1("dumps", (obj,))?.extract()?;
    serde_json::from_str(&text)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("not JSON-serializable: {e}")))
}

/// A `serde_json::Value` as the Python object it describes.
///
/// Through `json.loads`, so a list arrives as a list and an object as a
/// dict. The hand-written match this replaces ended in
/// `other => other.to_string()`: every array and every object reached
/// Python as the *string* of its JSON. A study whose search space held a
/// list gave `"[1, 2, 3]"` back from `trial["params"]`.
pub(crate) fn json_to_py(py: Python<'_>, v: &serde_json::Value) -> PyResult<PyObject> {
    let json = py.import("json")?;
    Ok(json.call_method1("loads", (v.to_string(),))?.unbind())
}

// ── Value conversion ──

/// A value's JSON form, if JSON can hold it without changing it.
///
/// The question is not "does `json.dumps` succeed" — it is lenient, and
/// turns a tuple into a list and an integer key into a string. It is
/// "does the value survive", because whatever this returns becomes the
/// value a loop reads a stop signal out of, a branch reads an arm label
/// out of, and a remote worker in another language receives. A value
/// that would come back changed keeps its pickle instead.
///
/// This used to answer by round-tripping: `json.dumps`, `json.loads`,
/// then `==` against the original. Correct, and three Python calls plus
/// a `serde_json` parse *per dict, per node hop* — on the path every
/// value takes between two filters. Walking the object once in Rust
/// answers the same question directly, and says which construct was the
/// problem rather than inferring it from an inequality.
///
/// Deliberately equivalent to the round-trip, case for case:
///
/// - a tuple became a list and compared unequal → rejected here too;
/// - an integer key became `"1"` and compared unequal → rejected;
/// - `NaN` and `±inf` are what `json.dumps` writes as bare `NaN`/`Infinity`,
///   and `NaN != NaN` made the round-trip reject them. They are rejected
///   here explicitly, because `serde_json` turns a non-finite float into
///   `null` — the same silent flattening that once gave two different
///   tensors one cache key;
/// - an integer too large for `i64`/`u64` round-tripped fine but has no
///   `serde_json::Number`, so it is rejected rather than quietly losing
///   precision as an `f64`.
pub(crate) fn as_json(obj: &Bound<'_, PyAny>) -> PyResult<Option<serde_json::Value>> {
    // Order matters: in Python `bool` is an `int`, so bool goes first.
    if obj.is_none() {
        return Ok(Some(serde_json::Value::Null));
    }
    if obj.is_instance_of::<pyo3::types::PyBool>() {
        return Ok(Some(serde_json::Value::Bool(obj.extract::<bool>()?)));
    }
    if obj.is_instance_of::<pyo3::types::PyInt>() {
        return Ok(obj
            .extract::<i64>()
            .ok()
            .map(serde_json::Value::from)
            .or_else(|| obj.extract::<u64>().ok().map(serde_json::Value::from)));
    }
    if obj.is_instance_of::<pyo3::types::PyFloat>() {
        let f = obj.extract::<f64>()?;
        return Ok(f.is_finite().then(|| serde_json::Value::from(f)));
    }
    if obj.is_instance_of::<pyo3::types::PyString>() {
        return Ok(Some(serde_json::Value::String(obj.extract::<String>()?)));
    }
    if obj.is_instance_of::<PyList>() {
        let mut out = Vec::new();
        for item in obj.try_iter()? {
            match as_json(&item?)? {
                Some(v) => out.push(v),
                None => return Ok(None),
            }
        }
        return Ok(Some(serde_json::Value::Array(out)));
    }
    if let Ok(dict) = obj.downcast::<PyDict>() {
        let mut map = serde_json::Map::with_capacity(dict.len());
        for (key, value) in dict.iter() {
            // A non-string key is what `json.dumps` would have stringified.
            let Ok(key) = key.extract::<String>() else {
                return Ok(None);
            };
            match as_json(&value)? {
                Some(v) => {
                    map.insert(key, v);
                }
                None => return Ok(None),
            }
        }
        return Ok(Some(serde_json::Value::Object(map)));
    }
    // A tuple, an ndarray, a dataclass, anything custom: JSON would either
    // refuse it or change it.
    Ok(None)
}

pub(crate) fn py_to_value(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Value> {
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

    // A non-numeric list — `["summarise", "critique"]`, a plan, a list of
    // records. Numeric lists were caught above and stay tensors, so no
    // existing cache key moves; this only rescues what used to be a flat
    // "cannot convert" on the most ordinary thing to hand a fan-out.
    if obj.is_instance_of::<PyList>()
        && let Some(json) = as_json(obj)?
    {
        return Ok(Value::json(json));
    }

    if obj.is_instance_of::<PyDict>() {
        // A dict that JSON can hold *becomes* JSON. An opaque pickle is
        // unreadable to everything outside this process: a loop cannot read
        // a stop signal out of it, a branch cannot read an arm label, a
        // report cannot show it and a remote worker in another language
        // cannot receive it. Round-tripping is what decides — a dict with
        // tuples or ndarrays inside comes back changed (or not at all), and
        // those keep the pickle.
        if let Some(json) = as_json(obj)? {
            return Ok(Value::json(json));
        }
        let pickle = py.import("pickle")?;
        let data: Vec<u8> = pickle.call_method1("dumps", (obj, 5i32))?.extract()?;
        return Ok(Value::object(data));
    }

    // A control value is usually one of these. They used to be outright
    // errors, so nothing can be relying on the old behaviour.
    if obj.is_none() {
        return Ok(Value::Empty);
    }
    if obj.is_instance_of::<pyo3::types::PyBool>() {
        return Ok(Value::json(serde_json::Value::Bool(obj.extract::<bool>()?)));
    }

    // A bare number, like a bare bool, used to be an outright error — which
    // made `Spawn([Run("worker", 3)])` fail on the most obvious thing anyone
    // would write. JSON rather than a 1-element tensor so it comes back as a
    // number, and so a loop or a branch can still read it.
    //
    // After the bool check, never before it: in Python `bool` *is* an `int`,
    // and `True` would extract as `1`.
    if obj.is_instance_of::<pyo3::types::PyInt>()
        && let Ok(i) = obj.extract::<i64>()
    {
        return Ok(Value::json(serde_json::Value::from(i)));
    }
    if obj.is_instance_of::<pyo3::types::PyFloat>()
        && let Ok(f) = obj.extract::<f64>()
    {
        // NaN and the infinities have no JSON spelling; they stay tensors
        // rather than becoming null and losing the value silently.
        return Ok(match serde_json::Number::from_f64(f) {
            Some(n) => Value::json(serde_json::Value::Number(n)),
            None => Value::tensor(vec![f], vec![1]),
        });
    }

    if let Ok(s) = obj.extract::<String>() {
        // A string that parses as JSON stays JSON — changing that would move
        // the cache key of every pipeline already passing JSON strings.
        // Anything else is plain text (a prompt, a label, a completion),
        // which used to be an outright error.
        return Ok(match serde_json::from_str(&s) {
            Ok(val) => Value::json(val),
            Err(_) => Value::text(s),
        });
    }

    Err(PyRuntimeError::new_err(
        "Cannot convert Python object to Value. Expected list, 2D list, dict, str, or JSON string.",
    ))
}

pub(crate) fn value_to_py(py: Python<'_>, val: &Value) -> PyResult<PyObject> {
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
                Ok(values.as_slice().into_pyobject(py)?.into_any().unbind())
            }
        }
        Value::Text(s) => Ok(s.as_ref().into_pyobject(py)?.into_any().unbind()),
        Value::Json(v) => {
            let json_str = v.to_string();
            let json_mod = py.import("json")?;
            let obj = json_mod.call_method1("loads", (json_str,))?;
            Ok(obj.unbind())
        }
        Value::Object(data) => {
            let pickle = py.import("pickle")?;
            let py_bytes = PyBytes::new(py, data.as_slice());
            let obj = pickle.call_method1("loads", (py_bytes,))?;
            Ok(obj.unbind())
        }
        Value::Bytes(b) => Ok(b.as_slice().into_pyobject(py)?.into_any().unbind()),
        Value::Empty => Ok(py.None()),
        _ => Ok(py.None()),
    }
}
