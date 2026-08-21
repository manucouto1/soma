//! Turning what only exists in this process into bytes, and back.
//!
//! `Opaque` was the one thing that could not leave: it carries a live Python
//! object, and neither a wire nor a store takes one. That frontier does not
//! disappear here, it **moves**: from "the variant" to "the variant nobody
//! registered a codec for", which is the more precise statement of the two —
//! what cannot travel is not a tensor, it is a tensor nobody said how to write
//! down.
//!
//! ```python
//! codec("torch.Tensor", torch.Tensor, dump=..., load=...)
//! ```
//!
//! # Two callers, one pair of passes
//!
//! | who | what it fills | what it does not learn |
//! |---|---|---|
//! | [`Packing`] | a [`Keeper`], decorating another one | `soma_next_store` never learns Python exists |
//! | [`Codecs`] | `soma_next_transport::Codec`, on both ends of a wire | the transport never learns what an opaque carries |
//!
//! One pass each way in both: opaques out on the way in, opaques back on the way
//! out. Underneath, a store sees a value made of maps and bytes and a socket
//! sees the same, and neither has any idea any of it was ever a tensor. It is
//! the same division as everywhere else — this crate translates, it does not
//! decide.
//!
//! What a tensor weighs written down is **one** question, so it has one answer
//! whether the bytes are going to a directory or down a socket, and the two
//! callers share it rather than each keeping a registry.
//!
//! What a packed opaque looks like, and what the risk of it is:
//!
//! ```text
//! {"__soma_opaque__": "torch.Tensor", "bytes": b"..."}
//! ```
//!
//! A user whose own map has that exact key gets it read back as an opaque. It is
//! known and accepted: the alternative is a variant in the core's `Value` that
//! exists only because Python is behind it.
//!
//! # What this does **not** give back
//!
//! A gradient. What comes out of `load` is a **leaf**: the graph that produced
//! it is gone, and the backward pass stops there. That is why a cached prefix
//! has to be frozen, and it is checked before running rather than discovered as
//! a net that quietly stopped training.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::sync::GILOnceCell;
use pyo3::types::{PyBytes, PyDict, PyTuple};
use soma_next_core::{Keeper, KeeperError, Kept, Key, Value};
use soma_next_transport::{Codec, CodecError};
use std::sync::Arc;

/// The reserved key that says a map is not a map.
const PACKED: &str = "__soma_opaque__";

/// Where the bytes are, next to it.
const BYTES: &str = "bytes";

/// What is registered, by the name it was registered under: `kind → (type,
/// dump, load)`.
///
/// A Python dict and not a Rust map because what it holds are Python objects,
/// and because the order it is walked in is the order they were registered:
/// the first type that matches wins, which is what makes registering a subclass
/// after its base a thing you can do.
static CODECS: GILOnceCell<Py<PyDict>> = GILOnceCell::new();

fn codecs(py: Python<'_>) -> &Bound<'_, PyDict> {
    CODECS.get_or_init(py, || PyDict::new(py).unbind()).bind(py)
}

/// Says how objects of a type are written down and read back.
///
/// `dump(obj) -> bytes` and `load(bytes) -> obj`. The `kind` is what gets
/// written beside the bytes, so it is what a store keeps forever: name it after
/// the type, not after the run.
#[pyfunction]
#[pyo3(signature = (kind, of_type, *, dump, load))]
pub fn codec(
    py: Python<'_>,
    kind: &str,
    of_type: PyObject,
    dump: PyObject,
    load: PyObject,
) -> PyResult<()> {
    let entry = PyTuple::new(py, [of_type, dump, load])?;
    codecs(py).set_item(kind, entry)
}

/// What has a codec registered today, in the order they were registered.
#[pyfunction]
pub fn codecs_registered(py: Python<'_>) -> Vec<String> {
    codecs(py)
        .keys()
        .iter()
        .filter_map(|kind| kind.extract::<String>().ok())
        .collect()
}

/// A keeper that writes opaques down before handing them on.
pub struct Packing<'a> {
    inner: &'a dyn Keeper,
}

impl<'a> Packing<'a> {
    /// The same keeper, with the codecs in front of it.
    pub fn over(inner: &'a dyn Keeper) -> Self {
        Self { inner }
    }
}

impl Keeper for Packing<'_> {
    fn key_of(&self, value: &Value) -> Option<Key> {
        self.inner.key_of(&pack(value).ok()?)
    }

    fn combine(&self, parts: &[&str]) -> Key {
        self.inner.combine(parts)
    }

    fn recall(&self, keys: &[&Key]) -> Result<Vec<Option<Kept>>, KeeperError> {
        self.inner
            .recall(keys)?
            .into_iter()
            .map(|kept| match kept {
                None => Ok(None),
                Some(kept) => Ok(Some(Kept {
                    value: unpack(&kept.value).map_err(KeeperError::new)?,
                    meta: kept.meta,
                })),
            })
            .collect()
    }

    fn keep(&self, key: &Key, value: &Value, meta: &[(&str, &str)]) -> Result<(), KeeperError> {
        self.inner
            .keep(key, &pack(value).map_err(KeeperError::new)?, meta)
    }
}

/// The same two passes, for a wire instead of a store.
///
/// Nothing of its own: what a tensor weighs in bytes is one question, and it has
/// one answer whether the bytes are going to a directory or down a socket. It is
/// a unit struct because the registry it reads is the process's, not this
/// object's.
pub struct Codecs;

impl Codec for Codecs {
    fn packed(&self, value: &Value) -> Result<Value, CodecError> {
        pack(value).map_err(CodecError::new)
    }

    fn unpacked(&self, value: &Value) -> Result<Value, CodecError> {
        unpack(value).map_err(CodecError::new)
    }
}

/// Every opaque in there, written down. Whatever carries none comes back
/// untouched and without the GIL ever being taken.
pub(crate) fn pack(value: &Value) -> Result<Value, String> {
    if value.travels() {
        return Ok(value.clone());
    }
    Python::with_gil(|py| written(py, value)).map_err(|e| e.to_string())
}

fn written(py: Python<'_>, value: &Value) -> PyResult<Value> {
    Ok(match value {
        Value::Opaque(_) => {
            let obj = value.downcast::<PyObject>().ok_or_else(|| {
                PyValueError::new_err(
                    "this opaque value was not put there by Python, so there is nothing \
                     here that knows how to write it down",
                )
            })?;
            let obj = obj.bind(py);
            let (kind, dump) = codec_for(py, obj)?;
            let written = dump.call1((obj,))?;
            let bytes = written.downcast::<PyBytes>().map_err(|_| {
                PyValueError::new_err(format!(
                    "the codec for `{kind}` has to return bytes, and it returned a `{}`",
                    written
                        .get_type()
                        .name()
                        .map(|n| n.to_string())
                        .unwrap_or_default()
                ))
            })?;
            Value::map(vec![
                (PACKED.to_string(), Value::text(kind)),
                (
                    BYTES.to_string(),
                    Value::Bytes(Arc::new(bytes.as_bytes().to_vec())),
                ),
            ])
        }
        Value::Map(pairs) => Value::map(
            pairs
                .iter()
                .map(|(key, value)| Ok((key.clone(), written(py, value)?)))
                .collect::<PyResult<Vec<_>>>()?,
        ),
        Value::List(items) => Value::list(
            items
                .iter()
                .map(|item| written(py, item))
                .collect::<PyResult<Vec<_>>>()?,
        ),
        other => other.clone(),
    })
}

/// The codec for this object: the first registered type it is an instance of.
fn codec_for<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
) -> PyResult<(String, Bound<'py, PyAny>)> {
    if let Some(found) = matching(py, obj)? {
        return Ok(found);
    }
    // The same second chance as reading back, asked the other way round: there
    // no object had arrived and the `kind` was the only name; here the object is
    // in hand and its type is the name. That the two meet is not luck — a kind
    // is named after the type, which is what makes a record readable by hand
    // years later.
    for name in type_names(obj)? {
        summon(py, &name);
    }
    if let Some(found) = matching(py, obj)? {
        return Ok(found);
    }
    Err(PyValueError::new_err(format!(
        "a `{}` cannot leave this process: nothing says how to write one down. \
         Register it with `codec(\"a name\", {0}, dump=..., load=...)`, which is \
         what `soma_next.torch` does for a tensor on being imported",
        obj.get_type().name()?
    )))
}

/// And the way back.
pub(crate) fn unpack(value: &Value) -> Result<Value, String> {
    if !packed_inside(value) {
        return Ok(value.clone());
    }
    Python::with_gil(|py| read(py, value)).map_err(|e| e.to_string())
}

fn read(py: Python<'_>, value: &Value) -> PyResult<Value> {
    if let Some((kind, bytes)) = as_packed(value) {
        let (_, load) = named(py, kind)?;
        let obj = load.call1((PyBytes::new(py, bytes),))?;
        return Ok(Value::opaque(obj.unbind()));
    }
    Ok(match value {
        Value::Map(pairs) => Value::map(
            pairs
                .iter()
                .map(|(key, value)| Ok((key.clone(), read(py, value)?)))
                .collect::<PyResult<Vec<_>>>()?,
        ),
        Value::List(items) => Value::list(
            items
                .iter()
                .map(|item| read(py, item))
                .collect::<PyResult<Vec<_>>>()?,
        ),
        other => other.clone(),
    })
}

/// The codec registered under that name.
fn named<'py>(py: Python<'py>, kind: &str) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let entry = match codecs(py).get_item(kind)? {
        Some(entry) => Some(entry),
        // Nothing registers it **yet**. What wrote these bytes may be a codec
        // this library ships and this process never had a reason to import —
        // a worker being the case, since it starts empty. Asked here and not on
        // standing up, because here it is known to be needed: something written
        // by it has just arrived.
        None => {
            summon(py, kind);
            codecs(py).get_item(kind)?
        }
    };
    let entry = entry.ok_or_else(|| {
        PyValueError::new_err(format!(
            "what is kept there was written by the codec for `{kind}`, and nothing \
             registers one now: importing whatever registered it is what is missing"
        ))
    })?;
    let entry = entry.downcast::<PyTuple>()?;
    Ok((entry.get_item(0)?, entry.get_item(2)?))
}

/// The first registered type this object is an instance of, if any is.
fn matching<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
) -> PyResult<Option<(String, Bound<'py, PyAny>)>> {
    for (kind, entry) in codecs(py).iter() {
        let entry = entry.downcast::<PyTuple>()?;
        if obj.is_instance(&entry.get_item(0)?)? {
            return Ok(Some((kind.extract()?, entry.get_item(1)?)));
        }
    }
    Ok(None)
}

/// What this object's type is called, and every type it inherits from, the way
/// a codec would have been named after it: `torch.Tensor`.
///
/// The whole line and not just the type, so that an `nn.Parameter` finds the
/// codec registered for a tensor rather than a codec of its own that nobody
/// wrote.
fn type_names(obj: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    let mut names = Vec::new();
    for class in obj.get_type().mro().iter() {
        let module = class
            .getattr("__module__")
            .and_then(|m| m.extract::<String>());
        let name = class
            .getattr("__qualname__")
            .and_then(|n| n.extract::<String>());
        if let (Ok(module), Ok(name)) = (module, name) {
            names.push(format!("{module}.{name}"));
        }
    }
    Ok(names)
}

/// Imports whoever registers this kind, if it is one of this library's.
///
/// Says nothing when it cannot: the caller is about to fail with the name of
/// what is missing, and that message is the better one.
fn summon(py: Python<'_>, kind: &str) {
    let _ = py
        .import("soma_next._codecs")
        .and_then(|module| module.call_method1("summon", (kind,)));
}

/// This value, if it is a written-down opaque and not a map somebody meant.
fn as_packed(value: &Value) -> Option<(&str, &[u8])> {
    let Value::Map(pairs) = value else {
        return None;
    };
    let kind = pairs.iter().find(|(key, _)| key == PACKED)?;
    let bytes = pairs.iter().find(|(key, _)| key == BYTES)?;
    match (&kind.1, &bytes.1) {
        (Value::Text(kind), Value::Bytes(bytes)) => Some((kind, bytes)),
        _ => None,
    }
}

/// Whether there is anything to read back in there at all, which is asked
/// before taking the GIL for nothing.
fn packed_inside(value: &Value) -> bool {
    if as_packed(value).is_some() {
        return true;
    }
    match value {
        Value::Map(pairs) => pairs.iter().any(|(_, value)| packed_inside(value)),
        Value::List(items) => items.iter().any(packed_inside),
        _ => false,
    }
}
