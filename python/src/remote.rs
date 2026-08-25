//! The seam with the transport: sending slices to another process from Python.
//!
//! Not one domain decision here, as in the rest of this crate — two adapters and
//! a calling convention:
//!
//! | from here | to there |
//! |---|---|
//! | [`PyWorker`] | a transport `Worker`, wrapped so Python can hold it |
//! | [`PyProvision`] | a Python object with `accepts` and `catalog`, seen as a `Provision` |
//!
//! `cloudpickle` appears nowhere because it is not Rust's business: the artifact
//! is a pile of bytes neither the core nor the transport looks at, and the one
//! that decides they are a pickle is `soma_next.worker`, in Python. The same
//! boundary as three other places — the core does not know what a node asks for,
//! what an `Opaque` carries, or what a serialized catalog is.

use crate::codec::Codecs;
use crate::node::PyNode;
use somatize_core::Packing;
use std::time::Duration;

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use somatize_core::Catalog;
use somatize_fabric_wire::{Provision, ProvisionError, Provisioned, Serving};
use somatize_store::{Cache, Store};
use std::sync::Arc;

/// A Python object with `accepts` and `provide`, from the Rust side.
///
/// - `accepts(runtime, kind)` → `None` if it accepts, or **how this worker
///   identifies itself** if not.
/// - `provide(kind, blob)` → a `dict` of `id → node`, and
///   whoever serves what they ask for, or `None`. It raises if it cannot.
pub struct PyProvision {
    obj: PyObject,
}

impl PyProvision {
    /// Wraps the object, checking that it knows what it has to know.
    pub fn new(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        for method in ["accepts", "provide"] {
            if !obj.hasattr(method)? {
                return Err(PyValueError::new_err(format!(
                    "a `{}` cannot provision a worker: it is missing {method}()",
                    obj.get_type().name()?
                )));
            }
        }
        Ok(Self {
            obj: obj.clone().unbind(),
        })
    }
}

impl Provision for PyProvision {
    fn accepts(&self, runtime: &str, kind: &str) -> Result<(), ProvisionError> {
        Python::with_gil(|py| {
            let answer = self
                .obj
                .call_method1(py, "accepts", (runtime, kind))
                .map_err(|e| ProvisionError::Broken(e.to_string()))?;
            match answer.extract::<Option<String>>(py) {
                Ok(None) => Ok(()),
                Ok(Some(worker)) => Err(ProvisionError::Incompatible {
                    client: runtime.to_string(),
                    worker,
                }),
                Err(e) => Err(ProvisionError::Broken(format!(
                    "accepts() must return None or a string: {e}"
                ))),
            }
        })
    }

    fn provide(&self, kind: &str, bytes: &[u8]) -> Result<Provisioned, ProvisionError> {
        Python::with_gil(|py| {
            let built = self
                .obj
                .call_method1(py, "provide", (kind, bytes))
                .map_err(|e| ProvisionError::Broken(e.to_string()))?;

            let dict = built.bind(py).downcast::<PyDict>().map_err(|_| {
                ProvisionError::Broken("the nodes have to be a dict of id → node".into())
            })?;
            // The same walk as a worker that brings its own catalog: how the
            // nodes got here is not what tells them apart.
            let catalog = catalog_of(dict).map_err(|e| ProvisionError::Broken(e.to_string()))?;
            Ok(Provisioned::new(catalog))
        })
    }
}

/// The nodes as a catalog the engine can execute. The one walk, for a worker
/// that brings its catalog and for one that is sent it alike.
fn catalog_of(nodes: &Bound<'_, PyDict>) -> PyResult<Catalog> {
    let mut catalog = Catalog::new();
    for (id, obj) in nodes.iter() {
        let named = id.extract::<String>().map_err(|_| {
            PyTypeError::new_err("a catalog is a dict of id → node, and a key is not text")
        })?;
        catalog.insert(named, Arc::new(PyNode::new(&obj)?));
    }
    Ok(catalog)
}

/// Serves slices with the catalog you pass it, `{id: node}`, until the client
/// closes.
#[pyfunction]
pub fn serve(py: Python<'_>, nodes: &Bound<'_, PyDict>) -> PyResult<()> {
    let catalog = catalog_of(nodes)?;
    // While this blocks on a read, a wave's threads need the interpreter.
    py.allow_threads(|| serving(&catalog).over_stdin())
        .map_err(|e| PyRuntimeError::new_err(format!("the worker was cut off: {e}")))
}

/// A worker with its own catalog.
fn serving(catalog: &Catalog) -> Serving<'_> {
    Serving::own(catalog).packing(&CODECS)
}

/// The same for a worker that is sent what to execute.
fn serving_provisioned(provision: &PyProvision) -> Serving<'_> {
    Serving::provisioned(provision).packing(&CODECS)
}

/// The codecs this side reads, which are the process's and not an object's.
///
/// It is installed on every worker stood up from Python and there is no way to
/// ask for one without it: the client always packs, so a worker that did not
/// unpack would be handed maps where its nodes expect what they were sent.
static CODECS: Codecs = Codecs;

/// The same worker, keeping what it is sent **and** what its nodes produce.
///
/// One directory answers the two questions and they stay two: a catalog that is
/// not sent twice, and a node that is not run twice. Neither is on unless a
/// `store` was given.
///
/// **With the codecs in front of the second**, exactly as the client puts them
/// in front of its own: without them a worker keeps what is made of numbers and
/// quietly keeps nothing else, so the same `.cached()` node would hit here and
/// miss there for no reason the user could see. What reaches the store is bytes
/// either way, and the store never learns Python exists.
fn keeping<'a>(
    serving: Serving<'a>,
    kept: Option<&'a Arc<dyn Store>>,
    packing: Option<&'a Packing<'a>>,
    reporting: Option<f64>,
) -> Serving<'a> {
    let serving = match kept {
        Some(kept) => serving.store(&**kept),
        None => serving,
    };
    let serving = match reporting {
        // Seconds, because that is what somebody standing a worker up thinks
        // in. Nothing happens without a store to write to, and `Serving` says
        // so rather than this asking twice.
        Some(every) if every > 0.0 => serving.reporting(Duration::from_secs_f64(every)),
        _ => serving,
    };
    match packing {
        Some(packing) => serving.keeping(packing),
        None => serving,
    }
}

/// Serves slices with what the client sends it: the generic worker. It starts
/// empty and `provision` turns whatever arrives into nodes and a driver.
///
#[pyfunction]
#[pyo3(signature = (provision, store = None, reporting = None))]
pub fn serve_provisioned(
    py: Python<'_>,
    provision: &Bound<'_, PyAny>,
    store: Option<&Bound<'_, PyAny>>,
    reporting: Option<f64>,
) -> PyResult<()> {
    let provision = PyProvision::new(provision)?;
    let kept = crate::store::opened(store)?;
    let cache = kept.as_ref().map(|kept| Cache::over(&**kept));
    let packing = cache.as_ref().map(|cache| Packing::over(cache, &Codecs));
    py.allow_threads(|| {
        keeping(
            serving_provisioned(&provision),
            kept.as_ref(),
            packing.as_ref(),
            reporting,
        )
        .over_stdin()
    })
    .map_err(|e| PyRuntimeError::new_err(format!("the worker was cut off: {e}")))
}

/// Stands on `addr` and serves whoever connects; it does not return. `opened`
/// is called once with the real address, so port `0` can be asked for.
#[pyfunction]
#[pyo3(signature = (addr, provision, opened = None, store = None, reporting = None))]
pub fn listen_provisioned(
    py: Python<'_>,
    addr: &str,
    provision: &Bound<'_, PyAny>,
    opened: Option<PyObject>,
    store: Option<&Bound<'_, PyAny>>,
    reporting: Option<f64>,
) -> PyResult<()> {
    let provision = PyProvision::new(provision)?;
    let kept = crate::store::opened(store)?;
    let cache = kept.as_ref().map(|kept| Cache::over(&**kept));
    let packing = cache.as_ref().map(|cache| Packing::over(cache, &Codecs));
    py.allow_threads(|| {
        keeping(
            serving_provisioned(&provision),
            kept.as_ref(),
            packing.as_ref(),
            reporting,
        )
        .listen_at(addr, |where_| {
            if let Some(notify) = opened {
                Python::with_gil(|py| {
                    let _ = notify.call1(py, (where_.to_string(),));
                });
            }
        })
    })
    .map_err(|e| PyRuntimeError::new_err(format!("the worker could not listen on `{addr}`: {e}")))
}

/// The same, with the catalog you already bring.
#[pyfunction]
#[pyo3(signature = (addr, nodes, opened = None))]
pub fn listen(
    py: Python<'_>,
    addr: &str,
    nodes: &Bound<'_, PyDict>,
    opened: Option<PyObject>,
) -> PyResult<()> {
    let catalog = catalog_of(nodes)?;
    py.allow_threads(|| {
        serving(&catalog).listen_at(addr, |where_| {
            if let Some(notify) = opened {
                Python::with_gil(|py| {
                    let _ = notify.call1(py, (where_.to_string(),));
                });
            }
        })
    })
    .map_err(|e| PyRuntimeError::new_err(format!("the worker could not listen on `{addr}`: {e}")))
}
