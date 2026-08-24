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
use soma_next_core::Packing;
use std::time::Duration;

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use soma_fabric_wire::{Artifact, Provision, ProvisionError, Provisioned, Serving, Worker};
use soma_next_core::Catalog;
use soma_next_store::{Cache, Store};
use std::process::Command;
use std::sync::Arc;

/// A process that gets sent slices of plan. `Arc` because the executor borrows
/// it while it runs and Python can drop its reference at any moment.
#[pyclass(
    name = "Worker",
    module = "soma_next._soma_next",
    frozen,
    subclass,
    dict
)]
pub struct PyWorker {
    inner: Arc<Worker>,
    /// Only for the `repr`: a worker does not say where it came from.
    where_: String,
}

/// What a client can carry to provision an empty worker: all three or none.
fn artifact(
    kind: Option<String>,
    id: Option<String>,
    blob: Option<&[u8]>,
) -> PyResult<Option<Artifact>> {
    match (kind, id, blob) {
        (None, None, None) => Ok(None),
        (Some(kind), Some(id), Some(blob)) => Ok(Some(Artifact::new(kind, id, blob.to_vec()))),
        _ => Err(PyValueError::new_err(
            "provisioning a worker needs `kind`, `id` and `blob`: all three or none",
        )),
    }
}

#[pymethods]
impl PyWorker {
    /// Opens the conversation with a worker.
    ///
    /// A **string** — `"node3:7000"` — connects to one already standing, which
    /// is the form that satisfies the use case; a **list** — `[sys.executable,
    /// "-m", "soma_next.worker"]` — starts a child, for testing.
    ///
    /// `kind`, `id` and `blob` are what to build its catalog from; normally
    /// `Graph.forward` sets them later. `runtime` is how this client identifies
    /// itself, and what the far side's `Provision` can reject.
    #[new]
    #[pyo3(signature = (target, *, kind = None, id = None, blob = None, runtime = "python"))]
    fn new(
        target: &Bound<'_, PyAny>,
        kind: Option<String>,
        id: Option<String>,
        blob: Option<&[u8]>,
        runtime: &str,
    ) -> PyResult<Self> {
        let carries = artifact(kind, id, blob)?;
        let (where_, inner) = match target.extract::<String>() {
            Ok(addr) => {
                let opened = Worker::connect(&addr).map_err(|e| {
                    PyRuntimeError::new_err(format!("nobody is listening on `{addr}`: {e}"))
                })?;
                (addr, opened)
            }
            Err(_) => {
                let argv: Vec<String> = target.extract().map_err(|_| {
                    PyValueError::new_err(
                        "a worker is opened with a `\"host:port\"` address or with an \
                         `argv` list",
                    )
                })?;
                let (program, rest) = argv
                    .split_first()
                    .ok_or_else(|| PyValueError::new_err("a worker needs at least a program"))?;
                let mut command = Command::new(program);
                command.args(rest);
                let started = Worker::spawn(command).map_err(|e| {
                    PyRuntimeError::new_err(format!("the worker could not be started: {e}"))
                })?;
                (argv.join(" "), started)
            }
        };

        // Always, and not on request: from Python an opaque carries a Python
        // object, and one that nobody registered a codec for is refused with its
        // type in front of you either way. There is nothing here to turn off.
        let inner = inner.packing(Arc::new(Codecs));
        Ok(Self {
            where_,
            inner: Arc::new(match carries {
                Some(a) => inner.carrying(a, runtime),
                None => inner,
            }),
        })
    }

    /// Tells it what to provision itself with, before the first job. The graph
    /// calls it once it knows which nodes go here.
    /// `blob` is borrowed and copied once, on purpose: taken as a `Vec<u8>`,
    /// PyO3 reads it out of the `bytes` **one byte at a time** — measured at
    /// 10 MB/s, which on a 4 MB artifact is 400 ms, three times a step, and was
    /// the whole cost of training across a wire.
    fn provision(&self, kind: String, id: String, blob: &[u8], runtime: &str) -> PyResult<()> {
        self.inner
            .offering(Artifact::new(kind, id, blob.to_vec()), runtime)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!("Worker({})", self.where_)
    }
}

impl PyWorker {
    /// The real worker, to lend to the executor.
    pub fn transport(&self) -> Arc<Worker> {
        Arc::clone(&self.inner)
    }
}

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
