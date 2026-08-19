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

use crate::node::{PyDriver, PyNode};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use soma_next_core::Catalog;
use soma_next_transport::{Artifact, Provision, ProvisionError, Provisioned, Serving, Worker};
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
    blob: Option<Vec<u8>>,
) -> PyResult<Option<Artifact>> {
    match (kind, id, blob) {
        (None, None, None) => Ok(None),
        (Some(kind), Some(id), Some(blob)) => Ok(Some(Artifact::new(kind, id, blob))),
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
        blob: Option<Vec<u8>>,
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
    fn provision(&self, kind: String, id: String, blob: Vec<u8>, runtime: &str) -> PyResult<()> {
        self.inner
            .offering(Artifact::new(kind, id, blob), runtime)
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
/// - `provide(kind, blob)` → `(nodes, driver)`: a `dict` of `id → node` and
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

            let (nodes, driver): (PyObject, Option<PyObject>) =
                built.extract(py).map_err(|_| {
                    ProvisionError::Broken("provide() must return (nodes, driver)".into())
                })?;

            let dict = nodes
                .bind(py)
                .downcast::<PyDict>()
                .map_err(|_| {
                    ProvisionError::Broken("the nodes have to be a dict of id → node".into())
                })?
                .clone();

            let mut catalog = Catalog::new();
            for (id, obj) in dict.iter() {
                let id: String = id.extract().map_err(|_| {
                    ProvisionError::Broken("the catalog's keys have to be text".into())
                })?;
                let node = PyNode::new(&obj).map_err(|e| ProvisionError::Broken(e.to_string()))?;
                catalog.insert(id, Arc::new(node));
            }

            let provisioned = Provisioned::new(catalog);
            Ok(match driver {
                None => provisioned,
                Some(obj) => {
                    let driver = PyDriver::new(obj.bind(py))
                        .map_err(|e| ProvisionError::Broken(e.to_string()))?;
                    provisioned.served_by(Arc::new(driver))
                }
            })
        })
    }
}

/// The nodes as a catalog the engine can execute.
fn catalog_of(nodes: &Bound<'_, PyDict>) -> PyResult<Catalog> {
    let mut catalog = Catalog::new();
    for (id, obj) in nodes.iter() {
        catalog.insert(id.extract::<String>()?, Arc::new(PyNode::new(&obj)?));
    }
    Ok(catalog)
}

/// Serves slices with the catalog you pass it, `{id: node}`, until the client
/// closes. `driver` is what serves whatever the steps here ask for.
#[pyfunction]
#[pyo3(signature = (nodes, driver = None))]
pub fn serve(
    py: Python<'_>,
    nodes: &Bound<'_, PyDict>,
    driver: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let catalog = catalog_of(nodes)?;
    let driver = driver.map(PyDriver::new).transpose()?;
    // While this blocks on a read, a wave's threads need the interpreter.
    py.allow_threads(|| serving(&catalog, driver.as_ref()).over_stdin())
        .map_err(|e| PyRuntimeError::new_err(format!("the worker was cut off: {e}")))
}

/// A worker with its own catalog, and its own driver if it was given one.
fn serving<'a>(catalog: &'a Catalog, driver: Option<&'a PyDriver>) -> Serving<'a> {
    let serving = Serving::own(catalog);
    match driver {
        Some(driver) => serving.driver(driver),
        None => serving,
    }
}

/// The same for a worker that is sent what to execute.
fn serving_provisioned<'a>(
    provision: &'a PyProvision,
    driver: Option<&'a PyDriver>,
) -> Serving<'a> {
    let serving = Serving::provisioned(provision);
    match driver {
        Some(driver) => serving.driver(driver),
        None => serving,
    }
}

/// Serves slices with what the client sends it: the generic worker. It starts
/// empty and `provision` turns whatever arrives into nodes and a driver.
///
/// `driver` here is only the fallback for clients that pack none: the one that
/// arrives in the artifact wins.
#[pyfunction]
#[pyo3(signature = (provision, driver = None))]
pub fn serve_provisioned(
    py: Python<'_>,
    provision: &Bound<'_, PyAny>,
    driver: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let provision = PyProvision::new(provision)?;
    let driver = driver.map(PyDriver::new).transpose()?;
    py.allow_threads(|| serving_provisioned(&provision, driver.as_ref()).over_stdin())
        .map_err(|e| PyRuntimeError::new_err(format!("the worker was cut off: {e}")))
}

/// Stands on `addr` and serves whoever connects; it does not return. `opened`
/// is called once with the real address, so port `0` can be asked for.
#[pyfunction]
#[pyo3(signature = (addr, provision, opened = None, driver = None))]
pub fn listen_provisioned(
    py: Python<'_>,
    addr: &str,
    provision: &Bound<'_, PyAny>,
    opened: Option<PyObject>,
    driver: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let provision = PyProvision::new(provision)?;
    let driver = driver.map(PyDriver::new).transpose()?;
    py.allow_threads(|| {
        serving_provisioned(&provision, driver.as_ref()).listen_at(addr, |where_| {
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
#[pyo3(signature = (addr, nodes, opened = None, driver = None))]
pub fn listen(
    py: Python<'_>,
    addr: &str,
    nodes: &Bound<'_, PyDict>,
    opened: Option<PyObject>,
    driver: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let catalog = catalog_of(nodes)?;
    let driver = driver.map(PyDriver::new).transpose()?;
    py.allow_threads(|| {
        serving(&catalog, driver.as_ref()).listen_at(addr, |where_| {
            if let Some(notify) = opened {
                Python::with_gil(|py| {
                    let _ = notify.call1(py, (where_.to_string(),));
                });
            }
        })
    })
    .map_err(|e| PyRuntimeError::new_err(format!("the worker could not listen on `{addr}`: {e}")))
}
