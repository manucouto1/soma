//! The seam with the rendezvous: finding out where a host is, from Python.
//!
//! Not one domain decision here, as in the rest of this crate — one adapter and
//! a calling convention. What a host resolves to, which hosts turn out to be
//! the same place, and when a wire is opened are all decided in
//! `somatize-fabric-broker`; this hands Python a door to them.
//!
//! # Why Python is given a token and not a path
//!
//! Deciding what to pack is Python's — it is the half that knows what a
//! `cloudpickle` is. But it must not decide *who shares a catalog*: a worker
//! has one, and two names for one process provisioned separately keeps only the
//! second half, taking every activation over there with it.
//!
//! So [`PyBroker::wire_token`] answers with opaque bytes that are equal exactly
//! when two hosts are one wire. Python groups by equality and never learns what
//! a path is, nor when two of them count as one.

use crate::codec::Codecs;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use somatize_fabric_broker::{Embedded, Endpoint, Host, Path, Reaching, Session};
use somatize_fabric_wire::Artifact;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

/// Where the hosts of a graph are, and how to reach them.
#[pyclass(name = "Broker", module = "somatize._somatize", frozen, subclass, dict)]
pub struct PyBroker {
    session: Arc<Session>,
    /// One handle per host, made the first time it is asked for and kept.
    ///
    /// Kept because two different callers need **the same one**: whoever
    /// provisions stages an artifact on it before the run, and the executor
    /// dispatches through it during. Two handles for one host would stage the
    /// catalog on the one nobody uses.
    handles: Mutex<BTreeMap<Host, Arc<Reaching>>>,
    /// Only for the `repr`.
    listed: Vec<String>,
}

#[pymethods]
impl PyBroker {
    /// A broker inside this process, knowing where these hosts are.
    ///
    /// The values are what the wire needs to get there: a `"host:port"` string
    /// for a worker that is already standing, or an `argv` list for one to be
    /// started as a child.
    ///
    /// Nothing is connected to here. What this costs is a thread and a map.
    #[new]
    #[pyo3(signature = (listing))]
    fn new(listing: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut listed = Vec::new();
        let mut hosts: Vec<(Host, Path)> = Vec::new();
        for (host, target) in listing.iter() {
            let host = Host::new(host.extract::<String>()?);
            let endpoint = endpoint_of(&host, &target)?;
            listed.push(format!("{host}={endpoint}"));
            hosts.push((host, Path::Direct { endpoint }));
        }
        listed.sort();
        Ok(Self {
            session: Arc::new(
                Session::with(Arc::new(Embedded::open(hosts))).packing(Arc::new(Codecs)),
            ),
            handles: Mutex::new(BTreeMap::new()),
            listed,
        })
    }

    /// Bytes that are equal for two hosts that share a wire, so whoever decides
    /// what to pack can group by them.
    ///
    /// This is where the rendezvous happens: **asking is eager**, because what
    /// gets packed for a host depends on which hosts are the same place, and
    /// that has to be settled before the first node runs. Connecting is not,
    /// and is not done here.
    fn wire_token<'py>(&self, py: Python<'py>, host: &str) -> PyResult<Bound<'py, PyBytes>> {
        let token = self
            .session
            .wire_token(&Host::new(host))
            .map_err(|why| PyRuntimeError::new_err(why.message().to_string()))?;
        Ok(PyBytes::new(py, &token))
    }

    /// Tells the host's wire what to provision the far side with, before the
    /// first job. Staged: nothing is sent until somebody dispatches, and even
    /// then only if the far side asks for it.
    fn provision(
        &self,
        host: &str,
        kind: String,
        id: String,
        blob: &[u8],
        runtime: &str,
    ) -> PyResult<()> {
        // `blob` is borrowed and copied once, on purpose: taken as a `Vec<u8>`,
        // PyO3 reads it out of the `bytes` one byte at a time — 10 MB/s
        // measured, which on a 4 MB artifact is 400 ms, three times a step.
        self.reaching(&Host::new(host))
            .offering(Artifact::new(kind, id, blob.to_vec()), runtime)
            .map_err(|why| PyValueError::new_err(why.message().to_string()))
    }

    fn __repr__(&self) -> String {
        format!("Broker(embedded: {})", self.listed.join(", "))
    }
}

impl PyBroker {
    /// The handle for this host, made once and kept.
    pub fn reaching(&self, host: &Host) -> Arc<Reaching> {
        let mut handles = match self.handles.lock() {
            Ok(handles) => handles,
            Err(poisoned) => poisoned.into_inner(),
        };
        Arc::clone(
            handles.entry(host.clone()).or_insert_with(|| {
                Arc::new(Reaching::new(Arc::clone(&self.session), host.clone()))
            }),
        )
    }
}

/// A `"host:port"` or an `argv`, the same two shapes a `Worker` was opened with.
fn endpoint_of(host: &Host, target: &Bound<'_, PyAny>) -> PyResult<Endpoint> {
    if let Ok(addr) = target.extract::<String>() {
        return Ok(Endpoint::Address(addr));
    }
    match target.extract::<Vec<String>>() {
        Ok(argv) if !argv.is_empty() => Ok(Endpoint::Command(argv)),
        Ok(_) => Err(PyValueError::new_err(format!(
            "`{host}` is listed as a command with no program in it"
        ))),
        Err(_) => Err(PyValueError::new_err(format!(
            "a broker lists a host as a `\"host:port\"` address or as an `argv` list; \
             for `{host}` a `{}` arrived",
            target.get_type().name()?
        ))),
    }
}
