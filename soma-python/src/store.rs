//! The store, from Python. A directory two machines can both see.
//!
//! Until now it was reachable only as a **string**: `forward(store=...)` and
//! `Trainer(store=...)`, a place the engine kept things in and nobody else could
//! open. That is enough while the only thing being kept is what the engine
//! decided to keep. It stops being enough the moment a training run has
//! something of its own to write down — its weights — and another machine has to
//! read it.
//!
//! # Two ways of asking, and they are the same store
//!
//! | asked in | what it deals in | who it is for |
//! |---|---|---|
//! | [`put`](PyStore::put) / [`get`](PyStore::get) / [`bind`](PyStore::bind) / [`resolve`](PyStore::resolve) | **bytes** | the store exactly as Rust sees it |
//! | [`keep`](PyStore::keep) / [`recall`](PyStore::recall) | **values**, tensors included | whoever has something to keep and not bytes |
//!
//! The first four are `somatize_store::Store`, one for one, and they invent no
//! vocabulary. The last two are `Keeper`'s, which is also not new, and they are
//! there because the thing this was opened for — an export — is a map of tensors
//! and not bytes. Without them everybody writes their own `torch.save`, and the
//! one that gets `weights_only` wrong writes a way into a shared directory.
//!
//! # Names and content, which stay two questions
//!
//! Bytes are known by **what they are** and names point at them. The same
//! weights written on two machines are one blob, and a name is how a round of
//! training is found again. It is git's model and it is the store's, and putting
//! it in front of Python changes neither.

use crate::codec::{pack, unpack};
use crate::value;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
// The store's record and PyO3's smart pointer are both called `Bound`, and one
// of them has to give: two same-named types in scope is the same trap the
// project's rules warn about for traits, one level down.
use somatize_store::Bound as Record;
use somatize_store::{
    Bucket, Credentials, Digest, Local, Meta, Store, UrlStyle, bytes_of, value_of,
};
use std::sync::Arc;

/// Something that keeps bytes by their content, and names that point at them.
///
/// A `dyn Store` and not a `Local`, since there are two: a directory and a
/// bucket. Which one this is does not reach the rest of Python — `take`,
/// `report` and `gather` take a store and never ask what kind it is, and that is
/// what lets a study run over a shared folder here and over S3 there without a
/// line of it changing.
#[pyclass(name = "Store", module = "somatize._somatize", frozen)]
pub struct PyStore {
    /// An `Arc` and not a `Box` because a store outlives the call it was passed
    /// to: a `Recorder` holds on to one across every `forward` of a training
    /// run, and the same store is still the caller's to read from.
    inner: Arc<dyn Store>,
    /// Only for the `repr`: a store does not say where it is.
    where_: String,
}

impl PyStore {
    /// The store itself, for whoever in this crate needs to hold on to one.
    pub fn shared(&self) -> Arc<dyn Store> {
        self.inner.clone()
    }
}

#[pymethods]
impl PyStore {
    /// Opens the directory, making it if it is not there.
    #[new]
    fn new(where_: &str) -> PyResult<Self> {
        Ok(Self {
            inner: Arc::new(Local::at(where_).map_err(failed)?),
            where_: where_.to_string(),
        })
    }

    /// The same store, on a bucket: S3, MinIO, R2 — for a cluster where there is
    /// no directory two machines can both see.
    ///
    /// `hosted=True` puts the bucket in the host name, which is what AWS wants
    /// for anything made recently; the default puts it in the path, which is
    /// what MinIO wants. Without `key`/`secret` it reads `AWS_ACCESS_KEY_ID` and
    /// `AWS_SECRET_ACCESS_KEY`, which is where everything else looks.
    ///
    /// **It talks to the endpoint before returning**: an endpoint that takes a
    /// conditional write and writes anyway would hand every trial to every
    /// machine and never say so, so it is tried rather than assumed.
    #[staticmethod]
    #[pyo3(signature = (endpoint, bucket, *, region = "us-east-1", key = None, secret = None, hosted = false))]
    fn on_bucket(
        endpoint: &str,
        bucket: &str,
        region: &str,
        key: Option<String>,
        secret: Option<String>,
        hosted: bool,
    ) -> PyResult<Self> {
        let named = |whose: &str, given: Option<String>| -> PyResult<String> {
            given.map(Ok).unwrap_or_else(|| {
                std::env::var(whose).map_err(|_| {
                    PyValueError::new_err(format!(
                        "no `{whose}` here and none was given: a bucket needs credentials"
                    ))
                })
            })
        };
        let credentials = Credentials::new(
            named("AWS_ACCESS_KEY_ID", key)?,
            named("AWS_SECRET_ACCESS_KEY", secret)?,
        );
        let style = if hosted {
            UrlStyle::VirtualHost
        } else {
            UrlStyle::Path
        };
        Ok(Self {
            inner: Arc::new(
                Bucket::at(endpoint, bucket, region, style, credentials).map_err(failed)?,
            ),
            where_: format!("{endpoint}/{bucket}"),
        })
    }

    /// Saves these bytes and gives back the digest that names them.
    ///
    /// Saving the same bytes twice is saving them once: that is what content
    /// addressing is for, and it is why a round of federated training that
    /// changed nothing costs nothing.
    fn put(&self, bytes: &[u8]) -> PyResult<String> {
        Ok(self.inner.put(bytes).map_err(failed)?.to_string())
    }

    /// The bytes, or `None` if this store does not have them.
    fn get(&self, py: Python<'_>, digest: &str) -> PyResult<Option<PyObject>> {
        let found = self.inner.get(&as_digest(digest)).map_err(failed)?;
        Ok(found.map(|bytes| PyBytes::new(py, &bytes).into()))
    }

    /// Points a name at some bytes, with whatever you want to remember beside
    /// it. Binding the same name again replaces it.
    #[pyo3(signature = (name, digest, meta = None))]
    fn bind(&self, name: &str, digest: &str, meta: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        self.inner
            .bind(name, &as_digest(digest), as_meta(meta)?)
            .map_err(failed)
    }

    /// Points a name at some bytes **only if nobody has**, and says whether it
    /// did. This is how work gets handed out.
    ///
    /// Not `resolve` and then `bind`: between the two somebody else does the
    /// same, and two machines train the same round while nobody trains the next
    /// one. Whoever is told `True` does the work; whoever is told `False` goes
    /// and asks for the next thing::
    ///
    ///     me = store.put(f"{socket.gethostname()}/{os.getpid()}".encode())
    ///     if store.claim(f"round/{r}/client/{k}", me):
    ///         ...
    #[pyo3(signature = (name, digest, meta = None))]
    fn claim(&self, name: &str, digest: &str, meta: Option<&Bound<'_, PyDict>>) -> PyResult<bool> {
        self.inner
            .claim(name, &as_digest(digest), as_meta(meta)?)
            .map_err(failed)
    }

    /// What that name points at, or `None`.
    fn resolve(&self, name: &str) -> PyResult<Option<PyBound>> {
        Ok(self.inner.resolve(name).map_err(failed)?.map(PyBound::of))
    }

    /// Everything bound here. A scan, and that is the point: the records are the
    /// truth and an index over them is something you can throw away.
    fn bound(&self) -> PyResult<Vec<PyBound>> {
        Ok(self
            .inner
            .bound()
            .map_err(failed)?
            .into_iter()
            .map(PyBound::of)
            .collect())
    }

    /// Keeps a value under a name — tensors and all, by the codecs.
    ///
    /// The two lines that make an export cross a machine::
    ///
    ///     store.keep("round/3", trainer.export())
    ///     trainer.load(store.recall("round/3"))
    ///
    /// What reaches the directory is bytes, and the directory never learns that
    /// any of it was a tensor. Something nobody registered a codec for is
    /// refused with its type in front of you, which is the same frontier as
    /// everywhere else.
    #[pyo3(signature = (name, what, meta = None))]
    fn keep(
        &self,
        name: &str,
        what: &Bound<'_, PyAny>,
        meta: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<String> {
        let written = pack(&value::to_be_kept(what)?).map_err(PyValueError::new_err)?;
        let bytes =
            bytes_of(&written).map_err(|e| PyValueError::new_err(e.message().to_string()))?;
        let digest = self.inner.put(&bytes).map_err(failed)?;
        self.inner
            .bind(name, &digest, as_meta(meta)?)
            .map_err(failed)?;
        Ok(digest.to_string())
    }

    /// What is kept under that name, alive again, or `None` if nothing is.
    fn recall(&self, py: Python<'_>, name: &str) -> PyResult<Option<PyObject>> {
        let Some(bound) = self.inner.resolve(name).map_err(failed)? else {
            return Ok(None);
        };
        let Some(bytes) = self.inner.get(&bound.digest).map_err(failed)? else {
            return Err(PyRuntimeError::new_err(format!(
                "`{name}` points at `{}` and this store does not have it: the record \
                 and the bytes are two things, and one of them is missing",
                bound.digest
            )));
        };
        let kept = value_of(&bytes).map_err(|e| PyValueError::new_err(e.message().to_string()))?;
        let alive = unpack(&kept).map_err(PyValueError::new_err)?;
        value::to_py(py, &alive).map(Some)
    }

    fn __repr__(&self) -> String {
        format!("Store({})", self.where_)
    }
}

/// A name, and what it points at.
#[pyclass(name = "Bound", module = "somatize._somatize", frozen, get_all)]
pub struct PyBound {
    /// The name somebody chose.
    name: String,
    /// The digest of the bytes it points at.
    digest: String,
    /// What was said beside it, in the order it was said.
    meta: Vec<(String, String)>,
    /// When it was bound, in seconds since the epoch.
    when: u64,
}

impl PyBound {
    fn of(bound: Record) -> Self {
        Self {
            name: bound.name,
            digest: bound.digest.to_string(),
            meta: bound.meta,
            when: bound.when,
        }
    }
}

#[pymethods]
impl PyBound {
    fn __repr__(&self) -> String {
        format!("Bound({} -> {})", self.name, self.digest)
    }
}

/// A digest as it is written down, and **not** checked: the store takes any
/// string on purpose, because what names bytes is not always a hash — an
/// artifact's id is whatever its producer wanted. A typo comes back as "there is
/// nothing there", which is the truth.
fn as_digest(digest: &str) -> Digest {
    Digest::parse(digest)
}

/// The metadata, whose vocabulary belongs to whoever is calling.
fn as_meta(meta: Option<&Bound<'_, PyDict>>) -> PyResult<Meta> {
    let Some(meta) = meta else {
        return Ok(Meta::new());
    };
    meta.iter()
        .map(|(key, value)| Ok((key.extract::<String>()?, value.extract::<String>()?)))
        .collect::<PyResult<Vec<_>>>()
        .map_err(|_: PyErr| {
            PyValueError::new_err("what is remembered beside a name is text to text")
        })
}

fn failed(e: somatize_store::StoreError) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// The store a call was pointed at: a `Store`, or a path to a directory.
///
/// **Both, and not one.** A path is what somebody standing a worker up on a
/// command line has, and a `Store` is what somebody who opened one by hand has —
/// and since [`PyStore::on_bucket`] exists, insisting on a path is insisting
/// that a cache lives on a disk. Two ways of saying the same thing, and only one
/// of them could say "a bucket".
pub fn opened(store: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Arc<dyn Store>>> {
    let Some(store) = store else {
        return Ok(None);
    };
    if let Ok(open) = store.downcast::<PyStore>() {
        return Ok(Some(open.get().shared()));
    }
    let where_: String = store.extract().map_err(|_| {
        PyValueError::new_err(format!(
            "`store` takes a directory or a Store, and a `{}` arrived",
            store
                .get_type()
                .name()
                .map(|name| name.to_string())
                .unwrap_or_default()
        ))
    })?;
    Ok(Some(Arc::new(Local::at(&where_).map_err(failed)?)))
}
