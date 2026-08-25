//! What happened, on its way to Python.
//!
//! Two things, and they are the two ends of the same seam: a [`Recorder`]
//! exposed so a training run can keep what happened, and an adapter that hands
//! every fact to whatever Python callable was given — which is what makes a
//! notebook able to draw a curve **while** it is being drawn.
//!
//! # One shape, in both directions
//!
//! A fact reaches Python as a `dict` with a `fact` key naming it and its fields
//! beside it, all text — which is **exactly what would be written down**, since
//! it is [`Fact::flattened`] and nothing else. So there is one shape to learn,
//! and what you print is what you would find in the store.
//!
//! It goes the other way too: [`PyRecorder`] is callable with that same `dict`,
//! which is how level 2 — a loss, a step, an update — gets into the record
//! without the core ever learning what a loss is. The `Trainer` calls
//! `watching(...)` and does not care which of the two it was handed.
//!
//! # Fanning out is a list, and it is not the core's problem
//!
//! `watching=[recorder, print]` builds one watcher holding two. The core
//! provides a hole and does not manage tenants; here is where the second one
//! goes.

use crate::store::PyStore;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use somatize_core::{Fact, Watcher};
use somatize_store::Recorder;
use std::sync::Arc;

/// Writes down what happened, one record per `forward`.
///
/// ```python
/// r = Recorder(store)                 # or Recorder(store, run="tuesday")
/// g.forward(x, watching=r)
/// store.resolve(f"run/{r.run}/0")     # what that forward did
/// ```
#[pyclass(name = "Recorder", module = "somatize._somatize", frozen)]
pub struct PyRecorder {
    inner: Arc<Recorder>,
}

#[pymethods]
impl PyRecorder {
    /// A recorder over this store. Without a `run` it makes a name up, and
    /// [`run`](Self::run) says which — a `forward` in a notebook has no reason
    /// to invent one and still has to be findable afterwards.
    ///
    /// `summarising` names the kinds of fact that go **into the record** as
    /// `<kind>.<field>` and not only into its blob, which is what makes reading
    /// them back cost one scan instead of one fetch per `forward`::
    ///
    ///     Recorder(store, run="tuesday", summarising=["loss"])
    ///
    /// That is what a training run wants, and it is said here rather than
    /// guessed there: `loss` is this side's word, and the store does not learn
    /// it.
    #[new]
    #[pyo3(signature = (store, *, run = None, summarising = None))]
    fn new(store: &PyStore, run: Option<String>, summarising: Option<Vec<String>>) -> Self {
        let store = store.shared();
        let recorder = match run {
            Some(run) => Recorder::named(store, run),
            None => Recorder::over(store),
        };
        Self {
            inner: Arc::new(match summarising {
                Some(kinds) => recorder.summarising(kinds),
                None => recorder,
            }),
        }
    }

    /// What this run is called, which is the first half of every name it writes.
    #[getter]
    fn run(&self) -> &str {
        self.inner.run()
    }

    /// One fact from a vocabulary that is not the engine's.
    ///
    /// The same `dict` the engine's facts arrive as: a `fact` key naming it,
    /// and text beside it. That is the whole of how level 2 meets level 1 —
    /// **in the record**, and not in a type either of them shares.
    fn __call__(&self, fact: &Bound<'_, PyDict>) -> PyResult<()> {
        let kind = match fact.get_item("fact")? {
            Some(kind) => kind.extract::<String>()?,
            None => {
                return Err(PyValueError::new_err(
                    "a fact says what it is: give it a `fact` key naming it",
                ));
            }
        };
        let mut fields = Vec::new();
        for (name, what) in fact.iter() {
            let name: String = name.extract()?;
            if name == "fact" {
                continue;
            }
            // Text to text, like everything else in a record: the vocabulary is
            // the caller's, and what it is written as is not.
            fields.push((name, what.str()?.extract::<String>()?));
        }
        self.inner.said(&kind, fields);
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!("Recorder({})", self.inner.run())
    }
}

/// The recorder as the engine sees it.
///
/// A wrapper and not the `Arc` itself, because both of them belong to other
/// crates and a trait cannot be implemented for a foreign type from here.
struct Recording(Arc<Recorder>);

impl Watcher for Recording {
    fn saw(&self, fact: &Fact) {
        self.0.saw(fact);
    }
}

/// Hands every fact to a Python callable.
///
/// It takes the GIL to do it, from whatever thread the engine is on — including
/// a wave's, which is why the call is short and does nothing but build a `dict`.
/// Whatever the callable then does with it is Python's business, and if it wants
/// to be asynchronous about it that is where the queue goes.
struct Calling(Py<PyAny>);

impl Watcher for Calling {
    fn saw(&self, fact: &Fact) {
        Python::with_gil(|py| {
            if let Err(e) = self.0.call1(py, (as_dict(py, fact),)) {
                // A watcher cannot fail by contract, and stopping a training run
                // because a print raised would be the observability layer
                // breaking the thing it observes. It is said once, loudly, where
                // a worker's own complaints go.
                e.print(py);
            }
        });
    }
}

/// Several of them, in the order they were given.
struct Several(Vec<Box<dyn Watcher>>);

impl Watcher for Several {
    fn saw(&self, fact: &Fact) {
        for one in &self.0 {
            one.saw(fact);
        }
    }
}

/// One fact as the `dict` Python sees, which is the record's own shape.
fn as_dict<'py>(py: Python<'py>, fact: &Fact) -> Bound<'py, PyDict> {
    let (kind, fields) = fact.flattened();
    let said = PyDict::new(py);
    let _ = said.set_item("fact", kind);
    for (name, what) in fields {
        let _ = said.set_item(name, what);
    }
    said
}

/// Whatever was passed as `watching=`, as something the engine can be told to
/// use: a [`PyRecorder`], any callable, or a list of either.
pub fn watching(given: &Bound<'_, PyAny>) -> PyResult<Box<dyn Watcher>> {
    if let Ok(recorder) = given.downcast::<PyRecorder>() {
        // The typed path, which costs no GIL per fact — and there are as many
        // facts as there are nodes times steps.
        return Ok(Box::new(Recording(recorder.get().inner.clone())));
    }
    if given.is_instance_of::<PyList>() || given.is_instance_of::<PyTuple>() {
        let several = given
            .try_iter()?
            .map(|one| watching(&one?))
            .collect::<PyResult<Vec<_>>>()?;
        return Ok(Box::new(Several(several)));
    }
    if !given.is_callable() {
        return Err(PyValueError::new_err(
            "`watching` takes a Recorder, anything callable, or a list of them; \
             what arrived is none of those",
        ));
    }
    Ok(Box::new(Calling(given.clone().unbind())))
}
