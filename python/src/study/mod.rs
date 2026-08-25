//! The seam for what is above one training run.
//!
//! It translates and it does not decide: where to look next, how the samples are
//! cut, and when a trial is not worth another epoch — all of it is in
//! `somatize_study`, which has no dependencies and does not know Python exists.
//!
//! What arrives here is counts and lists of small numbers, and what leaves is
//! pairs of indices, a configuration, or a reason. That is the shape that lets
//! the loop stay in Python — where torch is — without a single tensor crossing,
//! and without anything calling back the other way.

mod partition;
mod pruner;
mod sampler;
mod space;

pub use partition::PyPartition;
pub use pruner::PyPruner;
pub use sampler::PySampler;
pub use space::{PyPoint, PySpace};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use somatize_study::Goal;

/// Which way is better, from the word Python wrote.
fn read(goal: &str) -> PyResult<Goal> {
    goal.parse().map_err(to_py_err)
}

/// The text form **is** the identity of every one of these families — it is what
/// a cache key and a record are made of — so it is what a hash is taken over.
/// Deriving over the fields would need `Hash` on an `f64`, which does not exist
/// for the same reason `NaN != NaN`.
fn hashed(what: &impl std::fmt::Display) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    what.to_string().hash(&mut hasher);
    hasher.finish()
}

/// Whatever cannot be honoured, said as the exception a Python user expects.
/// The message is the library's: it already says which call supplies what is
/// missing.
fn to_py_err(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}
