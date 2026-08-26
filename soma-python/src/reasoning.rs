//! The reasoning, on its way to Python. A translation and nothing else.
//!
//! It crosses as JSON for the reason the plan does: the reader is `json.loads`,
//! nobody has to install anything to look at a shape, and the derivations —
//! what stands where, what folds, what a scope reaches — stay in the one place
//! that already computes them for the terminal.

use crate::store::PyStore;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use somatize_tree::reasoning::reasoned;

/// A whole reasoning as JSON: its moves, what was said, and what folds.
#[pyfunction]
pub fn reasoning(store: &PyStore, tree: &str) -> PyResult<String> {
    let read = reasoned(tree, store.shared().as_ref())
        .map_err(|why| PyRuntimeError::new_err(why.to_string()))?;
    serde_json::to_string(&read).map_err(|why| PyRuntimeError::new_err(why.to_string()))
}

/// What a scope with those roots reaches, in the order the moves were made.
#[pyfunction]
pub fn reasoning_covers(store: &PyStore, tree: &str, by: Vec<String>) -> PyResult<Vec<String>> {
    let read = reasoned(tree, store.shared().as_ref())
        .map_err(|why| PyRuntimeError::new_err(why.to_string()))?;
    read.covers(&by)
        .map_err(|why| PyValueError::new_err(why.to_string()))
}
