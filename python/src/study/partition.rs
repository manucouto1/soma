//! `soma_next.study.Partition` — how the samples are cut into folds.

use super::{hashed, to_py_err};
use pyo3::prelude::*;
use somatize_study::{
    Grouped, KFold, Partition, Samples, Stratified, StratifiedGrouped, TimeSeries,
};

/// `soma_next.study.Partition` — how the samples are cut into folds.
#[pyclass(name = "Partition", module = "soma_next._soma_next", frozen)]
pub struct PyPartition {
    cut: Partition,
}

#[pymethods]
impl PyPartition {
    /// `k` folds over the samples, each held out in turn. `shuffle` is the
    /// seed, and without one the order they came in is kept.
    #[staticmethod]
    #[pyo3(signature = (k, *, shuffle = None))]
    fn kfold(k: usize, shuffle: Option<u64>) -> Self {
        KFold { k, shuffle }.into()
    }

    /// `k` folds where every class keeps the share it has in the whole. Needs
    /// `classes=`.
    #[staticmethod]
    #[pyo3(signature = (k, *, shuffle = None))]
    fn stratified(k: usize, shuffle: Option<u64>) -> Self {
        Stratified { k, shuffle }.into()
    }

    /// `k` folds where all the samples of a group land on the same side. Needs
    /// `groups=`, and takes no seed: it places the biggest groups first, which
    /// is what keeps the folds comparable.
    #[staticmethod]
    fn grouped(k: usize) -> Self {
        Grouped { k }.into()
    }

    /// Groups whole, and among the ways of doing that the one leaving the
    /// classes most even. Needs both.
    #[staticmethod]
    fn stratified_grouped(k: usize) -> Self {
        StratifiedGrouped { k }.into()
    }

    /// `k` growing prefixes, so nothing is ever trained on its own future.
    /// `gap` drops that many samples between the two sides.
    #[staticmethod]
    #[pyo3(signature = (k, *, gap = 0))]
    fn time_series(k: usize, gap: usize) -> Self {
        TimeSeries { k, gap }.into()
    }

    /// The folds as `(train, test)` pairs of indices — sklearn's shape, so a
    /// loop written against `KFold().split()` reads the same.
    ///
    /// `classes=` and `groups=` are one small integer per sample. They are
    /// **numbers, not labels**: turn `y` into them where `y` already is, with
    /// `.tolist()` or a dictionary, and no tensor crosses.
    #[pyo3(signature = (n, *, classes = None, groups = None))]
    fn folds(
        &self,
        n: usize,
        classes: Option<Vec<u32>>,
        groups: Option<Vec<u32>>,
    ) -> PyResult<Vec<(Vec<usize>, Vec<usize>)>> {
        let mut samples = Samples::of(n);
        if let Some(classes) = classes {
            samples = samples.by_class(classes).map_err(to_py_err)?;
        }
        if let Some(groups) = groups {
            samples = samples.in_groups(groups).map_err(to_py_err)?;
        }
        Ok(self
            .cut
            .folds(&samples)
            .map_err(to_py_err)?
            .into_iter()
            .map(|fold| (fold.train, fold.test))
            .collect())
    }

    /// How many folds it produces, without producing them.
    #[getter]
    fn k(&self) -> usize {
        self.cut.k()
    }

    /// How it is written down: what goes into a cache key and into the record
    /// of a run. Two cuts that differ are written differently.
    fn __str__(&self) -> String {
        self.cut.to_string()
    }

    fn __repr__(&self) -> String {
        format!("Partition({})", self.cut)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.cut == other.cut
    }

    fn __hash__(&self) -> u64 {
        hashed(&self.cut)
    }
}

/// Every constructor arrives here: the scheme is chosen in Python, so what
/// crosses is the family and not one of the five types.
impl<C: Into<Partition>> From<C> for PyPartition {
    fn from(cut: C) -> Self {
        Self { cut: cut.into() }
    }
}
