//! The seam for what is above one training run.
//!
//! It translates and it does not decide: the five ways of cutting, the three
//! ways of giving up on a trial, and what can be honoured of either — all of it
//! is in `soma_next_study`, which has no dependencies and does not know Python
//! exists.
//!
//! What arrives here is a count and lists of small numbers, and what leaves is
//! pairs of indices or a reason. That is the shape that lets the loop stay in
//! Python — where torch is — without a single tensor crossing, and without
//! anything calling back the other way.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use soma_next_study::{
    Goal, Grouped, KFold, Partition, Patience, Percentile, Pruner, Samples, Stratified,
    StratifiedGrouped, Threshold, TimeSeries,
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

/// `soma_next.study.Pruner` — whether a trial that is going badly is worth
/// another epoch.
#[pyclass(name = "Pruner", module = "soma_next._soma_next", frozen)]
pub struct PyPruner {
    rule: Pruner,
}

#[pymethods]
impl PyPruner {
    /// Prune what is behind the median of the trials that already finished.
    /// `Pruner.percentile(50, …)` with a name on it.
    #[staticmethod]
    #[pyo3(signature = (*, goal = "min", warmup = 0, startup = 1))]
    fn median(goal: &str, warmup: usize, startup: usize) -> PyResult<Self> {
        Ok(Percentile::median(read(goal)?, warmup, startup).into())
    }

    /// The same with the share that survives said out loud: **smaller prunes
    /// more**. At `50` the better half stays.
    #[staticmethod]
    #[pyo3(signature = (p, *, goal = "min", warmup = 0, startup = 1))]
    fn percentile(p: f64, goal: &str, warmup: usize, startup: usize) -> PyResult<Self> {
        Ok(Percentile {
            p,
            goal: read(goal)?,
            warmup,
            startup,
        }
        .into())
    }

    /// Prune what leaves bounds you already know are hopeless. The only scheme
    /// that needs no other trial, so it works on the very first.
    #[staticmethod]
    #[pyo3(signature = (*, lower = None, upper = None))]
    fn threshold(lower: Option<f64>, upper: Option<f64>) -> Self {
        Threshold { lower, upper }.into()
    }

    /// Only what blew up: no bounds, so nothing goes but a loss that is not a
    /// number.
    #[staticmethod]
    fn diverged() -> Self {
        Threshold::diverged().into()
    }

    /// Prune what has stopped improving on its own best — early stopping.
    /// `steps` cannot be zero.
    #[staticmethod]
    #[pyo3(signature = (steps, *, min_delta = 0.0, goal = "min"))]
    fn patience(steps: std::num::NonZeroUsize, min_delta: f64, goal: &str) -> PyResult<Self> {
        Ok(Patience {
            steps,
            min_delta,
            goal: read(goal)?,
        }
        .into())
    }

    /// Why this trial is not worth another epoch, or `None` to carry on.
    ///
    /// `mine` is what it has reported so far, in order; `others` the same for
    /// the trials that already finished. A "step" is the **n-th report**, so
    /// trials have to report on the same schedule for the comparison across
    /// them to mean anything.
    ///
    /// Nothing is stopped here. You stop calling the trainer::
    ///
    ///     if why := pruner.verdict(reported, finished):
    ///         break
    #[pyo3(signature = (mine, others = None))]
    fn verdict(&self, mine: Vec<f64>, others: Option<Vec<Vec<f64>>>) -> Option<String> {
        self.rule
            .verdict(&mine, &others.unwrap_or_default())
            .reason()
            .map(ToString::to_string)
    }

    /// How it is written down: what goes into the record of a run.
    fn __str__(&self) -> String {
        self.rule.to_string()
    }

    fn __repr__(&self) -> String {
        format!("Pruner({})", self.rule)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.rule == other.rule
    }

    fn __hash__(&self) -> u64 {
        hashed(&self.rule)
    }
}

/// Every constructor arrives here, the same as for a cut.
impl<R: Into<Pruner>> From<R> for PyPruner {
    fn from(rule: R) -> Self {
        Self { rule: rule.into() }
    }
}

/// Which way is better, from the word Python wrote.
fn read(goal: &str) -> PyResult<Goal> {
    goal.parse().map_err(to_py_err)
}

/// The text form **is** the identity of both families — it is what a cache key
/// and a record are made of — so it is what a hash is taken over. Deriving over
/// the fields would need `Hash` on an `f64`, which does not exist for the same
/// reason `NaN != NaN`.
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
