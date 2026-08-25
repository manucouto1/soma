//! `somatize.study.Pruner` — whether a trial going badly is worth another epoch.

use super::{hashed, read};
use pyo3::prelude::*;
use somatize_study::{Patience, Percentile, Pruner, Threshold};

/// `somatize.study.Pruner` — whether a trial that is going badly is worth
/// another epoch.
#[pyclass(name = "Pruner", module = "somatize._somatize", frozen)]
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
