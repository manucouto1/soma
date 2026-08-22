//! `soma_next.study.Sampler` — where to look next.

use super::space::{PyPoint, PySpace};
use super::{hashed, read};
use pyo3::prelude::*;
use soma_next_study::{Grid, Halton, Point, Random, Sampler, Sobol, Tpe};

/// `soma_next.study.Sampler` — where to look for the next configuration.
#[pyclass(name = "Sampler", module = "soma_next._soma_next", frozen)]
pub struct PySampler {
    how: Sampler,
}

#[pymethods]
impl PySampler {
    /// Every combination, then nothing. `steps` says how finely a continuous
    /// knob is cut; an `int` narrower than that is taken whole.
    #[staticmethod]
    #[pyo3(signature = (steps = 5))]
    fn grid(steps: usize) -> Self {
        Grid { steps }.into()
    }

    /// Uniform in every knob, looking at nothing else — and over a space where
    /// only a few knobs matter, that beats a grid on the same budget.
    #[staticmethod]
    #[pyo3(signature = (*, seed = 0))]
    fn random(seed: u64) -> Self {
        Random { seed }.into()
    }

    /// Cover the space evenly instead of drawing from it evenly, one prime per
    /// knob.
    ///
    /// A uniform draw is even *in expectation*: nothing stops the next two
    /// trials from landing on top of each other, it is only unlikely. This is
    /// even *for every prefix*, which is what a study handed out of a shared
    /// folder wants — two machines taking different numbers do not collide, and
    /// not because collision is improbable.
    ///
    /// Its cover thins once there are many knobs, and it has no ceiling.
    #[staticmethod]
    #[pyo3(signature = (*, seed = 0))]
    fn halton(seed: u64) -> Self {
        Halton { seed }.into()
    }

    /// The same, without the seam — and with a ceiling of 32 knobs.
    ///
    /// Every knob is read in base two and told apart by a table of direction
    /// numbers (Joe and Kuo, 2008), so nothing thins out as the knobs grow. Past
    /// what the table reaches, `ask` answers `None` from the very first trial.
    #[staticmethod]
    #[pyo3(signature = (*, seed = 0))]
    fn sobol(seed: u64) -> Self {
        Sobol { seed }.into()
    }

    /// Guided by what already worked: model the good trials, model the bad ones,
    /// and propose where the first is likely and the second is not.
    ///
    /// Random until `startup` trials have finished. Unlike the other two its
    /// answer depends on what the asking machine has seen, which is what being
    /// guided means.
    #[staticmethod]
    #[pyo3(signature = (*, goal = "min", startup = 10, candidates = 24, quantile = 0.25, seed = 0))]
    fn tpe(
        goal: &str,
        startup: usize,
        candidates: usize,
        quantile: f64,
        seed: u64,
    ) -> PyResult<Self> {
        Ok(Tpe {
            goal: read(goal)?,
            startup,
            candidates,
            quantile,
            seed,
        }
        .into())
    }

    /// Where to look for the `trial`-th time, or `None` when there is nowhere
    /// left — which is a grid saying it is done, and how a `for` stops without
    /// being told a number.
    ///
    /// `seen` is `(point, score)` for the places somebody has already been.
    /// **A score of `None` means the trial is still running** — another machine
    /// is trying it and nobody knows yet how it will do, which is what
    /// `in_flight` gives back. Four of the five schemes ignore the whole
    /// argument, and that is the point of having five.
    #[pyo3(signature = (space, trial, seen = None))]
    fn ask(
        &self,
        space: &PySpace,
        trial: usize,
        seen: Option<Vec<(PyRef<'_, PyPoint>, Option<f64>)>>,
    ) -> Option<PyPoint> {
        let seen: Vec<(Point, Option<f64>)> = seen
            .unwrap_or_default()
            .into_iter()
            .map(|(point, score)| (point.point.clone(), score))
            .collect();
        self.how.ask(&space.space, trial, &seen).map(PyPoint::from)
    }

    /// How many combinations there are, for the one scheme that has an answer —
    /// `None` for the two that never run out. What a `range()` wants.
    fn total(&self, space: &PySpace) -> Option<usize> {
        match &self.how {
            Sampler::Grid(grid) => Some(grid.total(&space.space)),
            _ => None,
        }
    }

    /// How it is written down: what goes into the record of a run.
    fn __str__(&self) -> String {
        self.how.to_string()
    }

    fn __repr__(&self) -> String {
        format!("Sampler({})", self.how)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.how == other.how
    }

    fn __hash__(&self) -> u64 {
        hashed(&self.how)
    }
}

/// Every constructor arrives here, the same as for a cut and for a pruner.
impl<H: Into<Sampler>> From<H> for PySampler {
    fn from(how: H) -> Self {
        Self { how: how.into() }
    }
}
