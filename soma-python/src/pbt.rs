//! `Pbt` — Population-Based Training from Python.
//!
//! `PbtRunner` has worked, and been tested, for as long as the strategy
//! layer has existed. Nothing could reach it: it is a Rust type with no
//! binding, and `TrainingStrategy::PopulationBased` — the enum variant
//! that names it — refuses, because a *strategy* would have to apply a
//! different set of hyperparameters to the graph on each worker, and the
//! wire protocol carries a plan, not a parameterization.
//!
//! Which is the same shape as `Study`: the sampler proposes parameters,
//! and a **callback the user wrote** turns them into a run. So PBT is
//! exposed the way `Study` is, as an executor driven from Python, rather
//! than as a strategy that cannot see the graph it is evolving.

use crate::prelude::*;
use somatize_core::strategy::{ExploitStrategy, ExploreStrategy};
use somatize_runtime::{EventBus, PbtConfig, PbtExecutor, PbtRunner, PopulationMember};
use std::sync::Arc;

/// Population-Based Training: train, evaluate, exploit, explore.
///
///   pbt = soma.Pbt(
///       search_space=[{"type": "float", "name": "lr",
///                      "low": 1e-4, "high": 1e-1, "scale": "log"}],
///       population_size=8, generations=5,
///   )
///   best = pbt.run(train, evaluate)
///
/// `train(member)` receives `{"id", "params", "state", "fitness"}` and
/// returns the member's new state (anything soma can carry). `evaluate`
/// returns a number, **higher is better** — the one convention worth
/// stating twice, because a loss handed back unnegated evolves the
/// population towards the worst member.
#[pyclass(name = "Pbt")]
pub(crate) struct PyPbt {
    config: PbtConfig,
}

/// Bridges the two Python callables into the Rust trait.
struct PyPbtExecutor {
    train: PyObject,
    evaluate: PyObject,
}

/// A member as Python sees it.
fn member_to_py(py: Python<'_>, member: &PopulationMember) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("id", &member.id)?;
    let params = PyDict::new(py);
    for (key, value) in &member.params {
        params.set_item(key, json_to_py(py, value)?)?;
    }
    dict.set_item("params", params)?;
    dict.set_item("state", value_to_py(py, &member.state)?)?;
    dict.set_item("fitness", member.fitness)?;
    Ok(dict.into())
}

impl PbtExecutor for PyPbtExecutor {
    fn train(&self, member: &PopulationMember) -> somatize_core::error::Result<Value> {
        Python::with_gil(|py| {
            let arg = member_to_py(py, member).map_err(py_err_to_soma)?;
            let out = self
                .train
                .bind(py)
                .call1((arg,))
                .map_err(|e| soma_error(format!("PBT train({}) failed: {e}", member.id)))?;
            py_to_value(py, &out).map_err(py_err_to_soma)
        })
    }

    fn evaluate(&self, member: &PopulationMember) -> somatize_core::error::Result<f64> {
        Python::with_gil(|py| {
            let arg = member_to_py(py, member).map_err(py_err_to_soma)?;
            let out = self
                .evaluate
                .bind(py)
                .call1((arg,))
                .map_err(|e| soma_error(format!("PBT evaluate({}) failed: {e}", member.id)))?;
            out.extract::<f64>().map_err(|_| {
                soma_error(format!(
                    "PBT evaluate({}) returned {}, not a number. Fitness is a \
                     single score, and higher is better",
                    member.id,
                    out.get_type()
                        .name()
                        .map(|n| n.to_string())
                        .unwrap_or_default()
                ))
            })
        })
    }
}

fn soma_error(message: String) -> somatize_core::error::SomaError {
    somatize_core::error::SomaError::Other(message)
}

#[pymethods]
impl PyPbt {
    #[new]
    #[pyo3(signature = (
        search_space,
        population_size=8,
        generations=5,
        exploit="truncation",
        explore="perturbation",
        fraction=0.25,
        factor=0.2,
        train_steps_per_generation=1,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        search_space: &Bound<'_, PyAny>,
        population_size: usize,
        generations: usize,
        exploit: &str,
        explore: &str,
        fraction: f64,
        factor: f64,
        train_steps_per_generation: usize,
    ) -> PyResult<Self> {
        if population_size == 0 {
            return Err(PyValueError::new_err(
                "population_size must be at least 1: there is nothing to evolve",
            ));
        }
        // The same parser `Study` uses. A second reading of the same
        // dicts would be a second definition of what a dimension is.
        let mut space = SearchSpace::new();
        for item in search_space.try_iter()? {
            space.add(crate::study::parse_py_search_dim(py, &item?)?);
        }
        if space.dimensions.is_empty() {
            return Err(PyValueError::new_err(
                "search_space is empty: every member would carry the same \
                 hyperparameters, so exploit and explore would have nothing to \
                 copy or mutate",
            ));
        }
        let exploit = match exploit {
            "truncation" => ExploitStrategy::Truncation { fraction },
            "binary" => ExploitStrategy::Binary,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown exploit {other:?}. Available: truncation, binary"
                )));
            }
        };
        let explore = match explore {
            "perturbation" => ExploreStrategy::Perturbation { factor },
            "resample" => ExploreStrategy::Resample,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown explore {other:?}. Available: perturbation, resample"
                )));
            }
        };
        Ok(Self {
            config: PbtConfig {
                population_size,
                generations,
                exploit,
                explore,
                search_space: space,
                train_steps_per_generation,
            },
        })
    }

    /// Evolve the population, returning it sorted best-first.
    ///
    /// Each entry is `{"id", "params", "fitness"}`. The state is not
    /// returned: it is whatever `train` handed back, it can be large, and
    /// the caller already has it.
    fn run(&self, py: Python<'_>, train: PyObject, evaluate: PyObject) -> PyResult<Vec<PyObject>> {
        let executor = PyPbtExecutor { train, evaluate };
        let runner = PbtRunner::new(Arc::new(EventBus::new(256)));
        let population = runner
            .run(&self.config, &executor)
            .map_err(soma_err_to_py)?;

        population
            .iter()
            .map(|member| {
                let dict = PyDict::new(py);
                dict.set_item("id", &member.id)?;
                let params = PyDict::new(py);
                for (key, value) in &member.params {
                    params.set_item(key, json_to_py(py, value)?)?;
                }
                dict.set_item("params", params)?;
                dict.set_item("fitness", member.fitness)?;
                Ok(dict.into())
            })
            .collect()
    }

    /// How many members and how many generations, for `repr`.
    fn __repr__(&self) -> String {
        format!(
            "Pbt(population_size={}, generations={}, dimensions={})",
            self.config.population_size,
            self.config.generations,
            self.config.search_space.dimensions.len()
        )
    }
}
