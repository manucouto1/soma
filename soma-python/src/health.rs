//! The verdict, on its way to Python. A thin translation and nothing else: the
//! numbers arrive as a dict, the bounds as a `Thresholds`, and what comes back
//! is a list of strings.
//!
//! As a dict because that is how they were **written down** — a health fact is a
//! record entry like any other. So a diagnosis over a stored record and one over
//! a run in flight take the same argument, which is what makes the invariant
//! testable rather than aspirational.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
// Aliased, because the function this crate exposes to Python has the same
// name — it is the same question asked from the other side of the wall.
use somatize_health::{Seen, Thresholds, verdict as taken};

/// The bounds a verdict is taken at. **The whole of the opinion, and it is
/// data**: change one and the same record answers differently, without
/// training again.
#[pyclass(name = "Thresholds", module = "somatize._somatize")]
#[derive(Clone)]
pub struct PyThresholds {
    inner: Thresholds,
}

#[pymethods]
impl PyThresholds {
    /// The defaults, with whatever you name overridden.
    ///
    /// They come from the original soma — tuned for LayerNorm-ish activations
    /// and Adam-sized steps — plus the literature for the three it did not
    /// have. They are a starting point and they are meant to be argued with.
    #[new]
    #[pyo3(signature = (**over))]
    fn new(over: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut inner = Thresholds::default();
        let Some(over) = over else {
            return Ok(Self { inner });
        };
        for (name, what) in over.iter() {
            let name: String = name.extract()?;
            let what: f64 = what.extract()?;
            match name.as_str() {
                "grad_low" => inner.grad_low = what,
                "grad_high" => inner.grad_high = what,
                "dead_eps" => inner.dead_eps = what,
                "dead_frac" => inner.dead_frac = what,
                "saturated_at" => inner.saturated_at = what,
                "saturated_frac" => inner.saturated_frac = what,
                "update_low" => inner.update_low = what,
                "update_high" => inner.update_high = what,
                "dormant_tau" => inner.dormant_tau = what,
                "dormant_frac" => inner.dormant_frac = what,
                "leakage_cka" => inner.leakage_cka = what,
                "narrowing_of_usual" => inner.narrowing_of_usual = what,
                "plasticity_growth" => inner.plasticity_growth = what,
                "gain_drift" => inner.gain_drift = what,
                other => {
                    // Named, and refused. A threshold quietly ignored is an
                    // argument somebody thinks they won.
                    return Err(PyValueError::new_err(format!(
                        "`{other}` is not a threshold; there are: grad_low, grad_high, dead_eps, \
                         dead_frac, saturated_at, saturated_frac, update_low, update_high, \
                         dormant_tau, dormant_frac, leakage_cka, narrowing_of_usual, \
                         plasticity_growth, gain_drift"
                    )));
                }
            }
        }
        Ok(Self { inner })
    }

    /// What they are, as a dict — for writing them down beside a diagnosis, so
    /// that an alarm from last week can be read with the bounds it was taken
    /// at.
    fn as_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let said = PyDict::new(py);
        let t = &self.inner;
        for (name, what) in [
            ("grad_low", t.grad_low),
            ("grad_high", t.grad_high),
            ("dead_eps", t.dead_eps),
            ("dead_frac", t.dead_frac),
            ("saturated_at", t.saturated_at),
            ("saturated_frac", t.saturated_frac),
            ("update_low", t.update_low),
            ("update_high", t.update_high),
            ("dormant_tau", t.dormant_tau),
            ("dormant_frac", t.dormant_frac),
            ("leakage_cka", t.leakage_cka),
            ("narrowing_of_usual", t.narrowing_of_usual),
            ("plasticity_growth", t.plasticity_growth),
            ("gain_drift", t.gain_drift),
        ] {
            said.set_item(name, what)?;
        }
        Ok(said)
    }

    fn __repr__(&self) -> String {
        format!(
            "Thresholds(grad_low={:.0e}, grad_high={:.0e}, dead_frac={}, update_low={:.0e}, \
             update_high={:.0e})",
            self.inner.grad_low,
            self.inner.grad_high,
            self.inner.dead_frac,
            self.inner.update_low,
            self.inner.update_high,
        )
    }
}

/// What is wrong with these numbers, as a list of names. Named `verdict` like
/// the Rust it wraps, so `somatize.health.diagnose` is the only `diagnose` there
/// is. Anything missing is **not measured**, which is not zero and not healthy.
#[pyfunction]
#[pyo3(signature = (seen, thresholds = None))]
pub fn verdict(
    seen: &Bound<'_, PyDict>,
    thresholds: Option<PyThresholds>,
) -> PyResult<Vec<String>> {
    let bounds = thresholds.map(|t| t.inner).unwrap_or_default();
    let number = |name: &str| -> PyResult<Option<f64>> {
        match seen.get_item(name)? {
            None => Ok(None),
            Some(what) if what.is_none() => Ok(None),
            Some(what) => Ok(what.extract::<f64>().ok().filter(|one| one.is_finite())),
        }
    };
    let count = |name: &str| -> PyResult<usize> {
        Ok(number(name)?.filter(|one| *one > 0.0).unwrap_or(0.0) as usize)
    };
    let flag = |name: &str| -> PyResult<bool> {
        Ok(match seen.get_item(name)? {
            None => false,
            Some(what) => what.is_truthy().unwrap_or(false) && !what.is_none(),
        })
    };
    let seen = Seen {
        nan: flag("nan")?,
        inf: flag("inf")?,
        grad_norm: number("grad_norm")?,
        zero_frac_max: number("zero_frac_max")?,
        sat_frac_max: number("sat_frac_max")?,
        update_ratio: number("update_ratio")?,
        dead_channels: count("dead_channels")?,
        ignored_channels: count("ignored_channels")?,
        dormancy_frac: number("dormancy_frac")?,
        group_cka: number("group_cka")?,
        eff_rank: number("eff_rank")?,
        eff_rank_slope: number("eff_rank_slope")?,
        param_norm_slope: number("param_norm_slope")?,
        update_rank: number("update_rank")?,
        update_rank_usual: number("update_rank_usual")?,
        signal_gain: number("signal_gain")?,
        jacobian_gain: number("jacobian_gain")?,
        jacobian_spread: number("jacobian_spread")?,
    };
    Ok(taken(&seen, &bounds)
        .iter()
        .map(ToString::to_string)
        .collect())
}

/// What each input turned out to be worth: its name, its share, and the raw
/// difference it made.
///
/// A `type` and not a bare tuple in the signature, because clippy is right:
/// three unnamed numbers in a return position is a thing nobody can read twice.
type Worth = (String, f64, f64);

/// What each input is worth, and what is wrong with that.
///
/// `drops` is `{name: how much worse the score gets without it}`, measured by
/// whoever has the data — this side only divides and compares. What comes back
/// is `(shares, flags)`: the share each input is of what all of them are worth,
/// and the two ways that goes wrong.
#[pyfunction]
#[pyo3(signature = (drops, thresholds = None))]
pub fn leaning(
    drops: &Bound<'_, PyDict>,
    thresholds: Option<PyThresholds>,
) -> PyResult<(Vec<Worth>, Vec<String>)> {
    let bounds = thresholds.map(|t| t.inner).unwrap_or_default();
    let mut said = Vec::new();
    for (name, drop) in drops.iter() {
        said.push((name.extract::<String>()?, drop.extract::<f64>()?));
    }
    let shares = somatize_health::shares(&said);
    let flags = somatize_health::leaning(&shares, &bounds)
        .iter()
        .map(ToString::to_string)
        .collect();
    Ok((
        shares
            .into_iter()
            .map(|one| (one.name, one.share, one.drop))
            .collect(),
        flags,
    ))
}

/// Which family of trouble a flag belongs to, by name.
///
/// So a figure can give each family a colour: six alarms that all look the same
/// are one alarm, and the first question anybody asks of four findings is
/// whether they are four problems or one.
#[pyfunction]
pub fn family(flag: &str) -> PyResult<String> {
    Ok(known(flag)?.family().to_string())
}

/// What a flag means and what to do about it, by name.
///
/// Beside the flag and not in whoever draws it: the bounds and the advice are
/// one opinion, and splitting them is how a dashboard ends up saying something
/// this library never said.
#[pyfunction]
pub fn about(flag: &str) -> PyResult<String> {
    Ok(known(flag)?.about().to_string())
}

/// The flag that name is, whatever it counts.
///
/// A list and not a `match`, because it goes the other way: a name arrives as
/// text out of a record and has to find its variant — so **the compiler does not
/// keep it**, and a variant added to the enum and not here parses as nothing. It
/// has happened. The tests that catch it ask every flag a run raises, and every
/// flag a **probe** raises, what to do about itself.
fn known(flag: &str) -> PyResult<somatize_health::Flag> {
    use somatize_health::Flag;
    let bare = flag.split('(').next().unwrap_or(flag);
    [
        Flag::Nan,
        Flag::Inf,
        Flag::Vanishing,
        Flag::Exploding,
        Flag::Dead,
        Flag::Saturated,
        Flag::Stalled,
        Flag::Overstepping,
        Flag::DeadChannels(0),
        Flag::IgnoredChannels(0),
        Flag::MissingNormalisation,
        Flag::Leakage,
        Flag::IgnoredInput(String::new()),
        Flag::SoleReliance(String::new()),
        Flag::Narrowing,
        Flag::LosingPlasticity,
    ]
    .into_iter()
    .find(|one| one.name() == bare)
    .ok_or_else(|| PyValueError::new_err(format!("`{flag}` is not a flag this library raises")))
}
