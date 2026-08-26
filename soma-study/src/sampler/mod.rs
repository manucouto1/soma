//! Where to look next.
//!
//! Each scheme is a type of its own with its own `ask`, and [`Sampler`] is the
//! family for when the scheme arrives as data — the same shape as
//! [`Partition`](crate::Partition) and [`Pruner`](crate::Pruner).
//!
//! | scheme | looks at | covers | runs out | derives from the index |
//! |---|---|---|---|---|
//! | [`Grid`] | **the space's shape** | exactly, where it cut | yes | yes |
//! | [`Random`] | **nothing** | in expectation | no | yes |
//! | [`Halton`] | **nothing** | every prefix, thinning with the knobs | no | yes |
//! | [`Sobol`] | **nothing** | every prefix, up to [`KNOBS`] of them | no | yes |
//! | [`Tpe`] | **what already happened** | not what it is for | no | no |
//!
//! Three look at nothing and are still three schemes, which is why *looks at* is
//! not the only column: [`Random`] is uniform in expectation, the other two by
//! construction for every prefix. They pay differently — [`Halton`] is
//! arithmetic with no ceiling whose cover thins with many knobs, [`Sobol`] has no
//! seam but carries a table and answers nothing past [`KNOBS`].
//!
//! **`ask` is a function of the index**, not of what came before. The original's
//! took `&mut self` and had a `prepare`; this takes neither, so asking for trial
//! 7 without having asked for the first six gives the same answer. That is what
//! lets a study over a shared folder work with no coordinator. [`Tpe`] is the
//! honest exception and says so.
//!
//! A sampler starts nothing: `ask` gives back a [`Point`], or `None` — a
//! [`Grid`] with no combinations left, which is how a `for` knows to stop.

mod drawing;
mod grid;
mod halton;
mod random;
mod sobol;
mod tpe;

pub use grid::Grid;
pub use halton::Halton;
pub use random::Random;
pub use sobol::{KNOBS, Sobol};
pub use tpe::Tpe;

use crate::{Point, Space};
use std::fmt;

/// Whichever of the schemes a sampler is.
#[derive(Debug, Clone, PartialEq)]
pub enum Sampler {
    /// [`Grid`]: every combination, then nothing.
    Grid(Grid),
    /// [`Random`]: uniform, looking at nothing.
    Random(Random),
    /// [`Halton`]: spread on purpose, one prime per knob.
    Halton(Halton),
    /// [`Sobol`]: spread on purpose, and without Halton's seam.
    Sobol(Sobol),
    /// [`Tpe`]: guided by what already worked.
    Tpe(Tpe),
}

impl Sampler {
    /// Where to look for the `trial`-th time, or `None` when there is nowhere
    /// left. `seen` is the points somebody has already been to, and **a score
    /// that is not there means the trial is still running**. Four of the five
    /// ignore the argument, which is the point of having five.
    pub fn ask(&self, space: &Space, trial: usize, seen: &[(Point, Option<f64>)]) -> Option<Point> {
        match self {
            Self::Grid(how) => how.ask(space, trial, seen),
            Self::Random(how) => how.ask(space, trial, seen),
            Self::Halton(how) => how.ask(space, trial, seen),
            Self::Sobol(how) => how.ask(space, trial, seen),
            Self::Tpe(how) => how.ask(space, trial, seen),
        }
    }
}

impl fmt::Display for Sampler {
    /// As text, which is the form that goes into the record of a run.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Grid(how) => write!(f, "grid:{}", how.steps),
            Self::Random(how) => write!(f, "random:{}", how.seed),
            Self::Halton(how) => write!(f, "halton:{}", how.seed),
            Self::Sobol(how) => write!(f, "sobol:{}", how.seed),
            Self::Tpe(how) => write!(
                f,
                "tpe:{}:startup:{}:candidates:{}:quantile:{}:seed:{}",
                how.goal, how.startup, how.candidates, how.quantile, how.seed
            ),
        }
    }
}
