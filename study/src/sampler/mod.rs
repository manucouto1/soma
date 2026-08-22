//! Where to look next.
//!
//! Each scheme is a type of its own with its own `ask`, in its own file, and
//! [`Sampler`] is the family for when the scheme arrives as data — the same
//! shape as [`Partition`](crate::Partition) and [`Pruner`](crate::Pruner), for
//! the same three reasons.
//!
//! # Five schemes, and what tells them apart is not one thing
//!
//! | scheme | looks at | covers | runs out | derives from the index |
//! |---|---|---|---|---|
//! | [`Grid`] | **the space's shape** | exactly, and only where it cut | yes | yes |
//! | [`Random`] | **nothing** | in expectation | no | yes |
//! | [`Halton`] | **nothing** | every prefix, thinning with the knobs | no | yes |
//! | [`Sobol`] | **nothing** | every prefix, up to [`KNOBS`] of them | no | yes |
//! | [`Tpe`] | **what already happened** | not what it is for | no | no |
//!
//! Three of them look at nothing and they are still three different schemes,
//! which is why "looks at" is not the only column. [`Random`] is uniform *in
//! expectation*: nothing stops the next two trials from landing on top of each
//! other, it is only unlikely. The other two are uniform *by construction, for
//! every prefix* — and for a study handed out of a shared folder that is the
//! difference between collisions being improbable and there being no arrangement
//! of the indices that makes one.
//!
//! They pay for it differently, which is why both are here: [`Halton`] is
//! arithmetic with no ceiling whose cover thins once there are many knobs, and
//! [`Sobol`] has no such seam but carries a table, so past [`KNOBS`] knobs it has
//! nothing to answer with.
//!
//! # `ask` is a function of the index, not of what came before
//!
//! The original's `Sampler` took `&mut self` and had a `prepare` to build its
//! state up front. This one takes neither: a grid's combination is arithmetic on
//! the index, and a drawn point comes from `(seed, trial)`. Asking for trial 7
//! twice gives the same answer, and **asking for it without having asked for the
//! first six gives the same answer too**.
//!
//! That is not tidiness, it is what makes a study spread over a shared folder
//! work without a coordinator: `claim` hands a machine the number 7 and it
//! derives the point on its own, exactly as CU15's federated round has nobody in
//! charge. [`Tpe`] is the honest exception — it is guided, so it depends on what
//! the asking machine had already seen, and it says so.
//!
//! # It answers, and nothing else
//!
//! Like a pruner, a sampler starts nothing. `ask` gives back a [`Point`] and the
//! loop does what it likes with it — including `None`, which is a [`Grid`]
//! saying there are no combinations left and is how a `for` knows to stop
//! without being told a number.

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
    /// left — whichever scheme this is.
    ///
    /// `seen` is the points somebody has already been to. **A score that is not
    /// there means the trial is still running**: another machine is trying it
    /// and nobody knows yet how it will do. Four of the five ignore the whole
    /// argument, and that is the point of having five.
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
