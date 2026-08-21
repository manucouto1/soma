//! Every combination, in order, until there are none left.

use super::Sampler;
use super::drawing::span;
use crate::{Dimension, Point, Setting, Space};

/// Walk the whole space and stop.
///
/// The only scheme that **runs out**: `ask` answers `None` once every
/// combination has been handed out, and that is how a study written as a `for`
/// knows when to stop without being told a number.
///
/// What is continuous has to be cut to be enumerated, and `steps` says how
/// finely. An `Int` narrower than that is taken whole — a range of five values
/// is five points, not `steps` of them.
///
/// **The first dimension varies fastest**, so consecutive trials differ in the
/// knob declared first. Worth knowing when a grid is stopped early.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Grid {
    /// How many values to take from each continuous knob.
    pub steps: usize,
}

impl Grid {
    /// The `trial`-th combination, or `None` when there are no more.
    ///
    /// It looks at neither the finished trials nor a seed: a grid is a function
    /// of the space and the index alone.
    pub fn ask(&self, space: &Space, trial: usize, _finished: &[(Point, f64)]) -> Option<Point> {
        if space.is_empty() || trial >= self.total(space) {
            return None;
        }
        let mut left = trial;
        let mut settings = Vec::with_capacity(space.len());
        for (name, dimension) in space.dimensions() {
            let many = dimension.grid_of(self.steps);
            settings.push((name.clone(), nth(dimension, left % many, many)));
            left /= many;
        }
        Some(Point::of(settings))
    }

    /// How many combinations there are — which is how many trials a grid search
    /// **is**, and something a caller wants before it starts one.
    pub fn total(&self, space: &Space) -> usize {
        if space.is_empty() {
            return 0;
        }
        space
            .dimensions()
            .iter()
            .map(|(_, dimension)| dimension.grid_of(self.steps))
            .product()
    }
}

/// The `which`-th of `many` values of this knob, ends included.
fn nth(dimension: &Dimension, which: usize, many: usize) -> Setting {
    if let Dimension::Choice(options) = dimension {
        return Setting::Choice(options[which.min(options.len() - 1)].clone());
    }
    let (from, to) = span(dimension);
    // With a single value there is no interval to divide, and the bottom is a
    // less surprising answer than the middle.
    let place = if many <= 1 {
        from
    } else {
        from + (to - from) * which as f64 / (many - 1) as f64
    };
    super::drawing::settle(dimension, place)
}

impl From<Grid> for Sampler {
    fn from(how: Grid) -> Self {
        Self::Grid(how)
    }
}
