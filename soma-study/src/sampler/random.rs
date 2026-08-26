//! Drawn from the space, looking at nothing else.

use super::Sampler;
use super::drawing::{draw, stream, unit};
use crate::{Point, Space};

/// Uniform in every knob, independently.
///
/// The baseline and not a straw man: over a space where few knobs matter, random
/// search beats a grid, which spends its budget re-testing the ones that do not
/// (Bergstra and Bengio, 2012). It is what [`Tpe`](super::Tpe) falls back to
/// before it has anything to learn from. Its point is a function of the **seed
/// and the index**, so two machines drawing trial 7 draw the same point.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Random {
    /// The seed. There is no "unseeded": a search you cannot re-run is a result
    /// you cannot check.
    pub seed: u64,
}

impl Random {
    /// The `trial`-th point. It never runs out, and it never looks at what the
    /// finished trials did — which is the whole of what it is.
    pub fn ask(
        &self,
        space: &Space,
        trial: usize,
        _seen: &[(Point, Option<f64>)],
    ) -> Option<Point> {
        if space.is_empty() {
            return None;
        }
        let mut state = stream(self.seed, trial);
        Some(Point::of(
            space
                .dimensions()
                .iter()
                .map(|(name, dimension)| (name.clone(), draw(dimension, unit(&mut state))))
                .collect(),
        ))
    }
}

impl From<Random> for Sampler {
    fn from(how: Random) -> Self {
        Self::Random(how)
    }
}
