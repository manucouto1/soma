//! Drawn from what already worked.

use super::drawing::{bell, coordinate, settle, span, stream, unit};
use super::{Random, Sampler};
use crate::{Dimension, Goal, Point, Setting, Space};

/// Tree-structured Parzen Estimator: model what the good trials did, model what
/// the bad ones did, and propose where the first is likely and the second is not.
///
/// The one that looks at **what already happened**, which is also its one honest
/// cost: it cannot derive trial 7 from the seed and the index alone, so a study
/// spread over a folder gets a different search than one in a single process.
/// That is what being guided means, not a bug to fix.
#[derive(Debug, Clone, PartialEq)]
pub struct Tpe {
    /// Which way is better.
    pub goal: Goal,
    /// Draw at random until this many trials have finished. Below two there is
    /// nothing to split into good and bad, so two is the floor whatever is said.
    pub startup: usize,
    /// How many places to consider before proposing one. More is a better
    /// proposal and costs nothing but arithmetic — no trial is run for it.
    pub candidates: usize,
    /// What share of the finished trials counts as good. `0.25` keeps the best
    /// quarter as the thing to imitate.
    pub quantile: f64,
    /// The seed of the draws.
    pub seed: u64,
}

impl Tpe {
    /// The `trial`-th point. Random until `startup`, guided after.
    pub fn ask(&self, space: &Space, trial: usize, seen: &[(Point, Option<f64>)]) -> Option<Point> {
        if space.is_empty() {
            return None;
        }
        // A trial that reported nothing comparable says nothing about where to
        // look: it is dropped rather than counted as terrible.
        let scored: Vec<(&Point, f64)> = seen
            .iter()
            .filter_map(|(point, at)| at.filter(|at| !at.is_nan()).map(|at| (point, at)))
            .collect();
        if scored.len() < self.startup.max(2) {
            return Random { seed: self.seed }.ask(space, trial, seen);
        }

        let (good, mut bad) = self.split(&scored);
        // In flight: somebody is trying it and nobody knows how it will do. It
        // goes in the pile to keep away from — that is *constant liar* — but it
        // **does not vote on how big the other pile is**. Counted, it would push
        // the quantile up and promote a trial out of the bad pile; if that trial
        // sat next to the one in flight, the warning would pull the search
        // towards it. Measured: one proposal in two hundred became thirty-nine.
        bad.extend(
            seen.iter()
                .filter(|(_, at)| at.is_none())
                .map(|(point, _)| point),
        );
        let mut state = stream(self.seed, trial);
        let mut best: Option<Point> = None;
        let mut best_gain = f64::NEG_INFINITY;

        for _ in 0..self.candidates.max(1) {
            let mut settings = Vec::with_capacity(space.len());
            let mut gain = 0.0;
            for (name, dimension) in space.dimensions() {
                let (setting, said) = propose(
                    dimension,
                    &placed(&good, name, dimension),
                    &placed(&bad, name, dimension),
                    &mut state,
                );
                gain += said;
                settings.push((name.clone(), setting));
            }
            if gain > best_gain {
                best_gain = gain;
                best = Some(Point::of(settings));
            }
        }
        best
    }

    /// The finished trials in two piles: the ones worth imitating and the rest.
    /// Both piles are non-empty — with everything good there is nothing to
    /// prefer it to.
    fn split<'a>(&self, scored: &[(&'a Point, f64)]) -> (Vec<&'a Point>, Vec<&'a Point>) {
        let mut order: Vec<(&Point, f64)> = scored.to_vec();
        order.sort_by(|(_, one), (_, other)| match self.goal {
            Goal::Minimize => one.total_cmp(other),
            Goal::Maximize => other.total_cmp(one),
        });
        let many = (self.quantile.clamp(0.0, 1.0) * order.len() as f64).ceil() as usize;
        let many = many.clamp(1, order.len() - 1);
        let (good, bad) = order.split_at(many);
        (
            good.iter().map(|(point, _)| *point).collect(),
            bad.iter().map(|(point, _)| *point).collect(),
        )
    }
}

/// Where on this knob's line each of those trials sat. A trial that has no value
/// for it, or one of another kind — a point recorded against a space that has
/// since changed — is simply not there.
fn placed(points: &[&Point], name: &str, dimension: &Dimension) -> Vec<f64> {
    points
        .iter()
        .filter_map(|point| point.get(name))
        .filter_map(|setting| coordinate(dimension, setting))
        .collect()
}

/// A value for this knob, and how much better the good pile likes it than the
/// bad one. Summed over the knobs, that is what picks the candidate.
fn propose(dimension: &Dimension, good: &[f64], bad: &[f64], state: &mut u64) -> (Setting, f64) {
    if good.is_empty() {
        // Nothing to imitate for this knob: draw it from the space and let the
        // others decide the candidate.
        let (from, to) = span(dimension);
        return (settle(dimension, from + unit(state) * (to - from)), 0.0);
    }
    match dimension {
        Dimension::Choice(options) => among(options, good, bad, state),
        _ => along(dimension, good, bad, state),
    }
}

/// A knob whose values lie on a line: two Parzen windows, one per pile.
fn along(dimension: &Dimension, good: &[f64], bad: &[f64], state: &mut u64) -> (Setting, f64) {
    let (from, to) = span(dimension);
    let place = drawn_from(good, from, to, state);
    let gain = density(good, from, to, place).ln() - density(bad, from, to, place).ln();
    (settle(dimension, place), gain)
}

/// A knob that is a list of names: counts, with one imaginary observation of
/// each so an option nobody tried is unlikely rather than impossible.
fn among(options: &[String], good: &[f64], bad: &[f64], state: &mut u64) -> (Setting, f64) {
    let tally = |seen: &[f64]| {
        let mut counts = vec![1.0; options.len()];
        for &which in seen {
            counts[(which as usize).min(options.len() - 1)] += 1.0;
        }
        let total: f64 = counts.iter().sum();
        (counts, total)
    };
    let (liked, liked_total) = tally(good);
    let (disliked, disliked_total) = tally(bad);

    let mut left = unit(state) * liked_total;
    let mut which = options.len() - 1;
    for (option, count) in liked.iter().enumerate() {
        if left < *count {
            which = option;
            break;
        }
        left -= count;
    }

    let gain = (liked[which] / liked_total).ln() - (disliked[which] / disliked_total).ln();
    (Setting::Choice(options[which].clone()), gain)
}

/// One draw from the window the good pile makes: pick an observation and land
/// near it, or fall back on the prior that keeps the whole range reachable.
fn drawn_from(values: &[f64], from: f64, to: f64, state: &mut u64) -> f64 {
    let prior = 1.0 / (values.len() as f64 + 1.0);
    let place = if unit(state) < prior {
        bell(state, (from + to) / 2.0, (to - from) / 2.0)
    } else {
        let which = ((unit(state) * values.len() as f64) as usize).min(values.len() - 1);
        bell(state, values[which], width_of(values, to - from))
    };
    place.clamp(from, to)
}

/// How likely that place is under the window those values make. The prior — a
/// wide bell over the range, weighted as one extra observation — is what stops
/// three trials declaring the rest of the space impossible.
fn density(values: &[f64], from: f64, to: f64, place: f64) -> f64 {
    if values.is_empty() {
        return bell_at(place, (from + to) / 2.0, (to - from) / 2.0);
    }
    let many = values.len() as f64;
    let prior = 1.0 / (many + 1.0);
    let width = width_of(values, to - from);
    let mut how = prior * bell_at(place, (from + to) / 2.0, (to - from) / 2.0);
    for &value in values {
        how += (1.0 - prior) / many * bell_at(place, value, width);
    }
    how.max(f64::MIN_POSITIVE)
}

/// How wide each bell is: Scott's rule, floored so identical observations do not
/// make a spike of zero width and ceilinged at the range so a single one does
/// not flatten into nothing.
fn width_of(values: &[f64], span: f64) -> f64 {
    let many = values.len() as f64;
    let mean = values.iter().sum::<f64>() / many;
    let spread = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / many).sqrt();
    (1.06 * spread * many.powf(-0.2)).clamp(span / 100.0, span)
}

/// The height of a bell of that width, centred there.
fn bell_at(place: f64, centre: f64, width: f64) -> f64 {
    let from_centre = (place - centre) / width;
    (-0.5 * from_centre * from_centre).exp() / (width * std::f64::consts::TAU.sqrt())
}

impl From<Tpe> for Sampler {
    fn from(how: Tpe) -> Self {
        Self::Tpe(how)
    }
}
