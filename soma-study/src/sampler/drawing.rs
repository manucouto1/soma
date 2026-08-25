//! The mechanics every scheme shares: a stream of numbers that is the same on
//! every machine, and moving between a knob and the line a number lives on.

use crate::{Dimension, Setting};

/// splitmix64, the reference constants. Ours rather than a crate's for the same
/// reason the shuffle of a fold is: a seed has to mean the same thing on every
/// machine that reads the same record, and that is not something `rand`
/// promises across versions.
pub(super) fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A number in `0.0..1.0`, from the top 53 bits, which is every one a `f64` can
/// tell apart.
pub(super) fn unit(state: &mut u64) -> f64 {
    (splitmix(state) >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// The stream this trial draws from. It is a function of the seed and the
/// **index**, never of what has been asked before, so the point of trial 7 is
/// the same whoever asks and in whatever order — which is what lets a machine
/// that claimed trial 7 from a shared folder derive it without replaying six.
pub(super) fn stream(seed: u64, trial: usize) -> u64 {
    seed ^ (trial as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// A number in `0.0..1.0` placed into the knob.
pub(super) fn draw(dimension: &Dimension, u: f64) -> Setting {
    match dimension {
        Dimension::Real { low, high, log } => {
            let (from, to) = span(dimension);
            let value = from + u * (to - from);
            Setting::Real(if *log {
                value.exp().clamp(*low, *high)
            } else {
                value.clamp(*low, *high)
            })
        }
        Dimension::Int { low, high } => {
            let many = (high - low + 1) as f64;
            Setting::Int((low + (u * many) as i64).min(*high))
        }
        Dimension::Choice(options) => {
            let which = ((u * options.len() as f64) as usize).min(options.len() - 1);
            Setting::Choice(options[which].clone())
        }
    }
}

/// The two ends of the line a knob's values live on: the logarithms when it is
/// logarithmic, the indices when it is a choice.
pub(super) fn span(dimension: &Dimension) -> (f64, f64) {
    match dimension {
        Dimension::Real {
            low,
            high,
            log: true,
        } => (low.ln(), high.ln()),
        Dimension::Real {
            low,
            high,
            log: false,
        } => (*low, *high),
        Dimension::Int { low, high } => (*low as f64, *high as f64),
        Dimension::Choice(options) => (0.0, (options.len() - 1) as f64),
    }
}

/// Where on that line a setting sits, or `None` if it is not this knob's kind —
/// which is what a point recorded against a space that has since changed looks
/// like.
pub(super) fn coordinate(dimension: &Dimension, setting: &Setting) -> Option<f64> {
    match (dimension, setting) {
        (Dimension::Real { log: true, .. }, Setting::Real(value)) if *value > 0.0 => {
            Some(value.ln())
        }
        (Dimension::Real { log: false, .. }, Setting::Real(value)) => Some(*value),
        (Dimension::Int { .. }, Setting::Int(value)) => Some(*value as f64),
        (Dimension::Choice(options), Setting::Choice(option)) => options
            .iter()
            .position(|taken| taken == option)
            .map(|which| which as f64),
        _ => None,
    }
}

/// Back from that line to a value of the knob, whatever falls off the ends
/// pulled back onto them.
pub(super) fn settle(dimension: &Dimension, place: f64) -> Setting {
    match dimension {
        Dimension::Real { low, high, log } => Setting::Real(if *log {
            place.exp().clamp(*low, *high)
        } else {
            place.clamp(*low, *high)
        }),
        Dimension::Int { low, high } => Setting::Int((place.round() as i64).clamp(*low, *high)),
        Dimension::Choice(options) => {
            let which = (place.round().max(0.0) as usize).min(options.len() - 1);
            Setting::Choice(options[which].clone())
        }
    }
}

/// One draw from a bell around `centre`, by Box-Muller. Two uniforms in, one
/// number out; the second of the pair is thrown away, which costs nothing here
/// and saves carrying a half-used draw around.
pub(super) fn bell(state: &mut u64, centre: f64, width: f64) -> f64 {
    let (one, other) = (unit(state).max(f64::MIN_POSITIVE), unit(state));
    centre + width * (-2.0 * one.ln()).sqrt() * (std::f64::consts::TAU * other).cos()
}
