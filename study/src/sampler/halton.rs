//! Spread on purpose, one prime per knob.

use super::Sampler;
use super::drawing::{draw, splitmix};
use crate::{Point, Space};

/// Cover the space evenly instead of drawing from it evenly.
///
/// The second of the three that look at **nothing**, and what separates it from
/// [`Random`](super::Random) is not what it looks at but what it promises.
/// Random is uniform *in expectation*: over enough trials every corner gets its
/// share, and nothing stops the next two from landing on top of each other. This
/// is uniform *by construction, for every prefix* — of the first `base²` trials
/// exactly one lands in each cell of a `base²` grid, and there is no arrangement
/// of the indices that makes it otherwise.
///
/// That promise is what a study spread over a shared folder wants. Two machines
/// taking different numbers out of the folder do not collide, and it is not that
/// collision is unlikely: it cannot be arranged.
///
/// Knob `d` is read in base the `d`-th prime, which is why the promise thins out
/// once there are many knobs — the high primes need a long prefix before they
/// look like anything, and two of them drift into step. The scramble is what
/// holds it together, and a hyperparameter space with more than a handful of
/// knobs is rare enough that the seam is somewhere nobody stands.
/// [`Sobol`](super::Sobol) is the same idea without that seam, at the price of a
/// table.
///
/// Its point is a function of the **seed and the index**, so it derives like the
/// other two and a machine that claimed trial 7 needs nobody.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Halton {
    /// The seed, which here permutes the digits rather than drawing them: with
    /// no scramble a Halton sequence is one fixed sequence, and two studies of
    /// the same space would search it in exactly the same order.
    pub seed: u64,
}

impl Halton {
    /// The `trial`-th point. It never runs out, and it never looks at what the
    /// finished trials did.
    pub fn ask(&self, space: &Space, trial: usize, _finished: &[(Point, f64)]) -> Option<Point> {
        if space.is_empty() {
            return None;
        }
        Some(Point::of(
            space
                .dimensions()
                .iter()
                .enumerate()
                .map(|(which, (name, dimension))| {
                    let u = radical(prime(which), trial as u64, self.seed, which);
                    (name.clone(), draw(dimension, u))
                })
                .collect(),
        ))
    }
}

/// The index written in `base`, read back with its digits reversed and
/// scrambled — a number in `0.0..1.0`.
///
/// Reversing is the whole trick: consecutive indices differ in their *last*
/// digit, which after the reversal is the *first*, so they land far apart.
///
/// **Every place is scrambled, including the zeroes the index never reaches.**
/// That is not a detail: scrambling only the digits actually written down puts a
/// one-digit index and a two-digit one on two different grids, and then the
/// first `base²` trials stop landing one per cell — which is the whole of what
/// this is for. It also settles trial zero, which has no digits at all: it comes
/// out as the offsets themselves, a point like any other rather than the corner
/// of the space.
fn radical(base: u64, index: u64, seed: u64, which: usize) -> f64 {
    let mut state = seed ^ (which as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    // Never zero, so `digit -> multiplier * digit + offset` is a permutation of
    // the digits and not a mangling of them: the cover survives the seed. And it
    // is one multiplier per knob, which is what pulls two large primes back out
    // of step with each other.
    let multiplier = 1 + splitmix(&mut state) % (base - 1);
    let (mut left, mut place, mut value) = (index, 1.0 / base as f64, 0.0);
    for _ in 0..places(base) {
        // A fresh offset per place, so it is a permutation *of each place* and
        // not one rotation of the whole number.
        let offset = splitmix(&mut state) % base;
        value += ((multiplier * (left % base) + offset) % base) as f64 * place;
        left /= base;
        place /= base as f64;
    }
    value
}

/// How many places are worth scrambling: the first count whose grid is finer
/// than an `f64` tells apart. Past it every digit falls below the rounding.
fn places(base: u64) -> u32 {
    let (mut span, mut count) = (1u64, 0);
    while span < 1 << 53 {
        span *= base;
        count += 1;
    }
    count
}

/// The `which`-th prime, counting from zero. Found rather than tabulated: there
/// is no ceiling to write down, and a space has a handful of knobs.
fn prime(which: usize) -> u64 {
    let (mut found, mut candidate) = (0, 1u64);
    loop {
        candidate += 1;
        if (2u64..)
            .take_while(|by| by * by <= candidate)
            .all(|by| candidate % by != 0)
        {
            if found == which {
                return candidate;
            }
            found += 1;
        }
    }
}

impl From<Halton> for Sampler {
    fn from(how: Halton) -> Self {
        Self::Halton(how)
    }
}
