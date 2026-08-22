//! Spread on purpose, and without a seam.

use super::Sampler;
use super::drawing::{draw, splitmix};
use crate::{Point, Space};

/// Cover the space evenly, one binary bisection per knob.
///
/// The same promise as [`Halton`](super::Halton) — uniform *by construction for
/// every prefix*, not uniform in expectation — and the reason both are here is
/// where each one pays for it. Halton reads knob `d` in base the `d`-th prime,
/// so the promise thins out as the primes grow. This one reads **every** knob in
/// base two and gets the knobs to differ some other way: each has its own set of
/// direction numbers, chosen so that no two of them fall into step.
///
/// Chosen, and that is the price. Those numbers are a **table**, and a table is
/// data that has to be right: a Sobol sequence built on the wrong ones does not
/// fail, it just covers worse and nobody finds out. This one is Joe and Kuo's
/// (2008), and there is a test that walks the first dimensions against published
/// values, because reading it is not a way of checking it.
///
/// Its ceiling is that table: past [`KNOBS`] dimensions `ask` answers `None`,
/// from the very first trial rather than quietly somewhere in the middle. Halton
/// has no ceiling, which is the other half of why there are two.
///
/// Its point is a function of the **seed and the index**, so it derives like the
/// rest and a machine that claimed trial 7 needs nobody.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Sobol {
    /// The seed, which here shifts the digits rather than drawing them: without
    /// it a Sobol sequence is one fixed sequence and two studies of the same
    /// space would walk it in exactly the same order.
    pub seed: u64,
}

/// How many knobs the table reaches, and so how many this can search.
pub const KNOBS: usize = 32;

/// How many numbers each knob's direction gets, which is the width of the whole
/// arithmetic and so how many trials there are before it comes round again.
const BITS: usize = 32;

impl Sobol {
    /// The `trial`-th point, or `None` when there are more knobs than the table
    /// has. It never runs out, and it never looks at what the finished trials
    /// did.
    pub fn ask(&self, space: &Space, trial: usize, _finished: &[(Point, f64)]) -> Option<Point> {
        if space.is_empty() || space.len() > KNOBS {
            return None;
        }
        // Gray code: consecutive trials differ in exactly one bit, so each point
        // is the one before it with a single direction number flipped into it —
        // which is what keeps every prefix balanced and not just the whole.
        let step = trial as u64 ^ (trial as u64 >> 1);
        Some(Point::of(
            space
                .dimensions()
                .iter()
                .enumerate()
                .map(|(which, (name, dimension))| {
                    let numbers = directions(which);
                    let mut place = shift(self.seed, which);
                    for (bit, number) in numbers.iter().enumerate() {
                        if (step >> bit) & 1 == 1 {
                            place ^= number;
                        }
                    }
                    let u = place as f64 / (1u64 << BITS) as f64;
                    (name.clone(), draw(dimension, u))
                })
                .collect(),
        ))
    }
}

/// This knob's direction numbers: the `i`-th is what gets folded in when bit `i`
/// of the step is set.
///
/// The first `s` come from the table; the rest are the recurrence the primitive
/// polynomial stands for, `a` being its middle coefficients read as bits.
fn directions(which: usize) -> [u32; BITS] {
    let (a, m) = DIRECTIONS[which];
    let mut numbers = [0u32; BITS];
    // The first knob has no polynomial and no recurrence: its numbers are a
    // single one walking rightwards, which is plain bisection.
    if m.is_empty() {
        for (i, number) in numbers.iter_mut().enumerate() {
            *number = 1 << (BITS - 1 - i);
        }
        return numbers;
    }
    let s = m.len();
    for i in 0..s {
        numbers[i] = m[i] << (BITS - 1 - i);
    }
    for i in s..BITS {
        numbers[i] = numbers[i - s] ^ (numbers[i - s] >> s);
        for k in 1..s {
            numbers[i] ^= ((a >> (s - 1 - k)) & 1) * numbers[i - k];
        }
    }
    numbers
}

/// What this knob's numbers are shifted by. A digital shift and not a
/// rearrangement: xoring a constant into every point moves the whole set without
/// disturbing which cell each one is in, so the cover survives the seed intact.
fn shift(seed: u64, which: usize) -> u32 {
    let mut state = seed ^ (which as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    splitmix(&mut state) as u32
}

/// Joe and Kuo's direction numbers, `(the polynomial's middle coefficients, the
/// numbers it starts from)`, one row per knob.
///
/// *Constructing Sobol sequences with better two-dimensional projections*, SIAM
/// J. Sci. Comput. 30, 2635–2654 (2008) — the `new-joe-kuo-6` table, which is
/// the one the rest of the world means when it says Sobol.
const DIRECTIONS: [(u32, &[u32]); KNOBS] = [
    (0, &[]),
    (0, &[1]),
    (1, &[1, 3]),
    (1, &[1, 3, 1]),
    (2, &[1, 1, 1]),
    (1, &[1, 1, 3, 3]),
    (4, &[1, 3, 5, 13]),
    (2, &[1, 1, 5, 5, 17]),
    (4, &[1, 1, 5, 5, 5]),
    (7, &[1, 1, 7, 11, 19]),
    (11, &[1, 1, 5, 1, 1]),
    (13, &[1, 1, 1, 3, 11]),
    (14, &[1, 3, 5, 5, 31]),
    (1, &[1, 3, 3, 9, 7, 49]),
    (13, &[1, 1, 1, 15, 21, 21]),
    (16, &[1, 3, 1, 13, 27, 49]),
    (19, &[1, 1, 1, 15, 7, 5]),
    (22, &[1, 3, 1, 15, 13, 25]),
    (25, &[1, 1, 5, 5, 19, 61]),
    (1, &[1, 3, 7, 11, 23, 15, 103]),
    (4, &[1, 3, 7, 13, 13, 15, 69]),
    (7, &[1, 1, 3, 13, 7, 35, 63]),
    (8, &[1, 3, 5, 9, 1, 25, 53]),
    (14, &[1, 3, 1, 13, 9, 35, 107]),
    (19, &[1, 3, 1, 5, 27, 61, 31]),
    (21, &[1, 1, 5, 11, 19, 41, 61]),
    (28, &[1, 3, 5, 3, 3, 13, 69]),
    (31, &[1, 1, 7, 13, 1, 19, 1]),
    (32, &[1, 3, 7, 5, 13, 19, 59]),
    (37, &[1, 1, 3, 9, 25, 29, 41]),
    (41, &[1, 3, 5, 13, 23, 1, 55]),
    (42, &[1, 3, 7, 3, 13, 59, 17]),
];

impl From<Sobol> for Sampler {
    fn from(how: Sobol) -> Self {
        Self::Sobol(how)
    }
}
