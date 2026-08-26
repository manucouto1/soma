//! The mechanics every scheme shares: dealing indices out, and putting the folds
//! back together. Free functions because they are not anybody's — having
//! `assemble` in **one** place is what makes both sides of every fold ascending
//! by construction rather than by each scheme remembering to.

use super::{Fold, PartitionError};
use std::collections::BTreeMap;

/// Fewer than two folds is not a cut, and more folds than samples leaves one
/// empty. Both are the declaration being wrong, not the data.
pub(super) fn checked(k: usize, n: usize) -> Result<(), PartitionError> {
    if k < 2 {
        Err(PartitionError::TooFewFolds { k })
    } else if k > n {
        Err(PartitionError::MoreFoldsThanSamples { k, n })
    } else {
        Ok(())
    }
}

/// `0..n`.
pub(super) fn in_order(n: usize) -> Vec<usize> {
    (0..n).collect()
}

/// The order the samples are dealt in: as they came, or shuffled from a seed.
pub(super) fn ordering(n: usize, shuffle: Option<u64>) -> Vec<usize> {
    let mut order = in_order(n);
    if let Some(seed) = shuffle {
        // Fisher-Yates over splitmix64: ten lines against a dependency, and the
        // seed has to mean the same thing on every machine that reads the same
        // record — which rules out whatever `rand` defaults to this year.
        let mut state = seed;
        for i in (1..order.len()).rev() {
            order.swap(i, (next(&mut state) % (i as u64 + 1)) as usize);
        }
    }
    order
}

/// splitmix64, the reference constants.
fn next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The indices sharing each key, keyed in ascending key order so the result
/// does not depend on which sample happened to come first.
pub(super) fn grouped_by(keys: &[u32], order: &[usize]) -> BTreeMap<u32, Vec<usize>> {
    let mut out: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for &index in order {
        out.entry(keys[index]).or_default().push(index);
    }
    out
}

/// The groups biggest first, ties by key, and the check that there are enough
/// of them: with fewer groups than folds one fold gets nothing.
pub(super) fn heaviest_first(
    members: BTreeMap<u32, Vec<usize>>,
    k: usize,
) -> Result<Vec<(u32, Vec<usize>)>, PartitionError> {
    if members.len() < k {
        return Err(PartitionError::TooFewGroups {
            groups: members.len(),
            k,
        });
    }
    let mut ordered: Vec<_> = members.into_iter().collect();
    ordered.sort_by(|(a, one), (b, other)| other.len().cmp(&one.len()).then(a.cmp(b)));
    Ok(ordered)
}

/// `items` into `k` parts, the first `len % k` of them one longer. That is what
/// makes the folds differ by at most one when the count does not divide.
pub(super) fn deal(items: &[usize], k: usize) -> Vec<Vec<usize>> {
    let (size, extra) = (items.len() / k, items.len() % k);
    let mut out = Vec::with_capacity(k);
    let mut cut = 0;
    for part in 0..k {
        let take = size + usize::from(part < extra);
        out.push(items[cut..cut + take].to_vec());
        cut += take;
    }
    out
}

/// From what is held out to the folds: whoever is not held out, trains.
pub(super) fn assemble(n: usize, tests: Vec<Vec<usize>>) -> Vec<Fold> {
    tests
        .into_iter()
        .map(|test| {
            let mut held = vec![false; n];
            for &index in &test {
                held[index] = true;
            }
            let mut test = test;
            test.sort_unstable();
            Fold {
                train: (0..n).filter(|index| !held[*index]).collect(),
                test,
            }
        })
        .collect()
}
