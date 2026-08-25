//! What has to be true of a cut, whichever scheme made it.
//!
//! Written once because it is one invariant: **k folds are a partition of the
//! samples**. Each index is held out exactly once, and on the fold where it is
//! held out it does not train. The only scheme this is deliberately not true of
//! is `TimeSeries`, and its own tests say so.

use somatize_study::Fold;

/// Each index held out exactly once across the folds, and never held out and
/// training at the same time.
pub fn is_a_partition(folds: &[Fold], n: usize) {
    let mut held = vec![0usize; n];
    for fold in folds {
        for &index in &fold.test {
            held[index] += 1;
        }
        assert_eq!(
            fold.train.len() + fold.test.len(),
            n,
            "a fold has to account for every sample"
        );
        for &index in &fold.train {
            assert!(
                !fold.test.contains(&index),
                "{index} both trains and is held out"
            );
        }
    }
    assert!(
        held.iter().all(|&times| times == 1),
        "every sample is held out exactly once: {held:?}"
    );
}

/// Both sides come out ascending, whatever produced them.
pub fn is_ordered(folds: &[Fold]) {
    for fold in folds {
        assert!(
            fold.train.windows(2).all(|w| w[0] < w[1]),
            "train is ascending"
        );
        assert!(
            fold.test.windows(2).all(|w| w[0] < w[1]),
            "test is ascending"
        );
    }
}
