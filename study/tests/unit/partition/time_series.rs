//! Growing prefixes: the one scheme that is deliberately not a partition.

use somatize_study::{PartitionError, Samples, TimeSeries};

#[test]
fn it_never_trains_on_its_own_future() {
    let folds = TimeSeries { k: 3, gap: 0 }.folds(&Samples::of(12)).unwrap();

    assert_eq!(folds.len(), 3);
    for fold in &folds {
        assert!(
            fold.train.iter().max() < fold.test.iter().min(),
            "everything trained on comes before everything held out"
        );
    }
    assert_eq!(
        folds.iter().map(|f| f.train.len()).collect::<Vec<_>>(),
        vec![3, 6, 9]
    );
    assert_eq!(folds[2].test, vec![9, 10, 11]);
}

#[test]
fn the_first_block_only_ever_trains_and_that_is_the_point() {
    // There is nothing before it to learn from, so it is never held out.
    // Asserted here so the day someone "fixes" it, the reason is on the record.
    let folds = TimeSeries { k: 2, gap: 0 }.folds(&Samples::of(9)).unwrap();

    let held: Vec<usize> = folds.iter().flat_map(|f| f.test.clone()).collect();
    assert_eq!(held, vec![3, 4, 5, 6, 7, 8]);
    assert!(!held.contains(&0));
}

#[test]
fn a_gap_drops_what_sits_between_the_two_sides() {
    // Purged and embargoed cross-validation, which is a parameter and not a
    // scheme of its own.
    let folds = TimeSeries { k: 2, gap: 2 }.folds(&Samples::of(9)).unwrap();

    assert_eq!(folds[0].train, vec![0]);
    assert_eq!(folds[0].test, vec![3, 4, 5]);
    assert_eq!(folds[1].train, vec![0, 1, 2, 3]);
    assert_eq!(folds[1].test, vec![6, 7, 8]);
}

#[test]
fn a_gap_that_eats_the_first_training_set_is_an_error() {
    assert_eq!(
        TimeSeries { k: 2, gap: 5 }.folds(&Samples::of(9)),
        Err(PartitionError::GapTooLarge { gap: 5, k: 2 })
    );
    assert_eq!(
        TimeSeries { k: 20, gap: 0 }.folds(&Samples::of(10)),
        Err(PartitionError::MoreFoldsThanSamples { k: 20, n: 10 })
    );
}

#[test]
fn one_split_is_meaningful_here_and_nowhere_else() {
    // Unlike the rest: a single growing prefix is a plain holdout of the tail,
    // whereas a single k-fold would hold out everything and train on nothing.
    let folds = TimeSeries { k: 1, gap: 0 }.folds(&Samples::of(10)).unwrap();

    assert_eq!(folds.len(), 1);
    assert_eq!(folds[0].train, vec![0, 1, 2, 3, 4]);
    assert_eq!(folds[0].test, vec![5, 6, 7, 8, 9]);
}
