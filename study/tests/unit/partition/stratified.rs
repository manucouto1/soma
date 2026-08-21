//! A k-fold inside each class, and the refusal sklearn turns into a warning.

use crate::invariants::{is_a_partition, is_ordered};
use soma_next_study::{KFold, PartitionError, Samples, Stratified};

#[test]
fn every_class_keeps_its_share_in_every_fold() {
    // Six of one class and three of the other, three folds: 2 and 1 each time.
    let strata = vec![0, 0, 0, 0, 0, 0, 1, 1, 1];
    let samples = Samples::of(9).by_class(strata.clone()).unwrap();

    let folds = Stratified {
        k: 3,
        shuffle: None,
    }
    .folds(&samples)
    .unwrap();

    is_a_partition(&folds, 9);
    is_ordered(&folds);
    for fold in &folds {
        let zeros = fold.test.iter().filter(|&&i| strata[i] == 0).count();
        let ones = fold.test.iter().filter(|&&i| strata[i] == 1).count();
        assert_eq!((zeros, ones), (2, 1));
    }
}

#[test]
fn a_class_that_cannot_reach_every_fold_is_an_error_and_not_a_warning() {
    // sklearn warns and carries on, which leaves you with a result you cannot
    // tell from a good one. Here the declaration is simply wrong.
    let samples = Samples::of(10)
        .by_class(vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        .unwrap();

    assert_eq!(
        Stratified {
            k: 3,
            shuffle: None
        }
        .folds(&samples),
        Err(PartitionError::ClassTooSmall {
            class: 1,
            count: 1,
            k: 3
        })
    );
}

#[test]
fn without_the_classes_it_says_which_call_supplies_them() {
    assert_eq!(
        Stratified {
            k: 2,
            shuffle: None
        }
        .folds(&Samples::of(10)),
        Err(PartitionError::NeedsClasses)
    );
}

#[test]
fn it_is_what_the_plain_cut_is_not_when_the_classes_are_clumped() {
    // The case stratifying exists for: cut plainly, the first fold is all of
    // one class and has never seen the other.
    let strata = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1];
    let samples = Samples::of(12).by_class(strata.clone()).unwrap();

    let plain = KFold {
        k: 2,
        shuffle: None,
    }
    .folds(&samples)
    .unwrap();
    let stratified = Stratified {
        k: 2,
        shuffle: None,
    }
    .folds(&samples)
    .unwrap();

    assert!(plain[0].test.iter().all(|&i| strata[i] == 0));
    for fold in &stratified {
        assert_eq!(fold.test.iter().filter(|&&i| strata[i] == 1).count(), 2);
    }
}
