//! The plain cut, and everything the shuffle is and is not.

use crate::invariants::{is_a_partition, is_ordered};
use soma_next_study::{KFold, PartitionError, Samples};

#[test]
fn k_folds_are_a_partition_of_the_samples() {
    let folds = KFold {
        k: 5,
        shuffle: None,
    }
    .folds(&Samples::of(20))
    .unwrap();

    assert_eq!(folds.len(), 5);
    is_a_partition(&folds, 20);
    is_ordered(&folds);
    assert!(folds.iter().all(|f| f.test.len() == 4));
}

#[test]
fn what_does_not_divide_is_spread_one_at_a_time() {
    // 10 over 3 is 4-3-3 and not 4-4-2: folds that differ by more than one are
    // not comparable, which is the whole reason for cutting this way.
    let folds = KFold {
        k: 3,
        shuffle: None,
    }
    .folds(&Samples::of(10))
    .unwrap();

    assert_eq!(
        folds.iter().map(|f| f.test.len()).collect::<Vec<_>>(),
        vec![4, 3, 3]
    );
    is_a_partition(&folds, 10);
}

#[test]
fn without_a_seed_the_folds_are_the_order_they_came_in() {
    let folds = KFold {
        k: 2,
        shuffle: None,
    }
    .folds(&Samples::of(6))
    .unwrap();

    assert_eq!(folds[0].test, vec![0, 1, 2]);
    assert_eq!(folds[1].test, vec![3, 4, 5]);
}

#[test]
fn the_seed_decides_who_and_not_the_order_they_are_listed_in() {
    let folds = KFold {
        k: 2,
        shuffle: Some(7),
    }
    .folds(&Samples::of(6))
    .unwrap();

    is_a_partition(&folds, 6);
    is_ordered(&folds);
    assert_ne!(folds[0].test, vec![0, 1, 2], "it did shuffle");
}

#[test]
fn the_same_seed_gives_the_same_cut_and_a_different_one_does_not() {
    // It is what makes a fold reproducible from the record of a run, on any
    // machine: the generator is ours precisely so the answer does not move.
    let cut = |seed| {
        KFold {
            k: 4,
            shuffle: Some(seed),
        }
        .folds(&Samples::of(40))
        .unwrap()
    };

    assert_eq!(cut(1), cut(1));
    assert_ne!(cut(1), cut(2));
}

#[test]
fn leave_one_out_is_k_equal_to_n_and_not_a_scheme_of_its_own() {
    let folds = KFold {
        k: 6,
        shuffle: None,
    }
    .folds(&Samples::of(6))
    .unwrap();

    assert_eq!(folds.len(), 6);
    assert!(
        folds
            .iter()
            .all(|f| f.test.len() == 1 && f.train.len() == 5)
    );
    is_a_partition(&folds, 6);
}

#[test]
fn a_cut_that_cannot_be_honoured_fails_before_a_single_index_comes_out() {
    assert_eq!(
        KFold {
            k: 1,
            shuffle: None
        }
        .folds(&Samples::of(10)),
        Err(PartitionError::TooFewFolds { k: 1 })
    );
    assert_eq!(
        KFold {
            k: 20,
            shuffle: None
        }
        .folds(&Samples::of(10)),
        Err(PartitionError::MoreFoldsThanSamples { k: 20, n: 10 })
    );
}
