//! A k-fold over the groups, and why it takes no seed.

use crate::invariants::{is_a_partition, is_ordered};
use somatize_study::{Grouped, PartitionError, Samples};

#[test]
fn a_group_never_ends_up_on_both_sides() {
    let groups = vec![1, 1, 1, 2, 2, 3, 3, 4, 4, 4];
    let samples = Samples::of(10).in_groups(groups.clone()).unwrap();

    let folds = Grouped { k: 2 }.folds(&samples).unwrap();

    is_a_partition(&folds, 10);
    is_ordered(&folds);
    for fold in &folds {
        let whole: Vec<u32> = fold.test.iter().map(|&i| groups[i]).collect();
        for (index, group) in groups.iter().enumerate() {
            assert_eq!(
                whole.contains(group),
                fold.test.contains(&index),
                "group {group} was cut in half"
            );
        }
    }
}

#[test]
fn groups_are_placed_heaviest_first_so_the_folds_come_out_comparable() {
    // 4-3-2-1 over two folds is 5 and 5, which greedy-by-arrival would not
    // give. It is also why there is no seed: shuffling this only makes the
    // folds less alike.
    let samples = Samples::of(10)
        .in_groups(vec![1, 1, 1, 1, 2, 2, 2, 3, 3, 4])
        .unwrap();

    let folds = Grouped { k: 2 }.folds(&samples).unwrap();

    assert_eq!(folds[0].test.len(), 5);
    assert_eq!(folds[1].test.len(), 5);
}

#[test]
fn fewer_groups_than_folds_is_an_error_because_a_group_does_not_split() {
    let samples = Samples::of(6).in_groups(vec![1, 1, 1, 2, 2, 2]).unwrap();

    assert_eq!(
        Grouped { k: 3 }.folds(&samples),
        Err(PartitionError::TooFewGroups { groups: 2, k: 3 })
    );
}

#[test]
fn without_the_groups_it_says_which_call_supplies_them() {
    assert_eq!(
        Grouped { k: 2 }.folds(&Samples::of(10)),
        Err(PartitionError::NeedsGroups)
    );
}
