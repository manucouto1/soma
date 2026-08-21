//! Both constraints at once, which is where they stop composing cleanly.

use crate::invariants::is_a_partition;
use soma_next_study::{PartitionError, Samples, StratifiedGrouped};

#[test]
fn the_groups_stay_whole_and_the_classes_come_out_as_even_as_that_allows() {
    let strata = vec![0, 0, 1, 1, 0, 0, 1, 1];
    let groups = vec![1, 1, 1, 1, 2, 2, 2, 2];
    let samples = Samples::of(8)
        .by_class(strata.clone())
        .unwrap()
        .in_groups(groups.clone())
        .unwrap();

    let folds = StratifiedGrouped { k: 2 }.folds(&samples).unwrap();

    is_a_partition(&folds, 8);
    for fold in &folds {
        assert_eq!(
            fold.test
                .iter()
                .map(|&i| groups[i])
                .collect::<Vec<_>>()
                .len(),
            fold.test.len()
        );
        let group = groups[fold.test[0]];
        assert!(
            fold.test.iter().all(|&i| groups[i] == group),
            "a group was cut in half"
        );
        assert_eq!(
            fold.test.iter().filter(|&&i| strata[i] == 0).count(),
            2,
            "and the classes came out even anyway"
        );
    }
}

#[test]
fn it_needs_both_and_names_the_first_one_missing() {
    let classed = Samples::of(4).by_class(vec![0, 0, 1, 1]).unwrap();
    let grouped = Samples::of(4).in_groups(vec![1, 1, 2, 2]).unwrap();

    assert_eq!(
        StratifiedGrouped { k: 2 }.folds(&grouped),
        Err(PartitionError::NeedsClasses)
    );
    assert_eq!(
        StratifiedGrouped { k: 2 }.folds(&classed),
        Err(PartitionError::NeedsGroups)
    );
}
