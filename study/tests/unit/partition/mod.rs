//! The family: what the enum adds over calling a scheme directly.
//!
//! Which is, deliberately, **nothing at all** as far as the folds go — there is
//! a test for that. What it adds happens at the edges: a name that is derived
//! rather than agreed, and being able to hold a cut that was decided by data.

mod grouped;
mod kfold;
mod stratified;
mod stratified_grouped;
mod time_series;

use crate::invariants::is_a_partition;
use soma_next_study::{
    Grouped, KFold, Partition, PartitionError, Samples, Stratified, StratifiedGrouped, TimeSeries,
};

/// One of each, which is what the exhaustiveness of the `match` is worth.
fn all() -> Vec<Partition> {
    vec![
        KFold {
            k: 5,
            shuffle: None,
        }
        .into(),
        Stratified {
            k: 5,
            shuffle: None,
        }
        .into(),
        Grouped { k: 5 }.into(),
        StratifiedGrouped { k: 5 }.into(),
        TimeSeries { k: 5, gap: 0 }.into(),
    ]
}

#[test]
fn going_through_the_enum_cuts_exactly_the_same_as_not_going_through_it() {
    // The dispatch is static either way: the enum is for when the scheme
    // arrives as data, and it must not become a second way of cutting.
    let samples = Samples::of(20)
        .by_class([0, 1].repeat(10))
        .unwrap()
        .in_groups((0..20).map(|i| i as u32 / 2).collect())
        .unwrap();

    let cut = KFold {
        k: 4,
        shuffle: Some(3),
    };
    assert_eq!(
        Partition::from(cut.clone()).folds(&samples),
        cut.folds(&samples)
    );

    let cut = Stratified {
        k: 4,
        shuffle: None,
    };
    assert_eq!(
        Partition::from(cut.clone()).folds(&samples),
        cut.folds(&samples)
    );

    let cut = Grouped { k: 4 };
    assert_eq!(
        Partition::from(cut.clone()).folds(&samples),
        cut.folds(&samples)
    );

    let cut = TimeSeries { k: 4, gap: 1 };
    assert_eq!(
        Partition::from(cut.clone()).folds(&samples),
        cut.folds(&samples)
    );
}

#[test]
fn a_cut_that_arrived_as_data_still_refuses_what_it_cannot_honour() {
    let cut: Partition = Stratified {
        k: 2,
        shuffle: None,
    }
    .into();

    assert_eq!(
        cut.folds(&Samples::of(10)),
        Err(PartitionError::NeedsClasses)
    );
}

#[test]
fn keys_that_are_there_and_are_not_needed_change_nothing() {
    // The asymmetry on purpose: what is missing fails, what is spare is
    // ignored. It is what lets one `Samples` be cut by several schemes to
    // compare them, and what makes stratifying by accident impossible — the
    // scheme asks for it, never the presence of `y`.
    let bare = Samples::of(6);
    let laden = Samples::of(6)
        .by_class(vec![0, 1, 0, 1, 0, 1])
        .unwrap()
        .in_groups(vec![1, 1, 2, 2, 3, 3])
        .unwrap();

    let cut: Partition = KFold {
        k: 3,
        shuffle: Some(4),
    }
    .into();

    assert_eq!(cut.folds(&bare), cut.folds(&laden));
    is_a_partition(&cut.folds(&bare).unwrap(), 6);
}

#[test]
fn it_writes_itself_down_so_a_key_can_be_made_of_it() {
    // The reason the family is an enum: the name is derived from the structure,
    // not agreed on by whoever writes a scheme. With a name that is a
    // convention, two that collide give the wrong fold back in silence.
    assert_eq!(
        Partition::from(KFold {
            k: 5,
            shuffle: None
        })
        .to_string(),
        "kfold:5"
    );
    assert_eq!(
        Partition::from(KFold {
            k: 5,
            shuffle: Some(7)
        })
        .to_string(),
        "kfold:5:shuffled:7"
    );
    assert_eq!(
        Partition::from(TimeSeries { k: 5, gap: 2 }).to_string(),
        "timeseries:5:gap:2"
    );

    // And a scheme writes itself the same whether or not it is wrapped.
    let cut = StratifiedGrouped { k: 5 };
    assert_eq!(cut.to_string(), Partition::from(cut.clone()).to_string());
}

#[test]
fn two_cuts_that_differ_are_written_down_differently() {
    // The property the cache key needs, said as a property and not as seven
    // strings that happen to differ today.
    let mut cuts = all();
    cuts.push(
        KFold {
            k: 5,
            shuffle: Some(0),
        }
        .into(),
    );
    cuts.push(
        KFold {
            k: 4,
            shuffle: None,
        }
        .into(),
    );
    cuts.push(TimeSeries { k: 5, gap: 1 }.into());

    let names: std::collections::BTreeSet<String> = cuts.iter().map(|c| c.to_string()).collect();
    assert_eq!(names.len(), cuts.len());
}

#[test]
fn how_many_folds_without_producing_them() {
    for cut in all() {
        assert_eq!(cut.k(), 5);
    }
    assert_eq!(Partition::from(TimeSeries { k: 3, gap: 1 }).k(), 3);
}
