//! What is known about the samples, and the one thing that can be wrong about it.

use somatize_study::{Samples, SamplesError};

#[test]
fn most_cuts_only_need_the_count() {
    let samples = Samples::of(10);
    assert_eq!(samples.n(), 10);
    assert_eq!(samples.strata(), None);
    assert_eq!(samples.groups(), None);
}

#[test]
fn a_class_and_a_group_are_said_separately_and_stack() {
    let samples = Samples::of(4)
        .by_class(vec![0, 0, 1, 1])
        .unwrap()
        .in_groups(vec![7, 7, 8, 8])
        .unwrap();
    assert_eq!(samples.strata(), Some(&[0, 0, 1, 1][..]));
    assert_eq!(samples.groups(), Some(&[7, 7, 8, 8][..]));
}

#[test]
fn one_key_per_sample_or_it_says_which_one_is_short() {
    // The realistic slip: `y` of a different length than the dataset. It is
    // caught here and not as a fold that quietly leaves samples out.
    assert_eq!(
        Samples::of(10).by_class(vec![0, 1, 0]),
        Err(SamplesError::Mismatch {
            what: "classes",
            given: 3,
            n: 10
        })
    );
    assert_eq!(
        Samples::of(3).in_groups(vec![1, 1, 2, 2]),
        Err(SamplesError::Mismatch {
            what: "groups",
            given: 4,
            n: 3
        })
    );
}

#[test]
fn the_values_are_opaque_and_need_not_start_at_zero() {
    // They are compared, never ordered or counted on: a class is a name that
    // happens to be a number, which is what makes turning `y` into these one
    // line on the Python side.
    let samples = Samples::of(4).by_class(vec![31337, 4, 31337, 4]).unwrap();
    assert_eq!(samples.strata(), Some(&[31337, 4, 31337, 4][..]));
}
