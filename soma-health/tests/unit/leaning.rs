//! What a model is leaning on, and the two ways that goes wrong.

use somatize_health::{Thresholds, leaning, shares};

fn drops(said: &[(&str, f64)]) -> Vec<somatize_health::Contribution> {
    shares(
        &said
            .iter()
            .map(|(n, d)| (n.to_string(), *d))
            .collect::<Vec<_>>(),
    )
}

fn said(what: &[(&str, f64)]) -> Vec<String> {
    leaning(&drops(what), &Thresholds::default())
        .iter()
        .map(ToString::to_string)
        .collect()
}

#[test]
fn shares_add_up_to_one_so_they_can_be_read_as_how_much_of_what_matters() {
    let out = drops(&[("a", 3.0), ("b", 1.0)]);

    assert_eq!(out[0].share, 0.75);
    assert_eq!(out[1].share, 0.25);
    assert_eq!(out[0].drop, 3.0, "and the raw number is kept");
}

#[test]
fn an_input_the_model_is_not_using_says_so() {
    // The finding this whole thing exists for: the channel the research was
    // about costs nothing to remove.
    assert_eq!(
        said(&[("symptoms", 0.001), ("text", 2.0)]),
        ["IGNORED_INPUT(symptoms)", "SOLE_RELIANCE(text)"]
    );
}

#[test]
fn and_two_inputs_that_share_the_work_say_nothing() {
    assert!(said(&[("a", 1.0), ("b", 1.2)]).is_empty());
}

#[test]
fn one_input_carrying_everything_is_worth_knowing_before_it_goes_missing() {
    let out = said(&[("a", 10.0), ("b", 0.3), ("c", 0.2)]);

    assert!(out.contains(&"SOLE_RELIANCE(a)".to_string()));
}

#[test]
fn a_model_that_loses_nothing_whatever_you_take_away_is_using_none_of_it() {
    // Every share zero rather than a division by nothing, and every input
    // flagged — which is exactly right.
    let out = said(&[("a", 0.0), ("b", -0.1)]);

    assert_eq!(out, ["IGNORED_INPUT(a)", "IGNORED_INPUT(b)"]);
}

#[test]
fn an_input_the_model_does_better_without_keeps_its_negative() {
    // Real, and worth seeing rather than clamped away: it means the channel is
    // actively getting in the way.
    let out = drops(&[("noise", -0.4), ("text", 2.0)]);

    assert!(out[0].drop < 0.0);
    assert!(out[0].share < 0.0);
}

#[test]
fn one_input_alone_says_nothing_because_there_is_nothing_to_compare() {
    assert!(said(&[("only", 5.0)]).is_empty());
    assert!(said(&[]).is_empty());
}

#[test]
fn the_bounds_are_data_here_too() {
    let strict = Thresholds {
        ignored_input: 0.4,
        ..Thresholds::default()
    };
    let said: Vec<String> = leaning(&drops(&[("a", 3.0), ("b", 1.0)]), &strict)
        .iter()
        .map(ToString::to_string)
        .collect();

    assert_eq!(said, ["IGNORED_INPUT(b)"], "a quarter is under four tenths");
}
