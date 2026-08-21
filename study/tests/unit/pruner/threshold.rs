//! Judged against a constant: the only scheme that works on the first trial.

use soma_next_study::{Reason, Threshold, Verdict};

fn between(lower: f64, upper: f64) -> Threshold {
    Threshold {
        lower: Some(lower),
        upper: Some(upper),
    }
}

#[test]
fn what_crosses_a_bound_goes_and_it_says_which() {
    assert_eq!(
        between(0.0, 10.0).verdict(&[11.0], &[]),
        Verdict::Prune(Reason::OutOfBounds {
            value: 11.0,
            bound: 10.0
        })
    );
    assert_eq!(
        between(0.0, 10.0).verdict(&[-1.0], &[]),
        Verdict::Prune(Reason::OutOfBounds {
            value: -1.0,
            bound: 0.0
        })
    );
}

#[test]
fn inside_the_bounds_it_carries_on_and_the_bounds_are_inclusive() {
    assert_eq!(between(0.0, 10.0).verdict(&[5.0], &[]), Verdict::Continue);
    assert_eq!(between(0.0, 10.0).verdict(&[10.0], &[]), Verdict::Continue);
    assert_eq!(between(0.0, 10.0).verdict(&[0.0], &[]), Verdict::Continue);
}

#[test]
fn one_bound_and_no_bound_are_both_sayable() {
    let ceiling = Threshold {
        lower: None,
        upper: Some(10.0),
    };

    assert_eq!(ceiling.verdict(&[-100.0], &[]), Verdict::Continue);
    assert!(ceiling.verdict(&[11.0], &[]).is_prune());
}

#[test]
fn with_no_bounds_at_all_it_still_stops_what_blew_up() {
    // A legitimate way to use it: "prune only what diverged".
    assert_eq!(
        Threshold::diverged().verdict(&[f64::NAN], &[]),
        Verdict::Prune(Reason::NotANumber { at: 0 })
    );
    assert_eq!(
        Threshold::diverged().verdict(&[1e30], &[]),
        Verdict::Continue
    );
}

#[test]
fn it_needs_no_other_trial_which_is_the_whole_point_of_it() {
    // Where a diverged configuration costs the most is the very first trial,
    // and there the other two schemes have nothing to compare against.
    assert!(between(0.0, 1.0).verdict(&[50.0], &[]).is_prune());
}

#[test]
fn nothing_is_judged_before_its_first_report() {
    assert_eq!(between(0.0, 1.0).verdict(&[], &[]), Verdict::Continue);
}
