//! Judged against the trials that already finished.

use somatize_study::{Goal, Percentile, Reason, Verdict};

/// Four finished trials, one report each.
fn four() -> Vec<Vec<f64>> {
    vec![vec![3.0], vec![1.0], vec![2.0], vec![4.0]]
}

fn median(warmup: usize, startup: usize) -> Percentile {
    Percentile::median(Goal::Minimize, warmup, startup)
}

#[test]
fn the_median_is_percentile_fifty_and_not_a_scheme_of_its_own() {
    assert_eq!(
        Percentile::median(Goal::Minimize, 1, 2),
        Percentile {
            p: 50.0,
            goal: Goal::Minimize,
            warmup: 1,
            startup: 2
        }
    );
}

#[test]
fn what_is_behind_the_bar_goes_and_what_is_not_stays() {
    // Four others at 1, 2, 3, 4: the median is 2.5, interpolated between the
    // middle two rather than picked from them.
    assert_eq!(
        median(0, 1).verdict(&[5.0], &four()),
        Verdict::Prune(Reason::Worse { than: 2.5, at: 0 })
    );
    assert_eq!(median(0, 1).verdict(&[2.0], &four()), Verdict::Continue);
}

#[test]
fn a_trial_that_ties_the_bar_is_not_pruned_for_tying() {
    assert_eq!(median(0, 1).verdict(&[2.5], &four()), Verdict::Continue);
}

#[test]
fn nothing_is_judged_before_its_first_report() {
    assert_eq!(median(0, 1).verdict(&[], &four()), Verdict::Continue);
}

#[test]
fn warmup_buys_a_slow_starter_the_epochs_it_needs() {
    let rule = median(2, 1);

    assert_eq!(rule.verdict(&[9.0], &four()), Verdict::Continue);
    assert_eq!(rule.verdict(&[9.0, 9.0], &four()), Verdict::Continue);
    // The third report is past the allowance, and there it is judged.
    let others = vec![vec![3.0; 3], vec![1.0; 3], vec![2.0; 3], vec![4.0; 3]];
    assert!(rule.verdict(&[9.0, 9.0, 9.0], &others).is_prune());
}

#[test]
fn startup_stops_the_first_trial_to_finish_becoming_the_bar() {
    assert_eq!(median(0, 5).verdict(&[9.0], &four()), Verdict::Continue);
    assert!(median(0, 4).verdict(&[9.0], &four()).is_prune());
}

#[test]
fn only_the_trials_that_got_this_far_have_a_say() {
    // Three stopped at report 0; one reached report 1. At report 1 there is a
    // single opinion, and with `startup: 2` that is not enough.
    let others = vec![vec![1.0], vec![2.0], vec![3.0], vec![1.0, 1.0]];

    assert_eq!(
        median(0, 2).verdict(&[9.0, 9.0], &others),
        Verdict::Continue
    );
    assert!(median(0, 1).verdict(&[9.0, 9.0], &others).is_prune());
}

#[test]
fn it_compares_the_best_so_far_and_not_the_latest() {
    // A run that already touched a good number has shown it can, and one bad
    // epoch is noise. Comparing the latest value would prune this.
    let others = vec![vec![5.0, 5.0], vec![5.0, 5.0], vec![5.0, 5.0]];

    assert_eq!(
        median(0, 1).verdict(&[1.0, 9.0], &others),
        Verdict::Continue
    );
}

#[test]
fn maximizing_is_the_same_thing_read_from_the_other_end() {
    let up = Percentile::median(Goal::Maximize, 0, 1);
    let flipped: Vec<Vec<f64>> = four()
        .iter()
        .map(|c| c.iter().map(|v| -v).collect())
        .collect();

    assert_eq!(
        up.verdict(&[-5.0], &flipped),
        Verdict::Prune(Reason::Worse { than: -2.5, at: 0 })
    );
    assert_eq!(up.verdict(&[-2.0], &flipped), Verdict::Continue);
}

#[test]
fn a_smaller_percentile_prunes_more_because_p_is_what_is_kept() {
    // The way round that is easy to get backwards, and optuna's: `p` is the
    // share that survives. Others at 1, 2, 3, 4.
    let with = |p| Percentile {
        p,
        goal: Goal::Minimize,
        warmup: 0,
        startup: 1,
    };

    // Keeping the best only: a 2.0 is behind the bar of 1.0 and goes.
    assert!(with(0.0).verdict(&[2.0], &four()).is_prune());
    // Keeping the better half: a 2.0 clears a bar of 2.5 and stays.
    assert_eq!(with(50.0).verdict(&[2.0], &four()), Verdict::Continue);
    // Keeping everything: only what is behind all four goes.
    assert_eq!(with(100.0).verdict(&[3.5], &four()), Verdict::Continue);
    assert!(with(100.0).verdict(&[4.5], &four()).is_prune());
}

#[test]
fn what_diverged_goes_even_inside_the_warmup() {
    // A NaN loss does not recover, and the epochs spent finding that out are
    // the cheapest a pruner can save.
    assert_eq!(
        median(10, 100).verdict(&[f64::NAN], &[]),
        Verdict::Prune(Reason::NotANumber { at: 0 })
    );
}
