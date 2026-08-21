//! Judged against itself: it is not going anywhere.

use soma_next_study::{Goal, Patience, Reason, Verdict};

fn patience(steps: usize, min_delta: f64, goal: Goal) -> Patience {
    Patience {
        steps: steps
            .try_into()
            .expect("a patience of zero cannot be written"),
        min_delta,
        goal,
    }
}

#[test]
fn while_it_keeps_improving_it_carries_on() {
    let rule = patience(2, 0.0, Goal::Minimize);

    assert_eq!(rule.verdict(&[5.0, 4.0, 3.0, 2.0], &[]), Verdict::Continue);
}

#[test]
fn when_the_best_stops_moving_it_goes_and_says_where_the_best_was() {
    let rule = patience(2, 0.0, Goal::Minimize);

    // Best at report 1, then two reports that do not beat it.
    assert_eq!(
        rule.verdict(&[5.0, 2.0, 3.0, 4.0], &[]),
        Verdict::Prune(Reason::NotImproving { since: 1, steps: 2 })
    );
    // One report short of the allowance.
    assert_eq!(rule.verdict(&[5.0, 2.0, 3.0], &[]), Verdict::Continue);
}

#[test]
fn an_improvement_resets_the_count() {
    let rule = patience(2, 0.0, Goal::Minimize);

    assert_eq!(rule.verdict(&[5.0, 5.0, 4.0], &[]), Verdict::Continue);
}

#[test]
fn a_delta_stops_noise_from_looking_like_progress() {
    // Without it, a run creeping down by 0.001 an epoch looks alive forever.
    let creeping = [5.0, 4.999, 4.998, 4.997];

    assert_eq!(
        patience(2, 0.0, Goal::Minimize).verdict(&creeping, &[]),
        Verdict::Continue
    );
    assert!(
        patience(2, 0.01, Goal::Minimize)
            .verdict(&creeping, &[])
            .is_prune()
    );
}

#[test]
fn maximizing_is_the_same_thing_the_other_way_up() {
    let rule = patience(2, 0.0, Goal::Maximize);

    assert_eq!(rule.verdict(&[0.1, 0.5, 0.9], &[]), Verdict::Continue);
    assert_eq!(
        rule.verdict(&[0.1, 0.9, 0.5, 0.4], &[]),
        Verdict::Prune(Reason::NotImproving { since: 1, steps: 2 })
    );
}

#[test]
fn it_needs_no_other_trial_and_prunes_what_the_others_would_not() {
    // A run doing perfectly well against the field, and simply going nowhere.
    let rule = patience(2, 0.0, Goal::Minimize);
    let excellent = vec![vec![100.0; 4]; 5];

    assert!(rule.verdict(&[1.0, 1.0, 1.0], &excellent).is_prune());
}

#[test]
fn what_diverged_goes_here_too() {
    assert_eq!(
        patience(50, 0.0, Goal::Minimize).verdict(&[1.0, f64::NAN], &[]),
        Verdict::Prune(Reason::NotANumber { at: 1 })
    );
}

#[test]
fn nothing_is_judged_before_its_first_report() {
    assert_eq!(
        patience(1, 0.0, Goal::Minimize).verdict(&[], &[]),
        Verdict::Continue
    );
}
