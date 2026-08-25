//! Which way is better, which no number says on its own.

use somatize_study::{Goal, GoalError};

#[test]
fn better_is_strict_so_a_tie_is_never_worse() {
    // It matters: a trial that ties the bar is not pruned for tying.
    assert!(Goal::Minimize.better(1.0, 2.0));
    assert!(!Goal::Minimize.better(2.0, 1.0));
    assert!(!Goal::Minimize.better(1.0, 1.0));

    assert!(Goal::Maximize.better(2.0, 1.0));
    assert!(!Goal::Maximize.better(1.0, 2.0));
    assert!(!Goal::Maximize.better(1.0, 1.0));
}

#[test]
fn the_best_of_them_goes_the_way_the_goal_says() {
    let values = [3.0, 1.0, 2.0];

    assert_eq!(Goal::Minimize.best_of(&values), Some(1.0));
    assert_eq!(Goal::Maximize.best_of(&values), Some(3.0));
    assert_eq!(Goal::Minimize.best_of(&[]), None);
}

#[test]
fn what_is_not_a_number_is_skipped_because_it_compares_to_nothing() {
    assert_eq!(Goal::Minimize.best_of(&[3.0, f64::NAN, 1.0]), Some(1.0));
    assert_eq!(Goal::Maximize.best_of(&[f64::NAN]), None);
}

#[test]
fn it_says_which_way_it_goes() {
    assert_eq!(Goal::Minimize.to_string(), "min");
    assert_eq!(Goal::Maximize.to_string(), "max");
}

#[test]
fn it_is_read_back_from_what_it_wrote() {
    for goal in [Goal::Minimize, Goal::Maximize] {
        assert_eq!(goal.to_string().parse(), Ok(goal));
    }
}

#[test]
fn a_typo_is_caught_where_it_was_typed_and_not_as_a_search_that_ran_backwards() {
    assert_eq!(
        "minimise".parse::<Goal>(),
        Err(GoalError::Unknown("minimise".into()))
    );
    assert!(
        GoalError::Unknown("loss".into())
            .to_string()
            .contains("`min`")
    );
}
