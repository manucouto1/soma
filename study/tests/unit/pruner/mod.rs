//! The family: what the enum adds over calling a scheme directly.
//!
//! Which is, deliberately, nothing at all as far as the verdict goes.

mod patience;
mod percentile;
mod threshold;

use soma_next_study::{Goal, Patience, Percentile, Pruner, Reason, Threshold, Verdict};

fn one_of_each() -> Vec<Pruner> {
    vec![
        Percentile::median(Goal::Minimize, 0, 1).into(),
        Threshold::diverged().into(),
        Patience {
            steps: 3.try_into().unwrap(),
            min_delta: 0.0,
            goal: Goal::Minimize,
        }
        .into(),
    ]
}

#[test]
fn going_through_the_enum_judges_exactly_the_same_as_not_going_through_it() {
    let mine = [5.0, 5.0, 5.0];
    let others = vec![vec![1.0; 3], vec![2.0; 3], vec![3.0; 3]];

    let rule = Percentile::median(Goal::Minimize, 0, 1);
    assert_eq!(
        Pruner::from(rule.clone()).verdict(&mine, &others),
        rule.verdict(&mine, &others)
    );

    let rule = Threshold {
        lower: None,
        upper: Some(1.0),
    };
    assert_eq!(
        Pruner::from(rule.clone()).verdict(&mine, &others),
        rule.verdict(&mine, &others)
    );

    let rule = Patience {
        steps: 2.try_into().unwrap(),
        min_delta: 0.0,
        goal: Goal::Minimize,
    };
    assert_eq!(
        Pruner::from(rule.clone()).verdict(&mine, &others),
        rule.verdict(&mine, &others)
    );
}

#[test]
fn the_three_compare_against_three_different_things() {
    // The whole reason there are three and not one with options: the field, a
    // constant, and the trial itself. Same trial, three verdicts.
    let mine = [1.0, 1.0, 1.0];
    let field = vec![vec![100.0; 3]; 5];

    // Doing far better than everyone: the field has no complaint.
    assert_eq!(
        Percentile::median(Goal::Minimize, 0, 1).verdict(&mine, &field),
        Verdict::Continue
    );
    // Under any bound worth declaring: neither has the constant.
    assert_eq!(
        Threshold {
            lower: None,
            upper: Some(10.0)
        }
        .verdict(&mine, &field),
        Verdict::Continue
    );
    // And it has not moved in three reports, which only it can see.
    assert!(
        Patience {
            steps: 2.try_into().unwrap(),
            min_delta: 0.0,
            goal: Goal::Minimize,
        }
        .verdict(&mine, &field)
        .is_prune()
    );
}

#[test]
fn a_verdict_says_whether_and_why_without_being_taken_apart() {
    let carry_on = Verdict::Continue;
    assert!(!carry_on.is_prune());
    assert_eq!(carry_on.reason(), None);

    let dropped = Verdict::Prune(Reason::NotANumber { at: 3 });
    assert!(dropped.is_prune());
    assert_eq!(dropped.reason(), Some(&Reason::NotANumber { at: 3 }));
}

#[test]
fn every_reason_says_enough_to_act_on_without_the_curve_in_front_of_you() {
    let said = |why: Reason| why.to_string();

    assert!(said(Reason::NotANumber { at: 3 }).contains("diverged"));
    assert!(said(Reason::Worse { than: 2.5, at: 4 }).contains("2.5"));
    assert!(
        said(Reason::OutOfBounds {
            value: 11.0,
            bound: 10.0
        })
        .contains("11")
    );
    assert!(said(Reason::NotImproving { since: 1, steps: 2 }).contains("report 1"));
}

#[test]
fn it_writes_itself_down_so_the_record_of_a_run_says_what_pruned_it() {
    assert_eq!(
        Pruner::from(Percentile::median(Goal::Minimize, 2, 5)).to_string(),
        "percentile:50:min:warmup:2:startup:5"
    );
    assert_eq!(
        Pruner::from(Threshold::diverged()).to_string(),
        "threshold:lower:none:upper:none"
    );
    assert_eq!(
        Pruner::from(Threshold {
            lower: Some(0.0),
            upper: Some(10.0)
        })
        .to_string(),
        "threshold:lower:0:upper:10"
    );
    assert_eq!(
        Pruner::from(Patience {
            steps: 3.try_into().unwrap(),
            min_delta: 0.01,
            goal: Goal::Maximize,
        })
        .to_string(),
        "patience:3:delta:0.01:max"
    );

    // And a scheme writes itself the same whether or not it is wrapped.
    let rule = Percentile::median(Goal::Maximize, 1, 1);
    assert_eq!(rule.to_string(), Pruner::from(rule.clone()).to_string());
}

#[test]
fn two_pruners_that_differ_are_written_down_differently() {
    let mut rules = one_of_each();
    rules.push(Percentile::median(Goal::Maximize, 0, 1).into());
    rules.push(Percentile::median(Goal::Minimize, 0, 2).into());
    rules.push(
        Percentile {
            p: 25.0,
            goal: Goal::Minimize,
            warmup: 0,
            startup: 1,
        }
        .into(),
    );

    let names: std::collections::BTreeSet<String> = rules.iter().map(|r| r.to_string()).collect();
    assert_eq!(names.len(), rules.len());
}
