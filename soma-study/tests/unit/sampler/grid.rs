//! Every combination, and then nothing.

use somatize_study::{Dimension, Grid, Setting, Space};

fn space() -> Space {
    Space::new()
        .with(
            "lr",
            Dimension::Real {
                low: 0.0,
                high: 1.0,
                log: false,
            },
        )
        .unwrap()
        .with("opt", Dimension::Choice(vec!["adam".into(), "sgd".into()]))
        .unwrap()
}

#[test]
fn how_many_combinations_there_are_is_askable_before_starting() {
    // Which is how many trials a grid search *is*, and a caller wants it before
    // it opens the loop.
    assert_eq!(Grid { steps: 3 }.total(&space()), 6);
    assert_eq!(Grid { steps: 3 }.total(&Space::new()), 0);
}

#[test]
fn it_walks_every_combination_exactly_once() {
    let grid = Grid { steps: 3 };
    let space = space();

    let walked: Vec<String> = (0..grid.total(&space))
        .filter_map(|trial| grid.ask(&space, trial, &[]))
        .map(|point| point.to_string())
        .collect();

    assert_eq!(walked.len(), 6);
    assert_eq!(
        walked
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        6
    );
}

#[test]
fn it_runs_out_which_is_how_a_for_knows_to_stop_without_being_told_a_number() {
    let grid = Grid { steps: 3 };

    assert!(grid.ask(&space(), 5, &[]).is_some());
    assert_eq!(grid.ask(&space(), 6, &[]), None);
    assert_eq!(grid.ask(&Space::new(), 0, &[]), None);
}

#[test]
fn both_ends_of_a_range_are_taken_and_not_just_the_inside() {
    // Off-by-one here is a grid that never tries the highest learning rate.
    let grid = Grid { steps: 3 };
    let space = space();

    assert_eq!(
        grid.ask(&space, 0, &[]).unwrap().get("lr"),
        Some(&Setting::Real(0.0))
    );
    assert_eq!(
        grid.ask(&space, 2, &[]).unwrap().get("lr"),
        Some(&Setting::Real(1.0))
    );
}

#[test]
fn the_first_knob_varies_fastest() {
    // Worth knowing when a grid is stopped early: consecutive trials differ in
    // the knob declared first.
    let grid = Grid { steps: 3 };
    let space = space();
    let opt = |trial| grid.ask(&space, trial, &[]).unwrap().get("opt").cloned();

    assert_eq!(opt(0), opt(1));
    assert_eq!(opt(1), opt(2));
    assert_ne!(opt(2), opt(3));
}

#[test]
fn a_narrow_int_is_taken_whole_rather_than_cut_into_steps() {
    let space = Space::new()
        .with("layers", Dimension::Int { low: 1, high: 3 })
        .unwrap();

    assert_eq!(Grid { steps: 10 }.total(&space), 3);
    assert_eq!(
        (0..3)
            .map(|t| Grid { steps: 10 }
                .ask(&space, t, &[])
                .unwrap()
                .get("layers")
                .cloned())
            .collect::<Vec<_>>(),
        vec![
            Some(Setting::Int(1)),
            Some(Setting::Int(2)),
            Some(Setting::Int(3))
        ]
    );
}

#[test]
fn it_looks_at_neither_a_seed_nor_the_finished_trials() {
    let grid = Grid { steps: 3 };
    let space = space();
    let finished = vec![(grid.ask(&space, 0, &[]).unwrap(), Some(0.5))];

    assert_eq!(grid.ask(&space, 4, &[]), grid.ask(&space, 4, &finished));
}
